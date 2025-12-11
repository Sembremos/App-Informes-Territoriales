import re
import unicodedata
from typing import Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF


# -------------------------
# Normalización / parsing
# -------------------------
def norm_txt(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s


def pct_to_float(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip().replace("%", "").replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None


PCT_IN_TEXT = re.compile(r"(\d{1,3}[.,]\d{1,2})\s*%")
PCT_ALONE = re.compile(r"^\s*(\d{1,3}(?:[.,]\d{1,2})?)\s*%?\s*$")


# -------------------------
# PDF spans
# -------------------------
def extract_spans(page: fitz.Page) -> List[Dict]:
    d = page.get_text("dict")
    spans: List[Dict] = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for sp in line.get("spans", []):
                text = (sp.get("text") or "").strip()
                if not text:
                    continue
                x0, y0, x1, y1 = sp.get("bbox", (0, 0, 0, 0))
                spans.append(
                    {
                        "text": text,
                        "ntext": norm_txt(text),
                        "x": (x0 + x1) / 2.0,
                        "y": (y0 + y1) / 2.0,
                    }
                )
    return spans


def list_pct_points(spans: List[Dict]) -> List[Tuple[float, float, float]]:
    """(valor, x, y)"""
    out: List[Tuple[float, float, float]] = []
    for sp in spans:
        t = sp["text"].strip()

        m = PCT_IN_TEXT.search(t)
        if m:
            v = pct_to_float(m.group(1))
            if v is not None and 0 <= v <= 100:
                out.append((v, sp["x"], sp["y"]))
            continue

        m2 = PCT_ALONE.match(t)
        if m2:
            v = pct_to_float(m2.group(1))
            if v is not None and 0 <= v <= 100:
                out.append((v, sp["x"], sp["y"]))
    return out


def find_y(spans: List[Dict], variants: List[str]) -> Optional[float]:
    vv = [norm_txt(v) for v in variants]
    ys = [sp["y"] for sp in spans if any(v in sp["ntext"] for v in vv)]
    return min(ys) if ys else None


def slice_block_until_fuente(spans: List[Dict], start_variants: List[str]) -> List[Dict]:
    y_start = find_y(spans, start_variants)
    if y_start is None:
        return []

    fuente_y = None
    for sp in sorted(spans, key=lambda z: z["y"]):
        if "fuente:" in sp["ntext"] and sp["y"] > y_start:
            fuente_y = sp["y"]
            break

    y_end = fuente_y if fuente_y is not None else (max(sp["y"] for sp in spans) + 1)
    return [sp for sp in spans if y_start <= sp["y"] <= y_end]


def unique_top(vals: List[float], k: int, tol: float = 0.2) -> List[float]:
    out: List[float] = []
    for v in sorted(vals, reverse=True):
        if all(abs(v - u) > tol for u in out):
            out.append(v)
        if len(out) >= k:
            break
    return out


def extract_delegacion(doc: fitz.Document) -> str:
    for i in range(min(6, doc.page_count)):
        spans = extract_spans(doc[i])
        joined = " ".join(sp["text"] for sp in spans)
        m = re.search(r"\bD\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ\s]+", joined)
        if m:
            return " ".join(m.group(0).split()).strip()
    return "SIN_DELEGACIÓN"


# -------------------------
# Barras: match por etiqueta usando SOLO X
# -------------------------
def find_label_center_exact(right_block: List[Dict], patterns: List[str]) -> Optional[Tuple[float, float]]:
    """
    Encuentra la etiqueta por patrón EXACTO (normalizado) y toma la más baja (mayor y).
    patterns ya vienen normalizados (sin tildes).
    """
    cands = []
    for sp in right_block:
        t = sp["ntext"]
        for p in patterns:
            # match exacto de frase completa
            if t == p or (p in t and len(t) <= len(p) + 2):
                cands.append((sp["x"], sp["y"]))
                break
    if not cands:
        return None
    return sorted(cands, key=lambda z: z[1], reverse=True)[0]


def solve_assignment_min_dx(
    labels: List[Tuple[str, Tuple[float, float]]],
    pct_points: List[Tuple[float, float, float]],
    base_y: float,
    window_up: float = 320.0,
    max_dx: float = 260.0
) -> Dict[str, Optional[float]]:
    """
    Candidatos = % arriba de base_y (zona barras).
    Asignación 1-a-1 minimizando |px-lx| (solo X) para evitar el swap.
    """
    # filtrar candidatos por ventana vertical
    cand = [(v, x, y) for (v, x, y) in pct_points if (base_y - 15) > y >= (base_y - window_up)]
    if not cand:
        return {col: None for col, _ in labels}

    # quitar duplicados por valor (si el PDF repite el mismo % en varios spans)
    uniq = []
    for v, x, y in sorted(cand, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in uniq):
            uniq.append((v, x, y))

    # quedarnos con los 6 más grandes por valor para reducir ruido
    if len(uniq) > 6:
        uniq = sorted(uniq, key=lambda t: t[0], reverse=True)[:6]

    # probamos todas las asignaciones posibles (3! máximo)
    best = None  # (cost, mapping)
    from itertools import permutations

    cols = [c for c, _ in labels]
    lxs = {c: lx for c, (lx, ly) in labels}

    for perm in permutations(range(len(uniq)), r=len(labels)):
        used = set()
        cost = 0.0
        ok = True
        mapping = {}
        for col, idx in zip(cols, perm):
            if idx in used:
                ok = False
                break
            used.add(idx)
            v, px, py = uniq[idx]
            dx = abs(px - lxs[col])
            if dx > max_dx:
                ok = False
                break
            cost += dx
            mapping[col] = v

        if ok and (best is None or cost < best[0]):
            best = (cost, mapping)

    if best is None:
        # fallback: por cada etiqueta, el más cercano en X aunque repita
        out = {}
        for col, (lx, ly) in labels:
            best_dx = None
            best_v = None
            for v, px, py in uniq:
                dx = abs(px - lx)
                if dx <= max_dx and (best_dx is None or dx < best_dx):
                    best_dx = dx
                    best_v = v
            out[col] = best_v
        return out

    return {c: best[1].get(c) for c in cols}


# -------------------------
# Extracción Percepción ciudadana
# -------------------------
def extract_percepcion_ciudadana(doc: fitz.Document) -> Dict[str, Optional[str]]:
    out = {
        "Archivo": None,
        "Delegación": None,
        "Seguro en la comunidad (%)": None,
        "Inseguro en la comunidad (%)": None,
        "Comparación 2023 - Igual (%)": None,
        "Comparación 2023 - Menos seguro (%)": None,
        "Comparación 2023 - Más seguro (%)": None,
    }

    anchors = ["¿se siente de seguro en su comunidad", "percepcion ciudadana"]

    for pno in range(doc.page_count):
        page = doc[pno]
        spans = extract_spans(page)

        block = slice_block_until_fuente(spans, anchors)
        if not block:
            continue

        out["Delegación"] = extract_delegacion(doc)

        w = page.rect.width
        midx = w * 0.52
        left = [sp for sp in block if sp["x"] < midx]
        right = [sp for sp in block if sp["x"] >= midx]

        # PIE (izquierda)
        left_vals = [v for (v, _, _) in list_pct_points(left)]
        two = unique_top([v for v in left_vals if 0 <= v <= 100], 2)
        if len(two) == 2:
            out["Seguro en la comunidad (%)"] = f"{min(two):.2f}%"
            out["Inseguro en la comunidad (%)"] = f"{max(two):.2f}%"

        # BARRAS (derecha) por etiqueta, SIN depender del orden
        # patrones normalizados (sin tildes)
        p_igual = find_label_center_exact(right, ["igual"])
        p_menos = find_label_center_exact(right, ["menos seguro", "menosseguro"])
        p_mas = find_label_center_exact(right, ["mas seguro", "masseguro"])

        labels = []
        if p_igual:
            labels.append(("Comparación 2023 - Igual (%)", p_igual))
        if p_menos:
            labels.append(("Comparación 2023 - Menos seguro (%)", p_menos))
        if p_mas:
            labels.append(("Comparación 2023 - Más seguro (%)", p_mas))

        if len(labels) >= 2:
            # base_y: usamos el y más bajo de etiquetas
            base_y = max(ly for _, (lx, ly) in labels)
            pct_points = list_pct_points(right)
            matched = solve_assignment_min_dx(labels, pct_points, base_y)

            # si faltara 1 valor y tenemos 3 etiquetas, completar con el restante de los 3 principales
            if len(labels) == 3 and sum(v is not None for v in matched.values()) == 2:
                all_vals = [v for (v, _, _) in pct_points]
                top3 = unique_top(all_vals, 3)
                got = [v for v in matched.values() if v is not None]
                remaining = [v for v in top3 if all(abs(v - g) > 0.2 for g in got)]
                if remaining:
                    for k in matched:
                        if matched[k] is None:
                            matched[k] = remaining[0]

            for col, val in matched.items():
                if val is not None:
                    out[col] = f"{val:.2f}%"

        return out

    return out


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Percepción ciudadana — MPGP", layout="wide")
st.title("Percepción ciudadana — Extracción (Paso 1)")

files = st.file_uploader("Suba informes PDF", type="pdf", accept_multiple_files=True)

if files:
    rows = []
    for f in files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        row = extract_percepcion_ciudadana(doc)
        doc.close()

        row["Archivo"] = f.name
        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=[
            "Archivo",
            "Delegación",
            "Seguro en la comunidad (%)",
            "Inseguro en la comunidad (%)",
            "Comparación 2023 - Igual (%)",
            "Comparación 2023 - Menos seguro (%)",
            "Comparación 2023 - Más seguro (%)",
        ],
    )

    st.subheader("Resultados (tabla)")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "percepcion_ciudadana.csv",
        "text/csv",
    )
else:
    st.info("Cargue uno o más PDFs.")
