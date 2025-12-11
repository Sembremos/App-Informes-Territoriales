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
# Labels + matching robusto
# -------------------------
def find_label_center(right_block: List[Dict], variants: List[str]) -> Optional[Tuple[float, float]]:
    vv = [norm_txt(v) for v in variants]
    candidates = []
    for sp in right_block:
        if any(v == sp["ntext"] or v in sp["ntext"] for v in vv):
            candidates.append((sp["x"], sp["y"]))
    if not candidates:
        return None
    # etiqueta suele estar abajo: tomar la más baja (mayor y)
    return sorted(candidates, key=lambda t: t[1], reverse=True)[0]


def bar_candidates_for_label(
    pct_points: List[Tuple[float, float, float]],
    lx: float,
    ly: float,
    window_up: float = 320.0,
    max_dx: float = 220.0,
) -> List[Tuple[float, int]]:
    """
    Devuelve lista de (costo, idx_pct) para candidatos arriba de la etiqueta.
    Costo mezcla dx y dy para preferir el % "encima" de la barra.
    """
    cands = []
    for idx, (val, px, py) in enumerate(pct_points):
        if not (ly - window_up <= py < ly):
            continue
        dx = abs(px - lx)
        if dx > max_dx:
            continue
        dy = abs(ly - py)
        cost = dx * 1.0 + dy * 0.15
        cands.append((cost, idx))
    # mejores primero
    cands.sort(key=lambda t: t[0])
    return cands


def solve_best_assignment(
    pct_points: List[Tuple[float, float, float]],
    labels: List[Tuple[str, Tuple[float, float]]],
) -> Dict[str, Optional[float]]:
    """
    Asignación 1-a-1: cada etiqueta obtiene un porcentaje distinto.
    Como solo hay 3 etiquetas, probamos combinaciones de los top candidatos.
    """
    # candidatos por etiqueta (top 6 para limitar búsqueda)
    cand_lists = []
    for col, (lx, ly) in labels:
        c = bar_candidates_for_label(pct_points, lx, ly)
        cand_lists.append((col, c[:6]))

    best_total = None
    best_map: Dict[str, Optional[int]] = {col: None for col, _ in labels}

    # backtracking pequeño (3 etiquetas)
    used = set()

    def dfs(i: int, total_cost: float, cur_map: Dict[str, Optional[int]]):
        nonlocal best_total, best_map
        if i == len(cand_lists):
            if best_total is None or total_cost < best_total:
                best_total = total_cost
                best_map = cur_map.copy()
            return

        col, cands = cand_lists[i]

        # opción: no asignar (penaliza fuerte)
        # pero preferimos asignar si hay candidatos
        penalty = 9999.0
        if best_total is None or total_cost + penalty < best_total:
            cur_map[col] = None
            dfs(i + 1, total_cost + penalty, cur_map)
            cur_map.pop(col, None)

        for cost, idx_pct in cands:
            if idx_pct in used:
                continue
            if best_total is not None and total_cost + cost >= best_total:
                continue
            used.add(idx_pct)
            cur_map[col] = idx_pct
            dfs(i + 1, total_cost + cost, cur_map)
            cur_map.pop(col, None)
            used.remove(idx_pct)

    dfs(0, 0.0, {})

    # construir salida: col -> valor
    out: Dict[str, Optional[float]] = {}
    for col, _ in labels:
        idx_pct = best_map.get(col)
        out[col] = pct_points[idx_pct][0] if isinstance(idx_pct, int) else None

    return out


# -------------------------
# Percepción ciudadana (solo)
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

        # separar izquierda (pie) y derecha (barras)
        w = page.rect.width
        midx = w * 0.52
        left = [sp for sp in block if sp["x"] < midx]
        right = [sp for sp in block if sp["x"] >= midx]

        # PIE (izquierda): 2 valores
        left_vals = [v for (v, _, _) in list_pct_points(left)]
        two = unique_top([v for v in left_vals if 0 <= v <= 100], 2)
        if len(two) == 2:
            seguro = min(two)
            inseguro = max(two)
            out["Seguro en la comunidad (%)"] = f"{seguro:.2f}%"
            out["Inseguro en la comunidad (%)"] = f"{inseguro:.2f}%"

        # BARRAS (derecha): por etiqueta (orden no importa)
        p_igual = find_label_center(right, ["igual"])
        p_menos = find_label_center(right, ["menos seguro", "menosseguro"])
        p_mas = find_label_center(right, ["mas seguro", "más seguro", "masseguro", "másseguro"])

        # si falta una etiqueta, igual intentamos con las que existan
        labels = []
        if p_igual:
            labels.append(("Comparación 2023 - Igual (%)", p_igual))
        if p_menos:
            labels.append(("Comparación 2023 - Menos seguro (%)", p_menos))
        if p_mas:
            labels.append(("Comparación 2023 - Más seguro (%)", p_mas))

        pct_points = list_pct_points(right)

        if len(labels) >= 2 and pct_points:
            matched = solve_best_assignment(pct_points, labels)

            # Si queda una etiqueta sin valor, usamos fallback con el % restante (más bajo de los 3 principales)
            got = [v for v in matched.values() if v is not None]
            if len(got) == len(labels) - 1:
                all_vals = [v for (v, _, _) in pct_points]
                uniq3 = unique_top(all_vals, 3)  # los 3 del gráfico normalmente
                remaining = [v for v in uniq3 if all(abs(v - g) > 0.2 for g in got)]
                if remaining:
                    miss = remaining[0]  # normalmente queda 15.58
                    for k in matched:
                        if matched[k] is None:
                            matched[k] = miss

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
