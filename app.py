import re
import unicodedata
from typing import Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF


# -------------------------
# Normalización y parsing
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
                spans.append({
                    "text": text,
                    "ntext": norm_txt(text),
                    "x": (x0 + x1) / 2.0,
                    "y": (y0 + y1) / 2.0,
                })
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
    """Bloque = desde ancla hasta el primer 'Fuente:' posterior."""
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
# Matching de barras (robusto)
# -------------------------
def find_label_center_multi(right_block: List[Dict], variants: List[str]) -> Optional[Tuple[float, float]]:
    vv = [norm_txt(v) for v in variants]
    candidates = []
    for sp in right_block:
        if any(v in sp["ntext"] for v in vv):
            candidates.append((sp["x"], sp["y"]))
    if not candidates:
        return None
    # tomar el más abajo (etiqueta suele estar abajo del gráfico)
    return sorted(candidates, key=lambda t: t[1], reverse=True)[0]


def match_bars_by_nearest_x(
    right_block: List[Dict],
    labels: List[Tuple[str, Tuple[float, float]]],  # (col_name, (lx, ly))
    window_up: float = 300.0,
    max_dx: float = 240.0
) -> Dict[str, Optional[float]]:
    """
    Toma todos los % cercanos verticalmente arriba de la zona de etiquetas y
    los asigna a la etiqueta más cercana en X.
    """
    pct_points = list_pct_points(right_block)
    if not pct_points:
        return {col: None for col, _ in labels}

    max_label_y = max(ly for _, (lx, ly) in labels)

    candidates = []
    for val, px, py in pct_points:
        if py < max_label_y and py >= max_label_y - window_up:
            candidates.append((val, px, py))

    best_for = {col: None for col, _ in labels}  # col -> (score, val)
    for val, px, py in candidates:
        nearest = None
        for col, (lx, ly) in labels:
            dx = abs(px - lx)
            if dx <= max_dx:
                dy = abs(ly - py)
                score = dx * 1.0 + dy * 0.15
                if nearest is None or score < nearest[0]:
                    nearest = (score, col)
        if nearest is None:
            continue

        score, col = nearest
        cur = best_for[col]
        if cur is None or score < cur[0]:
            best_for[col] = (score, val)

    return {col: (best_for[col][1] if best_for[col] else None) for col in best_for}


# -------------------------
# Extracción Percepción ciudadana (solo)
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

        # ---- PIE (izquierda): 2 valores
        left_vals = [v for (v, _, _) in list_pct_points(left)]
        two = unique_top([v for v in left_vals if 0 <= v <= 100], 2)
        if len(two) == 2:
            seguro = min(two)
            inseguro = max(two)
            out["Seguro en la comunidad (%)"] = f"{seguro:.2f}%"
            out["Inseguro en la comunidad (%)"] = f"{inseguro:.2f}%"

        # ---- BARRAS (derecha): Igual / Menos / Más
        LABEL_IGUAL = ["igual"]
        LABEL_MENOS = ["menos seguro", "menosseguro"]
        LABEL_MAS = ["más seguro", "mas seguro", "másseguro", "masseguro", "más seguro", "mas seguro"]

        p_igual = find_label_center_multi(right, LABEL_IGUAL)
        p_menos = find_label_center_multi(right, LABEL_MENOS)
        p_mas = find_label_center_multi(right, LABEL_MAS)

        labels = []
        if p_igual:
            labels.append(("Comparación 2023 - Igual (%)", p_igual))
        if p_menos:
            labels.append(("Comparación 2023 - Menos seguro (%)", p_menos))
        if p_mas:
            labels.append(("Comparación 2023 - Más seguro (%)", p_mas))

        matched = {}
        if len(labels) >= 2:
            matched = match_bars_by_nearest_x(right, labels, window_up=320.0, max_dx=260.0)

            # Fallback: si falta exactamente 1 valor, usar el % restante más bajo (de los candidatos de barras)
            got = [v for v in matched.values() if v is not None]
            if len(got) == 2 and len(labels) == 3:
                all_bar_vals = [v for (v, _, _) in list_pct_points(right)]
                # quitar duplicados cercanos y quitar los ya tomados
                uniq = []
                for v in sorted(all_bar_vals, reverse=True):
                    if all(abs(v - u) > 0.2 for u in uniq):
                        uniq.append(v)

                remaining = [v for v in uniq if all(abs(v - g) > 0.2 for g in got)]
                if remaining:
                    missing_val = min(remaining)
                    for k in matched:
                        if matched[k] is None:
                            matched[k] = missing_val

            # escribir a salida
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
    prog = st.progress(0)
    status = st.empty()

    for i, f in enumerate(files, start=1):
        status.write(f"Procesando: **{f.name}** ({i}/{len(files)})")
        doc = fitz.open(stream=f.read(), filetype="pdf")
        row = extract_percepcion_ciudadana(doc)
        doc.close()

        row["Archivo"] = f.name
        rows.append(row)
        prog.progress(int(i / len(files) * 100))

    prog.empty()
    status.empty()

    df = pd.DataFrame(rows, columns=[
        "Archivo",
        "Delegación",
        "Seguro en la comunidad (%)",
        "Inseguro en la comunidad (%)",
        "Comparación 2023 - Igual (%)",
        "Comparación 2023 - Menos seguro (%)",
        "Comparación 2023 - Más seguro (%)",
    ])

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
