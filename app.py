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


def label_y_baseline(right_block: List[Dict]) -> Optional[float]:
    """
    Usa la y de las etiquetas (Igual/Menos/Más) para definir dónde están las barras arriba.
    """
    keys = ["igual", "menos seguro", "mas seguro", "más seguro"]
    ys = []
    for sp in right_block:
        if any(k in sp["ntext"] for k in keys):
            ys.append(sp["y"])
    return max(ys) if ys else None


def pick_bar_percentages_by_x(right_block: List[Dict]) -> List[Tuple[float, float, float]]:
    """
    Devuelve los 3 porcentajes de las barras, ordenados por X (izq->der),
    usando una ventana vertical arriba de las etiquetas.
    """
    base_y = label_y_baseline(right_block)
    if base_y is None:
        return []

    pcts = list_pct_points(right_block)

    # Solo porcentajes arriba de la línea de etiquetas y dentro de una ventana razonable
    window_up = 320.0
    cand = [(v, x, y) for (v, x, y) in pcts if (base_y - window_up) <= y < base_y]

    # Quitar duplicados cercanos por valor, quedándonos con el más “alto” (más arriba) si repite
    cand_sorted = sorted(cand, key=lambda t: (t[0], t[2]))  # por valor, luego por y (arriba primero)
    uniq: List[Tuple[float, float, float]] = []
    for v, x, y in cand_sorted:
        if all(abs(v - u[0]) > 0.2 for u in uniq):
            uniq.append((v, x, y))

    # Ahora tomamos los 3 que estén más “centrados” en la zona de barras:
    # en la práctica, ya quedan 3 (42.86, 41.56, 15.58). Si quedaran más, tomamos los 3 más grandes.
    if len(uniq) > 3:
        uniq = sorted(uniq, key=lambda t: t[0], reverse=True)[:3]

    # Orden final por posición X (izq->der) = Igual, Menos, Más
    uniq = sorted(uniq, key=lambda t: t[1])
    return uniq


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

        # BARRAS (derecha): 3 valores por X
        bars = pick_bar_percentages_by_x(right)
        if len(bars) == 3:
            v_igual, _, _ = bars[0]
            v_menos, _, _ = bars[1]
            v_mas, _, _ = bars[2]
            out["Comparación 2023 - Igual (%)"] = f"{v_igual:.2f}%"
            out["Comparación 2023 - Menos seguro (%)"] = f"{v_menos:.2f}%"
            out["Comparación 2023 - Más seguro (%)"] = f"{v_mas:.2f}%"

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
