import re
import unicodedata
from typing import Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF


# -------------------------
# Helpers
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
    s = s.replace("%", "").replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None


PCT_RE = re.compile(r"(\d{1,3}[.,]\d{1,2})\s*%")


def extract_spans(page: fitz.Page):
    d = page.get_text("dict")
    spans = []
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
                    "x": (x0 + x1) / 2,
                    "y": (y0 + y1) / 2,
                })
    return spans


def list_percentages(spans):
    vals = []
    for sp in spans:
        m = PCT_RE.search(sp["text"])
        if m:
            v = pct_to_float(m.group(1))
            if v is not None and 0 <= v <= 100:
                vals.append((v, sp["x"], sp["y"]))
    return vals


def unique_top(vals, k=3, tol=0.2):
    out = []
    for v in sorted(vals, reverse=True):
        if all(abs(v - u) > tol for u in out):
            out.append(v)
        if len(out) == k:
            break
    return out


def extract_delegacion(doc):
    for i in range(min(5, doc.page_count)):
        spans = extract_spans(doc[i])
        text = " ".join(sp["text"] for sp in spans)
        m = re.search(r"D\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ\s]+", text)
        if m:
            return " ".join(m.group(0).split())
    return "SIN_DELEGACIÓN"


# -------------------------
# Percepción ciudadana (FINAL)
# -------------------------
def extract_percepcion_ciudadana(doc: fitz.Document) -> Dict[str, Optional[float]]:
    for page in doc:
        spans = extract_spans(page)

        # buscar ancla
        if not any("percepcion ciudadana" in sp["ntext"] for sp in spans):
            continue

        w = page.rect.width
        midx = w * 0.52

        left = [sp for sp in spans if sp["x"] < midx]
        right = [sp for sp in spans if sp["x"] >= midx]

        # PIE (izquierda)
        pie_vals = [v for (v, _, _) in list_percentages(left)]
        pie = unique_top(pie_vals, 2)
        if len(pie) != 2:
            return {}

        perc_inseguro = max(pie)
        perc_seguro = min(pie)

        # BARRAS (derecha)
        bar_vals = [v for (v, _, _) in list_percentages(right)]
        bars = unique_top(bar_vals, 3)
        if len(bars) != 3:
            return {}

        # orden visual izquierda → derecha
        bars_sorted = sorted(
            bars,
            key=lambda v: min(abs(x - midx) for (vv, x, _) in list_percentages(right) if vv == v)
        )

        return {
            "Delegación": extract_delegacion(doc),
            "Seguro en la comunidad (%)": f"{perc_seguro:.2f}%",
            "Inseguro en la comunidad (%)": f"{perc_inseguro:.2f}%",
            "Comparación 2023 - Igual (%)": f"{bars_sorted[0]:.2f}%",
            "Comparación 2023 - Menos seguro (%)": f"{bars_sorted[1]:.2f}%",
            "Comparación 2023 - Más seguro (%)": f"{bars_sorted[2]:.2f}%",
        }

    return {}


# -------------------------
# UI
# -------------------------
st.set_page_config(layout="wide")
st.title("Percepción ciudadana – Validación de datos")

files = st.file_uploader("Suba informes PDF", type="pdf", accept_multiple_files=True)

if files:
    rows = []
    for f in files:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        row = extract_percepcion_ciudadana(doc)
        doc.close()
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "percepcion_ciudadana.csv",
        "text/csv",
    )
else:
    st.info("Cargue uno o más PDFs.")
