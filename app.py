import re
import math
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
    """(valor, x, y) detectados"""
    out: List[Tuple[float, float, float]] = []
    for sp in spans:
        t = sp["text"].strip()

        m = PCT_IN_TEXT.search(t)
        if m:
            v = pct_to_float(m.group(1))
            if v is not None:
                out.append((v, sp["x"], sp["y"]))
            continue

        m2 = PCT_ALONE.match(t)
        if m2:
            v = pct_to_float(m2.group(1))
            if v is not None:
                out.append((v, sp["x"], sp["y"]))
    return out


def find_y(spans: List[Dict], variants: List[str]) -> Optional[float]:
    vv = [norm_txt(v) for v in variants]
    ys = [sp["y"] for sp in spans if any(v in sp["ntext"] for v in vv)]
    return min(ys) if ys else None


def slice_block_until_fuente(spans: List[Dict], start_variants: List[str]) -> List[Dict]:
    """Bloque = desde ancla hasta 'Fuente:' posterior."""
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


def find_label_points(spans: List[Dict], label_variants: List[str]) -> List[Tuple[float, float]]:
    vv = [norm_txt(v) for v in label_variants]
    pts: List[Tuple[float, float]] = []
    for sp in spans:
        if any(v == sp["ntext"] or v in sp["ntext"] for v in vv):
            pts.append((sp["x"], sp["y"]))
    return pts


def nearest_pct_above_label(block: List[Dict], label_variants: List[str],
                            max_dx: float = 220.0, max_dy: float = 220.0) -> Optional[float]:
    """Barras: etiqueta abajo, % arriba"""
    pts = find_label_points(block, label_variants)
    pcts = list_pct_points(block)
    if not pts or not pcts:
        return None

    best = None
    for lx, ly in pts:
        for val, px, py in pcts:
            if py >= ly:
                continue
            dx = abs(px - lx)
            dy = abs(py - ly)
            if dx <= max_dx and dy <= max_dy:
                score = dx * 1.0 + dy * 0.25
                if best is None or score < best[0]:
                    best = (score, val)
    return best[1] if best else None


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

        m2 = re.search(r"Delegación\s+Policial\s+([A-Za-zÁÉÍÓÚÑáéíóúñ\s]+)", joined, flags=re.I)
        if m2:
            return f"Delegación Policial {m2.group(1).strip()}"

    return "SIN_DELEGACION"


# -------------------------
# Percepción ciudadana (solo)
# -------------------------
def extract_percepcion_ciudadana(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {
        "perc_seguro": None,
        "perc_inseguro": None,
        "comp_igual": None,
        "comp_menos": None,
        "comp_mas": None,
    }

    anchors = ["¿se siente de seguro en su comunidad", "percepcion ciudadana"]

    for pno in range(doc.page_count):
        page = doc[pno]
        spans = extract_spans(page)

        block = slice_block_until_fuente(spans, anchors)
        if not block:
            continue

        w = page.rect.width
        midx = w * 0.52  # separa pie (izq) vs barras (der)

        left = [sp for sp in block if sp["x"] < midx]
        right = [sp for sp in block if sp["x"] >= midx]

        # BARRAS (derecha)
        out["comp_igual"] = nearest_pct_above_label(right, ["Igual"])
        out["comp_menos"] = nearest_pct_above_label(right, ["Menos seguro"])
        out["comp_mas"] = nearest_pct_above_label(right, ["Más seguro", "Mas seguro", "Más Seguro"])

        # PIE (izquierda)
        left_vals = [v for (v, _, _) in list_pct_points(left)]
        left_vals = [v for v in left_vals if 0 <= v <= 100]

        two = unique_top(left_vals, 2)
        if len(two) == 2:
            # Comunidad: No = mayor, Sí = menor
            out["perc_inseguro"] = max(two)
            out["perc_seguro"] = min(two)

        return out

    return out


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Lector MPGP — Percepción ciudadana", layout="wide")
st.title("Lector MPGP — Percepción ciudadana (solo esta sección)")

uploaded = st.file_uploader("Subir PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded:
    rows = []
    prog = st.progress(0)
    status = st.empty()

    for idx, f in enumerate(uploaded, start=1):
        status.write(f"Procesando: **{f.name}** ({idx}/{len(uploaded)})")

        doc = fitz.open(stream=f.read(), filetype="pdf")
        row = {"archivo": f.name, "delegacion": extract_delegacion(doc)}
        row.update(extract_percepcion_ciudadana(doc))
        doc.close()

        rows.append(row)
        prog.progress(int(idx / len(uploaded) * 100))

    prog.empty()
    status.empty()

    df = pd.DataFrame(rows)
    st.subheader("Resultados (tabla)")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "percepcion_ciudadana.csv",
        "text/csv",
    )
else:
    st.info("Sube uno o más PDFs para iniciar.")
