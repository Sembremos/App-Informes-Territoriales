import re
import math
import unicodedata
from typing import Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF


# -------------------------
# Normalización / parsing
# -------------------------
PCT_RE = re.compile(r"^\s*(\d{1,3}(?:[.,]\d{1,2})?)\s*%?\s*$")

def norm_txt(s: str) -> str:
    """Lower + sin tildes + espacios compactos."""
    if not s:
        return ""
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s

def pct_to_float(s: str) -> Optional[float]:
    """'28,05%' -> 28.05"""
    if not s:
        return None
    s = s.strip().replace("%", "").replace(" ", "")
    # normaliza decimal
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None


# -------------------------
# Extracción por spans (dict)
# -------------------------
def extract_spans(page: fitz.Page) -> List[Dict]:
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
                    "x": (x0 + x1) / 2.0,
                    "y": (y0 + y1) / 2.0,
                    "bbox": (x0, y0, x1, y1),
                })
    return spans

def page_has(spans: List[Dict], phrase: str) -> bool:
    p = norm_txt(phrase)
    return any(p in sp["ntext"] for sp in spans)

def get_delegacion(doc: fitz.Document) -> str:
    # Busca en primeras páginas
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

def find_label_positions(spans: List[Dict], label_variants: List[str]) -> List[Tuple[float, float]]:
    """Devuelve centros (x,y) de las coincidencias de etiqueta."""
    variants = [norm_txt(v) for v in label_variants]
    pts = []
    for sp in spans:
        for v in variants:
            # match exacto o contenido
            if sp["ntext"] == v or v in sp["ntext"]:
                pts.append((sp["x"], sp["y"]))
                break
    return pts

def list_percent_spans(spans: List[Dict]) -> List[Dict]:
    out = []
    for sp in spans:
        t = sp["text"]
        if "%" in t:
            m = re.search(r"(\d{1,3}[.,]\d{1,2})\s*%", t)
            if m:
                out.append({**sp, "pct": pct_to_float(m.group(1))})
                continue
        # A veces viene sin % (ej: "66,24" y en otra línea "%")
        m2 = PCT_RE.match(t)
        if m2:
            val = pct_to_float(m2.group(1))
            if val is not None:
                out.append({**sp, "pct": val})
    return out

def nearest_pct_to_label(spans: List[Dict], label_variants: List[str], max_dist: float = 220.0) -> Optional[float]:
    """Encuentra el porcentaje más cercano (por distancia) a la etiqueta."""
    pts = find_label_positions(spans, label_variants)
    pcts = list_percent_spans(spans)
    if not pts or not pcts:
        return None

    best = None
    for (lx, ly) in pts:
        for p in pcts:
            dx = p["x"] - lx
            dy = p["y"] - ly
            dist = math.hypot(dx, dy)
            if dist <= max_dist:
                if best is None or dist < best[0]:
                    best = (dist, p["pct"])
    return None if best is None else best[1]

def nearest_pct_in_region(spans: List[Dict], anchor_variants: List[str], direction: str, max_dy: float = 120.0) -> Optional[float]:
    """
    Para casos donde la etiqueta está abajo y el % arriba (barras):
    - direction="up": buscar % con y menor (más arriba) cerca en x.
    """
    pts = find_label_positions(spans, anchor_variants)
    pcts = list_percent_spans(spans)
    if not pts or not pcts:
        return None

    best = None
    for (lx, ly) in pts:
        for p in pcts:
            # filtro dirección
            if direction == "up" and not (p["y"] < ly):
                continue
            if direction == "down" and not (p["y"] > ly):
                continue
            dy = abs(p["y"] - ly)
            if dy > max_dy:
                continue
            dx = abs(p["x"] - lx)
            # score: primero dx pequeño, luego dy
            score = dx * 1.0 + dy * 0.2
            if best is None or score < best[0]:
                best = (score, p["pct"])
    return None if best is None else best[1]


# -------------------------
# Extracción por bloque
# -------------------------
def extract_percepcion_ciudadana(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {
        "perc_seguro": None,
        "perc_inseguro": None,
        "comp_menos": None,
        "comp_igual": None,
        "comp_mas": None,
    }

    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        if not page_has(spans, "percepcion ciudadana"):
            continue

        # 1) Sí / No (pie)
        # En estos PDFs funciona mejor por cercanía general
        out["perc_seguro"] = nearest_pct_to_label(spans, ["Sí", "Si"])
        out["perc_inseguro"] = nearest_pct_to_label(spans, ["No"])

        # 2) Comparación (barras): el % suele estar arriba de cada barra y la etiqueta abajo
        out["comp_igual"] = nearest_pct_in_region(spans, ["Igual"], direction="up")
        out["comp_menos"] = nearest_pct_in_region(spans, ["Menos seguro"], direction="up")
        out["comp_mas"] = nearest_pct_in_region(spans, ["Más seguro", "Mas seguro", "Más Seguro"], direction="up")

        return out  # ya lo sacamos de la página correcta

    return out

def extract_servicio_policial(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {
        "fp_muy_mala": None,
        "fp_mala": None,
        "fp_regular": None,
        "fp_buena": None,
        "fp_excelente": None,
        "ult2_peor": None,
        "ult2_igual": None,
        "ult2_mejor": None,
    }

    # Página principal del servicio policial
    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        if page_has(spans, "percepcion del servicio policial"):
            # Barras: % arriba, etiqueta abajo
            out["fp_regular"] = nearest_pct_in_region(spans, ["Regular"], direction="up")
            out["fp_buena"] = nearest_pct_in_region(spans, ["Buena"], direction="up")
            out["fp_excelente"] = nearest_pct_in_region(spans, ["Excelente", "Muy buena", "Muy Buena"], direction="up")
            out["fp_mala"] = nearest_pct_in_region(spans, ["Mala"], direction="up")
            out["fp_muy_mala"] = nearest_pct_in_region(spans, ["Muy mala", "Muy Mala"], direction="up")
            break

    # Página de “últimos dos años” (a veces es la misma o la siguiente)
    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        if page_has(spans, "ultimos dos anos") or page_has(spans, "últimos dos años"):
            out["ult2_igual"] = nearest_pct_to_label(spans, ["Igual"])
            out["ult2_mejor"] = nearest_pct_to_label(spans, ["Mejor servicio", "Mejor"])
            out["ult2_peor"] = nearest_pct_to_label(spans, ["Peor servicio", "Peor"])
            break

    return out

def extract_comercio(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {"com_seguro": None, "com_inseguro": None}

    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        if not page_has(spans, "percepcion sector comercial"):
            continue

        # Pie: usamos cercanía
        out["com_seguro"] = nearest_pct_to_label(spans, ["Sí", "Si"])
        out["com_inseguro"] = nearest_pct_to_label(spans, ["No"])
        return out

    return out


# -------------------------
# Cálculos (tu Excel)
# -------------------------
def weighted_score(pairs: List[Tuple[Optional[float], float]]) -> Optional[float]:
    vals = [(p, w) for p, w in pairs if p is not None]
    if not vals:
        return None
    return round(sum(p * w for p, w in vals), 2)

def nivel_indice(x: Optional[float]) -> str:
    if x is None:
        return "SIN DATO"
    if 0 <= x <= 20:
        return "Crítico"
    if 20 < x <= 40:
        return "Bajo"
    if 40 < x <= 60:
        return "Medio"
    if 60 < x <= 80:
        return "Alto"
    if 80 < x <= 100:
        return "Muy Alto"
    return "FUERA DE RANGO"

def calcular_indices(row: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    p_perc = weighted_score([(row.get("perc_inseguro"), 0.0), (row.get("perc_seguro"), 1.0)])
    p_comp = weighted_score([(row.get("comp_menos"), 0.0), (row.get("comp_igual"), 0.5), (row.get("comp_mas"), 1.0)])

    p_fp = weighted_score([
        (row.get("fp_muy_mala"), 0.0),
        (row.get("fp_mala"), 0.0),
        (row.get("fp_regular"), 0.5),
        (row.get("fp_buena"), 0.75),
        (row.get("fp_excelente"), 1.0),
    ])

    p_ult2 = weighted_score([(row.get("ult2_peor"), 0.0), (row.get("ult2_igual"), 0.5), (row.get("ult2_mejor"), 1.0)])
    p_com = weighted_score([(row.get("com_inseguro"), 0.0), (row.get("com_seguro"), 1.0)])

    entorno_vals = [v for v in [p_perc, p_comp, p_com] if v is not None]
    des_vals = [v for v in [p_fp, p_ult2] if v is not None]

    i_entorno = round(sum(entorno_vals) / len(entorno_vals), 2) if entorno_vals else None
    i_des = round(sum(des_vals) / len(des_vals), 2) if des_vals else None
    glob_vals = [v for v in [i_entorno, i_des] if v is not None]
    i_global = round(sum(glob_vals) / len(glob_vals), 2) if glob_vals else None

    return {
        "punt_percepcion_general": p_perc,
        "punt_comparacion_2023": p_comp,
        "punt_calif_servicio_fp": p_fp,
        "punt_servicio_ult2": p_ult2,
        "punt_seguridad_comercio": p_com,
        "indice_percepcion_entorno": i_entorno,
        "indice_desempeno_policia": i_des,
        "indice_global": i_global,
        "nivel_indice": nivel_indice(i_global),
    }


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Lector MPGP (PDF) — Percepción", layout="wide")
st.title("Lector MPGP (PDF) — Percepción + Índice Global")

uploaded = st.file_uploader("Subir PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded:
    rows = []
    prog = st.progress(0)
    status = st.empty()

    for idx, f in enumerate(uploaded, start=1):
        status.write(f"Procesando: **{f.name}** ({idx}/{len(uploaded)})")
        file_bytes = f.read()

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        deleg = get_delegacion(doc)

        row = {"archivo": f.name, "delegacion": deleg}
        row.update(extract_percepcion_ciudadana(doc))
        row.update(extract_servicio_policial(doc))
        row.update(extract_comercio(doc))
        row.update(calcular_indices(row))

        doc.close()
        rows.append(row)
        prog.progress(int(idx / len(uploaded) * 100))

    prog.empty()
    status.empty()

    df = pd.DataFrame(rows)
    st.subheader("Resultados (tabla)")
    st.dataframe(df, use_container_width=True)

    st.subheader("Descarga")
    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "resultados_percepcion_indices.csv",
        "text/csv",
    )
else:
    st.info("Sube uno o más PDFs para iniciar.")
