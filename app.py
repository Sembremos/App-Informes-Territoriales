import re
import math
import unicodedata
from typing import Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF


# -------------------------
# Normalización
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
# Spans
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
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "x": (x0 + x1) / 2.0,
                    "y": (y0 + y1) / 2.0,
                })
    return spans


def list_pct_points(spans: List[Dict]) -> List[Tuple[float, float, float]]:
    out = []
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


def find_y_of_phrase(spans: List[Dict], phrase_variants: List[str]) -> Optional[float]:
    variants = [norm_txt(v) for v in phrase_variants]
    ys = []
    for sp in spans:
        for v in variants:
            if v in sp["ntext"]:
                ys.append(sp["y"])
                break
    return min(ys) if ys else None


def slice_block_until_fuente(spans: List[Dict], start_phrases: List[str]) -> List[Dict]:
    """
    Bloque = desde la primera ocurrencia de start_phrases hasta el primer "Fuente:" posterior.
    Si no hay Fuente:, usa hasta final de página.
    """
    y_start = find_y_of_phrase(spans, start_phrases)
    if y_start is None:
        return []

    # buscar fuente DESPUÉS del inicio
    fuente_y = None
    for sp in spans:
        if "fuente:" in sp["ntext"] and sp["y"] > y_start:
            fuente_y = sp["y"]
            break

    y_end = fuente_y if fuente_y is not None else (max(sp["y"] for sp in spans) + 1)
    return [sp for sp in spans if y_start <= sp["y"] <= y_end]


def get_delegacion(doc: fitz.Document) -> str:
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
# Match helpers
# -------------------------
def find_label_points(spans: List[Dict], label_variants: List[str]) -> List[Tuple[float, float]]:
    variants = [norm_txt(v) for v in label_variants]
    pts = []
    for sp in spans:
        for v in variants:
            # exact o contenido
            if sp["ntext"] == v or v in sp["ntext"]:
                pts.append((sp["x"], sp["y"]))
                break
    return pts


def nearest_pct_above_label(block: List[Dict], label_variants: List[str],
                            max_dx: float = 200.0, max_dy: float = 200.0) -> Optional[float]:
    """
    Barras: etiqueta abajo y % arriba
    """
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


def nearest_pct_to_label(block: List[Dict], label_variants: List[str], max_dist: float = 220.0) -> Optional[float]:
    """
    Pie/leyenda: % cerca de la etiqueta
    """
    pts = find_label_points(block, label_variants)
    pcts = list_pct_points(block)
    if not pts or not pcts:
        return None

    best = None
    for lx, ly in pts:
        for val, px, py in pcts:
            dist = math.hypot(px - lx, py - ly)
            if dist <= max_dist:
                if best is None or dist < best[0]:
                    best = (dist, val)
    return best[1] if best else None


def remove_vals(vals: List[float], used: List[Optional[float]], tol: float = 0.2) -> List[float]:
    out = vals[:]
    for u in used:
        if u is None:
            continue
        out = [v for v in out if abs(v - u) > tol]
    return out


# -------------------------
# Extracción por sección (ANCLADA)
# -------------------------
def extract_percepcion_ciudadana(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {"perc_seguro": None, "perc_inseguro": None, "comp_menos": None, "comp_igual": None, "comp_mas": None}

    anchors = ["¿se siente de seguro en su comunidad", "percepcion ciudadana"]
    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        block = slice_block_until_fuente(spans, anchors)
        if not block:
            continue

        # Comparación 2023 (barras)
        comp_igual = nearest_pct_above_label(block, ["Igual"])
        comp_menos = nearest_pct_above_label(block, ["Menos seguro"])
        comp_mas = nearest_pct_above_label(block, ["Más Seguro", "Más seguro", "Mas seguro"])

        # Pie Sí/No (tomar de los % restantes del bloque hasta Fuente)
        all_vals = [v for (v, _, _) in list_pct_points(block)]
        all_vals = [v for v in all_vals if 0 <= v <= 100]
        remain = remove_vals(all_vals, [comp_igual, comp_menos, comp_mas])

        # tomar 2 únicos más fuertes
        uniq = []
        for v in sorted(remain, reverse=True):
            if all(abs(v - u) > 0.2 for u in uniq):
                uniq.append(v)
            if len(uniq) == 2:
                break

        if len(uniq) == 2:
            # En comunidad: "No" suele ser el mayor, "Sí" el menor
            perc_inseguro = max(uniq)
            perc_seguro = min(uniq)
        else:
            perc_seguro = None
            perc_inseguro = None

        out.update({
            "perc_seguro": perc_seguro,
            "perc_inseguro": perc_inseguro,
            "comp_igual": comp_igual,
            "comp_menos": comp_menos,
            "comp_mas": comp_mas
        })
        return out

    return out


def extract_fp_calificacion(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {"fp_muy_mala": None, "fp_mala": None, "fp_regular": None, "fp_buena": None, "fp_excelente": None}

    anchors = ["¿como califica el servicio de fuerza publica", "percepcion del servicio policial"]
    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        block = slice_block_until_fuente(spans, anchors)
        if not block:
            continue

        out["fp_regular"] = nearest_pct_above_label(block, ["Regular"])
        out["fp_buena"] = nearest_pct_above_label(block, ["Buena"])
        out["fp_excelente"] = nearest_pct_above_label(block, ["Excelente", "Muy buena", "Muy Buena"])
        out["fp_muy_mala"] = nearest_pct_above_label(block, ["Muy mala", "Muy Mala"])
        out["fp_mala"] = nearest_pct_above_label(block, ["Mala"])

        return out

    return out


def extract_fp_ult2(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {"ult2_peor": None, "ult2_igual": None, "ult2_mejor": None}

    anchors = ["calificacion del servicio policial de los ultimos dos anos", "ultimos dos anos"]
    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        block = slice_block_until_fuente(spans, anchors)
        if not block:
            continue

        # Aquí usamos leyenda, que es lo más estable:
        out["ult2_igual"] = nearest_pct_to_label(block, ["Igual"])
        out["ult2_mejor"] = nearest_pct_to_label(block, ["Mejor servicio", "Mejor"])
        out["ult2_peor"] = nearest_pct_to_label(block, ["Peor servicio", "Peor"])

        # Fallback: si por alguna razón la leyenda no sale, tomar 3 valores únicos
        if out["ult2_igual"] is None or out["ult2_mejor"] is None or out["ult2_peor"] is None:
            vals = [v for (v, _, _) in list_pct_points(block)]
            vals = [v for v in vals if 0 <= v <= 100]
            uniq = []
            for v in sorted(vals, reverse=True):
                if all(abs(v - u) > 0.2 for u in uniq):
                    uniq.append(v)
                if len(uniq) == 3:
                    break
            if len(uniq) == 3:
                out["ult2_igual"] = max(uniq)
                out["ult2_peor"] = min(uniq)
                out["ult2_mejor"] = sorted(uniq)[1]

        return out

    return out


def extract_comercio(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {"com_seguro": None, "com_inseguro": None}
    anchors = ["¿se siente seguro en su establecimiento comercial", "percepcion sector comercial"]
    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        block = slice_block_until_fuente(spans, anchors)
        if not block:
            continue

        vals = [v for (v, _, _) in list_pct_points(block)]
        vals = [v for v in vals if 0 <= v <= 100]
        uniq = []
        for v in sorted(vals, reverse=True):
            if all(abs(v - u) > 0.2 for u in uniq):
                uniq.append(v)
            if len(uniq) == 2:
                break

        if len(uniq) == 2:
            # En comercio: "Sí" suele ser el mayor (ej 66.24)
            out["com_seguro"] = max(uniq)
            out["com_inseguro"] = min(uniq)

        return out

    return out


# -------------------------
# Cálculos (Excel)
# -------------------------
def weighted_score(pairs: List[Tuple[Optional[float], float]]) -> Optional[float]:
    ok = [(p, w) for p, w in pairs if p is not None]
    if not ok:
        return None
    return round(sum(p * w for p, w in ok), 2)


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

    p_ult2 = weighted_score([
        (row.get("ult2_peor"), 0.0),
        (row.get("ult2_igual"), 0.5),
        (row.get("ult2_mejor"), 1.0),
    ])

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
# UI
# -------------------------
st.set_page_config(page_title="Lector MPGP (PDF) — Percepción", layout="wide")
st.title("Lector MPGP (PDF) — Percepción + Índice Global (anclado a pregunta + Fuente)")

uploaded = st.file_uploader("Subir PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded:
    rows = []
    prog = st.progress(0)
    status = st.empty()

    for idx, f in enumerate(uploaded, start=1):
        status.write(f"Procesando: **{f.name}** ({idx}/{len(uploaded)})")
        file_bytes = f.read()

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        row = {"archivo": f.name, "delegacion": get_delegacion(doc)}

        row.update(extract_percepcion_ciudadana(doc))
        row.update(extract_fp_calificacion(doc))
        row.update(extract_fp_ult2(doc))
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
