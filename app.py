import re
import math
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
# Spans por coordenadas
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


def page_has(spans: List[Dict], phrase: str) -> bool:
    p = norm_txt(phrase)
    return any(p in sp["ntext"] for sp in spans)


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


def find_first_y(spans: List[Dict], phrase: str) -> Optional[float]:
    p = norm_txt(phrase)
    ys = [sp["y"] for sp in spans if p in sp["ntext"]]
    return min(ys) if ys else None


def section_spans_by_titles(spans: List[Dict], start_title: str, end_titles: List[str]) -> List[Dict]:
    """
    Devuelve spans dentro del rango vertical (y) de una sección:
    - inicio: primera ocurrencia de start_title
    - fin: primera ocurrencia (más cercana hacia abajo) de cualquiera de end_titles o 'Fuente:'
    """
    y_start = find_first_y(spans, start_title)
    if y_start is None:
        return []

    # candidatos a fin
    y_candidates = []
    for t in end_titles + ["Fuente:"]:
        y = find_first_y(spans, t)
        if y is not None and y > y_start:
            y_candidates.append(y)

    y_end = min(y_candidates) if y_candidates else (max(sp["y"] for sp in spans) + 1)

    return [sp for sp in spans if y_start <= sp["y"] <= y_end]


def list_pct_spans(spans: List[Dict]) -> List[Tuple[float, float, float]]:
    """
    Lista de (pct, x, y) detectados en spans.
    """
    out = []
    for sp in spans:
        t = sp["text"].strip()
        m = PCT_IN_TEXT.search(t)
        if m:
            val = pct_to_float(m.group(1))
            if val is not None:
                out.append((val, sp["x"], sp["y"]))
                continue
        m2 = PCT_ALONE.match(t)
        if m2:
            val = pct_to_float(m2.group(1))
            if val is not None:
                out.append((val, sp["x"], sp["y"]))
    return out


def find_label_points(spans: List[Dict], variants: List[str], exact: bool = False) -> List[Tuple[float, float]]:
    v = [norm_txt(x) for x in variants]
    pts = []
    for sp in spans:
        for a in v:
            if exact:
                if sp["ntext"] == a:
                    pts.append((sp["x"], sp["y"]))
                    break
            else:
                if sp["ntext"] == a or a in sp["ntext"]:
                    pts.append((sp["x"], sp["y"]))
                    break
    return pts


def nearest_pct_above_label(
    spans: List[Dict],
    label_variants: List[str],
    max_dx: float = 180.0,
    max_dy: float = 160.0,
) -> Optional[float]:
    """
    Para barras: etiqueta abajo, % arriba.
    Busca el % con (y menor) más cercano por x.
    """
    pts = find_label_points(spans, label_variants, exact=False)
    pcts = list_pct_spans(spans)
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


def remove_values(all_vals: List[float], to_remove: List[Optional[float]], tol: float = 0.15) -> List[float]:
    """
    Quita valores que ya asignamos (por tolerancia).
    """
    cleaned = all_vals[:]
    for r in to_remove:
        if r is None:
            continue
        cleaned = [v for v in cleaned if abs(v - r) > tol]
    return cleaned


# -------------------------
# Extracción por bloque (FIX)
# -------------------------
def extract_percepcion_ciudadana(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {"perc_seguro": None, "perc_inseguro": None, "comp_menos": None, "comp_igual": None, "comp_mas": None}

    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        if not page_has(spans, "Percepción ciudadana"):
            continue

        sec = section_spans_by_titles(
            spans,
            start_title="Percepción ciudadana",
            end_titles=["Percepción del Servicio Policial", "Percepción Sector Comercial"]
        )
        if not sec:
            sec = spans

        # 1) Barras comparación 2023 (robusto)
        comp_igual = nearest_pct_above_label(sec, ["Igual"])
        comp_menos = nearest_pct_above_label(sec, ["Menos seguro"])
        comp_mas = nearest_pct_above_label(sec, ["Más Seguro", "Más seguro", "Mas seguro"])

        # 2) Pie (Sí/No) usando "resto de porcentajes" en la sección
        all_pcts = [v for (v, _, _) in list_pct_spans(sec)]
        all_pcts = [v for v in all_pcts if 0 <= v <= 100]
        remain = remove_values(all_pcts, [comp_igual, comp_menos, comp_mas])

        # Tomamos los 2 más “relevantes” del pie (normalmente 2 únicos)
        uniq = []
        for v in sorted(remain, reverse=True):
            if all(abs(v - u) > 0.15 for u in uniq):
                uniq.append(v)
            if len(uniq) >= 2:
                break

        if len(uniq) == 2:
            # En estos informes: "Sí" suele ser el porcentaje menor del pie
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
            "comp_mas": comp_mas,
        })
        return out

    return out


def extract_servicio_policial(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {
        "fp_muy_mala": None, "fp_mala": None, "fp_regular": None, "fp_buena": None, "fp_excelente": None,
        "ult2_peor": None, "ult2_igual": None, "ult2_mejor": None,
    }

    # 1) Página/bloque de barras del servicio policial
    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        if not page_has(spans, "Percepción del Servicio Policial"):
            continue

        sec = section_spans_by_titles(
            spans,
            start_title="Percepción del Servicio Policial",
            end_titles=["Calificación del Servicio Policial de los Últimos Dos Años", "Percepción Sector Comercial"]
        )
        if not sec:
            sec = spans

        out["fp_regular"] = nearest_pct_above_label(sec, ["Regular"])
        out["fp_buena"] = nearest_pct_above_label(sec, ["Buena"])
        out["fp_excelente"] = nearest_pct_above_label(sec, ["Excelente", "Muy buena", "Muy Buena"])
        # Evita que "Mala" capture "Muy mala": primero Muy mala, luego Mala con exact más estricto
        out["fp_muy_mala"] = nearest_pct_above_label(sec, ["Muy mala", "Muy Mala"])
        out["fp_mala"] = nearest_pct_above_label(sec, ["Mala"])

        break

    # 2) Pie de últimos 2 años (asignación por orden de magnitud)
    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        if not (page_has(spans, "Últimos Dos Años") or page_has(spans, "Ultimos Dos Años")):
            continue

        sec = section_spans_by_titles(
            spans,
            start_title="Últimos Dos Años",
            end_titles=["Percepción Sector Comercial", "Percepción ciudadana"]
        )
        if not sec:
            sec = spans

        vals = [v for (v, _, _) in list_pct_spans(sec)]
        vals = [v for v in vals if 0 <= v <= 100]

        # únicos (3)
        uniq = []
        for v in sorted(vals, reverse=True):
            if all(abs(v - u) > 0.15 for u in uniq):
                uniq.append(v)
            if len(uniq) >= 3:
                break

        if len(uniq) == 3:
            # patrón típico: Igual = mayor, Mejor = medio, Peor = menor
            out["ult2_igual"] = max(uniq)
            out["ult2_peor"] = min(uniq)
            mid = sorted(uniq)[1]
            out["ult2_mejor"] = mid

        break

    return out


def extract_comercio(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {"com_seguro": None, "com_inseguro": None}

    for i in range(doc.page_count):
        spans = extract_spans(doc[i])
        if not page_has(spans, "Percepción Sector Comercial"):
            continue

        sec = section_spans_by_titles(
            spans,
            start_title="Percepción Sector Comercial",
            end_titles=[]
        )
        if not sec:
            sec = spans

        # Pie: dos porcentajes, asignamos seguro=min? NO: en comercio "Sí" suele ser mayor (66.24)
        vals = [v for (v, _, _) in list_pct_spans(sec)]
        vals = [v for v in vals if 0 <= v <= 100]
        uniq = []
        for v in sorted(vals, reverse=True):
            if all(abs(v - u) > 0.15 for u in uniq):
                uniq.append(v)
            if len(uniq) >= 2:
                break

        if len(uniq) == 2:
            out["com_seguro"] = max(uniq)
            out["com_inseguro"] = min(uniq)

        return out

    return out


# -------------------------
# Cálculos (tu Excel)
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
st.title("Lector MPGP (PDF) — Percepción + Índice Global (corregido)")

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
