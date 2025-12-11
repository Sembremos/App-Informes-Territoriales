import re
from typing import Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st

import fitz  # PyMuPDF


# -----------------------------
# Utilidades
# -----------------------------
def pct_to_float(s: str) -> Optional[float]:
    """Convierte '28,05%' / '28.05%' / '28,05' a float (28.05)."""
    if s is None:
        return None
    s = s.strip().replace("%", "").replace(" ", "")
    if not s:
        return None

    # normaliza formato decimal
    if "," in s and "." in s:
        # ejemplo raro 1.234,56 -> 1234.56
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")

    try:
        return float(s)
    except:
        return None


def extract_pdf_text(file_bytes: bytes) -> str:
    """Extrae texto completo del PDF (todas las páginas) usando PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts = []
    for page in doc:
        t = page.get_text("text") or ""
        if t:
            parts.append(t)
    doc.close()
    return "\n".join(parts)


def extract_delegacion(text: str) -> str:
    """Intenta obtener 'D-6 Hatillo' / 'D-5 San Sebastián' del texto del PDF."""
    m = re.search(r"\bD\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑÜáéíóúñü\s]+", text)
    if m:
        return " ".join(m.group(0).split()).strip()

    m2 = re.search(
        r"Delegación\s+Policial\s+([A-Za-zÁÉÍÓÚÑÜáéíóúñü\s]+)",
        text,
        flags=re.IGNORECASE,
    )
    if m2:
        name = " ".join(m2.group(1).split()).strip()
        return f"Delegación Policial {name}"

    return "SIN_DELEGACION"


def first_section_until_fuente(text: str) -> str:
    idx = text.lower().find("fuente:")
    return text if idx == -1 else text[:idx]


def slice_around(text: str, start_phrase: str, length: int = 900) -> str:
    pos = text.lower().find(start_phrase.lower())
    if pos == -1:
        return ""
    return text[pos : pos + length]


def find_value_near_label(section: str, label: str) -> Optional[float]:
    """
    Busca porcentaje asociado a etiqueta:
      - 'Etiqueta ... 28,05%'
      - '28,05% ... Etiqueta'
    """
    p1 = re.search(
        rf"{re.escape(label)}\s*[:\-]?\s*(\d{{1,3}}[.,]\d{{1,2}})\s*%?",
        section,
        flags=re.IGNORECASE,
    )
    if p1:
        return pct_to_float(p1.group(1))

    p2 = re.search(
        rf"(\d{{1,3}}[.,]\d{{1,2}})\s*%?\s*{re.escape(label)}",
        section,
        flags=re.IGNORECASE,
    )
    if p2:
        return pct_to_float(p2.group(1))

    return None


def compute_weighted_score(pairs: List[Tuple[Optional[float], float]]) -> Optional[float]:
    ok = [(p, w) for (p, w) in pairs if p is not None]
    if not ok:
        return None
    return round(sum((p * w) for (p, w) in ok), 2)


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


# -----------------------------
# Extracción por bloques (SIN comercio)
# -----------------------------
def extract_percepcion_ciudadana(full_text: str) -> Dict[str, Optional[float]]:
    """
    Obtiene:
      - perc_seguro, perc_inseguro
      - comp_menos, comp_igual, comp_mas
    """
    out = {
        "perc_seguro": None,
        "perc_inseguro": None,
        "comp_menos": None,
        "comp_igual": None,
        "comp_mas": None,
    }

    block = slice_around(full_text, "Percepción ciudadana", length=1400)
    if not block:
        return out

    block_main = first_section_until_fuente(block)

    out["perc_seguro"] = find_value_near_label(block_main, "Sí") or find_value_near_label(block_main, "Si")
    out["perc_inseguro"] = find_value_near_label(block_main, "No")

    out["comp_menos"] = find_value_near_label(block_main, "Menos seguro")
    out["comp_igual"] = find_value_near_label(block_main, "Igual")
    out["comp_mas"] = (
        find_value_near_label(block_main, "Más Seguro")
        or find_value_near_label(block_main, "Más seguro")
        or find_value_near_label(block_main, "Mas seguro")
    )

    return out


def extract_servicio_fp(full_text: str) -> Dict[str, Optional[float]]:
    """
    Obtiene:
      - fp_muy_mala, fp_mala, fp_regular, fp_buena, fp_excelente
      - ult2_peor, ult2_igual, ult2_mejor
    """
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

    block = slice_around(full_text, "Percepción del Servicio Policial", length=2000)
    if not block:
        return out

    main = first_section_until_fuente(block)

    # Calificación principal
    out["fp_excelente"] = find_value_near_label(main, "Excelente") or find_value_near_label(main, "Muy Buena") or find_value_near_label(main, "Muy buena")
    out["fp_regular"] = find_value_near_label(main, "Regular") or find_value_near_label(main, "Medio")
    out["fp_buena"] = find_value_near_label(main, "Buena")
    out["fp_mala"] = find_value_near_label(main, "Mala")
    out["fp_muy_mala"] = find_value_near_label(main, "Muy Mala") or find_value_near_label(main, "Muy mala")

    # Últimos 2 años
    ult = slice_around(block, "Últimos Dos Años", length=900) or slice_around(block, "Ultimos Dos Años", length=900)
    if ult:
        out["ult2_peor"] = find_value_near_label(ult, "Peor servicio") or find_value_near_label(ult, "Peor")
        out["ult2_igual"] = find_value_near_label(ult, "Igual")
        out["ult2_mejor"] = find_value_near_label(ult, "Mejor servicio") or find_value_near_label(ult, "Mejor")

    return out


# -----------------------------
# Cálculos (SIN comercio)
# -----------------------------
def calcular_indices(row: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    # Puntaje percepción general
    p_perc = compute_weighted_score([
        (row.get("perc_inseguro"), 0.0),
        (row.get("perc_seguro"), 1.0),
    ])

    # Puntaje comparación año anterior
    p_comp = compute_weighted_score([
        (row.get("comp_menos"), 0.0),
        (row.get("comp_igual"), 0.5),
        (row.get("comp_mas"), 1.0),
    ])

    # Puntaje servicio FP
    p_fp = compute_weighted_score([
        (row.get("fp_muy_mala"), 0.0),
        (row.get("fp_mala"), 0.0),
        (row.get("fp_regular"), 0.5),
        (row.get("fp_buena"), 0.75),
        (row.get("fp_excelente"), 1.0),
    ])

    # Puntaje servicio últimos 2 años
    p_ult2 = compute_weighted_score([
        (row.get("ult2_peor"), 0.0),
        (row.get("ult2_igual"), 0.5),
        (row.get("ult2_mejor"), 1.0),
    ])

    # Índice percepción del entorno (SIN comercio): promedio(percepción general, comparación)
    entorno_vals = [v for v in [p_perc, p_comp] if v is not None]
    i_entorno = round(sum(entorno_vals) / len(entorno_vals), 2) if entorno_vals else None

    # Índice desempeño policía: promedio(servicio FP, últimos 2 años)
    des_vals = [v for v in [p_fp, p_ult2] if v is not None]
    i_desempeno = round(sum(des_vals) / len(des_vals), 2) if des_vals else None

    # Índice global: promedio(entorno, desempeño)
    glob_vals = [v for v in [i_entorno, i_desempeno] if v is not None]
    i_global = round(sum(glob_vals) / len(glob_vals), 2) if glob_vals else None

    return {
        "punt_percepcion_general": p_perc,
        "punt_comparacion_2023": p_comp,
        "punt_calif_servicio_fp": p_fp,
        "punt_servicio_ult2": p_ult2,
        "indice_percepcion_entorno": i_entorno,
        "indice_desempeno_policia": i_desempeno,
        "indice_global": i_global,
        "nivel_indice": nivel_indice(i_global),
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Lector MPGP - Percepción (PDF)", layout="wide")
st.title("Lector MPGP — Percepción (PDF) + Índices (sin comercio)")

st.write(
    "Sube múltiples PDFs. La app extrae porcentajes de **Percepción ciudadana** y **Servicio Policial**, "
    "calcula puntajes e índices como tu Excel y clasifica el nivel."
)

uploaded = st.file_uploader("Subir PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded:
    rows = []
    progress = st.progress(0)
    status = st.empty()

    for i, f in enumerate(uploaded, start=1):
        status.write(f"Procesando: **{f.name}** ({i}/{len(uploaded)})")

        file_bytes = f.read()
        full_text = extract_pdf_text(file_bytes)

        data = {
            "archivo": f.name,
            "delegacion": extract_delegacion(full_text),
        }

        data.update(extract_percepcion_ciudadana(full_text))
        data.update(extract_servicio_fp(full_text))
        data.update(calcular_indices(data))

        rows.append(data)
        progress.progress(int(i / len(uploaded) * 100))

    progress.empty()
    status.empty()

    df = pd.DataFrame(rows)

    st.subheader("Resultados (tabla)")
    st.dataframe(df, use_container_width=True)

    st.subheader("Descarga")
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Descargar CSV",
        data=csv,
        file_name="resultados_percepcion_indices.csv",
        mime="text/csv",
    )
else:
    st.info("Sube uno o varios PDFs para empezar.")
