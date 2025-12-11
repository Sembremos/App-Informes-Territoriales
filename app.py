import re
from typing import Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st
import fitz  # PyMuPDF


# =============================
# UTILIDADES
# =============================
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


def extract_pdf_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text("text"))
    doc.close()
    return "\n".join(text)


def extract_delegacion(text: str) -> str:
    m = re.search(r"\bD\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ\s]+", text)
    if m:
        return " ".join(m.group(0).split())
    m2 = re.search(r"Delegación\s+Policial\s+([A-Za-zÁÉÍÓÚÑáéíóúñ\s]+)", text, re.I)
    if m2:
        return f"Delegación Policial {m2.group(1).strip()}"
    return "SIN_DELEGACION"


def find_value(section: str, label: str) -> Optional[float]:
    p = re.search(rf"{label}.*?(\d{{1,3}}[.,]\d{{1,2}})\s*%", section, re.I)
    if p:
        return pct_to_float(p.group(1))
    return None


# =============================
# EXTRACCIÓN DE DATOS
# =============================
def extract_percepcion(text: str) -> Dict[str, Optional[float]]:
    block = text[text.lower().find("percepción ciudadana") :][:1500]
    return {
        "perc_seguro": find_value(block, "Sí") or find_value(block, "Si"),
        "perc_inseguro": find_value(block, "No"),
        "comp_menos": find_value(block, "Menos seguro"),
        "comp_igual": find_value(block, "Igual"),
        "comp_mas": find_value(block, "Más seguro") or find_value(block, "Mas seguro"),
    }


def extract_servicio_fp(text: str) -> Dict[str, Optional[float]]:
    block = text[text.lower().find("percepción del servicio policial") :][:2000]
    ult = text[text.lower().find("últimos dos años") :][:800]

    return {
        "fp_muy_mala": find_value(block, "Muy mala"),
        "fp_mala": find_value(block, "Mala"),
        "fp_regular": find_value(block, "Regular"),
        "fp_buena": find_value(block, "Buena"),
        "fp_excelente": find_value(block, "Excelente") or find_value(block, "Muy buena"),
        "ult2_peor": find_value(ult, "Peor"),
        "ult2_igual": find_value(ult, "Igual"),
        "ult2_mejor": find_value(ult, "Mejor"),
    }


# =============================
# CÁLCULOS
# =============================
def score(pairs: List[Tuple[Optional[float], float]]) -> Optional[float]:
    vals = [(p, w) for p, w in pairs if p is not None]
    if not vals:
        return None
    return round(sum(p * w for p, w in vals), 2)


def nivel(x: Optional[float]) -> str:
    if x is None:
        return "SIN DATO"
    if x <= 20:
        return "Crítico"
    if x <= 40:
        return "Bajo"
    if x <= 60:
        return "Medio"
    if x <= 80:
        return "Alto"
    return "Muy Alto"


def calcular(row: dict) -> dict:
    p_perc = score([(row["perc_inseguro"], 0), (row["perc_seguro"], 1)])
    p_comp = score([(row["comp_menos"], 0), (row["comp_igual"], 0.5), (row["comp_mas"], 1)])
    p_fp = score([
        (row["fp_muy_mala"], 0),
        (row["fp_mala"], 0),
        (row["fp_regular"], 0.5),
        (row["fp_buena"], 0.75),
        (row["fp_excelente"], 1),
    ])
    p_ult = score([(row["ult2_peor"], 0), (row["ult2_igual"], 0.5), (row["ult2_mejor"], 1)])

    entorno = round((p_perc + p_comp) / 2, 2)
    desempeno = round((p_fp + p_ult) / 2, 2)
    global_i = round((entorno + desempeno) / 2, 2)

    return {
        "percepcion_entorno": entorno,
        "desempeno_policia": desempeno,
        "indice_global": global_i,
        "nivel": nivel(global_i),
    }


# =============================
# STREAMLIT APP
# =============================
st.set_page_config(layout="wide")
st.title("Lector MPGP — Índice de Percepción (PDF)")

files = st.file_uploader("Suba los informes PDF", type="pdf", accept_multiple_files=True)

if files:
    data = []
    for f in files:
        text = extract_pdf_text(f.read())
        row = {"archivo": f.name, "delegacion": extract_delegacion(text)}
        row.update(extract_percepcion(text))
        row.update(extract_servicio_fp(text))
        row.update(calcular(row))
        data.append(row)

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "indices_percepcion.csv",
        "text/csv",
    )
else:
    st.info("Suba uno o más PDFs para iniciar.")

