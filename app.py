import re
import io
from typing import Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st
import pdfplumber


# -----------------------------
# Utilidades
# -----------------------------
def pct_to_float(s: str) -> Optional[float]:
    """
    Convierte '28,05%' o '28,05' o '28.05%' a float (28.05).
    """
    if s is None:
        return None
    s = s.strip().replace("%", "").replace(" ", "")
    if not s:
        return None
    s = s.replace(".", "") if s.count(",") == 1 and s.count(".") >= 1 else s  # por si viniera 1.234,56
    s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None


def extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extrae el texto completo del PDF (todas las páginas) en un solo string.
    """
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def extract_delegacion(text: str) -> str:
    """
    Intenta obtener 'D-6 Hatillo' / 'D-5 San Sebastián' del PDF.
    Si no, devuelve 'SIN_DELEGACION'.
    """
    # patrón típico: D-6 Hatillo / D- 6 Hatillo / D6 Hatillo
    m = re.search(r"\bD\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑÜáéíóúñü\s]+", text)
    if m:
        return " ".join(m.group(0).split()).strip()
    # fallback: 'Delegación Policial <Nombre>'
    m2 = re.search(r"Delegación\s+Policial\s+([A-Za-zÁÉÍÓÚÑÜáéíóúñü\s]+)", text, flags=re.IGNORECASE)
    if m2:
        name = " ".join(m2.group(1).split()).strip()
        return f"Delegación Policial {name}"
    return "SIN_DELEGACION"


def first_section_until_fuente(text: str) -> str:
    """
    Devuelve la sección desde el inicio del bloque hasta el primer 'Fuente:'.
    Útil porque los gráficos principales suelen ir antes de la fuente.
    """
    idx = text.lower().find("fuente:")
    if idx == -1:
        return text
    return text[:idx]


def slice_around(text: str, start_phrase: str, length: int = 700) -> str:
    """
    Recorta un segmento de texto a partir de una frase, para reducir ruido.
    """
    pos = text.lower().find(start_phrase.lower())
    if pos == -1:
        return ""
    return text[pos:pos + length]


def find_value_near_label(section: str, label: str) -> Optional[float]:
    """
    Busca un porcentaje asociado a una etiqueta, en dos variantes:
    1) 'Etiqueta ... 28,05%'
    2) '28,05% ... Etiqueta'
    Devuelve float o None.
    """
    # etiqueta seguida de número
    p1 = re.search(
        rf"{re.escape(label)}\s*[:\-]?\s*(\d{{1,3}}[.,]\d{{1,2}})\s*%?",
        section,
        flags=re.IGNORECASE
    )
    if p1:
        return pct_to_float(p1.group(1))

    # número seguido de etiqueta
    p2 = re.search(
        rf"(\d{{1,3}}[.,]\d{{1,2}})\s*%?\s*{re.escape(label)}",
        section,
        flags=re.IGNORECASE
    )
    if p2:
        return pct_to_float(p2.group(1))

    return None


def compute_weighted_score(pairs: List[Tuple[Optional[float], float]]) -> Optional[float]:
    """
    pairs = [(porcentaje, peso), ...]
    Devuelve puntaje 0-100 si hay datos, si no None.
    """
    ok = [(p, w) for (p, w) in pairs if p is not None]
    if not ok:
        return None
    # porcentaje viene como 0..100
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
# Extracción por bloques (según tus PDFs)
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

    block = slice_around(full_text, "Percepción ciudadana", length=1200)
    if not block:
        return out

    block_main = first_section_until_fuente(block)

    # “Sí / No” del gráfico principal (solo en este bloque)
    out["perc_seguro"] = find_value_near_label(block_main, "Sí")
    out["perc_inseguro"] = find_value_near_label(block_main, "No")

    # Comparación año anterior (etiquetas pueden venir en distinto orden)
    out["comp_menos"] = find_value_near_label(block_main, "Menos seguro")
    out["comp_igual"] = find_value_near_label(block_main, "Igual")
    out["comp_mas"] = find_value_near_label(block_main, "Más Seguro") or find_value_near_label(block_main, "Más seguro")

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

    block = slice_around(full_text, "Percepción del Servicio Policial", length=1600)
    if not block:
        return out

    # Parte 1: Calificación principal (antes de la primera Fuente)
    main = first_section_until_fuente(block)

    # Los PDFs a veces usan “Excelente” como el tope (equivale a Muy buena en tu Excel)
    out["fp_excelente"] = find_value_near_label(main, "Excelente")
    out["fp_regular"] = find_value_near_label(main, "Regular")
    out["fp_buena"] = find_value_near_label(main, "Buena")
    out["fp_mala"] = find_value_near_label(main, "Mala")
    out["fp_muy_mala"] = find_value_near_label(main, "Muy Mala") or find_value_near_label(main, "Muy mala")

    # Parte 2: “Calificación ... de los Últimos Dos Años”
    ult = slice_around(block, "de los Últimos Dos Años", length=900)
    if ult:
        # Aquí aparecen más preguntas SI/NO, por eso buscamos etiquetas exactas:
        out["ult2_peor"] = find_value_near_label(ult, "Peor servicio")
        out["ult2_igual"] = find_value_near_label(ult, "Igual")
        out["ult2_mejor"] = find_value_near_label(ult, "Mejor servicio")

    return out


def extract_comercio(full_text: str) -> Dict[str, Optional[float]]:
    """
    Obtiene:
      - com_seguro, com_inseguro
    """
    out = {"com_seguro": None, "com_inseguro": None}

    block = slice_around(full_text, "Percepción Sector Comercial", length=1300)
    if not block:
        return out

    # Nos enfocamos en el pedazo justo después de la pregunta principal
    sec = slice_around(block, "¿Se siente seguro en su establecimiento", length=500)
    if not sec:
        # fallback: tomar el inicio del bloque
        sec = block[:500]

    # Dentro de ese fragmento suelen aparecer:
    # - un número para No y uno para Si (pero también hay otra pregunta al lado).
    # Solución: usar la “primera” coincidencia de No/Si cerca de la pregunta.

    # Tomamos los primeros matches de porcentaje asociados a No/Si
    no_val = find_value_near_label(sec, "No")
    si_val = find_value_near_label(sec, "Si") or find_value_near_label(sec, "Sí")

    # En algunos PDFs, el “Si” viene pegado con el % separado en otra línea, así que ampliamos si falta
    if si_val is None or no_val is None:
        sec2 = sec + " " + block[:800]
        no_val = no_val if no_val is not None else find_value_near_label(sec2, "No")
        si_val = si_val if si_val is not None else (find_value_near_label(sec2, "Si") or find_value_near_label(sec2, "Sí"))

    out["com_seguro"] = si_val
    out["com_inseguro"] = no_val
    return out


# -----------------------------
# Cálculos (idénticos a tu Excel)
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

    # Puntaje servicio FP (Muy mala 0, Mala 0, Regular 0.5, Buena 0.75, Excelente 1)
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

    # Puntaje comercio (Seguro 1, Inseguro 0)
    p_com = compute_weighted_score([
        (row.get("com_inseguro"), 0.0),
        (row.get("com_seguro"), 1.0),
    ])

    # Índices agregados
    entorno_vals = [v for v in [p_perc, p_comp, p_com] if v is not None]
    des_vals = [v for v in [p_fp, p_ult2] if v is not None]

    i_entorno = round(sum(entorno_vals) / len(entorno_vals), 2) if entorno_vals else None
    i_desempeno = round(sum(des_vals) / len(des_vals), 2) if des_vals else None

    glob_vals = [v for v in [i_entorno, i_desempeno] if v is not None]
    i_global = round(sum(glob_vals) / len(glob_vals), 2) if glob_vals else None

    return {
        "punt_percepcion_general": p_perc,
        "punt_comparacion_2023": p_comp,
        "punt_calif_servicio_fp": p_fp,
        "punt_servicio_ult2": p_ult2,
        "punt_seguridad_comercio": p_com,
        "indice_percepcion_entorno": i_entorno,
        "indice_desempeno_policia": i_desempeno,
        "indice_global": i_global,
        "nivel_indice": nivel_indice(i_global),
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Lector MPGP - Percepción (PDF)", layout="wide")
st.title("Lector de datos (MPGP) — Percepción y Cálculo de Índice")

st.write(
    "Sube múltiples PDFs (90+). La app extrae porcentajes de percepción y calcula puntajes e índice global "
    "según la lógica de tu Excel."
)

uploaded = st.file_uploader(
    "Subir PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded:
    rows = []
    progress = st.progress(0)
    status = st.empty()

    for i, f in enumerate(uploaded, start=1):
        status.write(f"Procesando: **{f.name}** ({i}/{len(uploaded)})")

        file_bytes = f.read()
        full_text = extract_pdf_text(file_bytes)
        deleg = extract_delegacion(full_text)

        data = {"archivo": f.name, "delegacion": deleg}

        # extracción por bloques
        data.update(extract_percepcion_ciudadana(full_text))
        data.update(extract_servicio_fp(full_text))
        data.update(extract_comercio(full_text))

        # cálculos
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
        mime="text/csv"
    )

else:
    st.info("Sube uno o varios PDFs para empezar.")
