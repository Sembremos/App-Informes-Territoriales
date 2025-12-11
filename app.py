import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import re
import io

# =========================
# CONFIG OCR
# =========================
pytesseract.pytesseract.tesseract_cmd = "tesseract"

def ocr_percent(img: Image.Image):
    text = pytesseract.image_to_string(img, config="--psm 6 digits")
    match = re.search(r"(\d{1,3}[.,]\d{1,2})", text)
    if match:
        return float(match.group(1).replace(",", "."))
    return None


def render_page(pdf_bytes, page_number=0, zoom=2):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return img


def crop(img, x1, y1, x2, y2):
    w, h = img.size
    return img.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))


# =========================
# ZONAS DEL GRÁFICO
# (ajustadas a MPGP)
# =========================
ZONAS = {
    "seguro":        (0.05, 0.30, 0.30, 0.55),
    "inseguro":     (0.30, 0.30, 0.55, 0.55),

    "igual":        (0.60, 0.30, 0.70, 0.55),
    "menos_seguro": (0.72, 0.30, 0.82, 0.55),
    "mas_seguro":   (0.84, 0.30, 0.94, 0.55),
}


def extract_percepcion(pdf_bytes):
    img = render_page(pdf_bytes, page_number=0)

    resultados = {}

    for key, zona in ZONAS.items():
        recorte = crop(img, *zona)
        valor = ocr_percent(recorte)
        resultados[key] = valor

    return {
        "Seguro en la comunidad (%)": resultados["seguro"],
        "Inseguro en la comunidad (%)": resultados["inseguro"],
        "Comparación 2023 - Igual (%)": resultados["igual"],
        "Comparación 2023 - Menos seguro (%)": resultados["menos_seguro"],
        "Comparación 2023 - Más seguro (%)": resultados["mas_seguro"],
    }


# =========================
# STREAMLIT
# =========================
st.set_page_config(page_title="Extractor MPGP", layout="wide")
st.title("Extractor de Percepción Ciudadana – MPGP (OCR estable)")

files = st.file_uploader(
    "Suba informes PDF MPGP",
    type="pdf",
    accept_multiple_files=True
)

if files:
    rows = []

    for f in files:
        data = extract_percepcion(f.read())
        row = {
            "Archivo": f.name,
            **{k: f"{v:.2f}%" if v is not None else None for k, v in data.items()}
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    st.subheader("Resultados (tabla)")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "percepcion_ciudadana.csv",
        "text/csv"
    )
