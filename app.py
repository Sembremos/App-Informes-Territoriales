import re
import unicodedata
from typing import Dict, Optional, Tuple, List
import colorsys

import streamlit as st
import pandas as pd

# PDFs
import fitz  # PyMuPDF

# Imágenes + OCR
from PIL import Image
import numpy as np

try:
    import pytesseract
    TESSERACT_OK = True
except Exception:
    TESSERACT_OK = False


# =========================
# Helpers
# =========================
PCT_RE = re.compile(r"(\d{1,3}[.,]\d{1,2})\s*%")

def norm(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"\s+", " ", s)
    return s

def parse_pct(txt: str) -> Optional[float]:
    if not txt:
        return None
    m = PCT_RE.search(txt)
    if not m:
        return None
    s = m.group(1).replace(",", ".")
    try:
        v = float(s)
        return v if 0 <= v <= 100 else None
    except:
        return None

def fmt_pct(x) -> Optional[str]:
    if isinstance(x, (int, float)):
        return f"{x:.2f}%"
    return None

def classify_bar_color(rgb: Tuple[float, float, float]) -> str:
    """Clasifica color de barra (MPGP): gris=igual, celeste=mas, azul=menos."""
    r, g, b = rgb

    # gris: canales parecidos + baja saturación
    if abs(r - g) < 0.08 and abs(g - b) < 0.08:
        return "igual"

    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.18:
        return "igual"

    # azul oscuro vs celeste por brillo
    return "menos" if v < 0.55 else "mas"


# =========================
# PDF pipeline (sin OCR)
# =========================
def find_page_percepcion_ciudadana(doc: fitz.Document) -> Optional[int]:
    anchors = ["percepcion ciudadana", "se siente de seguro en su comunidad"]
    for i in range(doc.page_count):
        t = norm(doc[i].get_text("text") or "")
        if all(a in t for a in anchors):
            return i
    return None

def extract_delegacion_from_pdf(doc: fitz.Document) -> str:
    for i in range(min(8, doc.page_count)):
        t = doc[i].get_text("text") or ""
        m = re.search(r"\bD\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ\s]+", t)
        if m:
            return " ".join(m.group(0).split()).strip()
    return "SIN_DELEGACIÓN"

def extract_pie_from_pdf(page: fitz.Page, midx: float, y0: float, y1: float):
    words = page.get_text("words")
    vals = []
    for (x0, y0w, x1w, y1w, w, *_rest) in words:
        cx = (x0 + x1w) / 2.0
        cy = (y0w + y1w) / 2.0
        if cx >= midx:
            continue
        if cy < y0 or cy > y1:
            continue
        v = parse_pct(w)
        if v is not None:
            vals.append(v)

    uniq = []
    for v in sorted(vals, reverse=True):
        if all(abs(v - u) > 0.2 for u in uniq):
            uniq.append(v)
        if len(uniq) == 2:
            break

    if len(uniq) != 2:
        return None, None

    inseguro = max(uniq)
    seguro = min(uniq)
    return seguro, inseguro

def render_pdf_page(page: fitz.Page, zoom: float = 2.0) -> Tuple[np.ndarray, float]:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return img, zoom

def find_bar_rects_pdf(page: fitz.Page, midx: float, y0: float, y1: float) -> List[fitz.Rect]:
    drawings = page.get_drawings()
    bars = []
    for d in drawings:
        for it in d.get("items", []):
            if not it or it[0] != "re":
                continue
            r = it[1]
            cx = (r.x0 + r.x1) / 2.0
            cy = (r.y0 + r.y1) / 2.0
            if cx < midx:
                continue
            if cy < y0 or cy > y1:
                continue
            w = r.x1 - r.x0
            h = r.y1 - r.y0
            if h > 90 and w > 12 and w < 340:
                bars.append((h, cx, r))

    if not bars:
        return []
    bars.sort(key=lambda t: t[0], reverse=True)
    top = bars[:16]

    picked = []
    pxs = []
    for _h, cx, r in sorted(top, key=lambda t: t[1]):
        if all(abs(cx - x) > 22 for x in pxs):
            picked.append(r)
            pxs.append(cx)
        if len(picked) == 3:
            break
    return picked

def extract_bar_pcts_pdf(page: fitz.Page, bars: List[fitz.Rect], midx: float, y0: float, y1: float) -> List[Optional[float]]:
    words = page.get_text("words")
    pts = []
    for (x0, y0w, x1w, y1w, w, *_rest) in words:
        v = parse_pct(w)
        if v is None:
            continue
        cx = (x0 + x1w) / 2.0
        cy = (y0w + y1w) / 2.0
        if cx < midx or cy < y0 or cy > y1:
            continue
        pts.append((v, cx, cy))

    uniq = []
    for v, x, y in sorted(pts, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in uniq):
            uniq.append((v, x, y))

    out = []
    for r in bars:
        bx = (r.x0 + r.x1) / 2.0
        best = None
        for (v, x, y) in uniq:
            if y >= r.y0:
                continue
            if y < r.y0 - 520:
                continue
            dx = abs(x - bx)
            if dx > 300:
                continue
            score = dx + (r.y0 - y) * 0.05
            if best is None or score < best[0]:
                best = (score, v)
        out.append(best[1] if best else None)
    return out

def extract_comp_pdf(page: fitz.Page, midx: float, y0: float, y1: float) -> Dict[str, Optional[float]]:
    bars = find_bar_rects_pdf(page, midx, y0, y1)
    if len(bars) != 3:
        return {"igual": None, "mas": None, "menos": None}

    pcts = extract_bar_pcts_pdf(page, bars, midx, y0, y1)

    img, zoom = render_pdf_page(page, zoom=2.0)
    assigned = {"igual": None, "mas": None, "menos": None}

    for r, pct in zip(bars, pcts):
        cx = (r.x0 + r.x1) / 2.0
        y_inside = r.y0 + (r.y1 - r.y0) * 0.35

        px = int(cx * zoom)
        py = int(y_inside * zoom)
        px = max(0, min(img.shape[1] - 1, px))
        py = max(0, min(img.shape[0] - 1, py))

        # promedio local
        patch = img[max(0, py-5):py+6, max(0, px-5):px+6, :]
        mean = patch.mean(axis=(0, 1)) / 255.0
        cat = classify_bar_color((float(mean[0]), float(mean[1]), float(mean[2])))

        if assigned[cat] is None or (assigned[cat] is not None and pct is not None):
            assigned[cat] = pct

    return assigned

def extract_from_pdf(file_bytes: bytes) -> Dict[str, Optional[float]]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    out = {
        "Delegación": extract_delegacion_from_pdf(doc),
        "Seguro en la comunidad (%)": None,
        "Inseguro en la comunidad (%)": None,
        "Comparación 2023 - Igual (%)": None,
        "Comparación 2023 - Más seguro (%)": None,
        "Comparación 2023 - Menos seguro (%)": None,
    }

    pidx = find_page_percepcion_ciudadana(doc)
    if pidx is None:
        doc.close()
        return out

    page = doc[pidx]
    y0 = page.rect.height * 0.16
    y1 = page.rect.height * 0.92
    midx = page.rect.width * 0.52

    seguro, inseguro = extract_pie_from_pdf(page, midx, y0, y1)
    out["Seguro en la comunidad (%)"] = seguro
    out["Inseguro en la comunidad (%)"] = inseguro

    comp = extract_comp_pdf(page, midx, y0, y1)
    out["Comparación 2023 - Igual (%)"] = comp["igual"]
    out["Comparación 2023 - Más seguro (%)"] = comp["mas"]
    out["Comparación 2023 - Menos seguro (%)"] = comp["menos"]

    doc.close()
    return out


# =========================
# IMAGEN pipeline (OCR + color)
# =========================
def ocr_percentages_image(img: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    Devuelve lista de (pct, cx, cy) en coords de imagen.
    Requiere Tesseract. Si no existe, lanza excepción controlada.
    """
    if not TESSERACT_OK:
        raise RuntimeError("pytesseract no disponible")

    pil = Image.fromarray(img)
    try:
        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, config="--psm 6")
    except Exception as e:
        # típico: TesseractNotFoundError
        raise RuntimeError(str(e))

    out = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        v = parse_pct(txt)
        if v is None:
            continue
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        cx = x + w / 2.0
        cy = y + h / 2.0
        out.append((v, cx, cy))
    return out

def mean_rgb_patch(img: np.ndarray, x: float, y: float, r: int = 6) -> Tuple[float, float, float]:
    h, w, _ = img.shape
    xi = int(max(0, min(w - 1, x)))
    yi = int(max(0, min(h - 1, y)))
    patch = img[max(0, yi-r):yi+r+1, max(0, xi-r):xi+r+1, :]
    mean = patch.mean(axis=(0, 1)) / 255.0
    return (float(mean[0]), float(mean[1]), float(mean[2]))

def extract_from_image(img: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Extrae los mismos campos desde una captura/imagen de la página.
    Asume layout MPGP (pastel izq, barras der).
    """
    H, W, _ = img.shape
    out = {
        "Delegación": "DESDE_IMAGEN",
        "Seguro en la comunidad (%)": None,
        "Inseguro en la comunidad (%)": None,
        "Comparación 2023 - Igual (%)": None,
        "Comparación 2023 - Más seguro (%)": None,
        "Comparación 2023 - Menos seguro (%)": None,
    }

    # OCR de porcentajes
    pts = ocr_percentages_image(img)

    # ventana vertical aproximada del bloque
    y0 = H * 0.15
    y1 = H * 0.92
    midx = W * 0.52

    # 1) Pastel: tomar 2 % del lado izquierdo
    left = [v for (v, x, y) in pts if x < midx and y0 <= y <= y1]
    uniq = []
    for v in sorted(left, reverse=True):
        if all(abs(v - u) > 0.2 for u in uniq):
            uniq.append(v)
        if len(uniq) == 2:
            break
    if len(uniq) == 2:
        out["Inseguro en la comunidad (%)"] = max(uniq)
        out["Seguro en la comunidad (%)"] = min(uniq)

    # 2) Comparación: 3 % del lado derecho + clasificar por color (bajo cada %)
    right = [(v, x, y) for (v, x, y) in pts if x >= midx and y0 <= y <= y1]

    # quedarnos con 3 porcentajes más “grandes” en vertical (normalmente son 3)
    # dedupe por valor
    dedup = []
    for v, x, y in sorted(right, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in dedup):
            dedup.append((v, x, y))
    # si hay más de 3, elegir 3 por estar más juntos arriba del gráfico (menor y)
    if len(dedup) > 3:
        dedup = sorted(dedup, key=lambda t: t[2])[:3]

    assigned = {"igual": None, "mas": None, "menos": None}
    for v, x, y in dedup[:3]:
        # muestrear color “debajo” del % para caer dentro de la barra
        y_in = y + (H * 0.10)
        rgb = mean_rgb_patch(img, x, y_in, r=7)
        cat = classify_bar_color(rgb)
        assigned[cat] = v

    out["Comparación 2023 - Igual (%)"] = assigned["igual"]
    out["Comparación 2023 - Más seguro (%)"] = assigned["mas"]
    out["Comparación 2023 - Menos seguro (%)"] = assigned["menos"]

    return out


# =========================
# UI
# =========================
st.set_page_config(page_title="MPGP Extractor", layout="wide")
st.title("Extractor MPGP — Percepción ciudadana (PDF o Imágenes)")

st.info(
    "✅ PDFs: funciona sin OCR.\n"
    "✅ Imágenes: requiere OCR (Tesseract). Si no está instalado, la app te lo dirá y no se cae."
)

files = st.file_uploader(
    "Subí PDFs o imágenes (PNG/JPG). Podés subir muchos.",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

rows = []
errors = []

if files:
    prog = st.progress(0)
    for i, f in enumerate(files, start=1):
        name = f.name
        suffix = name.lower().split(".")[-1]

        try:
            if suffix == "pdf":
                data = extract_from_pdf(f.read())
            else:
                img = Image.open(f).convert("RGB")
                arr = np.array(img)
                data = extract_from_image(arr)

            rows.append({"Archivo": name, **data})

        except Exception as e:
            errors.append((name, str(e)))

        prog.progress(int(i / len(files) * 100))
    prog.empty()

if rows:
    df = pd.DataFrame(rows)

    # orden
    cols = [
        "Archivo",
        "Delegación",
        "Seguro en la comunidad (%)",
        "Inseguro en la comunidad (%)",
        "Comparación 2023 - Igual (%)",
        "Comparación 2023 - Más seguro (%)",
        "Comparación 2023 - Menos seguro (%)",
    ]
    df = df[[c for c in cols if c in df.columns]]

    # formato %
    for c in cols[2:]:
        if c in df.columns:
            df[c] = df[c].map(fmt_pct)

    st.subheader("Resultados (tabla)")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="percepcion_ciudadana.csv",
        mime="text/csv",
    )

if errors:
    st.subheader("Errores / Avisos")
    for n, msg in errors:
        st.write(f"- **{n}**: {msg}")

    if any("tesseract" in m.lower() for _, m in errors) or (not TESSERACT_OK):
        st.warning(
            "Para procesar imágenes, necesitás Tesseract instalado.\n\n"
            "Si estás en Streamlit Cloud, te recomiendo agregar un archivo **packages.txt** con:\n"
            "`tesseract-ocr`\n"
        )
