import io
import re
import unicodedata
import colorsys
from typing import Dict, Optional, Tuple, List

import streamlit as st
import pandas as pd

# --- regex % ---
PCT_RE = re.compile(r"(\d{1,3}[.,]\d{1,2})\s*%")

# =========================
# Utils generales
# =========================
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

def classify_bar_color(rgb01: Tuple[float, float, float]) -> str:
    """
    MPGP:
    - gris -> igual
    - celeste -> mas seguro
    - azul oscuro -> menos seguro
    """
    r, g, b = rgb01

    # gris: canales parecidos
    if abs(r - g) < 0.08 and abs(g - b) < 0.08:
        return "igual"

    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.18:
        return "igual"

    return "menos" if v < 0.55 else "mas"


# =========================
# Chequeo OCR (para imágenes)
# =========================
def ocr_available() -> Tuple[bool, str]:
    try:
        import pytesseract  # noqa
    except Exception as e:
        return False, f"pytesseract no disponible: {e}"

    # En Cloud el binario es lo que suele faltar
    try:
        import shutil
        if shutil.which("tesseract") is None:
            return False, "No se encontró el binario 'tesseract' (falta packages.txt con tesseract-ocr)."
    except Exception:
        pass

    return True, "OK"


# =========================
# PDF EXTRACT (sin OCR)
# =========================
def extract_from_pdf(file_bytes: bytes) -> Dict[str, Optional[float]]:
    try:
        import fitz  # PyMuPDF
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"Faltan dependencias PDF (PyMuPDF/numpy): {e}")

    def find_page(doc: "fitz.Document") -> Optional[int]:
        anchors = ["percepcion ciudadana", "se siente de seguro en su comunidad"]
        for i in range(doc.page_count):
            t = norm(doc[i].get_text("text") or "")
            if all(a in t for a in anchors):
                return i
        return None

    def delegacion(doc: "fitz.Document") -> str:
        for i in range(min(8, doc.page_count)):
            t = doc[i].get_text("text") or ""
            m = re.search(r"\bD\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ\s]+", t)
            if m:
                return " ".join(m.group(0).split()).strip()
        return "SIN_DELEGACIÓN"

    def extract_pie(page: "fitz.Page", midx: float, y0: float, y1: float):
        words = page.get_text("words")
        vals = []
        for (x0, y0w, x1w, y1w, w, *_r) in words:
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
        return min(uniq), max(uniq)  # seguro, inseguro

    def render(page: "fitz.Page", zoom: float = 2.0):
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return img, zoom

    def find_bars(page: "fitz.Page", midx: float, y0: float, y1: float) -> List["fitz.Rect"]:
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
        picked, pxs = [], []
        for _h, cx, r in sorted(top, key=lambda t: t[1]):
            if all(abs(cx - x) > 22 for x in pxs):
                picked.append(r)
                pxs.append(cx)
            if len(picked) == 3:
                break
        return picked

    def bar_pcts(page: "fitz.Page", bars: List["fitz.Rect"], midx: float, y0: float, y1: float):
        words = page.get_text("words")
        pts = []
        for (x0, y0w, x1w, y1w, w, *_r) in words:
            v = parse_pct(w)
            if v is None:
                continue
            cx = (x0 + x1w) / 2.0
            cy = (y0w + y1w) / 2.0
            if cx < midx or cy < y0 or cy > y1:
                continue
            pts.append((v, cx, cy))

        # dedupe
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

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    out = {
        "Delegación": delegacion(doc),
        "Seguro en la comunidad (%)": None,
        "Inseguro en la comunidad (%)": None,
        "Comparación 2023 - Igual (%)": None,
        "Comparación 2023 - Más seguro (%)": None,
        "Comparación 2023 - Menos seguro (%)": None,
    }

    pidx = find_page(doc)
    if pidx is None:
        doc.close()
        return out

    page = doc[pidx]
    y0 = page.rect.height * 0.16
    y1 = page.rect.height * 0.92
    midx = page.rect.width * 0.52

    seguro, inseguro = extract_pie(page, midx, y0, y1)
    out["Seguro en la comunidad (%)"] = seguro
    out["Inseguro en la comunidad (%)"] = inseguro

    bars = find_bars(page, midx, y0, y1)
    if len(bars) == 3:
        pcts = bar_pcts(page, bars, midx, y0, y1)
        img, zoom = render(page, zoom=2.0)

        assigned = {"igual": None, "mas": None, "menos": None}
        for r, pct in zip(bars, pcts):
            cx = (r.x0 + r.x1) / 2.0
            y_inside = r.y0 + (r.y1 - r.y0) * 0.35
            px = int(cx * zoom)
            py = int(y_inside * zoom)
            px = max(0, min(img.shape[1] - 1, px))
            py = max(0, min(img.shape[0] - 1, py))
            patch = img[max(0, py-6):py+7, max(0, px-6):px+7, :]
            mean = patch.mean(axis=(0, 1)) / 255.0
            cat = classify_bar_color((float(mean[0]), float(mean[1]), float(mean[2])))
            assigned[cat] = pct

        out["Comparación 2023 - Igual (%)"] = assigned["igual"]
        out["Comparación 2023 - Más seguro (%)"] = assigned["mas"]
        out["Comparación 2023 - Menos seguro (%)"] = assigned["menos"]

    doc.close()
    return out


# =========================
# IMAGE EXTRACT (con OCR)
# =========================
def extract_from_image(file_bytes: bytes) -> Dict[str, Optional[float]]:
    ok, msg = ocr_available()
    if not ok:
        raise RuntimeError(msg)

    import numpy as np
    from PIL import Image
    import pytesseract

    pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = np.array(pil)
    H, W, _ = img.shape

    # OCR data
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT, config="--psm 6")

    # extraer porcentajes con centroides
    pts: List[Tuple[float, float, float]] = []
    for i in range(len(data["text"])):
        txt = (data["text"][i] or "").strip()
        v = parse_pct(txt)
        if v is None:
            continue
        x = data["left"][i]
        y = data["top"][i]
        w = data["width"][i]
        h = data["height"][i]
        pts.append((v, x + w/2.0, y + h/2.0))

    # Ventana aproximada (layout MPGP)
    y0 = H * 0.12
    y1 = H * 0.94
    midx = W * 0.52

    out = {
        "Delegación": "DESDE_IMAGEN",
        "Seguro en la comunidad (%)": None,
        "Inseguro en la comunidad (%)": None,
        "Comparación 2023 - Igual (%)": None,
        "Comparación 2023 - Más seguro (%)": None,
        "Comparación 2023 - Menos seguro (%)": None,
    }

    # Pastel: 2 % a la izquierda
    left_vals = [v for (v, x, y) in pts if x < midx and y0 <= y <= y1]
    uniq = []
    for v in sorted(left_vals, reverse=True):
        if all(abs(v - u) > 0.2 for u in uniq):
            uniq.append(v)
        if len(uniq) == 2:
            break
    if len(uniq) == 2:
        out["Inseguro en la comunidad (%)"] = max(uniq)
        out["Seguro en la comunidad (%)"] = min(uniq)

    # Comparación: 3 % a la derecha (encima de barras)
    right = [(v, x, y) for (v, x, y) in pts if x >= midx and y0 <= y <= y1]
    # dedupe por valor
    dedup = []
    for v, x, y in sorted(right, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in dedup):
            dedup.append((v, x, y))

    # nos quedamos con 3 más “arriba” (menor y)
    dedup = sorted(dedup, key=lambda t: t[2])[:3]

    def mean_rgb_patch(x: float, y: float, r: int = 7) -> Tuple[float, float, float]:
        xi = int(max(0, min(W - 1, x)))
        yi = int(max(0, min(H - 1, y)))
        patch = img[max(0, yi-r):yi+r+1, max(0, xi-r):xi+r+1, :]
        mean = patch.mean(axis=(0, 1)) / 255.0
        return (float(mean[0]), float(mean[1]), float(mean[2]))

    assigned = {"igual": None, "mas": None, "menos": None}
    for v, x, y in dedup:
        # muestrear color dentro de la barra (debajo del %)
        y_in = y + H * 0.10
        rgb = mean_rgb_patch(x, y_in, r=8)
        cat = classify_bar_color(rgb)
        assigned[cat] = v

    out["Comparación 2023 - Igual (%)"] = assigned["igual"]
    out["Comparación 2023 - Más seguro (%)"] = assigned["mas"]
    out["Comparación 2023 - Menos seguro (%)"] = assigned["menos"]

    return out


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="MPGP Extractor", layout="wide")
st.title("Extractor MPGP — Percepción ciudadana (PDF o Imágenes)")

ok_ocr, msg_ocr = ocr_available()
if ok_ocr:
    st.success("OCR para imágenes: OK (tesseract disponible).")
else:
    st.warning(f"OCR para imágenes: NO disponible. Motivo: {msg_ocr}")

st.caption("Extrae: Seguro/Inseguro y Comparación 2023 (Igual/Más/Menos).")

files = st.file_uploader(
    "Subí archivos (PDF, PNG, JPG, JPEG). Podés subir muchos.",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

rows = []
errs = []

if files:
    prog = st.progress(0)
    for i, f in enumerate(files, start=1):
        name = f.name
        ext = name.lower().split(".")[-1]

        try:
            b = f.read()
            if ext == "pdf":
                data = extract_from_pdf(b)
            else:
                data = extract_from_image(b)

            rows.append({"Archivo": name, **data})

        except Exception as e:
            errs.append((name, str(e)))

        prog.progress(int(i / len(files) * 100))
    prog.empty()

if rows:
    df = pd.DataFrame(rows)
    order = [
        "Archivo", "Delegación",
        "Seguro en la comunidad (%)", "Inseguro en la comunidad (%)",
        "Comparación 2023 - Igual (%)", "Comparación 2023 - Más seguro (%)", "Comparación 2023 - Menos seguro (%)",
    ]
    df = df[[c for c in order if c in df.columns]]

    for c in order[2:]:
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

if errs:
    st.subheader("Errores por archivo")
    for n, m in errs:
        st.error(f"{n}: {m}")

if not files:
    st.info("Subí PDFs o imágenes para procesar.")
