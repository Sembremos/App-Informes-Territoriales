import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import colorsys

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st


# =========================
# Utils
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

def render_page(page: fitz.Page, zoom: float = 2.0) -> Tuple[fitz.Pixmap, float]:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix, zoom

def sample_rgb(pix: fitz.Pixmap, x: float, y: float, radius: int = 4) -> Tuple[float, float, float]:
    """Promedio RGB (0..1) alrededor de (x,y) en coordenadas de pixmap."""
    w, h = pix.width, pix.height
    x = int(max(0, min(w - 1, x)))
    y = int(max(0, min(h - 1, y)))

    samples = pix.samples
    stride = pix.stride

    r_sum = g_sum = b_sum = 0
    n = 0
    for dy in range(-radius, radius + 1):
        yy = y + dy
        if yy < 0 or yy >= h:
            continue
        row = yy * stride
        for dx in range(-radius, radius + 1):
            xx = x + dx
            if xx < 0 or xx >= w:
                continue
            idx = row + xx * 3
            r_sum += samples[idx]
            g_sum += samples[idx + 1]
            b_sum += samples[idx + 2]
            n += 1

    if n == 0:
        return (0.0, 0.0, 0.0)
    return (r_sum / n / 255.0, g_sum / n / 255.0, b_sum / n / 255.0)


# =========================
# Identificar página + delegación
# =========================
def find_page_percepcion_ciudadana(doc: fitz.Document) -> Optional[int]:
    anchors = ["percepcion ciudadana", "se siente de seguro en su comunidad"]
    for i in range(doc.page_count):
        t = norm(doc[i].get_text("text") or "")
        if all(a in t for a in anchors):
            return i
    return None

def extract_delegacion(doc: fitz.Document) -> str:
    for i in range(min(8, doc.page_count)):
        t = doc[i].get_text("text") or ""
        m = re.search(r"\bD\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ\s]+", t)
        if m:
            return " ".join(m.group(0).split()).strip()
    return "SIN_DELEGACIÓN"


# =========================
# Pastel (Seguro / Inseguro)
# =========================
def extract_pie_seguro_inseguro(page: fitz.Page, midx: float, y_start: float, y_end: float) -> Tuple[Optional[float], Optional[float]]:
    # Tomamos porcentajes del lado izquierdo (pastel). Elegimos 2 distintos.
    words = page.get_text("words")
    vals: List[float] = []
    for (x0, y0, x1, y1, w, *_rest) in words:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if cx >= midx:
            continue
        if cy < y_start or cy > y_end:
            continue
        v = parse_pct(w)
        if v is not None:
            vals.append(v)

    uniq: List[float] = []
    for v in sorted(vals, reverse=True):
        if all(abs(v - u) > 0.2 for u in uniq):
            uniq.append(v)
        if len(uniq) == 2:
            break

    if len(uniq) != 2:
        return (None, None)

    inseguro = max(uniq)
    seguro = min(uniq)
    return (seguro, inseguro)


# =========================
# Barras (Comparación 2023)
# =========================
def find_bar_rects(page: fitz.Page, midx: float, y_start: float, y_end: float) -> List[fitz.Rect]:
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
            if cy < y_start or cy > y_end:
                continue

            w = r.x1 - r.x0
            h = r.y1 - r.y0

            # barras típicas
            if h > 90 and w > 12 and w < 340:
                bars.append((h, cx, r))

    if not bars:
        return []

    # coger top por altura y quedarnos con 3 x distintos
    bars.sort(key=lambda t: t[0], reverse=True)
    top = bars[:16]

    picked: List[fitz.Rect] = []
    picked_x: List[float] = []
    for _h, cx, r in sorted(top, key=lambda t: t[1]):
        if all(abs(cx - px) > 22 for px in picked_x):
            picked.append(r)
            picked_x.append(cx)
        if len(picked) == 3:
            break

    return picked

def extract_bar_percentages(page: fitz.Page, bars: List[fitz.Rect], midx: float, y_start: float, y_end: float) -> List[Optional[float]]:
    words = page.get_text("words")
    pct_points = []
    for (x0, y0, x1, y1, w, *_rest) in words:
        v = parse_pct(w)
        if v is None:
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if cx < midx or cy < y_start or cy > y_end:
            continue
        pct_points.append((v, cx, cy))

    # dedupe por valor
    uniq = []
    for v, x, y in sorted(pct_points, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in uniq):
            uniq.append((v, x, y))

    out = []
    for r in bars:
        bx = (r.x0 + r.x1) / 2.0
        best = None
        for (v, x, y) in uniq:
            # debe estar sobre la barra
            if y >= r.y0:
                continue
            # ventana vertical razonable
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

def classify_bar_color(rgb: Tuple[float, float, float]) -> str:
    """
    Clasificación robusta por color (SIN leyenda):
    - gris = Igual
    - celeste = Más seguro
    - azul oscuro = Menos seguro
    """
    r, g, b = rgb

    # "gris": canales parecidos y baja saturación
    if abs(r - g) < 0.08 and abs(g - b) < 0.08:
        return "igual"

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # si saturación es muy baja, también gris
    if s < 0.18:
        return "igual"

    # tonos azules: hue aprox 0.50 a 0.75
    # (no lo forzamos demasiado; MPGP es azul/celeste)
    # diferenciar celeste vs azul oscuro por "v" (brillo)
    if v < 0.55:
        return "menos"  # azul más oscuro
    return "mas"        # celeste más brillante

def extract_comp_comparison_by_bar_color(page: fitz.Page, midx: float, y_start: float, y_end: float) -> Dict[str, Optional[float]]:
    bars = find_bar_rects(page, midx, y_start, y_end)
    if len(bars) != 3:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
        }

    bar_pcts = extract_bar_percentages(page, bars, midx, y_start, y_end)

    pix, zoom = render_page(page, zoom=2.0)

    assigned = {"igual": None, "mas": None, "menos": None}

    for r, pct in zip(bars, bar_pcts):
        cx = (r.x0 + r.x1) / 2.0
        # muestrear dentro, cerca del tercio superior (evita sombras)
        y_inside = r.y0 + (r.y1 - r.y0) * 0.35
        rgb = sample_rgb(pix, cx * zoom, y_inside * zoom, radius=6)
        cat = classify_bar_color(rgb)

        # si por alguna razón dos barras caen en misma clase, guardamos la más "coherente":
        # regla: si ya existe y el nuevo pct es None, no sobreescribir
        if assigned[cat] is None or (assigned[cat] is not None and pct is not None):
            assigned[cat] = pct

    return {
        "Comparación 2023 - Igual (%)": assigned["igual"],
        "Comparación 2023 - Más seguro (%)": assigned["mas"],
        "Comparación 2023 - Menos seguro (%)": assigned["menos"],
    }


# =========================
# Extract principal (solo estos datos)
# =========================
def extract_percepcion_ciudadana(doc: fitz.Document) -> Dict[str, Optional[float]]:
    out = {
        "Delegación": extract_delegacion(doc),
        "Seguro en la comunidad (%)": None,
        "Inseguro en la comunidad (%)": None,
        "Comparación 2023 - Igual (%)": None,
        "Comparación 2023 - Más seguro (%)": None,
        "Comparación 2023 - Menos seguro (%)": None,
    }

    pidx = find_page_percepcion_ciudadana(doc)
    if pidx is None:
        return out

    page = doc[pidx]

    # Ventana amplia estable (incluye todo el bloque)
    y_start = page.rect.height * 0.16
    y_end   = page.rect.height * 0.92
    midx    = page.rect.width  * 0.52

    seguro, inseguro = extract_pie_seguro_inseguro(page, midx, y_start, y_end)
    out["Seguro en la comunidad (%)"] = seguro
    out["Inseguro en la comunidad (%)"] = inseguro

    out.update(extract_comp_comparison_by_bar_color(page, midx, y_start, y_end))
    return out


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="MPGP - Extractor", layout="wide")
st.title("MPGP — Extractor (Percepción ciudadana)")

st.caption(
    "Extrae: Seguro/Inseguro y Comparación 2023 (Igual, Más seguro, Menos seguro). "
    "NO usa OCR. NO depende de la leyenda. Clasifica por color de barras."
)

files = st.file_uploader(
    "Suba uno o varios PDF",
    type=["pdf"],
    accept_multiple_files=True
)

if files:
    rows = []
    prog = st.progress(0)
    for i, f in enumerate(files, start=1):
        doc = fitz.open(stream=f.read(), filetype="pdf")
        data = extract_percepcion_ciudadana(doc)
        doc.close()
        rows.append({"Archivo": f.name, **data})
        prog.progress(int(i / len(files) * 100))
    prog.empty()

    df = pd.DataFrame(rows)

    # ordenar columnas como querés verlas
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

    # Formato %
    for col in cols[2:]:
        if col in df.columns:
            df[col] = df[col].map(fmt_pct)

    st.subheader("Resultados (tabla)")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="percepcion_ciudadana.csv",
        mime="text/csv",
    )
else:
    st.info("Cargá PDFs para ver los resultados.")
