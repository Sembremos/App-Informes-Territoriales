import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st


# =========================================================
# Utilidades básicas
# =========================================================
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

def color_dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5


# =========================================================
# Render + muestreo de color (sin OCR)
# =========================================================
def render_page(page: fitz.Page, zoom: float = 2.0) -> Tuple[fitz.Pixmap, float]:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix, zoom

def sample_rgb(pix: fitz.Pixmap, x: float, y: float, radius: int = 3) -> Tuple[float, float, float]:
    w, h = pix.width, pix.height
    x = int(max(0, min(w - 1, x)))
    y = int(max(0, min(h - 1, y)))

    r_sum = g_sum = b_sum = 0
    n = 0

    samples = pix.samples
    stride = pix.stride

    for dy in range(-radius, radius + 1):
        yy = y + dy
        if yy < 0 or yy >= h:
            continue
        row_start = yy * stride
        for dx in range(-radius, radius + 1):
            xx = x + dx
            if xx < 0 or xx >= w:
                continue
            idx = row_start + xx * 3
            r_sum += samples[idx]
            g_sum += samples[idx + 1]
            b_sum += samples[idx + 2]
            n += 1

    if n == 0:
        return (0.0, 0.0, 0.0)

    return (r_sum / n / 255.0, g_sum / n / 255.0, b_sum / n / 255.0)


# =========================================================
# Búsqueda de página y delegación
# =========================================================
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


# =========================================================
# Extracción: pastel + comparación
# =========================================================
def extract_pie_seguro_inseguro(page: fitz.Page, midx: float, y_start: float, y_end: float) -> Tuple[Optional[float], Optional[float]]:
    words = page.get_text("words")
    vals = []
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

    uniq = []
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


def find_legend_label_positions(page: fitz.Page, midx: float, y_start: float, y_end: float) -> Dict[str, Tuple[float, float]]:
    words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,wordno
    by_line = {}
    for (x0, y0, x1, y1, w, b, ln, wn) in words:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if cx < midx:
            continue
        if cy < y_start or cy > y_end:
            continue
        by_line.setdefault((b, ln), []).append((x0, y0, x1, y1, cx, cy, w))

    best = {}
    for (_b, _ln), items in by_line.items():
        items = sorted(items, key=lambda t: t[0])
        line_txt = norm(" ".join(w for *_, w in items))
        cx_line = sum(t[4] for t in items) / len(items)
        cy_line = sum(t[5] for t in items) / len(items)

        if ("igual" == line_txt) or line_txt.startswith("igual "):
            if "igual" not in best or cy_line > best["igual"][1]:
                best["igual"] = (cx_line, cy_line)
        if "mas seguro" in line_txt:
            if "mas" not in best or cy_line > best["mas"][1]:
                best["mas"] = (cx_line, cy_line)
        if "menos seguro" in line_txt:
            if "menos" not in best or cy_line > best["menos"][1]:
                best["menos"] = (cx_line, cy_line)

    return best


def find_nearest_legend_square(page: fitz.Page, target_xy: Tuple[float, float], midx: float, y_start: float, y_end: float) -> Optional[fitz.Rect]:
    tx, ty = target_xy
    drawings = page.get_drawings()
    candidates = []

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

            if w < 70 and h < 70:
                if cx < tx and abs(cy - ty) <= 30 and (tx - cx) <= 260:
                    score = abs(cy - ty) * 10 + (tx - cx)
                    candidates.append((score, r))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


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
            if h > 90 and w > 15 and w < 280:
                bars.append((h, cx, r))

    if not bars:
        return []

    bars.sort(key=lambda t: t[0], reverse=True)
    top = bars[:12]

    picked: List[fitz.Rect] = []
    picked_cx: List[float] = []

    for _h, cx, r in sorted(top, key=lambda t: t[1]):
        if all(abs(cx - pcx) > 25 for pcx in picked_cx):
            picked.append(r)
            picked_cx.append(cx)
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

    uniq = []
    for v, x, y in sorted(pct_points, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in uniq):
            uniq.append((v, x, y))

    out = []
    for r in bars:
        bx = (r.x0 + r.x1) / 2.0
        best = None
        for (v, x, y) in uniq:
            if y >= r.y0:
                continue
            if y < r.y0 - 420:
                continue
            dx = abs(x - bx)
            if dx > 260:
                continue
            score = dx + (r.y0 - y) * 0.05
            if best is None or score < best[0]:
                best = (score, v)
        out.append(best[1] if best else None)

    return out


def extract_comp_comparison(page: fitz.Page, midx: float, y_start: float, y_end: float) -> Dict[str, Optional[float]]:
    labels = find_legend_label_positions(page, midx, y_start, y_end)
    if not all(k in labels for k in ("igual", "mas", "menos")):
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
        }

    sq_rects = {}
    for cat in ("igual", "mas", "menos"):
        sq_rects[cat] = find_nearest_legend_square(page, labels[cat], midx, y_start, y_end)

    if any(sq_rects[c] is None for c in sq_rects):
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
        }

    bars = find_bar_rects(page, midx, y_start, y_end)
    if len(bars) != 3:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
        }

    pix, zoom = render_page(page, zoom=2.0)

    # colores de leyenda (muestreo REAL)
    legend_colors = {}
    for cat, r in sq_rects.items():
        cx = (r.x0 + r.x1) / 2.0
        cy = (r.y0 + r.y1) / 2.0
        legend_colors[cat] = sample_rgb(pix, cx * zoom, cy * zoom, radius=4)

    # colores de barras (muestreo REAL dentro de la barra)
    bar_colors = []
    for r in bars:
        cx = (r.x0 + r.x1) / 2.0
        cy = (r.y0 + r.y1) / 2.0
        # sample más adentro para evitar borde/sombra
        bar_colors.append(sample_rgb(pix, cx * zoom, cy * zoom, radius=6))

    # porcentajes por barra
    bar_pcts = extract_bar_percentages(page, bars, midx, y_start, y_end)

    # asignación por distancia de color (con garantía: 1 barra por categoría)
    assigned = {"igual": None, "mas": None, "menos": None}
    used_bar = set()

    def pick_best_bar_for_cat(cat: str) -> Optional[int]:
        best = None
        for i, c in enumerate(bar_colors):
            if i in used_bar:
                continue
            d = color_dist(c, legend_colors[cat])
            if best is None or d < best[0]:
                best = (d, i)
        return best[1] if best else None

    for cat in ("igual", "mas", "menos"):
        idx = pick_best_bar_for_cat(cat)
        if idx is not None:
            used_bar.add(idx)
            assigned[cat] = bar_pcts[idx]

    return {
        "Comparación 2023 - Igual (%)": assigned["igual"],
        "Comparación 2023 - Más seguro (%)": assigned["mas"],
        "Comparación 2023 - Menos seguro (%)": assigned["menos"],
    }


# =========================================================
# Extract principal (solo estos datos)
# =========================================================
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

    # =========================
    # FIX CLAVE: NO cortar por "Fuente:"
    # Usar ventana amplia y estable que incluye leyenda SIEMPRE
    # =========================
    y_start = page.rect.height * 0.18
    y_end   = page.rect.height * 0.90
    midx    = page.rect.width  * 0.52

    # Pastel
    seguro, inseguro = extract_pie_seguro_inseguro(page, midx, y_start, y_end)
    out["Seguro en la comunidad (%)"] = seguro
    out["Inseguro en la comunidad (%)"] = inseguro

    # Barras comparación
    out.update(extract_comp_comparison(page, midx, y_start, y_end))

    return out


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="MPGP - Extractor", layout="wide")
st.title("MPGP — Extractor (Percepción ciudadana)")

st.caption(
    "Extrae: Seguro/Inseguro y Comparación 2023 (Igual, Más seguro, Menos seguro). "
    "Sin OCR, soporta cambios de orden y de formato."
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

    # Formato %
    for col in [
        "Seguro en la comunidad (%)",
        "Inseguro en la comunidad (%)",
        "Comparación 2023 - Igual (%)",
        "Comparación 2023 - Más seguro (%)",
        "Comparación 2023 - Menos seguro (%)",
    ]:
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
