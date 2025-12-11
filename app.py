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
# Render + muestreo de color (CLAVE para que no falle)
# =========================================================
def render_page(page: fitz.Page, zoom: float = 2.0) -> Tuple[fitz.Pixmap, float]:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix, zoom

def sample_rgb(pix: fitz.Pixmap, x: float, y: float, radius: int = 3) -> Tuple[float, float, float]:
    """
    Devuelve color promedio (0..1) alrededor del punto (x,y) en coordenadas de pixmap.
    """
    w, h = pix.width, pix.height
    x = int(max(0, min(w - 1, x)))
    y = int(max(0, min(h - 1, y)))

    r_sum = g_sum = b_sum = 0
    n = 0

    # pix.samples es bytes RGBRGB...
    samples = pix.samples
    stride = pix.stride  # bytes por fila

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
            r = samples[idx]
            g = samples[idx + 1]
            b = samples[idx + 2]
            r_sum += r
            g_sum += g
            b_sum += b
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
# Detección de zonas y extracción
# =========================================================
def get_block_bounds_until_fuente(page: fitz.Page) -> Tuple[float, float]:
    """
    Retorna (y_start, y_end) aproximados para el bloque de Percepción ciudadana:
    y_start = línea del título
    y_end   = línea de 'Fuente:' si existe; si no, final de página.
    """
    lines = page.get_text("dict").get("blocks", [])
    y_start = 0.0
    found = False

    # 1) encontrar y_start por "Percepción ciudadana"
    textdict = page.get_text("dict")
    for b in textdict.get("blocks", []):
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            line_text = " ".join((sp.get("text") or "").strip() for sp in ln.get("spans", []) if (sp.get("text") or "").strip())
            if "percepcion ciudadana" in norm(line_text):
                # bbox de la línea
                y0s = [sp["bbox"][1] for sp in ln.get("spans", []) if "bbox" in sp]
                y1s = [sp["bbox"][3] for sp in ln.get("spans", []) if "bbox" in sp]
                if y0s and y1s:
                    y_start = (min(y0s) + max(y1s)) / 2.0
                    found = True
                    break
        if found:
            break

    # 2) encontrar y_end por "Fuente:"
    y_end = page.rect.height
    words = page.get_text("words")
    fuente_candidates = []
    for (x0, y0, x1, y1, w, *_rest) in words:
        if "fuente:" in norm(w):
            cy = (y0 + y1) / 2.0
            if cy > y_start:
                fuente_candidates.append(cy)
    if fuente_candidates:
        y_end = min(fuente_candidates)

    return y_start, y_end

def extract_pie_seguro_inseguro(page: fitz.Page, midx: float, y_start: float, y_end: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Toma el gráfico de pastel (lado izquierdo): devuelve (seguro, inseguro)
    usando los 2 porcentajes más relevantes del lado izquierdo.
    """
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

    # quedarnos con 2 valores distintos
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
    """
    Encuentra coordenadas (cx,cy) de las etiquetas de leyenda:
    igual / mas / menos (en el bloque derecho abajo).
    """
    words = page.get_text("words")  # incluye block,line
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
    # buscamos líneas que contengan las palabras clave
    for (_b, _ln), items in by_line.items():
        items = sorted(items, key=lambda t: t[0])
        line_txt = norm(" ".join(w for *_, w in items))
        cx_line = sum(t[4] for t in items) / len(items)
        cy_line = sum(t[5] for t in items) / len(items)

        if "igual" == line_txt or line_txt.startswith("igual "):
            # nos quedamos con la más baja (normalmente leyenda abajo)
            if "igual" not in best or cy_line > best["igual"][1]:
                best["igual"] = (cx_line, cy_line)
        if "mas seguro" in line_txt:
            if "mas" not in best or cy_line > best["mas"][1]:
                best["mas"] = (cx_line, cy_line)
        if "menos seguro" in line_txt:
            if "menos" not in best or cy_line > best["menos"][1]:
                best["menos"] = (cx_line, cy_line)

    return best  # cat -> (cx,cy)

def find_nearest_legend_square(page: fitz.Page, target_xy: Tuple[float, float], midx: float, y_start: float, y_end: float) -> Optional[fitz.Rect]:
    """
    Busca un rectángulo pequeño (cuadrito) cercano a la izquierda de la etiqueta.
    No usamos fill del vector: solo geometría. Luego muestreamos color renderizado.
    """
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

            # cuadrito típico
            if w < 60 and h < 60:
                # debe estar relativamente cerca y a la izquierda del texto
                if cx < tx and abs(cy - ty) <= 25 and (tx - cx) <= 220:
                    score = abs(cy - ty) * 10 + (tx - cx)
                    candidates.append((score, r))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]

def find_bar_rects(page: fitz.Page, midx: float, y_start: float, y_end: float) -> List[fitz.Rect]:
    """
    Encuentra barras (rectángulos altos) del gráfico de comparación.
    """
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
            if h > 90 and w > 15 and w < 260:
                bars.append((h, cx, r))

    if not bars:
        return []

    # ordenar por altura y elegir hasta 6 (por si hay duplicados/sombras)
    bars.sort(key=lambda t: t[0], reverse=True)
    top = bars[:10]

    # quedarnos con 3 barras con centros X distintos
    picked: List[fitz.Rect] = []
    picked_cx: List[float] = []

    for _h, cx, r in sorted(top, key=lambda t: t[1]):  # por x
        if all(abs(cx - pcx) > 25 for pcx in picked_cx):
            picked.append(r)
            picked_cx.append(cx)
        if len(picked) == 3:
            break

    return picked

def extract_bar_percentages(page: fitz.Page, bars: List[fitz.Rect], midx: float, y_start: float, y_end: float) -> List[Optional[float]]:
    """
    Asigna % a cada barra usando el texto % por cercanía en X y arriba de la barra.
    Devuelve lista [p1,p2,p3] alineada a 'bars' (mismo orden).
    """
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
            # debe estar arriba del tope de la barra
            if y >= r.y0:
                continue
            # ventana vertical razonable
            if y < r.y0 - 360:
                continue
            dx = abs(x - bx)
            if dx > 230:
                continue
            score = dx + (r.y0 - y) * 0.05
            if best is None or score < best[0]:
                best = (score, v)
        out.append(best[1] if best else None)

    return out

def extract_comp_comparison(page: fitz.Page, midx: float, y_start: float, y_end: float) -> Dict[str, Optional[float]]:
    """
    Extrae Igual/Más/Menos correcto usando:
    - leyenda por texto
    - cuadritos (geom)
    - colores renderizados reales (sampling)
    - barras renderizadas reales (sampling)
    - emparejar barra->categoría por distancia de color
    """
    # 1) posiciones de etiquetas leyenda
    labels = find_legend_label_positions(page, midx, y_start, y_end)
    if not all(k in labels for k in ("igual", "mas", "menos")):
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
        }

    # 2) encontrar cuadrito de leyenda cercano a cada etiqueta
    sq_rects = {}
    for cat in ("igual", "mas", "menos"):
        sq = find_nearest_legend_square(page, labels[cat], midx, y_start, y_end)
        sq_rects[cat] = sq

    if any(sq_rects[c] is None for c in sq_rects):
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
        }

    # 3) render + sample colores reales de la leyenda
    pix, zoom = render_page(page, zoom=2.0)

    legend_colors = {}
    for cat, r in sq_rects.items():
        cx = (r.x0 + r.x1) / 2.0
        cy = (r.y0 + r.y1) / 2.0
        legend_colors[cat] = sample_rgb(pix, cx * zoom, cy * zoom, radius=3)

    # 4) barras + sus colores reales
    bars = find_bar_rects(page, midx, y_start, y_end)
    if len(bars) != 3:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
        }

    bar_colors = []
    for r in bars:
        # muestrear en el centro “dentro” de la barra (evitar bordes)
        cx = (r.x0 + r.x1) / 2.0
        cy = (r.y0 + r.y1) / 2.0
        bar_colors.append(sample_rgb(pix, cx * zoom, cy * zoom, radius=4))

    # 5) porcentajes por barra
    bar_pcts = extract_bar_percentages(page, bars, midx, y_start, y_end)

    # 6) asignar barra -> categoría por color (más cercano al color de leyenda)
    assigned = {"igual": None, "mas": None, "menos": None}
    used_bar = set()

    def best_bar_for_cat(cat: str) -> Optional[int]:
        best = None
        for i, c in enumerate(bar_colors):
            if i in used_bar:
                continue
            d = color_dist(c, legend_colors[cat])
            if best is None or d < best[0]:
                best = (d, i)
        # umbral flexible: si es PDF raro, igual tomamos el más cercano
        return best[1] if best else None

    # primero asignar las 3 categorías al mejor match
    for cat in ("igual", "mas", "menos"):
        idx = best_bar_for_cat(cat)
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
    y_start, y_end = get_block_bounds_until_fuente(page)
    midx = page.rect.width * 0.52

    # Pastel
    seguro, inseguro = extract_pie_seguro_inseguro(page, midx, y_start, y_end)
    out["Seguro en la comunidad (%)"] = seguro
    out["Inseguro en la comunidad (%)"] = inseguro

    # Barras comparación
    comp = extract_comp_comparison(page, midx, y_start, y_end)
    out.update(comp)

    return out


# =========================================================
# Streamlit UI
# =========================================================
st.set_page_config(page_title="MPGP - Extractor", layout="wide")
st.title("MPGP — Extractor (Percepción ciudadana)")

st.caption(
    "Extrae: Seguro/Inseguro y Comparación 2023 (Igual, Más seguro, Menos seguro). "
    "No usa OCR. Funciona aunque cambie el orden del gráfico."
)

files = st.file_uploader(
    "Suba uno o varios PDF (pueden ser muchos)",
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
