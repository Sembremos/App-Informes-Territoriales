import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from collections import Counter

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st


# =========================
# Helpers
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


PCT_RE = re.compile(r"(\d{1,3}[.,]\d{1,2})\s*%")

def parse_pct(txt: str) -> Optional[float]:
    if not txt:
        return None
    m = PCT_RE.search(txt)
    if not m:
        return None
    s = m.group(1).replace(",", ".")
    try:
        v = float(s)
        if 0 <= v <= 100:
            return v
        return None
    except:
        return None


def extract_delegacion(doc: fitz.Document) -> str:
    for i in range(min(8, doc.page_count)):
        t = doc[i].get_text("text") or ""
        m = re.search(r"\bD\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ\s]+", t)
        if m:
            return " ".join(m.group(0).split()).strip()
    return "SIN_DELEGACIÓN"


def page_text_norm(doc: fitz.Document, i: int) -> str:
    return norm(doc[i].get_text("text") or "")


def find_page_percepcion_ciudadana(doc: fitz.Document) -> Optional[int]:
    anchors = ["percepcion ciudadana", "se siente de seguro en su comunidad"]
    for i in range(doc.page_count):
        t = page_text_norm(doc, i)
        if all(a in t for a in anchors):
            return i
    return None


def unique_top(vals: List[float], k: int, tol: float = 0.2) -> List[float]:
    out = []
    for v in sorted(vals, reverse=True):
        if all(abs(v - u) > tol for u in out):
            out.append(v)
        if len(out) >= k:
            break
    return out


# =========================
# Spans / lines (Hatillo: etiquetas debajo)
# =========================
def get_spans(page: fitz.Page) -> List[Dict]:
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
                    "ntext": norm(text),
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "cx": (x0 + x1) / 2.0,
                    "cy": (y0 + y1) / 2.0,
                })
    return spans


def build_lines_from_dict(page: fitz.Page) -> List[Dict]:
    d = page.get_text("dict")
    out = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            texts = []
            x0s, y0s, x1s, y1s = [], [], [], []
            cxs, cys = [], []
            for sp in line.get("spans", []):
                t = (sp.get("text") or "").strip()
                if not t:
                    continue
                x0, y0, x1, y1 = sp.get("bbox", (0, 0, 0, 0))
                texts.append(t)
                x0s.append(x0); y0s.append(y0); x1s.append(x1); y1s.append(y1)
                cxs.append((x0 + x1) / 2.0)
                cys.append((y0 + y1) / 2.0)
            if not texts:
                continue
            out.append({
                "text": " ".join(texts),
                "ntext": norm(" ".join(texts)),
                "cx": sum(cxs)/len(cxs),
                "cy": sum(cys)/len(cys),
                "x0": min(x0s), "y0": min(y0s),
                "x1": max(x1s), "y1": max(y1s),
            })
    return out


def slice_until_fuente(spans: List[Dict], y_start: float) -> List[Dict]:
    fuente_y = None
    for sp in sorted(spans, key=lambda z: z["cy"]):
        if "fuente:" in sp["ntext"] and sp["cy"] > y_start:
            fuente_y = sp["cy"]
            break
    y_end = fuente_y if fuente_y is not None else max(sp["cy"] for sp in spans) + 1
    return [sp for sp in spans if y_start <= sp["cy"] <= y_end]


def extract_comp_by_labels(page: fitz.Page, lines: List[Dict], midx: float, y_start: float, block_right: List[Dict]) -> Dict[str, Optional[float]]:
    y_max = max(sp["cy"] for sp in block_right) if block_right else y_start + 9999
    right_lines = [ln for ln in lines if ln["cx"] >= midx and (y_start <= ln["cy"] <= y_max + 10)]

    def find_label_line(needle: str) -> Optional[Tuple[float, float]]:
        cands = [ln for ln in right_lines if needle in ln["ntext"]]
        if not cands:
            return None
        cands.sort(key=lambda a: a["cy"], reverse=True)
        return (cands[0]["cx"], cands[0]["cy"])

    c_igual = find_label_line("igual")
    c_menos = find_label_line("menos seguro")
    c_mas = find_label_line("mas seguro")

    # porcentajes en lado derecho
    pct_points = []
    for sp in block_right:
        v = parse_pct(sp["text"])
        if v is not None:
            pct_points.append((v, sp["cx"], sp["cy"]))

    label_ys = [c[1] for c in [c_igual, c_menos, c_mas] if c]
    if label_ys:
        base_y = max(label_ys)
        candidates = [(v, x, y) for (v, x, y) in pct_points if y < base_y and y > base_y - 350]
    else:
        candidates = pct_points

    uniq = []
    for v, x, y in sorted(candidates, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in uniq):
            uniq.append((v, x, y))

    def best_for_label(lx, ly, used):
        best = None
        for i, (v, x, y) in enumerate(uniq):
            if i in used: 
                continue
            if y >= ly: 
                continue
            dx = abs(x - lx)
            dy = abs(ly - y)
            if dx > 260:
                continue
            score = dx + dy * 0.05
            if best is None or score < best[0]:
                best = (score, i, v)
        return best

    got = {"igual": None, "menos": None, "mas": None}
    used = set()
    for key, c in [("igual", c_igual), ("menos", c_menos), ("mas", c_mas)]:
        if not c:
            continue
        b = best_for_label(c[0], c[1], used)
        if b:
            used.add(b[1])
            got[key] = b[2]

    # anti-swap: mas suele ser el menor
    if all(got[k] is not None for k in got):
        m = min(got["igual"], got["menos"], got["mas"])
        if got["menos"] == m and got["mas"] != m:
            got["menos"], got["mas"] = got["mas"], got["menos"]

    return {
        "Comparación 2023 - Igual (%)": got["igual"],
        "Comparación 2023 - Menos seguro (%)": got["menos"],
        "Comparación 2023 - Más seguro (%)": got["mas"],
    }


# =========================
# Modo 2: leyenda por LÍNEAS (fix real)
# =========================
def rgb_tuple(c):
    if c is None:
        return None
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return (float(c[0]), float(c[1]), float(c[2]))
    return None


def color_dist(a, b) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5


def extract_comp_by_legend_lines(page: fitz.Page, midx: float, y_start: float, y_end: float) -> Dict[str, Optional[float]]:
    # words: x0,y0,x1,y1,word,block,line,wordno
    words = page.get_text("words")
    # agrupar por (block,line)
    by_line = {}
    for (x0, y0, x1, y1, w, b, ln, _wn) in words:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if cx < midx:
            continue
        if cy < y_start or cy > y_end:
            continue
        by_line.setdefault((b, ln), []).append((x0, y0, x1, y1, cx, cy, w))

    # encontrar líneas de leyenda que contengan las etiquetas
    targets = {}  # cat -> (cy, x_left)
    for (b, ln), items in by_line.items():
        items_sorted = sorted(items, key=lambda t: t[0])
        text_line = norm(" ".join(w for *_rest, w in items_sorted))
        cy_line = sum(t[5] for t in items_sorted) / len(items_sorted)
        x_left = min(t[0] for t in items_sorted)

        if "igual" == text_line or text_line.startswith("igual "):
            targets["igual"] = (cy_line, x_left)
        elif "mas seguro" in text_line:
            targets["mas"] = (cy_line, x_left)
        elif "menos seguro" in text_line:
            targets["menos"] = (cy_line, x_left)

    if len(targets) < 2:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
        }

    # rectángulos pequeños (cuadritos de color) cerca de la leyenda:
    drawings = page.get_drawings()
    legend_squares = []
    for d in drawings:
        fill = rgb_tuple(d.get("fill"))
        if fill is None:
            continue
        for it in d.get("items", []):
            if not it or it[0] != "re":
                continue
            r = it[1]
            w = r.x1 - r.x0
            h = r.y1 - r.y0
            cy = (r.y0 + r.y1) / 2.0
            cx = (r.x0 + r.x1) / 2.0
            if cx < midx:
                continue
            if cy < y_start or cy > y_end:
                continue
            if w < 50 and h < 50:
                legend_squares.append((r, fill))

    # mapear color de cada categoría buscando el cuadrito más cercano a la izquierda de su texto
    cat_to_color = {}
    for cat, (cy_line, x_left) in targets.items():
        best = None
        for (r, fill) in legend_squares:
            cy_sq = (r.y0 + r.y1) / 2.0
            cx_sq = (r.x0 + r.x1) / 2.0
            # debe estar a la izquierda del texto
            if cx_sq > x_left:
                continue
            dy = abs(cy_sq - cy_line)
            dx = abs(x_left - cx_sq)
            if dy > 18:
                continue
            score = dx + dy * 5
            if best is None or score < best[0]:
                best = (score, fill)
        if best:
            cat_to_color[cat] = best[1]

    if len(cat_to_color) < 2:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
        }

    # detectar barras (rect altos)
    bar_rects = []
    for d in drawings:
        fill = rgb_tuple(d.get("fill"))
        if fill is None:
            continue
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
            if h > 90 and w > 15 and w < 220:
                bar_rects.append((r, fill))

    # porcentajes (texto) en bloque derecho
    pct_points = []
    for (x0, y0, x1, y1, w, b, ln, wn) in words:
        v = parse_pct(w)
        if v is None:
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if cx < midx or cy < y_start or cy > y_end:
            continue
        pct_points.append((v, cx, cy))

    uniq_pcts = []
    for v, x, y in sorted(pct_points, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in uniq_pcts):
            uniq_pcts.append((v, x, y))

    # asignar color barra -> cat por cercanía a color de leyenda
    def nearest_cat(fill):
        best = None
        for cat, c in cat_to_color.items():
            d = color_dist(fill, c)
            if best is None or d < best[0]:
                best = (d, cat)
        return best[1] if best and best[0] < 0.35 else None

    results = {"igual": None, "menos": None, "mas": None}
    for r, fill in bar_rects:
        cat = nearest_cat(fill)
        if not cat:
            continue
        bx = (r.x0 + r.x1) / 2.0
        best = None
        for (v, x, y) in uniq_pcts:
            if y >= r.y0:
                continue
            if y < r.y0 - 320:
                continue
            dx = abs(x - bx)
            if dx > 190:
                continue
            score = dx + (r.y0 - y) * 0.05
            if best is None or score < best[0]:
                best = (score, v)
        if best and results[cat] is None:
            results[cat] = best[1]

    # fallback: completar con top3
    top3 = unique_top([v for v, _, _ in uniq_pcts], 3)
    got = [v for v in results.values() if v is not None]
    if len(top3) == 3 and len(got) == 2:
        rem = [v for v in top3 if all(abs(v - g) > 0.2 for g in got)]
        if rem:
            for k in results:
                if results[k] is None:
                    results[k] = rem[0]

    return {
        "Comparación 2023 - Igual (%)": results["igual"],
        "Comparación 2023 - Menos seguro (%)": results["menos"],
        "Comparación 2023 - Más seguro (%)": results["mas"],
    }


def comp_values_ok(comp: Dict[str, Optional[float]]) -> bool:
    vals = [
        comp.get("Comparación 2023 - Igual (%)"),
        comp.get("Comparación 2023 - Menos seguro (%)"),
        comp.get("Comparación 2023 - Más seguro (%)"),
    ]
    if any(v is None for v in vals):
        return False
    s = sum(vals)
    return 97.0 <= s <= 103.0


# =========================
# Extracción principal (solo lo que pediste)
# =========================
def extract_percepcion_ciudadana(doc: fitz.Document) -> Dict:
    out = {
        "Delegación": extract_delegacion(doc),
        "Seguro en la comunidad (%)": None,
        "Inseguro en la comunidad (%)": None,
        "Comparación 2023 - Igual (%)": None,
        "Comparación 2023 - Menos seguro (%)": None,
        "Comparación 2023 - Más seguro (%)": None,
    }

    pidx = find_page_percepcion_ciudadana(doc)
    if pidx is None:
        return out

    page = doc[pidx]
    spans = get_spans(page)
    lines = build_lines_from_dict(page)

    # ancla vertical
    y_start = None
    for ln in lines:
        if "percepcion ciudadana" in ln["ntext"]:
            y_start = ln["cy"]
            break
    if y_start is None:
        y_start = 0.0

    block = slice_until_fuente(spans, y_start)
    y_end = max(sp["cy"] for sp in block) if block else page.rect.height

    # dividir izquierda/derecha
    w = page.rect.width
    midx = w * 0.52
    left = [sp for sp in block if sp["cx"] < midx]
    right = [sp for sp in block if sp["cx"] >= midx]

    # PIE: 2 % principales
    left_pcts = []
    for sp in left:
        v = parse_pct(sp["text"])
        if v is not None:
            left_pcts.append(v)
    two = unique_top(left_pcts, 2)
    if len(two) == 2:
        out["Inseguro en la comunidad (%)"] = max(two)
        out["Seguro en la comunidad (%)"] = min(two)

    # Modo 1 (Hatillo)
    comp1 = extract_comp_by_labels(page, lines, midx, y_start, right)
    if comp_values_ok(comp1):
        out.update(comp1)
        return out

    # Modo 2 (leyenda por líneas)
    comp2 = extract_comp_by_legend_lines(page, midx, y_start, y_end)
    out.update(comp2)
    return out


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="MPGP - Extractor", layout="wide")
st.title("MPGP — Extraer Percepción ciudadana (solo estos datos)")

files = st.file_uploader("Suba uno o varios PDF", type=["pdf"], accept_multiple_files=True)

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

    pct_cols = [
        "Seguro en la comunidad (%)",
        "Inseguro en la comunidad (%)",
        "Comparación 2023 - Igual (%)",
        "Comparación 2023 - Menos seguro (%)",
        "Comparación 2023 - Más seguro (%)",
    ]
    for c in pct_cols:
        df[c] = df[c].map(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else None)

    st.subheader("Resultados (tabla)")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        "percepcion_ciudadana.csv",
        "text/csv",
    )
else:
    st.info("Cargá PDFs para extraer los datos.")
