import re
import unicodedata
from typing import Dict, List, Optional, Tuple

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
    anchors = [
        "percepcion ciudadana",
        "se siente de seguro en su comunidad",
    ]
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
# Spans / lines (para Hatillo)
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
    lines_out = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            texts = []
            x0s, y0s, x1s, y1s = [], [], [], []
            cxs, cys = [], []
            for sp in spans:
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
            txt = " ".join(texts)
            lines_out.append({
                "text": txt,
                "ntext": norm(txt),
                "cx": sum(cxs) / len(cxs),
                "cy": sum(cys) / len(cys),
                "x0": min(x0s), "y0": min(y0s),
                "x1": max(x1s), "y1": max(y1s),
            })
    return lines_out


def slice_until_fuente(spans: List[Dict], y_start: float) -> List[Dict]:
    fuente_y = None
    for sp in sorted(spans, key=lambda z: z["cy"]):
        if "fuente:" in sp["ntext"] and sp["cy"] > y_start:
            fuente_y = sp["cy"]
            break
    y_end = fuente_y if fuente_y is not None else max(sp["cy"] for sp in spans) + 1
    return [sp for sp in spans if y_start <= sp["cy"] <= y_end]


# =========================
# Modo 1: Hatillo (etiquetas debajo de barras)
# =========================
def extract_comp_by_labels(page: fitz.Page, spans: List[Dict], lines: List[Dict], midx: float, y_start: float, block_right: List[Dict]) -> Dict[str, Optional[float]]:
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

    pct_points = []
    for sp in block_right:
        v = parse_pct(sp["text"])
        if v is not None:
            pct_points.append((v, sp["cx"], sp["cy"]))

    label_ys = [c[1] for c in [c_igual, c_menos, c_mas] if c is not None]
    if label_ys:
        base_y = max(label_ys)
        candidates = [(v, x, y) for (v, x, y) in pct_points if y < base_y and y > base_y - 350]
    else:
        candidates = pct_points

    uniq = []
    for v, x, y in sorted(candidates, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in uniq):
            uniq.append((v, x, y))

    labels = []
    if c_igual: labels.append(("igual", c_igual))
    if c_menos: labels.append(("menos", c_menos))
    if c_mas: labels.append(("mas", c_mas))

    used = set()
    got = {"igual": None, "menos": None, "mas": None}

    def best_for_label(lx, ly):
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
            score = dx * 1.0 + dy * 0.05
            if best is None or score < best[0]:
                best = (score, i, v)
        return best

    for key, (lx, ly) in labels:
        b = best_for_label(lx, ly)
        if b:
            used.add(b[1])
            got[key] = b[2]

    top3 = unique_top([v for v, _, _ in uniq], 3)
    current = [v for v in got.values() if v is not None]
    if len(labels) == 3 and len(current) == 2 and len(top3) == 3:
        rem = [v for v in top3 if all(abs(v - c) > 0.2 for c in current)]
        if rem:
            for k in got:
                if got[k] is None:
                    got[k] = rem[0]

    # anti-swap: "más seguro" suele ser el menor
    if all(got[k] is not None for k in ["igual", "menos", "mas"]):
        menor = min(got["igual"], got["menos"], got["mas"])
        if got["menos"] == menor and got["mas"] != menor:
            got["menos"], got["mas"] = got["mas"], got["menos"]

    return {
        "Comparación 2023 - Igual (%)": got["igual"],
        "Comparación 2023 - Menos seguro (%)": got["menos"],
        "Comparación 2023 - Más seguro (%)": got["mas"],
    }


# =========================
# Modo 2: Leyenda por color (CORREGIDO)
# =========================
def rgb_tuple(c):
    if c is None:
        return None
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return (float(c[0]), float(c[1]), float(c[2]))
    return None


def color_dist(a, b) -> float:
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2) ** 0.5


def extract_comp_by_legend_color(page: fitz.Page, midx: float, y_start: float, y_end: float) -> Dict[str, Optional[float]]:
    words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,wordno

    # helper words centers
    words2 = []
    for (x0, y0, x1, y1, w, *_rest) in words:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        words2.append((x0, y0, x1, y1, cx, cy, w))

    drawings = page.get_drawings()
    filled_rects = []
    for d in drawings:
        fill = rgb_tuple(d.get("fill"))
        if fill is None:
            continue
        for it in d.get("items", []):
            if not it:
                continue
            if it[0] != "re":
                continue
            rect = it[1]
            if rect.x1 < midx:
                continue
            if rect.y1 < y_start or rect.y0 > y_end:
                continue
            w = rect.x1 - rect.x0
            h = rect.y1 - rect.y0
            filled_rects.append((rect, fill, w, h))

    if not filled_rects:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
        }

    legend_rects = []
    bar_rects = []
    for rect, fill, w, h in filled_rects:
        # leyenda: cuadrito pequeño y abajo
        if w < 45 and h < 45 and rect.y0 > (y_end - 320):
            legend_rects.append((rect, fill))
        # barras: rect alto
        elif h > 80 and w > 15 and rect.y0 > (y_start + 40):
            bar_rects.append((rect, fill))

    if not legend_rects or not bar_rects:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
        }

    # --- CORRECCIÓN CLAVE ---
    # En vez de una caja grande, tomamos SOLO palabras cuya cy esté cerca del cy del cuadrito
    def legend_label_for_rect(rect) -> str:
        cy0 = (rect.y0 + rect.y1) / 2.0
        x_min = rect.x1 + 4
        x_max = rect.x1 + 260
        band = 12  # estricto para no comerse otras líneas

        line_words = []
        for (x0, y0, x1, y1, cx, cy, w) in words2:
            if x0 < x_min or x1 > x_max:
                continue
            if abs(cy - cy0) > band:
                continue
            line_words.append((x0, w))

        line_words.sort(key=lambda t: t[0])
        return norm(" ".join(w for _, w in line_words))

    # mapa color_leyenda -> categoria
    color_to_cat = {}
    for rect, fill in legend_rects:
        label = legend_label_for_rect(rect)
        cat = None
        if "igual" in label:
            cat = "igual"
        elif "mas seguro" in label:
            cat = "mas"
        elif "menos seguro" in label:
            cat = "menos"

        if cat:
            # guardamos el color asociado a esa categoría
            color_to_cat[fill] = cat

    if not color_to_cat:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
        }

    # % en el bloque derecho
    pct_points = []
    for (x0, y0, x1, y1, cx, cy, w) in words2:
        v = parse_pct(w)
        if v is None:
            continue
        if cx < midx:
            continue
        if cy < y_start or cy > y_end:
            continue
        pct_points.append((v, cx, cy))

    uniq_pcts = []
    for v, x, y in sorted(pct_points, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in uniq_pcts):
            uniq_pcts.append((v, x, y))

    # asignación color barra -> categoria (por distancia con el color de leyenda)
    def nearest_cat(fill):
        best = None
        for c_leg, cat in color_to_cat.items():
            d = color_dist(fill, c_leg)
            if best is None or d < best[0]:
                best = (d, cat)
        return best[1] if best and best[0] < 0.35 else None

    results = {"igual": None, "menos": None, "mas": None}

    # para cada barra: buscar % arriba cercano al centro X de la barra
    for rect, fill in bar_rects:
        cat = nearest_cat(fill)
        if not cat:
            continue

        bx = (rect.x0 + rect.x1) / 2.0
        best = None
        for (v, x, y) in uniq_pcts:
            if y >= rect.y0:
                continue
            if y < rect.y0 - 280:
                continue
            dx = abs(x - bx)
            if dx > 170:
                continue
            score = dx + (rect.y0 - y) * 0.05
            if best is None or score < best[0]:
                best = (score, v)

        if best and results[cat] is None:
            results[cat] = best[1]

    # fallback top3
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
# Extracción principal
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

    y_start = None
    for ln in lines:
        if "percepcion ciudadana" in ln["ntext"]:
            y_start = ln["cy"]
            break
    if y_start is None:
        y_start = 0.0

    block = slice_until_fuente(spans, y_start)

    w = page.rect.width
    midx = w * 0.52
    left = [sp for sp in block if sp["cx"] < midx]
    right = [sp for sp in block if sp["cx"] >= midx]

    # PIE
    left_pcts = []
    for sp in left:
        v = parse_pct(sp["text"])
        if v is not None:
            left_pcts.append(v)

    two = unique_top(left_pcts, 2)
    if len(two) == 2:
        out["Inseguro en la comunidad (%)"] = max(two)
        out["Seguro en la comunidad (%)"] = min(two)

    y_end = max(sp["cy"] for sp in block) if block else page.rect.height

    # Modo 1
    comp1 = extract_comp_by_labels(page, spans, lines, midx, y_start, right)
    if comp_values_ok(comp1):
        out.update(comp1)
        return out

    # Modo 2 (corregido)
    comp2 = extract_comp_by_legend_color(page, midx, y_start, y_end)
    if comp_values_ok(comp2):
        out.update(comp2)
        return out

    # fallback: el que tenga más valores
    def count_vals(comp):
        return sum(1 for v in comp.values() if isinstance(v, (int, float)))

    out.update(comp2 if count_vals(comp2) >= count_vals(comp1) else comp1)
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

        row = {"Archivo": f.name, **data}
        rows.append(row)
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
