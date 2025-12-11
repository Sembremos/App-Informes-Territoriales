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
# Text spans/lines (para pie y Hatillo)
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


# =========================
# Modo Hatillo: etiquetas debajo
# =========================
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
            if dx > 260:
                continue
            score = dx + (ly - y) * 0.05
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

    return {
        "Comparación 2023 - Igual (%)": got["igual"],
        "Comparación 2023 - Menos seguro (%)": got["menos"],
        "Comparación 2023 - Más seguro (%)": got["mas"],
    }


# =========================
# Modo robusto: orden por LEYENDA (texto) + barras izq→der
# =========================
def extract_comp_by_legend_order(page: fitz.Page, midx: float, y_start: float, y_end: float) -> Dict[str, Optional[float]]:
    words = page.get_text("words")  # x0,y0,x1,y1,word,block,line,wordno

    # --- 1) leyenda: capturar posiciones de las 3 etiquetas por línea ---
    by_line = {}
    for (x0, y0, x1, y1, w, b, ln, wn) in words:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if cx < midx:
            continue
        if cy < y_start or cy > y_end:
            continue
        by_line.setdefault((b, ln), []).append((x0, y0, x1, y1, cx, cy, w))

    legend_hits = []  # (cat, x, y)
    for (_b, _ln), items in by_line.items():
        items = sorted(items, key=lambda t: t[0])
        line_txt = norm(" ".join(w for *_, w in items))
        cx_line = sum(t[4] for t in items) / len(items)
        cy_line = sum(t[5] for t in items) / len(items)

        # detecta categorías aunque vengan con variaciones leves
        if "igual" == line_txt or line_txt.startswith("igual "):
            legend_hits.append(("igual", cx_line, cy_line))
        if "mas seguro" in line_txt:
            legend_hits.append(("mas", cx_line, cy_line))
        if "menos seguro" in line_txt:
            legend_hits.append(("menos", cx_line, cy_line))

    # dejar solo 1 por categoría (por si hay repeticiones)
    best = {}
    for cat, x, y in legend_hits:
        # preferimos las que están más abajo (normalmente la leyenda va abajo del gráfico)
        if cat not in best or y > best[cat][1]:
            best[cat] = (x, y)

    if len(best) < 3:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
        }

    # determinar si la leyenda está vertical u horizontal
    xs = [best[c][0] for c in ["igual", "mas", "menos"]]
    ys = [best[c][1] for c in ["igual", "mas", "menos"]]
    spread_x = max(xs) - min(xs)
    spread_y = max(ys) - min(ys)

    # si están casi alineadas en X → lista vertical; si no, horizontal
    if spread_x < 70 and spread_y > 20:
        # vertical: arriba→abajo (y menor primero)
        legend_order = sorted(["igual", "mas", "menos"], key=lambda c: best[c][1])
    else:
        # horizontal: izq→der
        legend_order = sorted(["igual", "mas", "menos"], key=lambda c: best[c][0])

    # --- 2) barras: detectar 3 rectángulos altos y ordenar izq→der ---
    drawings = page.get_drawings()
    bars = []
    for d in drawings:
        fill = d.get("fill")
        for it in d.get("items", []):
            if not it or it[0] != "re":
                continue
            r = it[1]
            cx = (r.x0 + r.x1) / 2.0
            cy = (r.y0 + r.y1) / 2.0
            if cx < midx or cy < y_start or cy > y_end:
                continue
            w = r.x1 - r.x0
            h = r.y1 - r.y0
            # barras: altas y angostas
            if h > 90 and w > 15 and w < 240:
                bars.append((r, cx, h))

    if not bars:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
        }

    # tomar las 3 barras más altas (por si hay cosas extra dibujadas)
    bars.sort(key=lambda t: t[2], reverse=True)
    top = bars[:8]  # margen por si hay duplicados
    # quedarnos con 3 por centros X distintos
    picked = []
    for r, cx, h in sorted(top, key=lambda t: t[1]):
        if all(abs(cx - pcx) > 25 for (_r, pcx, _h) in picked):
            picked.append((r, cx, h))
        if len(picked) == 3:
            break

    if len(picked) != 3:
        return {
            "Comparación 2023 - Igual (%)": None,
            "Comparación 2023 - Menos seguro (%)": None,
            "Comparación 2023 - Más seguro (%)": None,
        }

    picked.sort(key=lambda t: t[1])  # izq→der

    # --- 3) porcentajes: asignar a cada barra por cercanía arriba ---
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

    # dedupe por valor
    uniq = []
    for v, x, y in sorted(pct_points, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in uniq):
            uniq.append((v, x, y))

    bar_vals = []
    for r, bx, _h in picked:
        bestp = None
        for (v, x, y) in uniq:
            if y >= r.y0:
                continue
            if y < r.y0 - 340:
                continue
            dx = abs(x - bx)
            if dx > 220:
                continue
            score = dx + (r.y0 - y) * 0.05
            if bestp is None or score < bestp[0]:
                bestp = (score, v)
        bar_vals.append(bestp[1] if bestp else None)

    # --- 4) mapear bar izq→der a legend order ---
    mapping = dict(zip(legend_order, bar_vals))

    return {
        "Comparación 2023 - Igual (%)": mapping.get("igual"),
        "Comparación 2023 - Menos seguro (%)": mapping.get("menos"),
        "Comparación 2023 - Más seguro (%)": mapping.get("mas"),
    }


# =========================
# Extract principal
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

    w = page.rect.width
    midx = w * 0.52

    left = [sp for sp in block if sp["cx"] < midx]
    right = [sp for sp in block if sp["cx"] >= midx]

    # PIE izquierda: 2 % principales
    left_pcts = []
    for sp in left:
        v = parse_pct(sp["text"])
        if v is not None:
            left_pcts.append(v)
    two = unique_top(left_pcts, 2)
    if len(two) == 2:
        out["Inseguro en la comunidad (%)"] = max(two)
        out["Seguro en la comunidad (%)"] = min(two)

    # Modo Hatillo primero
    comp1 = extract_comp_by_labels(page, lines, midx, y_start, right)
    # Si comp1 trae al menos 2 valores, lo usamos; si no, modo leyenda orden
    if sum(v is not None for v in comp1.values()) >= 2:
        out.update(comp1)
    else:
        out.update(extract_comp_by_legend_order(page, midx, y_start, y_end))

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
