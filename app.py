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
    m = PCT_RE.search(txt or "")
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
    # lo básico: intenta encontrar "D-6 Hatillo", etc.
    for i in range(min(8, doc.page_count)):
        t = doc[i].get_text("text") or ""
        m = re.search(r"\bD\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ\s]+", t)
        if m:
            return " ".join(m.group(0).split()).strip()
    return "SIN_DELEGACIÓN"


def page_text_norm(doc: fitz.Document, i: int) -> str:
    return norm(doc[i].get_text("text") or "")


def find_page_percepcion_ciudadana(doc: fitz.Document) -> Optional[int]:
    # anclas
    anchors = [
        "percepcion ciudadana",
        "se siente de seguro en su comunidad",
        "comparacion con el ano anterior",
    ]
    for i in range(doc.page_count):
        t = page_text_norm(doc, i)
        if all(a in t for a in anchors[:2]) or ("percepcion ciudadana" in t and "comparacion" in t):
            return i
    return None


# =========================
# Extraer spans y líneas
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
    """
    Reconstruye líneas usando el dict: junta spans en la misma línea.
    Esto es lo que evita confundir 'Menos' y 'seguro' cuando vienen separados.
    """
    d = page.get_text("dict")
    lines_out = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            texts = []
            xs, ys = [], []
            x0s, y0s, x1s, y1s = [], [], [], []
            for sp in spans:
                t = (sp.get("text") or "").strip()
                if not t:
                    continue
                x0, y0, x1, y1 = sp.get("bbox", (0, 0, 0, 0))
                texts.append(t)
                xs.append((x0 + x1) / 2.0)
                ys.append((y0 + y1) / 2.0)
                x0s.append(x0); y0s.append(y0); x1s.append(x1); y1s.append(y1)

            if not texts:
                continue

            txt = " ".join(texts)
            lines_out.append({
                "text": txt,
                "ntext": norm(txt),
                "cx": sum(xs)/len(xs),
                "cy": sum(ys)/len(ys),
                "x0": min(x0s), "y0": min(y0s),
                "x1": max(x1s), "y1": max(y1s),
            })
    return lines_out


# =========================
# Lógica de extracción (solo los datos que pediste)
# =========================
def unique_top(vals: List[float], k: int, tol: float = 0.2) -> List[float]:
    out = []
    for v in sorted(vals, reverse=True):
        if all(abs(v - u) > tol for u in out):
            out.append(v)
        if len(out) >= k:
            break
    return out


def slice_until_fuente(spans: List[Dict], y_start: float) -> List[Dict]:
    # corta hasta "Fuente:" posterior
    fuente_y = None
    for sp in sorted(spans, key=lambda z: z["cy"]):
        if "fuente:" in sp["ntext"] and sp["cy"] > y_start:
            fuente_y = sp["cy"]
            break
    y_end = fuente_y if fuente_y is not None else max(sp["cy"] for sp in spans) + 1
    return [sp for sp in spans if y_start <= sp["cy"] <= y_end]


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

    # ancla vertical: donde aparece el título del bloque
    y_start = None
    for ln in lines:
        if "percepcion ciudadana" in ln["ntext"]:
            y_start = ln["cy"]
            break
    if y_start is None:
        y_start = 0.0

    block = slice_until_fuente(spans, y_start)

    # dividir izquierda/derecha por ancho
    w = page.rect.width
    midx = w * 0.52
    left = [sp for sp in block if sp["cx"] < midx]
    right = [sp for sp in block if sp["cx"] >= midx]

    # 1) PIE izquierda: tomar 2 % principales
    left_pcts = []
    for sp in left:
        v = parse_pct(sp["text"])
        if v is not None:
            left_pcts.append(v)

    two = unique_top(left_pcts, 2)
    if len(two) == 2:
        out["Inseguro en la comunidad (%)"] = max(two)
        out["Seguro en la comunidad (%)"] = min(two)

    # 2) BARRAS derecha: detectar etiquetas por LÍNEA (no por span)
    right_lines = [ln for ln in lines if ln["cx"] >= midx and (y_start <= ln["cy"] <= max(sp["cy"] for sp in right) + 5)]
    # buscar líneas exactas
    def find_label_line(needle_norm: str) -> Optional[Tuple[float, float]]:
        # match fuerte: debe contener la frase completa
        cands = [ln for ln in right_lines if needle_norm in ln["ntext"]]
        if not cands:
            return None
        # escoger la más baja (etiquetas están abajo)
        cands.sort(key=lambda a: a["cy"], reverse=True)
        return (cands[0]["cx"], cands[0]["cy"])

    c_igual = find_label_line("igual")
    c_menos = find_label_line("menos seguro")
    c_mas = find_label_line("mas seguro") or find_label_line("más seguro".replace("á","a"))  # por si acaso

    # % candidatos en el lado derecho
    pct_points = []
    for sp in right:
        v = parse_pct(sp["text"])
        if v is not None:
            pct_points.append((v, sp["cx"], sp["cy"]))

    # filtrar % que están arriba de las etiquetas (zona barras)
    label_ys = [c[1] for c in [c_igual, c_menos, c_mas] if c is not None]
    if label_ys:
        base_y = max(label_ys)
        candidates = [(v, x, y) for (v, x, y) in pct_points if y < base_y and y > base_y - 350]
    else:
        candidates = pct_points[:]

    # quitar duplicados por valor
    uniq = []
    for v, x, y in sorted(candidates, key=lambda t: (t[0], t[2])):
        if all(abs(v - u[0]) > 0.2 for u in uniq):
            uniq.append((v, x, y))

    # resolver asignación por cercanía (X dominante)
    labels = []
    if c_igual: labels.append(("Comparación 2023 - Igual (%)", c_igual))
    if c_menos: labels.append(("Comparación 2023 - Menos seguro (%)", c_menos))
    if c_mas: labels.append(("Comparación 2023 - Más seguro (%)", c_mas))

    def best_for_label(lx, ly, used_idx):
        best = None
        for i, (v, x, y) in enumerate(uniq):
            if i in used_idx:
                continue
            if y >= ly:  # debe estar arriba
                continue
            dx = abs(x - lx)
            dy = abs(ly - y)
            if dx > 260:
                continue
            score = dx * 1.0 + dy * 0.05
            if best is None or score < best[0]:
                best = (score, i, v)
        return best

    used = set()
    got = {}
    for col, (lx, ly) in labels:
        b = best_for_label(lx, ly, used)
        if b:
            used.add(b[1])
            got[col] = b[2]
        else:
            got[col] = None

    # fallback con top3 si falta uno
    top3 = unique_top([v for v, _, _ in uniq], 3)
    current = [v for v in got.values() if v is not None]
    if len(labels) == 3 and len(current) == 2 and len(top3) == 3:
        rem = [v for v in top3 if all(abs(v - c) > 0.2 for c in current)]
        if rem:
            for k in got:
                if got[k] is None:
                    got[k] = rem[0]

    # extra regla anti-confusión: "Más seguro" suele ser el menor de los 3
    # (si detectamos 3 valores y más quedó mayor que menos, corregimos swap)
    if len(labels) == 3 and all(got.get(k) is not None for k in got):
        v_igual = got["Comparación 2023 - Igual (%)"]
        v_menos = got["Comparación 2023 - Menos seguro (%)"]
        v_mas = got["Comparación 2023 - Más seguro (%)"]
        # si el menor no quedó en "Más seguro", swap Menos/Más
        menor = min([v_igual, v_menos, v_mas])
        if v_menos == menor and v_mas != menor:
            got["Comparación 2023 - Menos seguro (%)"], got["Comparación 2023 - Más seguro (%)"] = v_mas, v_menos

    # asignar a salida
    out["Comparación 2023 - Igual (%)"] = got.get("Comparación 2023 - Igual (%)")
    out["Comparación 2023 - Menos seguro (%)"] = got.get("Comparación 2023 - Menos seguro (%)")
    out["Comparación 2023 - Más seguro (%)"] = got.get("Comparación 2023 - Más seguro (%)")

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

    # Formato final con %
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
