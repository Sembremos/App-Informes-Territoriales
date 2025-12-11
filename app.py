import io
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageOps

# OCR
import pytesseract


# ============================================================
# Utilidades
# ============================================================
def norm_txt(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s)
    return s


def pct_to_float(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.replace("%", "").strip()
    s = s.replace(" ", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        v = float(s)
        if 0 <= v <= 100:
            return v
        return None
    except:
        return None


PCT_RE = re.compile(r"(\d{1,3}[.,]\d{1,2})\s*%?")


def safe_float(v):
    try:
        return float(v)
    except:
        return None


def img_from_pix(pix: fitz.Pixmap) -> Image.Image:
    return Image.open(io.BytesIO(pix.tobytes("png")))


def render_page(doc: fitz.Document, page_index: int, zoom: float = 2.0) -> Image.Image:
    page = doc[page_index]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return img_from_pix(pix)


def crop_norm(img: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
    """box en coordenadas normalizadas (0-1): (x1,y1,x2,y2)"""
    w, h = img.size
    x1, y1, x2, y2 = box
    return img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))


def preprocess(img: Image.Image) -> Image.Image:
    """Mejorar OCR: gris + autocontraste + umbral suave"""
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    arr = np.array(g)
    thr = np.percentile(arr, 70)  # umbral adaptativo simple
    bw = (arr > thr).astype(np.uint8) * 255
    return Image.fromarray(bw)


@dataclass
class OCRToken:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


def ocr_tokens(img: Image.Image) -> List[OCRToken]:
    """OCR con bounding boxes"""
    img2 = preprocess(img)
    data = pytesseract.image_to_data(img2, output_type=pytesseract.Output.DICT, config="--psm 6")
    out: List[OCRToken] = []
    n = len(data["text"])
    for i in range(n):
        t = (data["text"][i] or "").strip()
        if not t:
            continue
        conf = safe_float(data["conf"][i])
        if conf is None:
            conf = 0.0
        out.append(
            OCRToken(
                text=t,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
                conf=float(conf),
            )
        )
    return out


def tokens_find_any(tokens: List[OCRToken], variants: List[str]) -> List[OCRToken]:
    vv = [norm_txt(v) for v in variants]
    found = []
    for tok in tokens:
        nt = norm_txt(tok.text)
        if any(v == nt or v in nt for v in vv):
            found.append(tok)
    return found


def tokens_extract_pcts(tokens: List[OCRToken]) -> List[Tuple[float, OCRToken]]:
    """devuelve (valor, token) para tokens con % o números tipo 42,86"""
    out = []
    for tok in tokens:
        m = PCT_RE.search(tok.text)
        if not m:
            continue
        v = pct_to_float(m.group(1))
        if v is None:
            continue
        out.append((v, tok))
    return out


def unique_vals(vals: List[float], tol: float = 0.2) -> List[float]:
    uniq = []
    for v in sorted(vals, reverse=True):
        if all(abs(v - u) > tol for u in uniq):
            uniq.append(v)
    return uniq


# ============================================================
# Detección de páginas por anclas
# ============================================================
def find_page_by_anchor(doc: fitz.Document, anchor_variants: List[str], max_pages: int = 60) -> Optional[int]:
    vv = [norm_txt(v) for v in anchor_variants]
    for i in range(min(max_pages, doc.page_count)):
        txt = norm_txt(doc[i].get_text("text") or "")
        if any(v in txt for v in vv):
            return i
    return None


def extract_delegacion(doc: fitz.Document) -> str:
    for i in range(min(8, doc.page_count)):
        t = doc[i].get_text("text") or ""
        m = re.search(r"\bD\s*-\s*\d+\s+[A-Za-zÁÉÍÓÚÑáéíóúñ\s]+", t)
        if m:
            return " ".join(m.group(0).split()).strip()
        m2 = re.search(r"Delegaci[oó]n\s+Policial\s+([A-Za-zÁÉÍÓÚÑáéíóúñ\s]+)", t, flags=re.I)
        if m2:
            return f"Delegación Policial {m2.group(1).strip()}"
    return "SIN_DELEGACIÓN"


# ============================================================
# Zonas (normalizadas) para OCR dentro de cada página
# Ajustables si algún informe trae márgenes distintos.
# ============================================================
DEFAULT_ZONES = {
    # Percepción ciudadana (página con el título)
    "pc_pie": (0.05, 0.30, 0.52, 0.72),      # área del pie (izquierda)
    "pc_barras": (0.52, 0.30, 0.96, 0.72),   # área barras + etiquetas (derecha)

    # Servicio Policial (barras con Regular/Buena/Excelente/Mala/Muy mala)
    "sp_barras": (0.06, 0.28, 0.96, 0.72),

    # Últimos dos años (pie con Igual/Mejor/Peor)
    "u2_pie": (0.20, 0.22, 0.82, 0.86),
}


# ============================================================
# Extracción Percepción ciudadana (OCR + coordenadas)
# ============================================================
def extract_percepcion_ciudadana(doc: fitz.Document, zones: Dict[str, Tuple[float, float, float, float]], debug=False):
    page_idx = find_page_by_anchor(doc, ["Percepción ciudadana", "¿Se siente de seguro en su comunidad?"])
    if page_idx is None:
        return None, {"pc_page": None}

    img = render_page(doc, page_idx, zoom=2.2)
    pie_img = crop_norm(img, zones["pc_pie"])
    bar_img = crop_norm(img, zones["pc_barras"])

    # PIE: extraemos 2 porcentajes del gráfico (sin depender del orden)
    pie_tokens = ocr_tokens(pie_img)
    pie_pcts = tokens_extract_pcts(pie_tokens)
    pie_vals = unique_vals([v for v, _ in pie_pcts], tol=0.2)
    pie_vals = sorted(pie_vals, reverse=True)[:2]  # dos principales

    seguro = None
    inseguro = None
    if len(pie_vals) == 2:
        # en tu plantilla: inseguro es mayor, seguro es menor
        inseguro = max(pie_vals)
        seguro = min(pie_vals)

    # BARRAS: usamos etiquetas y % con coordenadas
    bar_tokens = ocr_tokens(bar_img)
    pct_tokens = tokens_extract_pcts(bar_tokens)

    # Etiquetas (pueden venir con mayúsculas/tilde)
    lab_igual = tokens_find_any(bar_tokens, ["igual"])
    lab_menos = tokens_find_any(bar_tokens, ["menos seguro", "menos"])
    lab_mas = tokens_find_any(bar_tokens, ["más seguro", "mas seguro", "más", "mas"])

    def best_label_center(labs: List[OCRToken]) -> Optional[Tuple[float, float]]:
        if not labs:
            return None
        # etiqueta suele estar abajo, tomamos la de mayor y
        tok = sorted(labs, key=lambda t: t.y, reverse=True)[0]
        return (tok.cx, tok.cy)

    c_igual = best_label_center(lab_igual)
    c_menos = best_label_center(lab_menos)
    c_mas = best_label_center(lab_mas)

    def pick_pct_above(label_center: Tuple[float, float], used_idxs: set) -> Optional[float]:
        lx, ly = label_center
        best = None
        for idx, (v, tok) in enumerate(pct_tokens):
            if idx in used_idxs:
                continue
            # debe estar arriba de la etiqueta
            if tok.cy >= ly:
                continue
            dx = abs(tok.cx - lx)
            dy = abs(ly - tok.cy)
            # ventana razonable (ajustable)
            if dx > 220:
                continue
            if dy > 260:
                continue
            score = dx * 1.0 + dy * 0.15
            if best is None or score < best[0]:
                best = (score, idx, v)
        if best is None:
            return None
        used_idxs.add(best[1])
        return best[2]

    comp_igual = None
    comp_menos = None
    comp_mas = None

    used = set()
    if c_igual:
        comp_igual = pick_pct_above(c_igual, used)
    if c_menos:
        comp_menos = pick_pct_above(c_menos, used)
    if c_mas:
        comp_mas = pick_pct_above(c_mas, used)

    # Fallback: si falta uno, completamos con el % restante de los 3 principales del área de barras
    bar_vals = unique_vals([v for v, _ in pct_tokens], tol=0.2)
    top3 = sorted(bar_vals, reverse=True)[:3]
    got = [v for v in [comp_igual, comp_menos, comp_mas] if v is not None]
    if len(top3) == 3 and len(got) == 2:
        remaining = [v for v in top3 if all(abs(v - g) > 0.2 for g in got)]
        if remaining:
            miss = remaining[0]
            if comp_igual is None:
                comp_igual = miss
            elif comp_menos is None:
                comp_menos = miss
            elif comp_mas is None:
                comp_mas = miss

    # Último fallback (anti-swap): si detectamos {15.58, 41.56, 42.86} pero Menos/Más se cruzan,
    # los corregimos por lógica: Más seguro suele ser el menor de los 3.
    if comp_igual is not None and comp_menos is not None and comp_mas is not None:
        trio = sorted([comp_igual, comp_menos, comp_mas], reverse=True)
        # Igual suele ser el mayor (o de los mayores), Más suele ser el menor
        # No tocamos Igual; solo corregimos Menos/Más si el menor quedó en Menos.
        menor = min([comp_igual, comp_menos, comp_mas])
        if comp_menos == menor and comp_mas != menor:
            # swap Menos/Más
            comp_menos, comp_mas = comp_mas, comp_menos

    meta = {"pc_page": page_idx}
    if debug:
        meta.update({
            "pc_pie_vals_detectados": pie_vals,
            "pc_bar_top3_detectados": top3,
        })

    return {
        "Seguro en la comunidad (%)": seguro,
        "Inseguro en la comunidad (%)": inseguro,
        "Comparación 2023 - Igual (%)": comp_igual,
        "Comparación 2023 - Menos seguro (%)": comp_menos,
        "Comparación 2023 - Más seguro (%)": comp_mas,
    }, meta


# ============================================================
# Servicio Policial (OCR): Regular/Buena/Excelente/Mala/Muy mala
# ============================================================
def extract_servicio_policial(doc: fitz.Document, zones: Dict[str, Tuple[float, float, float, float]], debug=False):
    page_idx = find_page_by_anchor(doc, ["Percepción del Servicio Policial", "¿Cómo califica el servicio"])
    if page_idx is None:
        return None, {"sp_page": None}

    img = render_page(doc, page_idx, zoom=2.2)
    area = crop_norm(img, zones["sp_barras"])
    toks = ocr_tokens(area)
    pcts = tokens_extract_pcts(toks)

    # Etiquetas
    labs = {
        "Regular": tokens_find_any(toks, ["regular"]),
        "Buena": tokens_find_any(toks, ["buena"]),
        "Excelente": tokens_find_any(toks, ["excelente"]),
        "Mala": tokens_find_any(toks, ["mala"]),
        "Muy mala": tokens_find_any(toks, ["muy mala", "muymala"]),
    }

    def label_center(name: str) -> Optional[Tuple[float, float]]:
        L = labs[name]
        if not L:
            return None
        tok = sorted(L, key=lambda t: t.y, reverse=True)[0]
        return (tok.cx, tok.cy)

    centers = {k: label_center(k) for k in labs}

    used = set()

    def pick_pct(label: Optional[Tuple[float, float]]) -> Optional[float]:
        if not label:
            return None
        lx, ly = label
        best = None
        for idx, (v, tok) in enumerate(pcts):
            if idx in used:
                continue
            if tok.cy >= ly:
                continue
            dx = abs(tok.cx - lx)
            dy = abs(ly - tok.cy)
            if dx > 260 or dy > 280:
                continue
            score = dx * 1.0 + dy * 0.15
            if best is None or score < best[0]:
                best = (score, idx, v)
        if best is None:
            return None
        used.add(best[1])
        return best[2]

    vals = {k: pick_pct(centers[k]) for k in labs}

    # fallback: si faltan, completamos con los top5 del área de barras
    uniq = unique_vals([v for v, _ in pcts], tol=0.2)
    top5 = sorted(uniq, reverse=True)[:5]
    got = [v for v in vals.values() if v is not None]
    if len(top5) >= 4:
        for k in vals:
            if vals[k] is None:
                rem = [v for v in top5 if all(abs(v - g) > 0.2 for g in got)]
                if rem:
                    vals[k] = rem[0]
                    got.append(rem[0])

    meta = {"sp_page": page_idx}
    if debug:
        meta["sp_top5_detectados"] = top5

    return {
        "Servicio Policial - Regular (%)": vals["Regular"],
        "Servicio Policial - Buena (%)": vals["Buena"],
        "Servicio Policial - Excelente (%)": vals["Excelente"],
        "Servicio Policial - Mala (%)": vals["Mala"],
        "Servicio Policial - Muy mala (%)": vals["Muy mala"],
    }, meta


# ============================================================
# Últimos 2 años (pie): Igual / Mejor servicio / Peor servicio
# ============================================================
def extract_ultimos_2(doc: fitz.Document, zones: Dict[str, Tuple[float, float, float, float]], debug=False):
    page_idx = find_page_by_anchor(doc, ["Calificación del Servicio Policial de los Últimos Dos Años", "Últimos Dos Años"])
    if page_idx is None:
        return None, {"u2_page": None}

    img = render_page(doc, page_idx, zoom=2.2)
    area = crop_norm(img, zones["u2_pie"])
    toks = ocr_tokens(area)
    pcts = tokens_extract_pcts(toks)
    vals = unique_vals([v for v, _ in pcts], tol=0.2)
    top3 = sorted(vals, reverse=True)[:3]

    # En tu ejemplo: Igual 58.12, Mejor 26.20, Peor 15.68
    # OCR solo da números: los asociamos a etiquetas por cercanía
    labs_igual = tokens_find_any(toks, ["igual"])
    labs_mejor = tokens_find_any(toks, ["mejor"])
    labs_peor = tokens_find_any(toks, ["peor"])

    def best_center(L: List[OCRToken]) -> Optional[Tuple[float, float]]:
        if not L:
            return None
        tok = sorted(L, key=lambda t: t.y, reverse=True)[0]
        return (tok.cx, tok.cy)

    c_igual = best_center(labs_igual)
    c_mejor = best_center(labs_mejor)
    c_peor = best_center(labs_peor)

    used = set()

    def pick(label: Optional[Tuple[float, float]]) -> Optional[float]:
        if not label:
            return None
        lx, ly = label
        best = None
        for idx, (v, tok) in enumerate(pcts):
            if idx in used:
                continue
            # puede estar alrededor (en pie a veces no es estrictamente "arriba"), solo cercanía en X/Y
            dx = abs(tok.cx - lx)
            dy = abs(tok.cy - ly)
            if dx > 320 or dy > 320:
                continue
            score = dx * 1.0 + dy * 0.6
            if best is None or score < best[0]:
                best = (score, idx, v)
        if best is None:
            return None
        used.add(best[1])
        return best[2]

    v_igual = pick(c_igual)
    v_mejor = pick(c_mejor)
    v_peor = pick(c_peor)

    got = [v for v in [v_igual, v_mejor, v_peor] if v is not None]
    if len(top3) == 3 and len(got) == 2:
        rem = [v for v in top3 if all(abs(v - g) > 0.2 for g in got)]
        if rem:
            miss = rem[0]
            if v_igual is None:
                v_igual = miss
            elif v_mejor is None:
                v_mejor = miss
            elif v_peor is None:
                v_peor = miss

    meta = {"u2_page": page_idx}
    if debug:
        meta["u2_top3_detectados"] = top3

    return {
        "Últimos 2 años - Igual (%)": v_igual,
        "Últimos 2 años - Mejor servicio (%)": v_mejor,
        "Últimos 2 años - Peor servicio (%)": v_peor,
    }, meta


# ============================================================
# Puntajes e Índices (sin Comercio)
# ============================================================
def score_percepcion_general(seguro: Optional[float], inseguro: Optional[float]) -> Optional[float]:
    # Inseguro*0 + Seguro*1 = Seguro
    return seguro if seguro is not None else None


def score_comparacion(igual: Optional[float], menos: Optional[float], mas: Optional[float]) -> Optional[float]:
    # Menos*0 + Igual*0.5 + Más*1
    if igual is None or mas is None:
        return None
    return igual * 0.5 + mas * 1.0


def score_servicio_fp(reg: Optional[float], buena: Optional[float], exc: Optional[float], mala: Optional[float], muy_mala: Optional[float]) -> Optional[float]:
    # Muy mala*0 + Mala*0 + Regular*0.5 + Buena*0.75 + Excelente*1
    if reg is None or buena is None or exc is None:
        return None
    return reg * 0.5 + buena * 0.75 + exc * 1.0


def score_ult2(igual: Optional[float], mejor: Optional[float], peor: Optional[float]) -> Optional[float]:
    # Peor*0 + Igual*0.5 + Mejor*1
    if igual is None or mejor is None:
        return None
    return igual * 0.5 + mejor * 1.0


def avg2(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def nivel_indice(v: Optional[float]) -> Optional[str]:
    if v is None:
        return None
    if 0 <= v <= 20:
        return "Crítico"
    if 20 < v <= 40:
        return "Bajo"
    if 40 < v <= 60:
        return "Medio"
    if 60 < v <= 80:
        return "Alto"
    if 80 < v <= 100:
        return "Muy Alto"
    return None


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Extractor MPGP (OCR estable)", layout="wide")
st.title("Extractor MPGP — Percepción ciudadana + Servicio policial (OCR estable)")

st.caption(
    "Esta versión usa OCR con coordenadas (no texto suelto del PDF). "
    "Es lo que evita swaps y ‘None’ al procesar muchos informes."
)

with st.sidebar:
    st.header("Ajustes")
    debug = st.toggle("Modo Debug", value=False)
    zoom = st.slider("Zoom render (más alto = mejor OCR, más lento)", 1.6, 3.0, 2.2, 0.1)

    st.subheader("Zonas (plantilla)")
    st.write("Si algún PDF trae márgenes distintos, ajustá estas zonas.")
    zones = dict(DEFAULT_ZONES)

    # opción de ajustar rápido solo percepción ciudadana
    if st.checkbox("Mostrar ajustes de zonas", value=False):
        st.markdown("**Percepción ciudadana**")
        zones["pc_pie"] = (
            st.slider("pc_pie x1", 0.0, 1.0, zones["pc_pie"][0], 0.01),
            st.slider("pc_pie y1", 0.0, 1.0, zones["pc_pie"][1], 0.01),
            st.slider("pc_pie x2", 0.0, 1.0, zones["pc_pie"][2], 0.01),
            st.slider("pc_pie y2", 0.0, 1.0, zones["pc_pie"][3], 0.01),
        )
        zones["pc_barras"] = (
            st.slider("pc_barras x1", 0.0, 1.0, zones["pc_barras"][0], 0.01),
            st.slider("pc_barras y1", 0.0, 1.0, zones["pc_barras"][1], 0.01),
            st.slider("pc_barras x2", 0.0, 1.0, zones["pc_barras"][2], 0.01),
            st.slider("pc_barras y2", 0.0, 1.0, zones["pc_barras"][3], 0.01),
        )

        st.markdown("**Servicio policial**")
        zones["sp_barras"] = (
            st.slider("sp_barras x1", 0.0, 1.0, zones["sp_barras"][0], 0.01),
            st.slider("sp_barras y1", 0.0, 1.0, zones["sp_barras"][1], 0.01),
            st.slider("sp_barras x2", 0.0, 1.0, zones["sp_barras"][2], 0.01),
            st.slider("sp_barras y2", 0.0, 1.0, zones["sp_barras"][3], 0.01),
        )

        st.markdown("**Últimos 2 años**")
        zones["u2_pie"] = (
            st.slider("u2_pie x1", 0.0, 1.0, zones["u2_pie"][0], 0.01),
            st.slider("u2_pie y1", 0.0, 1.0, zones["u2_pie"][1], 0.01),
            st.slider("u2_pie x2", 0.0, 1.0, zones["u2_pie"][2], 0.01),
            st.slider("u2_pie y2", 0.0, 1.0, zones["u2_pie"][3], 0.01),
        )

uploaded = st.file_uploader("Suba uno o muchos PDF (90+)", type=["pdf"], accept_multiple_files=True)

if uploaded:
    rows = []
    meta_rows = []

    prog = st.progress(0)
    status = st.empty()

    for i, f in enumerate(uploaded, start=1):
        status.write(f"Procesando: **{f.name}** ({i}/{len(uploaded)})")

        pdf_bytes = f.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        deleg = extract_delegacion(doc)

        pc, pc_meta = extract_percepcion_ciudadana(doc, zones, debug=debug)
        sp, sp_meta = extract_servicio_policial(doc, zones, debug=debug)
        u2, u2_meta = extract_ultimos_2(doc, zones, debug=debug)

        # Cerrar
        doc.close()

        row = {"Archivo": f.name, "Delegación": deleg}

        # Unir resultados
        if pc:
            row.update(pc)
        if sp:
            row.update(sp)
        if u2:
            row.update(u2)

        # Puntajes
        punt_pg = score_percepcion_general(row.get("Seguro en la comunidad (%)"), row.get("Inseguro en la comunidad (%)"))
        punt_c23 = score_comparacion(
            row.get("Comparación 2023 - Igual (%)"),
            row.get("Comparación 2023 - Menos seguro (%)"),
            row.get("Comparación 2023 - Más seguro (%)"),
        )
        punt_sp = score_servicio_fp(
            row.get("Servicio Policial - Regular (%)"),
            row.get("Servicio Policial - Buena (%)"),
            row.get("Servicio Policial - Excelente (%)"),
            row.get("Servicio Policial - Mala (%)"),
            row.get("Servicio Policial - Muy mala (%)"),
        )
        punt_u2 = score_ult2(
            row.get("Últimos 2 años - Igual (%)"),
            row.get("Últimos 2 años - Mejor servicio (%)"),
            row.get("Últimos 2 años - Peor servicio (%)"),
        )

        # Índices (sin comercio)
        indice_entorno = avg2(punt_pg, punt_c23)
        indice_desempeno = avg2(punt_sp, punt_u2)
        indice_global = avg2(indice_entorno, indice_desempeno)

        row["Puntaje - Percepción general (0-100)"] = punt_pg
        row["Puntaje - Comparación 2023 (0-100)"] = punt_c23
        row["Puntaje - Servicio FP (0-100)"] = punt_sp
        row["Puntaje - Servicio últimos 2 (0-100)"] = punt_u2

        row["Índice - Percepción del entorno"] = indice_entorno
        row["Índice - Desempeño policía"] = indice_desempeno
        row["Índice Global"] = indice_global
        row["Nivel índice"] = nivel_indice(indice_global)

        rows.append(row)

        if debug:
            meta_rows.append({
                "Archivo": f.name,
                "Delegación": deleg,
                **pc_meta, **sp_meta, **u2_meta
            })

        prog.progress(int(i / len(uploaded) * 100))

    prog.empty()
    status.empty()

    df = pd.DataFrame(rows)

    # Formato % en columnas de porcentaje
    pct_cols = [c for c in df.columns if "(%)" in c]
    for c in pct_cols:
        df[c] = df[c].map(lambda x: (f"{x:.2f}%" if isinstance(x, (int, float)) else None))

    # Formato puntajes/índices
    score_cols = [c for c in df.columns if "Puntaje" in c or "Índice" in c]
    for c in score_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    st.subheader("Resultados (tabla)")
    st.dataframe(df, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.download_button(
            "Descargar CSV",
            df.to_csv(index=False).encode("utf-8-sig"),
            "resultados_mpgp.csv",
            "text/csv",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "Descargar Excel",
            df.to_excel(index=False, engine="openpyxl"),
            "resultados_mpgp.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    if debug and meta_rows:
        st.subheader("Debug (páginas detectadas y valores auxiliares)")
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)

else:
    st.info("Subí tus PDFs para extraer los datos.")
