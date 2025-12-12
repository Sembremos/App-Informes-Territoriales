import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Índice Territorial — Lectura masiva de Excel", layout="wide")

# ============================================================
# Utilidades
# ============================================================
def norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()

def to_pct(x: Any) -> Optional[float]:
    """Convierte a porcentaje 0-100. Acepta 71.95 o 0.7195."""
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    # Normaliza si viene en 0-1
    if 0 <= v <= 1.5:
        v = v * 100.0
    return v

def pick_sheet(xls: pd.ExcelFile) -> str:
    for name in xls.sheet_names:
        n = name.lower()
        if "información" in n or "informacion" in n:
            return name
    return xls.sheet_names[0]

def read_sheet_df(uploaded_file) -> pd.DataFrame:
    xls = pd.ExcelFile(uploaded_file, engine="openpyxl")
    sheet = pick_sheet(xls)
    df = pd.read_excel(xls, sheet_name=sheet, header=None, engine="openpyxl")
    return df

def find_cell_anywhere(df: pd.DataFrame, needles: List[str]) -> Optional[Tuple[int, int]]:
    """Busca cualquiera de los needles en cualquier celda. Devuelve (row, col)"""
    needles = [n.lower() for n in needles]
    for r in range(df.shape[0]):
        for c in range(df.shape[1]):
            txt = norm(df.iat[r, c])
            if not txt:
                continue
            for nd in needles:
                if nd in txt:
                    return (r, c)
    return None

def find_header_col(df: pd.DataFrame, start_r: int, start_c: int, header_needles: List[str], scan_rows: int = 6, scan_cols: int = 10) -> Optional[int]:
    """
    Desde un punto, busca una celda con 'porcentaje' o similar para encontrar la columna del porcentaje.
    """
    header_needles = [h.lower() for h in header_needles]
    r0 = max(0, start_r)
    c0 = max(0, start_c)
    r1 = min(df.shape[0], r0 + scan_rows)
    c1 = min(df.shape[1], c0 + scan_cols)

    for r in range(r0, r1):
        for c in range(c0, c1):
            txt = norm(df.iat[r, c])
            for h in header_needles:
                if h in txt:
                    return c
    return None

def extract_table_percentages(
    df: pd.DataFrame,
    title_needles: List[str],
    label_map: Dict[str, List[str]],
    header_needles: List[str] = ["porcentaje", "%"],
    scan_down_rows: int = 25,
) -> Tuple[Optional[Dict[str, float]], str]:
    """
    - Encuentra el título en cualquier celda
    - Encuentra columna de Porcentaje cerca
    - Lee filas debajo buscando etiquetas (con sinónimos)
    - Devuelve dict con claves normalizadas (las llaves del label_map)
    """
    pos = find_cell_anywhere(df, title_needles)
    if not pos:
        return None, f"No encontré el bloque: {title_needles[0]}"

    tr, tc = pos

    # Buscar columna "Porcentaje" cerca del título
    pct_col = find_header_col(df, tr, 0, header_needles, scan_rows=8, scan_cols=df.shape[1])
    if pct_col is None:
        # intento: buscar cerca (alrededor del título)
        pct_col = find_header_col(df, tr, tc, header_needles, scan_rows=10, scan_cols=12)

    if pct_col is None:
        return None, f"Encontré el título '{title_needles[0]}' pero no encontré la columna 'Porcentaje'."

    # Buscar etiquetas y recoger %
    out: Dict[str, float] = {}
    # invertimos sinónimos -> clave
    syn_to_key = {}
    for key, syns in label_map.items():
        for s in syns:
            syn_to_key[s.lower()] = key

    # recorrer filas debajo del título (y un poco arriba también por si el título queda en medio)
    start = max(0, tr - 2)
    end = min(df.shape[0], tr + scan_down_rows)

    for r in range(start, end):
        # buscar etiqueta en toda la fila (porque a veces está en col 0 o combinada)
        row_texts = [norm(df.iat[r, c]) for c in range(df.shape[1])]
        # detectar si alguna celda coincide con algún sinónimo
        found_key = None
        for cell_txt in row_texts:
            if not cell_txt:
                continue
            for syn, key in syn_to_key.items():
                if syn == cell_txt or syn in cell_txt:
                    found_key = key
                    break
            if found_key:
                break

        if found_key:
            v = to_pct(df.iat[r, pct_col] if pct_col < df.shape[1] else None)
            if v is not None:
                out[found_key] = float(v)

    # Validar que al menos 2 claves existan (para no agarrar basura)
    if len(out) < max(2, len(label_map) // 2):
        return None, f"Encontré '{title_needles[0]}', pero no pude leer suficientes porcentajes (revisar estructura)."

    return out, ""

def score_from_percentages(pcts: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for k, w in weights.items():
        total += float(pcts.get(k, 0.0) or 0.0) * w
    return total  # 0-100

def classify_index(x: float) -> str:
    if x <= 20:
        return "Crítico (0-20)"
    if x <= 40:
        return "Bajo (20,1-40)"
    if x <= 60:
        return "Medio (40,1-60)"
    if x <= 80:
        return "Alto (60,1-80)"
    return "Muy Alto (80,1-100)"

def level_color(level: str) -> str:
    if "Crítico" in level:
        return "#ff4d4d"
    if "Bajo" in level:
        return "#ffb84d"
    if "Medio" in level:
        return "#ffd84d"
    if "Alto" in level:
        return "#7bdc7b"
    return "#2ea44f"


# ============================================================
# Definición de bloques EXACTOS (como tus tablas)
# ============================================================
# 1) Percepción General: No / Sí
PG_TITLE = ["¿se siente seguro en su comunidad?", "se siente seguro en su comunidad", "seguro en su comunidad"]
PG_LABELS = {
    "inseguro": ["no"],
    "seguro": ["sí", "si"],
}
PG_WEIGHTS = {"inseguro": 0.0, "seguro": 1.0}

# 2) Comparación año anterior: Igual / Más / Menos
CA_TITLE = [
    "¿cómo se siente en cuanto a la seguridad en su barrio en comparación con el año anterior?",
    "comparación con el año anterior",
    "comparacion con el año anterior",
]
CA_LABELS = {
    "igual": ["igual"],
    "mas_seguro": ["más seguro", "mas seguro"],
    "menos_seguro": ["menos seguro"],
}
CA_WEIGHTS = {"menos_seguro": 0.0, "igual": 0.5, "mas_seguro": 1.0}

# 3) Percepción del Servicio Policial: Excelente / Buena / Regular / Mala / Muy Mala
SP_TITLE = ["percepción del servicio policial", "percepcion del servicio policial"]
SP_LABELS = {
    "excelente": ["excelente"],
    "buena": ["buena", "bueno"],
    "regular": ["regular"],
    "mala": ["mala"],
    "muy_mala": ["muy mala", "muy_mala", "muy mala "],
}
# Mapeo equivalente a tu escala: Excelente=1, Buena=0.75, Regular=0.5, Mala=0, Muy Mala=0
SP_WEIGHTS = {"excelente": 1.0, "buena": 0.75, "regular": 0.50, "mala": 0.0, "muy_mala": 0.0}

# 4) Calificación del Servicio Policial del Último Año: Igual / Mejor servicio / Peor servicio
UA_TITLE = [
    "calificación del servicio policial del último año",
    "calificacion del servicio policial del ultimo año",
    "calificacion del servicio policial del ultimo ano",
]
UA_LABELS = {
    "igual": ["igual"],
    "mejor": ["mejor servicio", "mejor"],
    "peor": ["peor servicio", "peor"],
}
UA_WEIGHTS = {"peor": 0.0, "igual": 0.5, "mejor": 1.0}


# ============================================================
# UI
# ============================================================
st.title("Índice Territorial — Lectura masiva de Excel")
st.caption("Ubica automáticamente los cuadros (tablas) y calcula puntajes + Índice Global.")

files = st.file_uploader(
    "Sube hasta 80 archivos Excel (.xlsx / .xlsm)",
    type=["xlsx", "xlsm"],
    accept_multiple_files=True
)

if not files:
    st.info("Sube uno o varios archivos para empezar.")
    st.stop()

results = []
fails = []

for f in files:
    try:
        df = read_sheet_df(f)

        pg, e1 = extract_table_percentages(df, PG_TITLE, PG_LABELS)
        ca, e2 = extract_table_percentages(df, CA_TITLE, CA_LABELS)
        sp, e3 = extract_table_percentages(df, SP_TITLE, SP_LABELS)
        ua, e4 = extract_table_percentages(df, UA_TITLE, UA_LABELS)

        errs = [e for e in [e1, e2, e3, e4] if e]
        if errs:
            fails.append({"archivo": f.name, "errores": errs})
            continue

        # Puntajes base
        score_pg = score_from_percentages(pg, PG_WEIGHTS)
        score_ca = score_from_percentages(ca, CA_WEIGHTS)
        score_sp = score_from_percentages(sp, SP_WEIGHTS)
        score_ua = score_from_percentages(ua, UA_WEIGHTS)

        # Índices como tu imagen
        entorno = (score_pg + score_ca) / 2.0
        policia = (score_sp + score_ua) / 2.0
        global_idx = (entorno + policia) / 2.0
        level = classify_index(global_idx)

        results.append({
            "archivo": f.name,
            "puntaje_percepcion_general": round(score_pg, 3),
            "puntaje_comparacion_anio_anterior": round(score_ca, 3),
            "puntaje_percepcion_servicio_policial": round(score_sp, 3),
            "puntaje_calificacion_ultimo_anio": round(score_ua, 3),
            "percepcion_del_entorno": round(entorno, 3),
            "desempeno_policia": round(policia, 3),
            "indice_global": round(global_idx, 3),
            "nivel_indice": level,
        })

    except Exception as e:
        fails.append({"archivo": f.name, "errores": [f"Error leyendo/calculando: {e}"]})

# Mostrar resultados
if results:
    st.subheader("Resultados")
    for r in results:
        color = level_color(r["nivel_indice"])
        st.markdown(
            f"""
            <div style="border:1px solid rgba(255,255,255,0.15); border-radius:14px; padding:14px; background:rgba(255,255,255,0.04); margin-bottom:12px;">
              <div style="font-weight:800; font-size:16px;">📄 {r["archivo"]}</div>

              <div style="margin-top:8px; display:flex; gap:10px; flex-wrap:wrap;">
                <div style="padding:10px; border-radius:12px; border:1px solid rgba(255,255,255,0.12); min-width:220px;">
                  <div style="opacity:0.85; font-size:12px;">Percepción del entorno</div>
                  <div style="font-weight:800; font-size:22px;">{r["percepcion_del_entorno"]:.2f}</div>
                </div>
                <div style="padding:10px; border-radius:12px; border:1px solid rgba(255,255,255,0.12); min-width:220px;">
                  <div style="opacity:0.85; font-size:12px;">Desempeño policía</div>
                  <div style="font-weight:800; font-size:22px;">{r["desempeno_policia"]:.2f}</div>
                </div>
                <div style="padding:10px; border-radius:12px; border:1px solid rgba(255,255,255,0.12); min-width:220px;">
                  <div style="opacity:0.85; font-size:12px;">Índice Global</div>
                  <div style="font-weight:800; font-size:22px;">{r["indice_global"]:.2f}</div>
                </div>
              </div>

              <div style="margin-top:10px;">
                <span style="display:inline-block; padding:6px 10px; border-radius:999px; font-weight:800; font-size:12px; background:{color}; color:#111;">
                  {r["nivel_indice"]}
                </span>
              </div>

              <div style="margin-top:10px; font-weight:700;">Puntajes por bloque (0-100):</div>
              <ul style="margin-top:6px;">
                <li>Percepción General (No/Sí): <b>{r["puntaje_percepcion_general"]:.2f}</b></li>
                <li>Comparación Año Anterior (Menos/Igual/Más): <b>{r["puntaje_comparacion_anio_anterior"]:.2f}</b></li>
                <li>Percepción Servicio Policial (Excelente…Muy Mala): <b>{r["puntaje_percepcion_servicio_policial"]:.2f}</b></li>
                <li>Calificación Servicio Policial del Último Año (Igual/Mejor/Peor): <b>{r["puntaje_calificacion_ultimo_anio"]:.2f}</b></li>
              </ul>

              <div style="opacity:0.75; font-size:12px;">
                Fórmulas: Entorno = promedio(Percepción General, Comparación). Policía = promedio(Servicio Policial, Último Año). Global = promedio(Entorno, Policía).
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Consolidado")
    df_out = pd.DataFrame(results).sort_values("indice_global", ascending=True)
    st.dataframe(df_out, use_container_width=True)

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="consolidado")
    st.download_button(
        "⬇️ Descargar consolidado (Excel)",
        data=bio.getvalue(),
        file_name="consolidado_indices.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if fails:
    st.subheader("Archivos que no calzaron con el formato (detalle real)")
    for item in fails:
        with st.expander(item["archivo"], expanded=False):
            for e in item["errores"]:
                st.write("•", e)
