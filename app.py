import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Índice Territorial (Excel)", layout="wide")


# ============================================================
# Pesos (según tu imagen)
# ============================================================
WEIGHTS = {
    "percepcion_general": {"inseguro": 0.0, "seguro": 1.0},
    "comparacion_anio_anterior": {"menos_seguro": 0.0, "igual": 0.5, "mas_seguro": 1.0},
    "calificacion_servicio_fp": {
        "muy_malo": 0.0, "malo": 0.0, "medio": 0.50, "buena": 0.75, "muy_buena": 1.0
    },
    "calificacion_servicio_ultimo_anio": {"peor": 0.0, "igual": 0.5, "mejor": 1.0},
}


# ============================================================
# Utilidades
# ============================================================
def safe_norm(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip().lower()


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


def calc_score(percentages: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for k, w in weights.items():
        p = float(percentages.get(k, 0.0) or 0.0)
        total += p * w
    return total  # ya queda 0-100 porque p ya viene en porcentaje


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


def find_row_any(df: pd.DataFrame, needles: List[str], col: int = 0) -> Optional[int]:
    """Encuentra una fila donde la celda (col) contiene cualquiera de los textos."""
    needles = [n.lower() for n in needles]
    for i in range(len(df)):
        txt = safe_norm(df.iat[i, col] if col < df.shape[1] else "")
        if not txt:
            continue
        for n in needles:
            if n in txt:
                return i
    return None


def read_cols_as_floats(df: pd.DataFrame, row: int, cols: List[int]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c in cols:
        if c >= df.shape[1]:
            out[c] = 0.0
            continue
        val = df.iat[row, c]
        try:
            out[c] = float(val)
        except Exception:
            out[c] = 0.0
    return out


# ============================================================
# Extracción de bloques (con nombres reales)
# ============================================================
def extract_blocks(df: pd.DataFrame) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    data: Dict[str, Any] = {}

    # 1) Percepción General (B,C)
    r = find_row_any(df, ["percepción general", "percepcion general"])
    if r is None:
        errors.append("No encontré 'Percepción General' en la columna A.")
    else:
        v = read_cols_as_floats(df, r, [1, 2])  # B,C
        data["percepcion_general"] = {"inseguro": v[1], "seguro": v[2]}

    # 2) Comparación Año Anterior (B,C,D)
    r = find_row_any(df, ["comparación año anterior", "comparacion año anterior", "comparacion ano anterior"])
    if r is None:
        errors.append("No encontré 'Comparación Año Anterior' en la columna A.")
    else:
        v = read_cols_as_floats(df, r, [1, 2, 3])  # B,C,D
        data["comparacion_anio_anterior"] = {"menos_seguro": v[1], "igual": v[2], "mas_seguro": v[3]}

    # 3) Calificación Servicio FP (B..F)
    # En algunos archivos puede venir como "Calificación Servicio FP" o "Percepción del Servicio Policial"
    r = find_row_any(df, ["calificación servicio fp", "calificacion servicio fp", "percepción del servicio policial", "percepcion del servicio policial"])
    if r is None:
        errors.append("No encontré 'Calificación Servicio FP' / 'Percepción del Servicio Policial' en la columna A.")
    else:
        v = read_cols_as_floats(df, r, [1, 2, 3, 4, 5])  # B..F
        data["calificacion_servicio_fp"] = {
            "muy_malo": v[1],
            "malo": v[2],
            "medio": v[3],
            "buena": v[4],
            "muy_buena": v[5],
        }

    # 4) Calificación del Servicio Policial del Último Año (B,C,D)
    # Tu captura muestra EXACTAMENTE ese título (con variaciones posibles de tildes / “del ultimo ano”)
    r = find_row_any(
        df,
        [
            "calificación del servicio policial del último año",
            "calificacion del servicio policial del ultimo año",
            "calificacion del servicio policial del ultimo ano",
            "calificación del servicio policial del ultimo año",
            "calificación del servicio policial del ultimo ano",
            "calificacion del servicio policial ultimo año",
            "calificacion del servicio policial ultimo ano",
        ],
    )
    if r is None:
        errors.append("No encontré 'Calificación del Servicio Policial del Último Año' en la columna A.")
    else:
        v = read_cols_as_floats(df, r, [1, 2, 3])  # B,C,D
        data["calificacion_servicio_ultimo_anio"] = {"igual": v[1], "mejor": v[2], "peor": v[3]}
        # Nota: tu tabla es Igual / Mejor / Peor (en ese orden). Los pesos se aplican por clave.

    return data, errors


def compute_all(blocks: Dict[str, Any]) -> Dict[str, Any]:
    # Puntajes base
    pg = calc_score(blocks["percepcion_general"], WEIGHTS["percepcion_general"])
    ca = calc_score(blocks["comparacion_anio_anterior"], WEIGHTS["comparacion_anio_anterior"])
    sfp = calc_score(blocks["calificacion_servicio_fp"], WEIGHTS["calificacion_servicio_fp"])
    ua = calc_score(blocks["calificacion_servicio_ultimo_anio"], WEIGHTS["calificacion_servicio_ultimo_anio"])

    # Agrupaciones como tu imagen
    percepcion_entorno = (pg + ca) / 2.0
    desempeno_policia = (sfp + ua) / 2.0
    indice_global = (percepcion_entorno + desempeno_policia) / 2.0

    return {
        "puntaje_percepcion_general": pg,
        "puntaje_comparacion_anio_anterior": ca,
        "puntaje_calificacion_servicio_fp": sfp,
        "puntaje_calificacion_servicio_ultimo_anio": ua,
        "percepcion_del_entorno": percepcion_entorno,
        "desempeno_policia": desempeno_policia,
        "indice_global": indice_global,
        "nivel_indice": classify_index(indice_global),
    }


# ============================================================
# UI
# ============================================================
st.title("Índice Territorial — Lectura masiva de Excel")
st.caption("Lee tus Excel (formato como el de las imágenes), calcula puntajes y el Índice Global.")

files = st.file_uploader("Sube hasta 80 archivos Excel (.xlsx / .xlsm)", type=["xlsx", "xlsm"], accept_multiple_files=True)

if not files:
    st.info("Sube uno o varios archivos para empezar.")
    st.stop()

results: List[Dict[str, Any]] = []
fails: List[Dict[str, Any]] = []

for f in files:
    try:
        df = read_sheet_df(f)
        blocks, errs = extract_blocks(df)

        if errs:
            fails.append({"archivo": f.name, "errores": errs})
            continue

        calc = compute_all(blocks)

        row = {
            "archivo": f.name,
            "puntaje_percepcion_general": round(calc["puntaje_percepcion_general"], 3),
            "puntaje_comparacion_anio_anterior": round(calc["puntaje_comparacion_anio_anterior"], 3),
            "puntaje_calificacion_servicio_fp": round(calc["puntaje_calificacion_servicio_fp"], 3),
            "puntaje_calificacion_servicio_ultimo_anio": round(calc["puntaje_calificacion_servicio_ultimo_anio"], 3),
            "percepcion_del_entorno": round(calc["percepcion_del_entorno"], 3),
            "desempeno_policia": round(calc["desempeno_policia"], 3),
            "indice_global": round(calc["indice_global"], 3),
            "nivel_indice": calc["nivel_indice"],
        }
        results.append(row)

    except Exception as e:
        fails.append({"archivo": f.name, "errores": [f"Error leyendo/calculando: {e}"]})

# ---- Mostrar resultados
if results:
    st.subheader("Resultados")
    for r in results:
        level = r["nivel_indice"]
        color = "#2ea44f"
        if "Crítico" in level:
            color = "#ff4d4d"
        elif "Bajo" in level:
            color = "#ffb84d"
        elif "Medio" in level:
            color = "#ffd84d"
        elif "Alto" in level:
            color = "#7bdc7b"

        st.markdown(
            f"""
            <div style="border:1px solid rgba(255,255,255,0.15); border-radius:14px; padding:14px; background:rgba(255,255,255,0.04); margin-bottom:12px;">
              <div style="font-weight:800; font-size:16px;">📄 {r["archivo"]}</div>
              <div style="margin-top:6px; display:flex; gap:10px; flex-wrap:wrap;">
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
                  {level}
                </span>
              </div>

              <div style="margin-top:10px; font-weight:700;">Puntajes por bloque (0-100):</div>
              <ul style="margin-top:6px;">
                <li>Percepción General: <b>{r["puntaje_percepcion_general"]:.2f}</b></li>
                <li>Comparación Año Anterior: <b>{r["puntaje_comparacion_anio_anterior"]:.2f}</b></li>
                <li>Calificación Servicio FP / Percepción Servicio Policial: <b>{r["puntaje_calificacion_servicio_fp"]:.2f}</b></li>
                <li>Calificación Servicio Policial del Último Año: <b>{r["puntaje_calificacion_servicio_ultimo_anio"]:.2f}</b></li>
              </ul>

              <div style="opacity:0.75; font-size:12px;">
                Fórmulas: Percepción del entorno = promedio(PG, Comparación). Desempeño policía = promedio(Calificación FP, Último Año). Índice Global = promedio(Entorno, Policía).
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Consolidado")
    df_out = pd.DataFrame(results).sort_values("indice_global", ascending=True)
    st.dataframe(df_out, use_container_width=True)

    # Descargar consolidado
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
    st.subheader("Archivos que no calzaron con el formato")
    for item in fails:
        with st.expander(item["archivo"], expanded=False):
            for e in item["errores"]:
                st.write("•", e)
