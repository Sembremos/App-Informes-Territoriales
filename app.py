import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Índice Territorial (Excel)", layout="wide")

CSS = """
<style>
.card {
  border: 1px solid rgba(255,255,255,0.15);
  border-radius: 14px;
  padding: 14px 14px;
  background: rgba(255,255,255,0.04);
}
.grid {display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;}
.kpi {
  border-radius: 12px;
  padding: 10px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.03);
}
.kpi .label {opacity:0.85; font-size: 12px;}
.kpi .value {font-weight:800; font-size: 22px; margin-top:2px;}
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  font-weight: 700; font-size: 12px;
  border: 1px solid rgba(0,0,0,0.2);
}
.table-like {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.12);
}
.table-like th, .table-like td {
  padding: 8px 10px;
  border-bottom: 1px solid rgba(255,255,255,0.10);
  font-size: 13px;
}
.table-like th {text-align:left; background: rgba(255,255,255,0.06);}
.muted {opacity:0.75; font-size: 12px;}
hr.sep {border: none; border-top: 1px solid rgba(255,255,255,0.12); margin: 12px 0;}
h3 {margin: 0 0 8px 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============================================================
# Constantes / Lógica de cálculo (según tu imagen)
# ============================================================
# PUNTAJE (0-100) = suma( porcentaje_opcion * peso_opcion ) / 100
WEIGHTS = {
    "percepcion_general": {"inseguro": 0.0, "seguro": 1.0},
    "comparacion_anio_anterior": {"menos_seguro": 0.0, "igual": 0.5, "mas_seguro": 1.0},
    "calificacion_servicio_fp": {
        "muy_malo": 0.0, "malo": 0.0, "medio": 0.50, "buena": 0.75, "muy_buena": 1.0
    },
    "calif_servicio_ultimos_2_anios": {"peor": 0.0, "igual": 0.5, "mejor": 1.0},
}

# Indice Global (tu imagen): promedio de 2 bloques
# Percepción del entorno = promedio(percepción general, comparación año anterior)
# Desempeño policía     = promedio(calificación servicio FP, calif. últimos 2 años)
# Índice Global         = promedio(percepción del entorno, desempeño policía)

def calc_score(percentages: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for k, w in weights.items():
        p = float(percentages.get(k, 0.0) or 0.0)
        total += p * w
    return total / 100.0 * 100.0  # sigue siendo 0-100


def classify_index(x: float) -> str:
    # Según tu tabla
    if x <= 20:
        return "Crítico (0-20)"
    if x <= 40:
        return "Bajo (20,1-40)"
    if x <= 60:
        return "Medio (40,1-60)"
    if x <= 80:
        return "Alto (60,1-80)"
    return "Muy Alto (80,1-100)"


def badge_color(level: str) -> str:
    # colores aproximados a tu semáforo
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
# Lectura y extracción por “anclas” (texto en col A)
# ============================================================
def _norm(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip().lower()


def pick_sheet(xls: pd.ExcelFile) -> str:
    for name in xls.sheet_names:
        n = name.lower()
        if "información" in n or "informacion" in n:
            return name
    return xls.sheet_names[0]


def read_df(file) -> pd.DataFrame:
    xls = pd.ExcelFile(file, engine="openpyxl")
    sheet = pick_sheet(xls)
    df = pd.read_excel(xls, sheet_name=sheet, header=None)
    return df


def find_row(df: pd.DataFrame, needle: str, col: int = 0) -> Optional[int]:
    needle = needle.lower()
    for i in range(len(df)):
        txt = _norm(df.iat[i, col] if col < df.shape[1] else "")
        if needle in txt:
            return i
    return None


def read_percent_row(df: pd.DataFrame, row: int, cols: List[int]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for c in cols:
        if c < df.shape[1]:
            val = df.iat[row, c]
            try:
                out[c] = float(val)
            except Exception:
                out[c] = 0.0
        else:
            out[c] = 0.0
    return out


def extract_blocks(df: pd.DataFrame) -> Tuple[Dict[str, Any], List[str]]:
    """
    Asume el layout como en tu imagen:
    - Percepción General: fila con texto "Percepción General" en col A
      porcentajes en B y C
    - Comparación Año Anterior: fila con texto "Comparación Año Anterior"
      porcentajes en B, C, D
    - Calificación Servicio FP: fila con texto "Calificación Servicio FP"
      porcentajes en B,C,D,E,F
    - Calif. Servicio Últimos 2 Años: fila con texto "Calif. Servicio Últimos 2 Años"
      porcentajes en B,C,D
    """
    errors: List[str] = []
    data: Dict[str, Any] = {}

    # Percepción General
    r = find_row(df, "percepción general")
    if r is None:
        r = find_row(df, "percepcion general")
    if r is None:
        errors.append("No encontré la fila de 'Percepción General'.")
    else:
        vals = read_percent_row(df, r, [1, 2])  # B,C
        data["percepcion_general"] = {"inseguro": vals[1], "seguro": vals[2]}

    # Comparación Año Anterior
    r = find_row(df, "comparación año anterior")
    if r is None:
        r = find_row(df, "comparacion año anterior")
    if r is None:
        r = find_row(df, "comparacion ano anterior")
    if r is None:
        errors.append("No encontré la fila de 'Comparación Año Anterior'.")
    else:
        vals = read_percent_row(df, r, [1, 2, 3])  # B,C,D
        data["comparacion_anio_anterior"] = {"menos_seguro": vals[1], "igual": vals[2], "mas_seguro": vals[3]}

    # Calificación Servicio FP
    r = find_row(df, "calificación servicio fp")
    if r is None:
        r = find_row(df, "calificacion servicio fp")
    if r is None:
        errors.append("No encontré la fila de 'Calificación Servicio FP'.")
    else:
        vals = read_percent_row(df, r, [1, 2, 3, 4, 5])  # B..F
        data["calificacion_servicio_fp"] = {
            "muy_malo": vals[1],
            "malo": vals[2],
            "medio": vals[3],
            "buena": vals[4],
            "muy_buena": vals[5],
        }

    # Calif. Servicio Últimos 2 Años
    r = find_row(df, "calif. servicio últimos 2 años")
    if r is None:
        r = find_row(df, "calif. servicio ultimos 2 anos")
    if r is None:
        r = find_row(df, "calif. servicio ultimos 2 años")
    if r is None:
        r = find_row(df, "calificación servicio últimos 2 años")
    if r is None:
        r = find_row(df, "calificacion servicio ultimos 2")
    if r is None:
        errors.append("No encontré la fila de 'Calif. Servicio Últimos 2 Años'.")
    else:
        vals = read_percent_row(df, r, [1, 2, 3])  # B,C,D
        data["calif_servicio_ultimos_2_anios"] = {"peor": vals[1], "igual": vals[2], "mejor": vals[3]}

    return data, errors


def compute_all(blocks: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Puntajes base
    pg = calc_score(blocks["percepcion_general"], WEIGHTS["percepcion_general"])
    ca = calc_score(blocks["comparacion_anio_anterior"], WEIGHTS["comparacion_anio_anterior"])
    sfp = calc_score(blocks["calificacion_servicio_fp"], WEIGHTS["calificacion_servicio_fp"])
    u2 = calc_score(blocks["calif_servicio_ultimos_2_anios"], WEIGHTS["calif_servicio_ultimos_2_anios"])

    out["puntaje_percepcion_general"] = pg
    out["puntaje_comparacion_anio_anterior"] = ca
    out["puntaje_calificacion_servicio_fp"] = sfp
    out["puntaje_calif_servicio_ultimos_2_anios"] = u2

    # Agrupados como tu imagen
    out["percepcion_del_entorno"] = (pg + ca) / 2.0
    out["desempeno_policia"] = (sfp + u2) / 2.0
    out["indice_global"] = (out["percepcion_del_entorno"] + out["desempeno_policia"]) / 2.0

    out["nivel_indice"] = classify_index(out["indice_global"])
    return out


# ============================================================
# UI
# ============================================================
st.title("Índice Territorial — Lectura masiva de Excel")
st.caption("Carga varios Excel, extrae los bloques de la hoja de Información Relevante, calcula puntajes e Índice Global.")

with st.expander("⚙️ Configuración (si algún Excel viene diferente)", expanded=False):
    st.write("Este primer entregable asume que todos los Excel vienen con el mismo formato que tu imagen.")
    st.write("Luego ajustamos detección por celdas/zonas si algún archivo trae variaciones.")

files = st.file_uploader("Sube hasta 80 archivos Excel", type=["xlsx", "xlsm"], accept_multiple_files=True)

if not files:
    st.info("Sube uno o varios archivos para empezar.")
    st.stop()

results_rows: List[Dict[str, Any]] = []
errors_all: List[Tuple[str, List[str]]] = []

for f in files:
    try:
        df = read_df(f)
        blocks, errs = extract_blocks(df)

        if errs:
            errors_all.append((f.name, errs))
            continue

        calc = compute_all(blocks)

        row = {
            "archivo": f.name,
            "puntaje_percepcion_general": round(calc["puntaje_percepcion_general"], 3),
            "puntaje_comparacion_anio_anterior": round(calc["puntaje_comparacion_anio_anterior"], 3),
            "puntaje_calificacion_servicio_fp": round(calc["puntaje_calificacion_servicio_fp"], 3),
            "puntaje_calif_servicio_ultimos_2_anios": round(calc["puntaje_calif_servicio_ultimos_2_anios"], 3),
            "percepcion_del_entorno": round(calc["percepcion_del_entorno"], 3),
            "desempeno_policia": round(calc["desempeno_policia"], 3),
            "indice_global": round(calc["indice_global"], 3),
            "nivel_indice": calc["nivel_indice"],
        }
        results_rows.append(row)

        # Tarjeta por archivo (similar a tu cuadro)
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader(f"📄 {f.name}")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown('<div class="kpi"><div class="label">Percepción del entorno</div>'
                            f'<div class="value">{row["percepcion_del_entorno"]:.2f}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown('<div class="kpi"><div class="label">Desempeño policía</div>'
                            f'<div class="value">{row["desempeno_policia"]:.2f}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="kpi"><div class="label">Índice Global</div>'
                            f'<div class="value">{row["indice_global"]:.2f}</div></div>', unsafe_allow_html=True)

            level = row["nivel_indice"]
            color = badge_color(level)
            st.markdown(
                f'<span class="badge" style="background:{color}; color:#111;">{level}</span>',
                unsafe_allow_html=True
            )
            st.markdown('<hr class="sep">', unsafe_allow_html=True)

            st.markdown("**Puntajes por bloque (0-100):**")
            tbl = f"""
            <table class="table-like">
              <tr><th>Bloque</th><th>Puntaje</th></tr>
              <tr><td>Percepción General</td><td>{row["puntaje_percepcion_general"]:.2f}</td></tr>
              <tr><td>Comparación Año Anterior</td><td>{row["puntaje_comparacion_anio_anterior"]:.2f}</td></tr>
              <tr><td>Calificación Servicio FP</td><td>{row["puntaje_calificacion_servicio_fp"]:.2f}</td></tr>
              <tr><td>Calificación Servicio Últimos 2 Años</td><td>{row["puntaje_calif_servicio_ultimos_2_anios"]:.2f}</td></tr>
            </table>
            <div class="muted" style="margin-top:8px;">
              Fórmulas: Percepción del entorno = promedio(Percepción General, Comparación Año Anterior).
              Desempeño policía = promedio(Calificación FP, Últimos 2 años).
              Índice Global = promedio(Percepción del entorno, Desempeño policía).
            </div>
            """
            st.markdown(tbl, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        errors_all.append((f.name, [f"Error leyendo/calculando: {e}"]))

st.divider()

# Tabla consolidada
if results_rows:
    st.subheader("📊 Consolidado (todos los Excel leídos)")
    res_df = pd.DataFrame(results_rows).sort_values(["indice_global"], ascending=True)
    st.dataframe(res_df, use_container_width=True)

    # Descargar excel consolidado
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        res_df.to_excel(writer, index=False, sheet_name="consolidado")
    st.download_button(
        "⬇️ Descargar consolidado (Excel)",
        data=out.getvalue(),
        file_name="consolidado_indices.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# Errores
if errors_all:
    st.subheader("⚠️ Archivos con problemas (no se pudieron leer con el formato esperado)")
    for name, errs in errors_all:
        with st.expander(name, expanded=False):
            for er in errs:
                st.write("• " + er)


