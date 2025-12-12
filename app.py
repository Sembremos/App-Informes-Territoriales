import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openpyxl import load_workbook

st.set_page_config(page_title="Índice Territorial — Lectura masiva de Excel", layout="wide")


# ============================================================
# Normalización / conversión
# ============================================================
def norm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    s = (
        s.replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u").replace("ü", "u").replace("ñ", "n")
    )
    return s

def to_pct(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if 0 <= v <= 1.5:
        v *= 100.0
    return v

def classify_index(x: float) -> str:
    if x <= 20:
        return "Crítico (0-20)"
    if x <= 40:
        return "Bajo (20,1-40)"
    if x <= 60:
        return "Medio (40,1-60)"
    if x <= 80:
        return "Alto (60,1-80)"
    return "Muy Alto (80-100)"

def level_color(level: str) -> str:
    lv = norm(level)
    if "critico" in lv:
        return "#ff4d4d"
    if "bajo" in lv:
        return "#ffb84d"
    if "medio" in lv:
        return "#ffd84d"
    if "alto" in lv:
        return "#7bdc7b"
    return "#2ea44f"


# ============================================================
# Lectura robusta: incluye merges (celdas combinadas)
# ============================================================
def sheet_to_matrix(ws) -> List[List[Any]]:
    max_r = ws.max_row or 1
    max_c = ws.max_column or 1

    merge_map = {}
    for mr in ws.merged_cells.ranges:
        top_val = ws.cell(mr.min_row, mr.min_col).value
        for r in range(mr.min_row, mr.max_row + 1):
            for c in range(mr.min_col, mr.max_col + 1):
                merge_map[(r, c)] = top_val

    mat = []
    for r in range(1, max_r + 1):
        row = []
        for c in range(1, max_c + 1):
            v = ws.cell(r, c).value
            if v is None and (r, c) in merge_map:
                v = merge_map[(r, c)]
            row.append(v)
        mat.append(row)
    return mat


# ============================================================
# Detección de tablas por título + columna porcentaje
# ============================================================
def find_title_cells(mat: List[List[Any]], title_needles: List[str]) -> List[Tuple[int, int]]:
    needles = [norm(n) for n in title_needles]
    hits = []
    for r, row in enumerate(mat):
        for c, cell in enumerate(row):
            t = norm(cell)
            if not t:
                continue
            for nd in needles:
                if nd and nd in t:
                    hits.append((r, c))
                    break
    return hits

def find_pct_column_near(mat: List[List[Any]], anchor_r: int, search_rows: int = 14) -> Optional[int]:
    candidates = []
    r0 = max(0, anchor_r)
    r1 = min(len(mat), anchor_r + search_rows)
    for r in range(r0, r1):
        row = mat[r]
        for c, cell in enumerate(row):
            t = norm(cell)
            if t == "%" or "porcentaje" in t:
                candidates.append(c)
    if not candidates:
        return None
    return max(set(candidates), key=candidates.count)

def read_table_down(mat: List[List[Any]], start_r: int, pct_col: int, max_rows: int = 40) -> Dict[str, float]:
    out: Dict[str, float] = {}
    r1 = min(len(mat), start_r + max_rows)

    for r in range(start_r + 1, r1):
        row = mat[r]
        pct = to_pct(row[pct_col] if pct_col < len(row) else None)

        label = ""
        for c in range(len(row)):
            txt = norm(row[c])
            if not txt:
                continue
            if txt in ("respuesta", "total", "comunidad", "comercio", "%"):
                continue
            if "porcentaje" in txt:
                continue
            label = txt
            break

        if label and pct is not None:
            out[label] = float(pct)

    return out

def match_labels(table: Dict[str, float], needed_any: List[str]) -> bool:
    keys = list(table.keys())
    hits = 0
    for n in needed_any:
        nn = norm(n)
        if any(nn == k or nn in k for k in keys):
            hits += 1
    return hits >= max(2, len(needed_any) // 2)

def get_value_contains(table: Dict[str, float], needle: str) -> float:
    nd = norm(needle)
    for k, v in table.items():
        if nd == k or nd in k:
            return float(v)
    return 0.0


# ============================================================
# Bloques a detectar
# ============================================================
BLOCKS = {
    "PG": {
        "titles": ["se siente seguro en su comunidad", "siente seguro en su comunidad"],
        "expect": ["no", "si", "sí"],
    },
    "CA": {
        "titles": ["comparacion con el ano anterior", "comparacion con el año anterior", "comparación con el año anterior"],
        "expect": ["igual", "mas seguro", "más seguro", "menos seguro"],
    },
    "SP": {
        "titles": ["percepcion del servicio policial", "percepción del servicio policial"],
        "expect": ["excelente", "buena", "regular", "mala", "muy mala"],
    },
    "UA": {
        "titles": ["calificacion del servicio policial del ultimo ano", "calificacion del servicio policial del ultimo de ano", "calificacion del servicio policial del ultimo año"],
        "expect": ["igual", "mejor", "peor"],
    },
}

# Pesos (según tu lógica)
PG_W = {"inseguro": 0.0, "seguro": 1.0}
CA_W = {"menos_seguro": 0.0, "igual": 0.5, "mas_seguro": 1.0}
SP_W = {"excelente": 1.0, "buena": 0.75, "regular": 0.50, "mala": 0.0, "muy_mala": 0.0}
UA_W = {"peor": 0.0, "igual": 0.5, "mejor": 1.0}

def score(table_map: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(float(table_map.get(k, 0.0) or 0.0) * w for k, w in weights.items())


def extract_from_workbook(wb) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
    """
    Devuelve:
      - res: puntajes + índices
      - raw: datos detectados (% por opción)
      - errores
    """
    found = {"PG": None, "CA": None, "SP": None, "UA": None}
    sheet_found = {"PG": None, "CA": None, "SP": None, "UA": None}

    for sname in wb.sheetnames:
        ws = wb[sname]
        mat = sheet_to_matrix(ws)

        for key, cfg in BLOCKS.items():
            if found[key] is not None:
                continue

            title_cells = find_title_cells(mat, cfg["titles"])
            for (tr, tc) in title_cells:
                pct_col = find_pct_column_near(mat, tr, search_rows=14)
                if pct_col is None:
                    continue
                tbl = read_table_down(mat, tr, pct_col, max_rows=40)
                if match_labels(tbl, cfg["expect"]):
                    found[key] = tbl
                    sheet_found[key] = sname
                    break

        if all(found[k] is not None for k in found):
            break

    errors = []
    for k, msg in [
        ("PG", "No detecté la tabla de Percepción General (No/Sí)."),
        ("CA", "No detecté la tabla de Comparación Año Anterior (Menos/Igual/Más)."),
        ("SP", "No detecté la tabla de Percepción del Servicio Policial (Excelente…Muy Mala)."),
        ("UA", "No detecté la tabla de Calificación del Último Año (Igual/Mejor/Peor)."),
    ]:
        if found[k] is None:
            errors.append(msg)

    if errors:
        return None, None, errors

    # -----------------------
    # RAW % (lo que lee)
    # -----------------------
    raw = {
        "hoja_pg": sheet_found["PG"],
        "hoja_ca": sheet_found["CA"],
        "hoja_sp": sheet_found["SP"],
        "hoja_ua": sheet_found["UA"],

        # Percepción general
        "pg_no": get_value_contains(found["PG"], "no"),
        "pg_si": get_value_contains(found["PG"], "si") or get_value_contains(found["PG"], "sí"),

        # Comparación año anterior
        "ca_igual": get_value_contains(found["CA"], "igual"),
        "ca_mas_seguro": get_value_contains(found["CA"], "mas seguro") or get_value_contains(found["CA"], "más seguro"),
        "ca_menos_seguro": get_value_contains(found["CA"], "menos seguro"),

        # Servicio policial
        "sp_excelente": get_value_contains(found["SP"], "excelente"),
        "sp_buena": get_value_contains(found["SP"], "buena"),
        "sp_regular": get_value_contains(found["SP"], "regular"),
        "sp_mala": get_value_contains(found["SP"], "mala"),
        "sp_muy_mala": get_value_contains(found["SP"], "muy mala"),

        # Último año
        "ua_igual": get_value_contains(found["UA"], "igual"),
        "ua_mejor": get_value_contains(found["UA"], "mejor"),
        "ua_peor": get_value_contains(found["UA"], "peor"),
    }

    # -----------------------
    # Mapeo para puntajes
    # -----------------------
    pg_map = {"inseguro": raw["pg_no"], "seguro": raw["pg_si"]}
    ca_map = {"igual": raw["ca_igual"], "mas_seguro": raw["ca_mas_seguro"], "menos_seguro": raw["ca_menos_seguro"]}
    sp_map = {
        "excelente": raw["sp_excelente"], "buena": raw["sp_buena"], "regular": raw["sp_regular"],
        "mala": raw["sp_mala"], "muy_mala": raw["sp_muy_mala"]
    }
    ua_map = {"igual": raw["ua_igual"], "mejor": raw["ua_mejor"], "peor": raw["ua_peor"]}

    s_pg = score(pg_map, PG_W)
    s_ca = score(ca_map, CA_W)
    s_sp = score(sp_map, SP_W)
    s_ua = score(ua_map, UA_W)

    entorno = (s_pg + s_ca) / 2.0
    policia = (s_sp + s_ua) / 2.0
    idx = (entorno + policia) / 2.0
    level = classify_index(idx)

    res = {
        "puntaje_percepcion_general": s_pg,
        "puntaje_comparacion_anio_anterior": s_ca,
        "puntaje_servicio_policial": s_sp,
        "puntaje_ultimo_anio": s_ua,
        "percepcion_del_entorno": entorno,
        "desempeno_policia": policia,
        "indice_global": idx,
        "nivel_indice": level,
    }
    return res, raw, []


# ============================================================
# UI
# ============================================================
st.title("Índice Territorial — Lectura masiva de Excel")
st.caption("Ahora también muestra los porcentajes detectados (datos leídos), además de los puntajes.")

files = st.file_uploader(
    "Sube hasta 80 archivos Excel (.xlsx / .xlsm)",
    type=["xlsx", "xlsm"],
    accept_multiple_files=True
)

if not files:
    st.stop()

results_rows = []
fails = []

for f in files:
    try:
        wb = load_workbook(f, data_only=True)
        res, raw, errs = extract_from_workbook(wb)
        if errs:
            fails.append({"archivo": f.name, "errores": errs})
            continue

        # fila consolidada con TODO
        results_rows.append({
            "archivo": f.name,

            # datos detectados
            "pg_no_%": round(raw["pg_no"], 3),
            "pg_si_%": round(raw["pg_si"], 3),

            "ca_igual_%": round(raw["ca_igual"], 3),
            "ca_mas_seguro_%": round(raw["ca_mas_seguro"], 3),
            "ca_menos_seguro_%": round(raw["ca_menos_seguro"], 3),

            "sp_excelente_%": round(raw["sp_excelente"], 3),
            "sp_buena_%": round(raw["sp_buena"], 3),
            "sp_regular_%": round(raw["sp_regular"], 3),
            "sp_mala_%": round(raw["sp_mala"], 3),
            "sp_muy_mala_%": round(raw["sp_muy_mala"], 3),

            "ua_igual_%": round(raw["ua_igual"], 3),
            "ua_mejor_%": round(raw["ua_mejor"], 3),
            "ua_peor_%": round(raw["ua_peor"], 3),

            # puntajes
            "puntaje_percepcion_general": round(res["puntaje_percepcion_general"], 3),
            "puntaje_comparacion_anio_anterior": round(res["puntaje_comparacion_anio_anterior"], 3),
            "puntaje_servicio_policial": round(res["puntaje_servicio_policial"], 3),
            "puntaje_ultimo_anio": round(res["puntaje_ultimo_anio"], 3),

            "percepcion_del_entorno": round(res["percepcion_del_entorno"], 3),
            "desempeno_policia": round(res["desempeno_policia"], 3),
            "indice_global": round(res["indice_global"], 3),
            "nivel_indice": res["nivel_indice"],
        })

    except Exception as e:
        fails.append({"archivo": f.name, "errores": [f"Error general leyendo archivo: {e}"]})


# Render
if results_rows:
    st.subheader("✅ Resultados (con datos detectados)")

    for r in results_rows:
        color = level_color(r["nivel_indice"])
        st.markdown(
            f"""
            <div style="border:1px solid rgba(255,255,255,0.15); border-radius:14px; padding:14px; background:rgba(255,255,255,0.04); margin-bottom:12px;">
              <div style="font-weight:800; font-size:16px;">📄 {r["archivo"]}</div>

              <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
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

              <div style="margin-top:14px; font-weight:800;">Datos detectados (porcentajes):</div>
              <ul style="margin-top:6px;">
                <li><b>Percepción General:</b> No={r["pg_no_%"]:.2f}% | Sí={r["pg_si_%"]:.2f}%</li>
                <li><b>Comparación Año Anterior:</b> Menos={r["ca_menos_seguro_%"]:.2f}% | Igual={r["ca_igual_%"]:.2f}% | Más={r["ca_mas_seguro_%"]:.2f}%</li>
                <li><b>Servicio Policial:</b> Excelente={r["sp_excelente_%"]:.2f}% | Buena={r["sp_buena_%"]:.2f}% | Regular={r["sp_regular_%"]:.2f}% | Mala={r["sp_mala_%"]:.2f}% | Muy Mala={r["sp_muy_mala_%"]:.2f}%</li>
                <li><b>Último Año:</b> Igual={r["ua_igual_%"]:.2f}% | Mejor={r["ua_mejor_%"]:.2f}% | Peor={r["ua_peor_%"]:.2f}%</li>
              </ul>

              <div style="margin-top:10px; font-weight:800;">Puntajes por bloque (0-100):</div>
              <ul style="margin-top:6px;">
                <li>Percepción General (No/Sí): <b>{r["puntaje_percepcion_general"]:.2f}</b></li>
                <li>Comparación Año Anterior (Menos/Igual/Más): <b>{r["puntaje_comparacion_anio_anterior"]:.2f}</b></li>
                <li>Servicio Policial (Excelente…Muy Mala): <b>{r["puntaje_servicio_policial"]:.2f}</b></li>
                <li>Último Año (Igual/Mejor/Peor): <b>{r["puntaje_ultimo_anio"]:.2f}</b></li>
              </ul>

              <div style="opacity:0.75; font-size:12px;">
                Fórmulas: Entorno = promedio(PG, Comparación). Policía = promedio(Servicio Policial, Último Año). Global = promedio(Entorno, Policía).
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("📊 Consolidado (incluye datos detectados)")
    df_out = pd.DataFrame(results_rows).sort_values("indice_global", ascending=True)
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
    st.subheader("❌ Archivos que no calzaron (detalle)")
    for item in fails:
        with st.expander(item["archivo"], expanded=True):
            for e in item["errores"]:
                st.write("•", e)
