import io
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openpyxl import load_workbook

st.set_page_config(page_title="Índice Territorial — Lectura masiva de Excel", layout="wide")


# ============================================================
# Normalización / utilidades
# ============================================================
def norm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    s = s.replace("\n", " ").replace("\r", " ")
    s = " ".join(s.split())
    return s

def to_pct(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    # si viene 0-1
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
# Cálculo (según tu lógica)
# ============================================================
PG_WEIGHTS = {"inseguro": 0.0, "seguro": 1.0}
CA_WEIGHTS = {"menos_seguro": 0.0, "igual": 0.5, "mas_seguro": 1.0}
SP_WEIGHTS = {"excelente": 1.0, "buena": 0.75, "regular": 0.50, "mala": 0.0, "muy_mala": 0.0}
UA_WEIGHTS = {"peor": 0.0, "igual": 0.5, "mejor": 1.0}

def score_from_percentages(pcts: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(float(pcts.get(k, 0.0) or 0.0) * w for k, w in weights.items())


# ============================================================
# Lectura OPENPYXL (celdas combinadas)
# ============================================================
def sheet_to_matrix(ws) -> List[List[Any]]:
    # Construye una matriz de valores (incluyendo merges “arrastrando” el valor al resto)
    max_r = ws.max_row or 1
    max_c = ws.max_column or 1

    # mapa para merges: (r,c) -> valor del topleft
    merge_map = {}
    for mr in ws.merged_cells.ranges:
        min_row, min_col, max_row, max_col = mr.min_row, mr.min_col, mr.max_row, mr.max_col
        top_val = ws.cell(min_row, min_col).value
        for r in range(min_row, max_row + 1):
            for c in range(min_col, max_col + 1):
                merge_map[(r, c)] = top_val

    matrix = []
    for r in range(1, max_r + 1):
        row_vals = []
        for c in range(1, max_c + 1):
            val = ws.cell(r, c).value
            if val is None and (r, c) in merge_map:
                val = merge_map[(r, c)]
            row_vals.append(val)
        matrix.append(row_vals)
    return matrix


# ============================================================
# Detección por estructura de tablas
# (Respuesta | Porcentaje) y etiquetas esperadas
# ============================================================
def find_header_positions(mat: List[List[Any]]) -> List[Tuple[int, int, int]]:
    """
    Encuentra filas donde existan headers tipo:
    - "respuesta" y "porcentaje" en la misma fila
    Retorna lista de tuplas: (row_index, col_respuesta, col_porcentaje)
    """
    positions = []
    for r, row in enumerate(mat):
        row_norm = [norm(x) for x in row]
        # buscar "respuesta"
        res_cols = [c for c, v in enumerate(row_norm) if v == "respuesta" or "respuesta" in v]
        pct_cols = [c for c, v in enumerate(row_norm) if v == "porcentaje" or v == "%" or "porcentaje" in v]
        if res_cols and pct_cols:
            # escogemos la primera combinación más cercana
            rc = res_cols[0]
            pc = min(pct_cols, key=lambda x: abs(x - rc))
            positions.append((r, rc, pc))
    return positions

def read_table(mat: List[List[Any]], header_r: int, col_label: int, col_pct: int, max_rows: int = 20) -> Dict[str, float]:
    """
    Lee debajo del header hasta que se acaben etiquetas.
    """
    out = {}
    for rr in range(header_r + 1, min(len(mat), header_r + 1 + max_rows)):
        label = norm(mat[rr][col_label] if col_label < len(mat[rr]) else None)
        pct = to_pct(mat[rr][col_pct] if col_pct < len(mat[rr]) else None)
        if not label:
            # si la fila está vacía, paramos
            # pero si hay números sueltos seguimos 1 fila más por si hay merges
            if pct is None:
                break
            continue
        if pct is None:
            continue
        out[label] = float(pct)
    return out

def match_block(table: Dict[str, float], expected_any: List[str]) -> bool:
    """
    Verifica si la tabla contiene suficientes etiquetas esperadas (por texto normalizado).
    """
    keys = set(table.keys())
    hits = 0
    for e in expected_any:
        e = norm(e)
        if any(e == k or e in k for k in keys):
            hits += 1
    return hits >= max(2, len(expected_any) // 2)

def extract_blocks_from_matrix(mat: List[List[Any]]) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[Dict[str, float]], Optional[Dict[str, float]], List[str]]:
    """
    Busca todas las tablas y asigna:
    - PG: No/Sí
    - CA: Igual/Más/Menos
    - SP: Excelente/Buena/Regular/Mala/Muy Mala
    - UA: Igual/Mejor/Peor
    """
    errors = []
    headers = find_header_positions(mat)

    pg = ca = sp = ua = None

    for (hr, c_label, c_pct) in headers:
        tbl = read_table(mat, hr, c_label, c_pct)

        # PG
        if pg is None and match_block(tbl, ["no", "sí", "si"]):
            # map
            pg = {
                "inseguro": tbl.get("no", 0.0),
                "seguro": tbl.get("sí", tbl.get("si", 0.0)),
            }
            continue

        # CA
        if ca is None and match_block(tbl, ["igual", "más seguro", "mas seguro", "menos seguro"]):
            # buscar por contains
            def get_contains(k):
                for kk, vv in tbl.items():
                    if k in kk:
                        return vv
                return 0.0
            ca = {
                "igual": get_contains("igual"),
                "mas_seguro": get_contains("mas seguro") or get_contains("más seguro"),
                "menos_seguro": get_contains("menos seguro"),
            }
            continue

        # SP
        if sp is None and match_block(tbl, ["excelente", "buena", "regular", "mala", "muy mala"]):
            def get_exact_or_contains(k):
                for kk, vv in tbl.items():
                    if kk == k or k in kk:
                        return vv
                return 0.0
            sp = {
                "excelente": get_exact_or_contains("excelente"),
                "buena": get_exact_or_contains("buena"),
                "regular": get_exact_or_contains("regular"),
                "mala": get_exact_or_contains("mala"),
                "muy_mala": get_exact_or_contains("muy mala"),
            }
            continue

        # UA
        if ua is None and match_block(tbl, ["igual", "mejor", "peor"]):
            def get_contains_one(k):
                for kk, vv in tbl.items():
                    if k in kk:
                        return vv
                return 0.0
            ua = {
                "igual": get_contains_one("igual"),
                "mejor": get_contains_one("mejor"),
                "peor": get_contains_one("peor"),
            }
            continue

    if pg is None: errors.append("No pude detectar la tabla de Percepción General (No/Sí).")
    if ca is None: errors.append("No pude detectar la tabla de Comparación Año Anterior (Igual/Más/Menos).")
    if sp is None: errors.append("No pude detectar la tabla de Percepción del Servicio Policial (Excelente…Muy Mala).")
    if ua is None: errors.append("No pude detectar la tabla de Calificación del Servicio del Último Año (Igual/Mejor/Peor).")

    return pg, ca, sp, ua, errors


# ============================================================
# UI
# ============================================================
st.title("Índice Territorial — Lectura masiva de Excel")
st.caption("Esta versión NO depende de títulos en una columna; detecta las tablas por estructura (Respuesta/Porcentaje) y escanea todas las hojas.")

debug = st.toggle("🔎 Mostrar debug (recomendado para detectar por qué no aparece)", value=True)

files = st.file_uploader(
    "Sube hasta 80 archivos Excel (.xlsx / .xlsm)",
    type=["xlsx", "xlsm"],
    accept_multiple_files=True
)

if not files:
    st.stop()

results = []
fails = []

for f in files:
    try:
        wb = load_workbook(f, data_only=True)
        sheet_names = wb.sheetnames

        found = None
        debug_info = []

        for sname in sheet_names:
            ws = wb[sname]
            mat = sheet_to_matrix(ws)
            pg, ca, sp, ua, errs = extract_blocks_from_matrix(mat)

            debug_info.append({
                "hoja": sname,
                "ok": len(errs) == 0,
                "errores": errs,
            })

            if len(errs) == 0:
                found = (sname, pg, ca, sp, ua)
                break

        if not found:
            fails.append({"archivo": f.name, "errores": ["No se detectaron las 4 tablas en ninguna hoja."], "debug": debug_info})
            continue

        sname, pg, ca, sp, ua = found

        score_pg = score_from_percentages(pg, PG_WEIGHTS)
        score_ca = score_from_percentages(ca, CA_WEIGHTS)
        score_sp = score_from_percentages(sp, SP_WEIGHTS)
        score_ua = score_from_percentages(ua, UA_WEIGHTS)

        entorno = (score_pg + score_ca) / 2.0
        policia = (score_sp + score_ua) / 2.0
        global_idx = (entorno + policia) / 2.0
        level = classify_index(global_idx)

        results.append({
            "archivo": f.name,
            "hoja_detectada": sname,
            "puntaje_percepcion_general": round(score_pg, 3),
            "puntaje_comparacion_anio_anterior": round(score_ca, 3),
            "puntaje_servicio_policial": round(score_sp, 3),
            "puntaje_ultimo_anio": round(score_ua, 3),
            "percepcion_del_entorno": round(entorno, 3),
            "desempeno_policia": round(policia, 3),
            "indice_global": round(global_idx, 3),
            "nivel_indice": level,
        })

    except Exception as e:
        fails.append({"archivo": f.name, "errores": [f"Error general: {e}"], "debug": []})

# Render
if results:
    st.subheader("✅ Resultados")
    for r in results:
        color = level_color(r["nivel_indice"])
        st.markdown(
            f"""
            <div style="border:1px solid rgba(255,255,255,0.15); border-radius:14px; padding:14px; background:rgba(255,255,255,0.04); margin-bottom:12px;">
              <div style="font-weight:800; font-size:16px;">📄 {r["archivo"]}</div>
              <div style="opacity:0.75; font-size:12px;">Hoja detectada: <b>{r["hoja_detectada"]}</b></div>

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

              <div style="margin-top:10px; font-weight:700;">Puntajes por bloque (0-100):</div>
              <ul style="margin-top:6px;">
                <li>Percepción General: <b>{r["puntaje_percepcion_general"]:.2f}</b></li>
                <li>Comparación Año Anterior: <b>{r["puntaje_comparacion_anio_anterior"]:.2f}</b></li>
                <li>Servicio Policial: <b>{r["puntaje_servicio_policial"]:.2f}</b></li>
                <li>Último Año: <b>{r["puntaje_ultimo_anio"]:.2f}</b></li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("📊 Consolidado")
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
    st.subheader("❌ Archivos que no calzaron (con evidencia)")
    for item in fails:
        with st.expander(item["archivo"], expanded=True):
            for e in item["errores"]:
                st.write("•", e)

            if debug and item.get("debug"):
                st.write("Debug por hoja (para ver dónde lo está buscando):")
                for d in item["debug"]:
                    st.write(f"- Hoja: **{d['hoja']}** | OK: **{d['ok']}**")
                    for er in d["errores"]:
                        st.write(f"   • {er}")
