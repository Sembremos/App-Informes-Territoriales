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
    # normaliza tildes comunes mínimas (para robustez sin librerías extra)
    s = s.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ü", "u").replace("ñ", "n")
    return s

def to_pct(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    # Si viene 0-1, lo pasamos a 0-100
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
    if "Critico" in norm(level):
        return "#ff4d4d"
    if "Bajo" in level:
        return "#ffb84d"
    if "Medio" in level:
        return "#ffd84d"
    if "Alto" in level:
        return "#7bdc7b"
    return "#2ea44f"


# ============================================================
# Lectura robusta: incluye merges (celdas combinadas)
# ============================================================
def sheet_to_matrix(ws) -> List[List[Any]]:
    max_r = ws.max_row or 1
    max_c = ws.max_column or 1

    # rellena merges: cada celda del rango recibe el valor del topleft
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
                # match por contains (robusto)
                if nd and nd in t:
                    hits.append((r, c))
                    break
    return hits

def find_pct_column_near(mat: List[List[Any]], anchor_r: int, search_rows: int = 12) -> Optional[int]:
    # Busca una columna con header "porcentaje" o "%" cerca del anchor
    candidates = []
    r0 = max(0, anchor_r)
    r1 = min(len(mat), anchor_r + search_rows)
    for r in range(r0, r1):
        row = mat[r]
        for c, cell in enumerate(row):
            t = norm(cell)
            if t in ("%",) or "porcentaje" in t:
                candidates.append(c)
    if not candidates:
        return None
    # elige la más frecuente
    return max(set(candidates), key=candidates.count)

def read_table_down(mat: List[List[Any]], start_r: int, pct_col: int, max_rows: int = 30) -> Dict[str, float]:
    """
    Lee filas debajo del título buscando:
      - etiqueta en cualquier columna de la fila
      - % en pct_col
    Retorna dict: etiqueta_normalizada -> porcentaje
    """
    out: Dict[str, float] = {}
    r1 = min(len(mat), start_r + max_rows)

    for r in range(start_r + 1, r1):
        row = mat[r]
        pct = to_pct(row[pct_col] if pct_col < len(row) else None)

        # etiqueta: buscamos la primera celda tipo texto “no vacía” en la fila
        label = ""
        for c in range(len(row)):
            txt = norm(row[c])
            # descarta headers típicos
            if txt and ("porcentaje" not in txt) and txt not in ("respuesta", "total", "comunidad", "comercio", "%"):
                label = txt
                break

        if not label and pct is None:
            # fin lógico de tabla
            continue

        # si hay label y porcentaje, lo guardamos
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
# Bloques que necesitamos (títulos + etiquetas esperadas)
# ============================================================
BLOCKS = {
    "PG": {
        "titles": [
            "se siente seguro en su comunidad",
            "se siente seguro en la comunidad",
            "siente seguro en su comunidad",
        ],
        "expect": ["no", "si", "sí"],
    },
    "CA": {
        "titles": [
            "en comparacion con el ano anterior",
            "en comparación con el año anterior",
            "comparacion con el ano anterior",
            "comparación con el año anterior",
        ],
        "expect": ["igual", "mas seguro", "más seguro", "menos seguro"],
    },
    "SP": {
        "titles": [
            "percepcion del servicio policial",
            "percepcion servicio policial",
            "percepción del servicio policial",
        ],
        "expect": ["excelente", "buena", "regular", "mala", "muy mala"],
    },
    "UA": {
        "titles": [
            "calificacion del servicio policial del ultimo ano",
            "calificacion del servicio policial del ultimo de ano",
            "calificación del servicio policial del ultimo año",
            "calificacion del servicio policial del ultimo año",
        ],
        "expect": ["igual", "mejor", "peor"],
    },
}

# Pesos
PG_W = {"inseguro": 0.0, "seguro": 1.0}
CA_W = {"menos_seguro": 0.0, "igual": 0.5, "mas_seguro": 1.0}
SP_W = {"excelente": 1.0, "buena": 0.75, "regular": 0.50, "mala": 0.0, "muy_mala": 0.0}
UA_W = {"peor": 0.0, "igual": 0.5, "mejor": 1.0}

def score(table_map: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(float(table_map.get(k, 0.0) or 0.0) * w for k, w in weights.items())

def extract_from_workbook(wb) -> Tuple[Optional[Dict[str, Any]], List[str], Dict[str, Any]]:
    """
    Devuelve:
      - result dict (si se pudo)
      - lista de errores
      - debug info (por si falla)
    """
    debug = {"hojas": []}

    found_pg = found_ca = found_sp = found_ua = None

    for sname in wb.sheetnames:
        ws = wb[sname]
        mat = sheet_to_matrix(ws)

        sheet_debug = {"hoja": sname, "hits": {}}

        # intentar cada bloque en esta hoja
        for key, cfg in BLOCKS.items():
            if (key == "PG" and found_pg) or (key == "CA" and found_ca) or (key == "SP" and found_sp) or (key == "UA" and found_ua):
                continue

            title_cells = find_title_cells(mat, cfg["titles"])
            sheet_debug["hits"][key] = len(title_cells)

            for (tr, tc) in title_cells:
                pct_col = find_pct_column_near(mat, tr, search_rows=14)
                if pct_col is None:
                    continue
                tbl = read_table_down(mat, tr, pct_col, max_rows=40)
                if match_labels(tbl, cfg["expect"]):
                    # asignar al bloque correspondiente
                    if key == "PG" and not found_pg:
                        found_pg = tbl
                    elif key == "CA" and not found_ca:
                        found_ca = tbl
                    elif key == "SP" and not found_sp:
                        found_sp = tbl
                    elif key == "UA" and not found_ua:
                        found_ua = tbl

        debug["hojas"].append(sheet_debug)

        # si ya están los 4, salimos
        if found_pg and found_ca and found_sp and found_ua:
            break

    errors = []
    if not found_pg: errors.append("No detecté la tabla de Percepción General (No/Sí).")
    if not found_ca: errors.append("No detecté la tabla de Comparación Año Anterior (Igual/Más/Menos).")
    if not found_sp: errors.append("No detecté la tabla de Percepción del Servicio Policial (Excelente…Muy Mala).")
    if not found_ua: errors.append("No detecté la tabla de Calificación del Último Año (Igual/Mejor/Peor).")

    if errors:
        return None, errors, debug

    # Mapear a llaves usadas para calcular
    pg_map = {
        "inseguro": get_value_contains(found_pg, "no"),
        "seguro": get_value_contains(found_pg, "si") or get_value_contains(found_pg, "sí"),
    }
    ca_map = {
        "igual": get_value_contains(found_ca, "igual"),
        "mas_seguro": get_value_contains(found_ca, "mas seguro") or get_value_contains(found_ca, "más seguro"),
        "menos_seguro": get_value_contains(found_ca, "menos seguro"),
    }
    sp_map = {
        "excelente": get_value_contains(found_sp, "excelente"),
        "buena": get_value_contains(found_sp, "buena"),
        "regular": get_value_contains(found_sp, "regular"),
        "mala": get_value_contains(found_sp, "mala"),
        "muy_mala": get_value_contains(found_sp, "muy mala"),
    }
    ua_map = {
        "igual": get_value_contains(found_ua, "igual"),
        "mejor": get_value_contains(found_ua, "mejor"),
        "peor": get_value_contains(found_ua, "peor"),
    }

    # Puntajes
    s_pg = score(pg_map, PG_W)
    s_ca = score(ca_map, CA_W)
    s_sp = score(sp_map, SP_W)
    s_ua = score(ua_map, UA_W)

    entorno = (s_pg + s_ca) / 2.0
    policia = (s_sp + s_ua) / 2.0
    idx = (entorno + policia) / 2.0
    level = classify_index(idx)

    result = {
        "puntaje_percepcion_general": s_pg,
        "puntaje_comparacion_anio_anterior": s_ca,
        "puntaje_servicio_policial": s_sp,
        "puntaje_ultimo_anio": s_ua,
        "percepcion_del_entorno": entorno,
        "desempeno_policia": policia,
        "indice_global": idx,
        "nivel_indice": level,
    }
    return result, [], debug


# ============================================================
# UI
# ============================================================
st.title("Índice Territorial — Lectura masiva de Excel")
st.caption("Detecta los cuadros aunque estén en posiciones diferentes. Carga hasta 80 Excel.")

show_debug = st.toggle("🔎 Mostrar debug (solo si falla)", value=False)

files = st.file_uploader(
    "Sube hasta 80 archivos Excel (.xlsx / .xlsm)",
    type=["xlsx", "xlsm"],
    accept_multiple_files=True
)

if not files:
    st.info("Sube uno o varios archivos para empezar.")
    st.stop()

results_rows = []
fails = []

for f in files:
    try:
        wb = load_workbook(f, data_only=True)
        res, errs, dbg = extract_from_workbook(wb)

        if errs:
            fails.append({"archivo": f.name, "errores": errs, "debug": dbg})
            continue

        results_rows.append({
            "archivo": f.name,
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
        fails.append({"archivo": f.name, "errores": [f"Error general leyendo archivo: {e}"], "debug": {}})

# Mostrar resultados
if results_rows:
    st.subheader("✅ Resultados")
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

              <div style="margin-top:10px; font-weight:700;">Puntajes por bloque (0-100):</div>
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

    st.subheader("📊 Consolidado")
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

            if show_debug and item.get("debug"):
                st.write("Debug (cuántos matches de títulos por hoja):")
                for h in item["debug"].get("hojas", []):
                    st.write(f"- Hoja: **{h['hoja']}** | hits: {h['hits']}")
