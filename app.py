import streamlit as st
import openai
import pandas as pd
import json
import ast
import io
import math
from datetime import datetime


# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Challenge Mentor AI - Matriz QFD", layout="wide")

API_KEY = st.secrets["OPENROUTER_API_KEY"]
API_BASE = st.secrets["OPENROUTER_API_BASE"]
MODEL_NAME = st.secrets["OPENROUTER_MODEL"]
# -----------------------------------------------------------------------------
# INSTRUCCIONES DEL SISTEMA
# -----------------------------------------------------------------------------
INSTRUCCIONES_SISTEMA = """
Eres un asistente técnico experto en integración de sistemas mecatrónicos para la generación de matrices QFD. Recibirás cuatro entradas estructuradas: (1) contexto del Cliente o Usuario Final, (2) pregunta esencial a resolver, (3) reto específico a resolver, y (4) necesidades del cliente. Con base en estos elementos, debes realizar lo siguiente:

1. Proponer requerimientos técnicos base principales (añade '(b)'), que representen características genéricas para un producto de este tipo, es decir, funcionalidades técnicas que cualquier producto similar debería tener. Además, propones requerimientos técnicos de valor agregado (añade '(v.a.)') que respondan al reto planteado y proporcionen ventajas adicionales.

2. Crear una matriz de relaciones QFD utilizando las necesidades del cliente como filas y los requerimientos técnicos (base y de valor agregado) como columnas. Evalúa de manera rigurosa cada intersección entre una necesidad y un requerimiento técnico, respondiendo a la pregunta: "¿Qué tanto este requerimiento técnico contribuye a satisfacer esta necesidad del cliente?". Asigna valores únicamente cuando exista una relación significativa:
   - 9: Relación fuerte
   - 3: Relación moderada
   - 1: Relación débil
   - 0: Sin relación significativa

3. Asignar un valor de importancia del 1 al 5 a cada necesidad del cliente con base en el contexto del Cliente o Usuario Final, la pregunta esencial y el reto específico. Asegúrate de usar toda la escala (del 1 al 5) para reflejar diferentes niveles de prioridad entre las necesidades.

4. Generar una lista de targets y unidades asociadas a cada requerimiento técnico (en el mismo orden en que los presentas). Los targets pueden ser valores puntuales o rangos, según la naturaleza del requerimiento. Si el requerimiento técnico puede implicar múltiples variantes (por ejemplo, sensores con diferentes resoluciones), expresa el target como un rango representativo o menciona varias opciones relevantes.

5. Regresa únicamente un JSON válido, sin texto adicional, sin Markdown y sin explicación. El JSON debe tener exactamente estas claves:
   - "necesidades_cliente": lista de necesidades del cliente,
   - "importancia_cliente": lista de valores del 1 al 5,
   - "req_tecnicos_b": lista de requerimientos técnicos base,
   - "req_tecnicos_va": lista de requerimientos técnicos de valor agregado,
   - "matriz_qfd": matriz de relaciones con valores 0, 1, 3, 9,
   - "targets": lista de valores objetivo para cada requerimiento técnico,
   - "unidades": lista de unidades para cada requerimiento técnico.
"""


# -----------------------------------------------------------------------------
# FUNCIONES
# -----------------------------------------------------------------------------
def obtener_respuesta_chat(messages):
    client = openai.OpenAI(api_key=API_KEY, base_url=API_BASE)
    errores_modelo = []

    for model_name in MODEL_NAMES:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": INSTRUCCIONES_SISTEMA}] + messages,
                temperature=0.2,
            )
            return completion.choices[0].message.content

        except openai.NotFoundError as e:
            errores_modelo.append(f"{model_name}: {str(e)}")
            continue

    raise RuntimeError(
        "No se encontró un endpoint disponible para los modelos configurados. "
        "Revisa el modelo en OpenRouter o define OPENROUTER_MODEL en Streamlit Secrets.\n\n"
        + "\n".join(errores_modelo)
    )


def extraer_info_completa(contexto, pregunta_esencial, reto_especifico, necesidades):
    prompt = f"""
A continuación se presenta la información estructurada que debes analizar para generar la matriz QFD:

Contexto del Cliente o Usuario Final:
{contexto}

Pregunta esencial:
{pregunta_esencial}

Reto específico:
{reto_especifico}

Lista de necesidades del cliente:
{necesidades}

Genera únicamente un JSON válido con las claves: necesidades_cliente, importancia_cliente, req_tecnicos_b, req_tecnicos_va, matriz_qfd, targets, unidades.
"""
    return obtener_respuesta_chat([{"role": "user", "content": prompt}])


def revalorar_importancia(contexto, pregunta_esencial, reto_especifico, necesidades_cliente):
    prompt = f"""
Con base en el siguiente contexto, pregunta esencial y reto específico, revalora el nivel de importancia de las siguientes necesidades del cliente. Asigna un nuevo ranking del 1 al N, donde 1 es la más importante y N la menos importante. Devuelve únicamente una lista JSON válida de enteros, sin texto adicional.

Contexto del Cliente o Usuario Final:
{contexto}

Pregunta esencial:
{pregunta_esencial}

Reto específico:
{reto_especifico}

Necesidades del cliente:
{json.dumps(necesidades_cliente, ensure_ascii=False)}
"""
    return obtener_respuesta_chat([{"role": "user", "content": prompt}])


def extraer_json_desde_texto(texto):
    """Intenta recuperar un JSON aunque el modelo lo devuelva dentro de texto extra."""
    if not texto:
        raise ValueError("La respuesta del modelo llegó vacía.")

    texto = texto.strip()

    # Quita cercas Markdown si aparecen.
    texto = texto.replace("```json", "```").replace("```JSON", "```")
    if texto.startswith("```") and texto.endswith("```"):
        texto = texto[3:-3].strip()

    # Intenta JSON directo.
    try:
        return json.loads(texto)
    except Exception:
        pass

    # Intenta extraer desde la primera llave hasta la última.
    inicio_json = texto.find("{")
    fin_json = texto.rfind("}") + 1
    if inicio_json >= 0 and fin_json > inicio_json:
        candidato = texto[inicio_json:fin_json]
        try:
            return json.loads(candidato)
        except Exception:
            try:
                return ast.literal_eval(candidato)
            except Exception:
                pass

    # Último intento con literal_eval del texto completo.
    return ast.literal_eval(texto)


def asegurar_lista(valor, longitud, relleno=""):
    if not isinstance(valor, list):
        valor = []
    if len(valor) < longitud:
        return valor + [relleno] * (longitud - len(valor))
    return valor[:longitud]


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🤖 AI Challenge Mentor - Matriz QFD")
st.markdown("Creadora: Dra. J. Isabel Méndez Garduño")
st.subheader("Guía interactiva que te sugiere requerimientos técnicos para tu QFD.")
st.markdown(
    "Este asistente te ayudará paso a paso a obtener tu listado de requerimientos para la matriz QFD "
    "con base en el contexto del Cliente o Usuario Final, la pregunta esencial, el reto específico a resolver "
    "y la lista de necesidades del cliente. Recibirás una **MATRIZ QFD** que te servirá de base para analizarla "
    "y proponer tu propia matriz QFD."
)

if "resultado_qfd" not in st.session_state:
    st.session_state.resultado_qfd = None

with st.form("formulario_qfd"):
    st.subheader("🧩 Información contextual")
    contexto = st.text_area("🏢 Contexto del Cliente o Usuario Final")
    pregunta_esencial = st.text_area("❓ Pregunta esencial a resolver")
    reto_especifico = st.text_area("🚩 Reto específico a resolver")
    necesidades = st.text_area("📋 Lista de necesidades del cliente o usuario final conforme a la entrevista")
    submitted = st.form_submit_button("Generar matriz QFD")

if submitted:
    if not contexto or not pregunta_esencial or not reto_especifico or not necesidades:
        st.warning("Por favor completa todos los campos.")
    else:
        try:
            with st.spinner("🔍 Analizando información con el modelo de IA..."):
                resultado_texto = extraer_info_completa(
                    contexto,
                    pregunta_esencial,
                    reto_especifico,
                    necesidades,
                )
        except Exception as e:
            st.error("❌ No se pudo obtener respuesta del modelo.")
            st.info(
                "Si el error dice NotFoundError o 'No endpoints found', cambia el modelo en Streamlit Secrets "
                "con la variable OPENROUTER_MODEL. Ejemplo: deepseek/deepseek-r1-0528:free"
            )
            st.code(str(e))
            st.stop()

        try:
            resultado = extraer_json_desde_texto(resultado_texto)
        except Exception as e:
            st.error("❌ No se pudo interpretar la respuesta del modelo como JSON.")
            st.code(str(e))
            st.markdown("#### Respuesta original del modelo")
            st.code(resultado_texto)
            st.stop()

        st.session_state.resultado_qfd = resultado

if st.session_state.resultado_qfd:
    resultado = st.session_state.resultado_qfd

    req_b = resultado.get("req_tecnicos_b", [])
    req_va = resultado.get("req_tecnicos_va", [])
    columnas = req_b + req_va
    num_cols = len(columnas)

    necesidades_cliente = resultado.get("necesidades_cliente", [])
    importancias = asegurar_lista(resultado.get("importancia_cliente", []), len(necesidades_cliente), 1)
    data = resultado.get("matriz_qfd", [])

    data_padded = [
        asegurar_lista(fila if isinstance(fila, list) else [], num_cols, "")
        for fila in data
    ]

    # Asegura que la matriz tenga tantas filas como necesidades.
    while len(data_padded) < len(necesidades_cliente):
        data_padded.append([""] * num_cols)
    data_padded = data_padded[:len(necesidades_cliente)]

    df = pd.DataFrame(data_padded, columns=columnas)

    symbol_map = {
        "9": "●", 9: "●",
        "3": "○", 3: "○",
        "1": "▽", 1: "▽",
        "0": " ", 0: " ",
        "": " ", None: " ",
    }

    df_visual = df.map(lambda x: symbol_map.get(x, x))
    df_visual.insert(0, "Necesidades del cliente", necesidades_cliente)
    df_visual.insert(0, "Importancia del cliente", importancias)

    df_numeric = pd.DataFrame(data_padded, columns=columnas).apply(pd.to_numeric, errors="coerce").fillna(0)
    df_numeric.insert(0, "Necesidades del cliente", necesidades_cliente)
    df_numeric.insert(0, "Importancia del cliente", importancias)

    importancia_tecnica = df_numeric[columnas].multiply(importancias, axis=0).sum(axis=0)

    targets = asegurar_lista(resultado.get("targets", []), num_cols, "")
    unidades = asegurar_lista(resultado.get("unidades", []), num_cols, "")

    df_visual.loc["Target"] = ["", "Target"] + targets
    df_visual.loc["Unidades"] = ["", "Unidades"] + unidades
    df_visual.loc["Calificación técnica"] = ["", "Calificación de importancia técnica"] + list(importancia_tecnica)

    # Cálculo del peso relativo en porcentaje, evitando división entre cero.
    total_importancia = importancia_tecnica.sum()
    if total_importancia > 0:
        pesos_relativos_raw = (importancia_tecnica / total_importancia) * 100
        pesos_redondeados = pesos_relativos_raw.round(1)
        diferencia = round(100 - pesos_redondeados.sum(), 1)

        if abs(diferencia) > 0:
            decimas = int(round(diferencia * 10))
            indices_ordenados = pesos_redondeados.sort_values(ascending=False).index.tolist()
            for i in range(abs(decimas)):
                idx = indices_ordenados[i % len(indices_ordenados)]
                if diferencia > 0:
                    pesos_redondeados[idx] += 0.1
                else:
                    pesos_redondeados[idx] -= 0.1
            pesos_redondeados = pesos_redondeados.round(1)
    else:
        pesos_redondeados = pd.Series([0.0] * num_cols, index=columnas)

    df_visual.loc["Peso relativo (%)"] = ["", "Peso relativo (%)"] + list(pesos_redondeados)

    max_blocks = 50

    def barra_unicode(v):
        blocks = max(1, math.ceil(v / 100 * max_blocks)) if v > 0 else 0
        return "█" * blocks

    df_visual.loc["Gráfica relativa"] = ["", "Gráfica relativa"] + [
        barra_unicode(v) for v in pesos_redondeados
    ]

    st.markdown("""
    ### 🔍 Leyenda de la matriz:
    - **●** : Relación fuerte  
    - **○** : Relación moderada  
    - **▽** : Relación débil  
    - *(espacio en blanco)* : Sin relación significativa
    """)

    st.markdown("### ✅ Matriz QFD Generada")
    st.dataframe(df_visual, use_container_width=True)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_visual.to_excel(writer, index=False, sheet_name="QFD")
    buffer.seek(0)

    nombre_archivo = f"{datetime.now().strftime('%Y%m%d-%H%M')}-matriz_qfd.xlsx"
    st.markdown("### 📥 Descargar Matriz")
    st.download_button(
        "📂 Descargar como Excel",
        data=buffer,
        file_name=nombre_archivo,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
