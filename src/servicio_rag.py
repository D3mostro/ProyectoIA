"""Gate 2: servicio RAG único, de punta a punta.

Une lo que ya existe, sin duplicarlo:
- índice persistido e incremental (B24/B25, `indexar_corpus.py`), con la limpieza de chunks
  y el tope de fichas de oferentes (27-11);
- recuperación con citas (B26) y logging con trace_id, consumo y filtro de inyección sobre
  los Document recuperados (B31): todo eso ya lo hace `logging_rag.procesar_consulta`;
- la versión de prompt aprobada en el registro (B32, `registro_prompts.prompt_activo`).

Lo que agrega este archivo:
- La respuesta siempre muestra el nombre de los archivos citados ("Fuentes: [1] archivo.txt").
  Si el modelo no citó ninguna fuente del contexto y tampoco dijo que el contexto no alcanza,
  el estado queda "sin_cita" (no "ok") y la respuesta trae un aviso para no usarla a ciegas.
- Elige la colección: usa `astroturismo_real` (corpus turístico, solo en el PC de German) y,
  si no existe o está vacía, `astroturismo` (los .md de corpus/, que vienen en el repo). Así un
  clon limpio funciona con `python src/indexar_corpus.py` y nada más.
- Modo `--simulado`: índice temporal con embeddings falsos y un modelo falso. Sin API ni cuota.

La supervisión humana (interrupt/Command) es para activar una versión de prompt, no para cada
consulta: está en `grafo_supervision.py` y la prueba `test_gate2.py`.

Uso (desde la raíz del repo):
    python src/servicio_rag.py "¿Qué tour en el Valle del Elqui acepta mascotas?"
        # real: 1 embedding + 1 llamada a Gemini por pregunta
    python src/servicio_rag.py --simulado          # sin API
    python src/servicio_rag.py --coleccion astroturismo "¿Cómo reservo un cupo?"
"""
import argparse
import os
import sys
import tempfile

from consumir_gemini import MODELO, enviar_peticion
from indexar_corpus import (
    COLECCIONES_PREFERIDAS,
    conectar,
    describir_error_api,
    elegir_coleccion,
    indexar,
    partir_en_chunks,
)
from logging_rag import ARCHIVO_LOGS, configurar_logger, procesar_consulta
from muestra_formato import PATRON_NO_ALCANZA
from recuperacion import K_POR_DEFECTO

PREGUNTA_POR_DEFECTO = "¿Qué necesito para reservar una visita a un observatorio?"
AVISO_SIN_CITA = ("Aviso: esta respuesta no cita ninguna fuente del contexto recuperado. "
                  "No la uses sin revisarla.")
MENSAJES = {
    "bloqueada": "No puedo procesar esa consulta: trae instrucciones para cambiar mi comportamiento.",
    "sin_contexto": "No encontré información confiable en el corpus para responder esa pregunta.",
}


def con_fuentes(texto, citas_validas):
    """Agrega los archivos citados al final: el nombre de la fuente queda en el texto."""
    if not citas_validas:
        return texto
    return texto.rstrip() + "\n\nFuentes: " + "; ".join(f"[{n}] {f}" for n, f in sorted(citas_validas.items()))


def ejecutar_servicio(pregunta, vectorstore, k=K_POR_DEFECTO, enviar=enviar_peticion, **opciones):
    """Corre una consulta completa. Devuelve el dict de `procesar_consulta` con dos campos más:

    - "texto": lo que se le muestra al turista (respuesta + fuentes, o el aviso que corresponda).
    - "fuentes": lista de archivos citados.

    Estados: "ok" (cita fuentes), "reconoce_limite" (dice que el contexto no alcanza),
    "sin_cita" (no cita ni lo reconoce), "bloqueada", "sin_contexto".
    """
    resultado = procesar_consulta(pregunta, vectorstore=vectorstore, k=k, enviar=enviar, **opciones)
    estado = resultado["estado"]
    if estado in MENSAJES:
        return {**resultado, "texto": MENSAJES[estado], "fuentes": []}
    validas = resultado.get("citas_validas") or {}
    texto = resultado["respuesta"]
    if validas:
        return {**resultado, "texto": con_fuentes(texto, validas), "fuentes": list(validas.values())}
    if PATRON_NO_ALCANZA.search(texto):
        return {**resultado, "estado": "reconoce_limite", "texto": texto, "fuentes": []}
    return {**resultado, "estado": "sin_cita", "texto": texto.rstrip() + "\n\n" + AVISO_SIN_CITA,
            "fuentes": []}


def preparar_simulado():
    """Índice temporal de corpus/*.md con embeddings falsos y un modelo falso que cita [1]."""
    from langchain_core.embeddings import DeterministicFakeEmbedding

    embeddings = DeterministicFakeEmbedding(size=16)
    carpeta = tempfile.mkdtemp(prefix="vs_simulado_")
    indexar(*partir_en_chunks(), embeddings, carpeta)

    def enviar_simulado(url, body):
        return {"candidates": [{"content": {"parts": [{"text": "[SIMULADO] Según el contexto [1], "
                                                                "revisa las condiciones antes de reservar."}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0,
                                  "thoughtsTokenCount": 0, "totalTokenCount": 0}}

    return conectar(embeddings, carpeta), "simulada", enviar_simulado


def preparar_real(coleccion=None):
    from dotenv import load_dotenv
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    from indexar_corpus import MODELO_EMBEDDINGS

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit('Falta GEMINI_API_KEY. Defínela en tu archivo ".env" y vuelve a ejecutar.')
    vectorstore, nombre = elegir_coleccion(GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS), pedida=coleccion)
    return vectorstore, nombre, enviar_peticion


def leer_argumentos(argv=None):
    parser = argparse.ArgumentParser(description="Servicio RAG de astroturismo (Gate 2).")
    parser.add_argument("preguntas", nargs="*", help="preguntas (1 embedding + 1 llamada cada una)")
    parser.add_argument("--simulado", action="store_true", help="sin API: embeddings y modelo falsos")
    parser.add_argument("--coleccion", default=None,
                        help=f"por defecto, la primera con chunks de: {', '.join(COLECCIONES_PREFERIDAS)}")
    parser.add_argument("-k", type=int, default=K_POR_DEFECTO)
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("-k debe ser al menos 1")
    return args


def main(argv=None, archivo_logs=ARCHIVO_LOGS):
    args = leer_argumentos(argv)
    configurar_logger(archivo_logs, consola=False)
    vectorstore, nombre, enviar = preparar_simulado() if args.simulado else preparar_real(args.coleccion)
    preguntas = args.preguntas or [PREGUNTA_POR_DEFECTO]
    total = vectorstore._collection.count()
    gasto = "sin API" if args.simulado else f"{len(preguntas)} embeddings + {len(preguntas)} llamadas a {MODELO}"
    print(f"Colección '{nombre}' ({total} chunks), k={args.k}. Gasto: {gasto}.")
    codigo = 0
    for pregunta in preguntas:
        try:
            r = ejecutar_servicio(pregunta, vectorstore, args.k, enviar)
        except Exception as error:  # borde del programa: se informa y se sigue con la siguiente
            print(describir_error_api(error))
            codigo = 1
            continue
        print(f"\n[{r['trace_id']}] estado: {r['estado']} | prompt: {r.get('prompt_huella', '-')}")
        print(r["texto"])
    if archivo_logs is not None:
        print(f"\nTrazas (sin texto de la pregunta ni de la respuesta) en {os.path.basename(archivo_logs)}.")
    if codigo == 0:
        print("Proceso terminado sin errores.")
    return codigo


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
