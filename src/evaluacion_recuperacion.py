"""Bloque 27 (Core): evalúa la recuperación con precision@k, recall@k y F1.

Corre un conjunto de preguntas etiquetadas a mano contra el retriever real del B26 y agrega
el resultado a `resultados_evaluacion.txt` (una corrida debajo de la otra, para comparar).

Diferencias con el código de la página (fallos verificados con embeddings falsos):
- Conecta con `conectar()` (colección explícita y ruta absoluta). La página abre la colección
  "langchain", vacía, y daría 0.0 en todo sin ningún error.
- Las etiquetas no son ids sueltos sino (fuente, frase de evidencia). Los ids relevantes se
  calculan contra el índice actual, así que siguen valiendo si cambias chunk_size u overlap
  (con ids fijos, "fase_lunar-1" pasa a ser otro texto al re-partir).
- Una etiqueta que no calza con ningún chunk detiene la evaluación: es un error de etiquetado,
  no un recall de 0.
- Las preguntas sin respuesta en el corpus no entran en los promedios (la página les pone
  precisión 0, que baja el promedio aunque la recuperación sea buena). Se cuentan aparte.
- Agrega al archivo en vez de sobrescribirlo, y escribe chunk_size/overlap reales.
- Recupera una sola vez con el k más grande y recorta para los demás: comparar k=3 y k=5
  no duplica el gasto de cuota.
- Con un solo chunk relevante por pregunta, precision@k tiene techo 1/k y no distingue nada.
  Por eso se guarda la posición del primer relevante y su MRR (1.º = 1, 2.º = 0,5, 3.º = 0,33,
  no llegó = 0), y se separan las preguntas parafraseadas (sin palabras del corpus) de las
  directas: si solo fallan las parafraseadas, el retriever depende de palabras compartidas.
- Preguntas "amplias" con 2 o 3 chunks relevantes (idea de la profesora): ahí la precisión
  sí puede pasar de 1/k y sirve para comparar configuraciones.
"""
import argparse
import os
from datetime import datetime

from dotenv import load_dotenv

from indexar_corpus import (
    CARPETA_VECTORSTORE,
    COLECCION,
    MODELO_EMBEDDINGS,
    RAIZ,
    SPLITTER,
    con_reintentos,
    conectar,
    describir_error_api,
)

ARCHIVO_RESULTADOS = RAIZ / "resultados_evaluacion.txt"
K_POR_DEFECTO = (3,)

# Etiquetado leyendo el corpus (corpus/*.md), no el retriever. Cada evidencia es
# (fuente, frase): son relevantes los chunks de esa fuente que contienen la frase.
CASOS_EVALUACION = [
    {"pregunta": "¿Qué responde la API cuando se pasa el límite de peticiones?",
     "evidencias": [("errores_y_cuota.md", "429 (RESOURCE_EXHAUSTED)")]},
    {"pregunta": "¿Qué hace el cliente si la clave de la API es inválida?",
     "evidencias": [("errores_y_cuota.md", "Una clave inválida no se reintenta"),
                    ("errores_y_cuota.md", "GEMINI_API_KEY en el archivo .env")]},
    {"pregunta": "¿Cuántas veces reintenta el cliente un error pasajero?",
     "evidencias": [("errores_y_cuota.md", "hasta tres intentos")]},
    {"pregunta": "¿Qué pasa con el resto de la cadena si un paso falla?",
     "evidencias": [("errores_y_cuota.md", "cada ruta tiene un respaldo")]},
    {"pregunta": "¿Cuánto pesa el archivo de efemérides que usa la fase lunar?",
     "evidencias": [("fase_lunar.md", "17 MB")]},
    {"pregunta": "¿Qué iluminación tendrá la Luna para las Gemínidas de 2026?",
     "evidencias": [("fase_lunar.md", "iluminada en un 30 %")]},
    {"pregunta": "¿Dónde se guarda el registro de cada consulta del agente?",
     "evidencias": [("memoria_y_trazas.md", "trazas_agente.jsonl")]},
    {"pregunta": "¿Qué se busca al revisar las trazas cada semana?",
     "evidencias": [("memoria_y_trazas.md", "Las trazas se revisan cada semana")]},
    {"pregunta": "¿Cuánto cuesta el millón de tokens de entrada del modelo?",
     "evidencias": [("modelo_y_costo.md", "0,30 USD por millón")]},
    {"pregunta": "¿Sirve cambiar la temperature para que las respuestas sean consistentes?",
     "evidencias": [("modelo_y_costo.md", "ignora el parámetro temperature")]},
    {"pregunta": "¿Con cuántos días de anticipación se puede pedir el pronóstico de nubes?",
     "evidencias": [("pronostico_nubes.md", "hasta 16 días")]},
    {"pregunta": "¿Qué lugares acepta la herramienta de pronóstico de nubes?",
     "evidencias": [("pronostico_nubes.md", "Vicuña, Pisco Elqui")]},
    {"pregunta": "¿Cuántas personas se pueden incluir en una reserva de observatorio?",
     "evidencias": [("reserva_observatorio.md", "entre 1 y 20 personas")]},
    {"pregunta": "¿Qué pasa si nadie aprueba la reserva a tiempo?",
     "evidencias": [("reserva_observatorio.md", "la reserva queda no aprobada")]},
    # Parafraseadas: dicen lo mismo con otras palabras, para que no gane quien comparte vocabulario.
    {"pregunta": "¿Por qué la app a veces se niega a contestar si hago muchas preguntas seguidas?",
     "parafrasis": True, "evidencias": [("errores_y_cuota.md", "429 (RESOURCE_EXHAUSTED)")]},
    {"pregunta": "¿Necesito internet para saber cuánta luna habrá una noche?",
     "parafrasis": True, "evidencias": [("fase_lunar.md", "no depende de ningún servicio externo")]},
    {"pregunta": "¿Conviene salir a mirar galaxias cuando la luna está redonda?",
     "parafrasis": True, "evidencias": [("fase_lunar.md", "con luna llena el cielo se aclara")]},
    {"pregunta": "Si apago la aplicación, ¿el asistente sigue acordándose de nuestra charla?",
     "parafrasis": True, "evidencias": [("memoria_y_trazas.md", "se pierde al cerrar el programa")]},
    {"pregunta": "¿Puedo saber si va a estar despejado dentro de tres semanas?",
     "parafrasis": True, "evidencias": [("pronostico_nubes.md", "hasta 16 días")]},
    {"pregunta": "¿Mi lugar en el observatorio queda apartado de verdad?",
     "parafrasis": True, "evidencias": [("reserva_observatorio.md", "Hoy es simulada")]},
    # Amplias (idea de la profesora, B27): la respuesta completa necesita 2 o 3 chunks.
    # Con un solo relevante, precision@3 tiene techo 0.33; con varios, sí distingue configuraciones.
    {"pregunta": "¿Cómo maneja la aplicación los errores de la API de Gemini?",
     "amplia": True, "evidencias": [("errores_y_cuota.md", "429 (RESOURCE_EXHAUSTED)"),
                                     ("errores_y_cuota.md", "Una clave inválida no se reintenta"),
                                     ("errores_y_cuota.md", "cada ruta tiene un respaldo")]},
    {"pregunta": "¿Qué controles pasa una reserva de observatorio antes de quedar aprobada?",
     "amplia": True, "evidencias": [("reserva_observatorio.md", "entre 1 y 20 personas"),
                                     ("reserva_observatorio.md", "un supervisor humano debe aprobar")]},
    {"pregunta": "¿Cómo se calcula la luna para una noche y cuándo conviene observar?",
     "amplia": True, "evidencias": [("fase_lunar.md", "calcula qué porcentaje de la Luna"),
                                     ("fase_lunar.md", "con luna llena el cielo se aclara")]},
    {"pregunta": "¿Qué impide que el agente invente datos de luna o de nubes?",
     "amplia": True, "evidencias": [("memoria_y_trazas.md", "eso significa que el agente la inventó"),
                                     ("pronostico_nubes.md", "tiene prohibido completar un pronóstico")]},
    {"pregunta": "¿Qué modelo usa la aplicación y cuánto cuesta cada consulta?",
     "amplia": True, "evidencias": [("modelo_y_costo.md", "El modelo de la aplicación es gemini-3.5-flash-lite"),
                                     ("modelo_y_costo.md", "el cliente calcula el costo de cada llamada")]},
    # Sin respuesta en el corpus: no entran en los promedios (ver evaluar()).
    {"pregunta": "¿Cuánto cuesta la entrada al observatorio Mamalluca?", "evidencias": []},
    {"pregunta": "¿Qué cámara conviene para fotografiar la Vía Láctea?", "evidencias": []},
]


class EtiquetaInvalida(ValueError):
    """Una evidencia no calza con ningún chunk del índice: hay que corregir el etiquetado."""


def precision_at_k(ids_recuperados, ids_relevantes):
    """Fracción de los recuperados que son relevantes. None si el caso no tiene relevantes
    (pregunta sin respuesta: cualquier chunk sería "irrelevante" y no mide la recuperación)."""
    if not ids_relevantes:
        return None
    if not ids_recuperados:
        return 0.0
    return len([i for i in ids_recuperados if i in ids_relevantes]) / len(ids_recuperados)


def recall_at_k(ids_recuperados, ids_relevantes):
    """Fracción de los relevantes que se recuperaron. None si el caso no tiene relevantes."""
    if not ids_relevantes:
        return None
    return len(set(ids_recuperados) & set(ids_relevantes)) / len(ids_relevantes)


def f1(precision, recall):
    """Media armónica. None si falta alguna de las dos."""
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def promediar(valores):
    """Promedio que ignora los None. None si no queda ningún valor (no 0.0: 0.0 sería un resultado)."""
    validos = [v for v in valores if v is not None]
    return sum(validos) / len(validos) if validos else None


def posicion_primer_relevante(ranking, ids_relevantes):
    """Posición (desde 1) del primer relevante en el ranking. None si no llegó o no hay relevantes."""
    for posicion, id_ in enumerate(ranking, start=1):
        if id_ in ids_relevantes:
            return posicion
    return None


def rango_reciproco(posicion, ids_relevantes):
    """1/posición; 0.0 si el relevante no llegó; None si la pregunta no tiene relevantes."""
    if not ids_relevantes:
        return None
    return 1 / posicion if posicion else 0.0


def grupo(caso):
    if not caso["evidencias"]:
        return "sin_respuesta"
    if caso.get("amplia"):
        return "amplia"
    return "parafraseada" if caso.get("parafrasis") else "directa"


def resolver_relevantes(vectorstore, casos):
    """Convierte las evidencias (fuente, frase) en ids del índice actual. Devuelve una lista de sets.

    Falla si una evidencia no calza con ningún chunk: con ids escritos a mano ese error pasaría
    callado como recall 0.
    """
    datos = vectorstore.get(include=["documents", "metadatas"])
    chunks = list(zip(datos["ids"], datos["documents"], datos["metadatas"]))
    if not chunks:
        raise EtiquetaInvalida("El índice está vacío: no hay chunks contra los cuales etiquetar.")
    resultado = []
    for caso in casos:
        ids = set()
        for fuente, frase in caso["evidencias"]:
            calzan = {i for i, texto, meta in chunks if (meta or {}).get("fuente") == fuente and frase in texto}
            if not calzan:
                raise EtiquetaInvalida(
                    f"La evidencia {frase!r} de {fuente!r} no aparece en ningún chunk "
                    f"(pregunta: {caso['pregunta']!r}). Corrige la etiqueta o revisa el índice."
                )
            ids |= calzan
        resultado.append(ids)
    return resultado


def evaluar(vectorstore, casos=CASOS_EVALUACION, ks=K_POR_DEFECTO):
    """Corre cada pregunta una sola vez con el k más grande y calcula las métricas para cada k.

    Devuelve {k: {"casos": [...], "precision": p, "recall": r, "f1": f, "mrr": m,
    "por_grupo": {grupo: {"recall": r, "mrr": m, "n": n}}, "sin_respuesta": n}}.
    La posición y el MRR se miden sobre el ranking hasta ese k.
    Gasta 1 embedding por pregunta, sin importar cuántos k se comparen.
    """
    ks = sorted(set(ks))
    if not ks or ks[0] < 1:
        raise ValueError(f"Cada k debe ser al menos 1 (recibido: {ks}).")
    relevantes = resolver_relevantes(vectorstore, casos)
    retriever = vectorstore.as_retriever(search_kwargs={"k": ks[-1]})
    rankings = []
    for caso in casos:
        chunks = con_reintentos(lambda: retriever.invoke(caso["pregunta"]))
        rankings.append([c.id for c in chunks])

    resultados = {}
    for k in ks:
        filas = []
        for caso, rel, ranking in zip(casos, relevantes, rankings):
            recuperados = ranking[:k]
            p, r = precision_at_k(recuperados, rel), recall_at_k(recuperados, rel)
            posicion = posicion_primer_relevante(recuperados, rel)
            filas.append({
                "pregunta": caso["pregunta"],
                "grupo": grupo(caso),
                "posicion": posicion,
                "rr": rango_reciproco(posicion, rel),
                "relevantes": sorted(rel),
                "recuperados": recuperados,
                "precision": p,
                "recall": r,
                "f1": f1(p, r),
            })
        resultados[k] = {
            "casos": filas,
            "precision": promediar(f["precision"] for f in filas),
            "recall": promediar(f["recall"] for f in filas),
            "f1": promediar(f["f1"] for f in filas),
            "mrr": promediar(f["rr"] for f in filas),
            "por_grupo": {
                g: {
                    "precision": promediar(f["precision"] for f in filas if f["grupo"] == g),
                    "recall": promediar(f["recall"] for f in filas if f["grupo"] == g),
                    "mrr": promediar(f["rr"] for f in filas if f["grupo"] == g),
                    "n": sum(1 for f in filas if f["grupo"] == g),
                }
                for g in ("directa", "parafraseada", "amplia")
            },
            "sin_respuesta": sum(1 for f in filas if f["recall"] is None),
        }
    return resultados


def _num(valor):
    return "n/a " if valor is None else f"{valor:.2f}"


def formatear_resultados(resultados, coleccion, total_chunks, fecha=None):
    """Texto de una corrida: fecha, configuración real del índice, una línea por pregunta y promedios."""
    fecha = fecha or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lineas = [
        "=" * 78,
        f"Fecha: {fecha}",
        f"Configuracion: coleccion={coleccion} ({total_chunks} chunks), "
        f"chunk_size={SPLITTER._chunk_size}, chunk_overlap={SPLITTER._chunk_overlap}, "
        f"embeddings={MODELO_EMBEDDINGS}",
    ]
    for k, res in resultados.items():
        lineas += ["", f"k={k}   precision  recall  f1    pos  tipo          pregunta"]
        for f in res["casos"]:
            fallo = "  <- no recuperó lo relevante" if f["recall"] == 0 else ""
            pos = "-" if f["posicion"] is None else str(f["posicion"])
            lineas.append(
                f"      {_num(f['precision'])}       {_num(f['recall'])}    {_num(f['f1'])}  {pos:>3}  "
                f"{f['grupo']:<13} {f['pregunta']}{fallo}"
            )
        lineas.append(
            f"promedio k={k} -> precision: {_num(res['precision'])}  recall: {_num(res['recall'])}  "
            f"f1: {_num(res['f1'])}  mrr: {_num(res['mrr'])}  "
            f"(sin respuesta en el corpus, fuera del promedio: {res['sin_respuesta']})"
        )
        for g, datos in res["por_grupo"].items():
            lineas.append(f"  {g:<13} (n={datos['n']:>2}) -> precision: {_num(datos['precision'])}  recall: {_num(datos['recall'])}  mrr: {_num(datos['mrr'])}")
    return "\n".join(lineas) + "\n"


def guardar(texto, archivo=ARCHIVO_RESULTADOS):
    """Agrega la corrida al final del archivo: las anteriores se conservan para comparar."""
    with open(archivo, "a", encoding="utf-8") as salida:
        salida.write(texto + "\n")
    return archivo


def leer_argumentos(argv=None):
    parser = argparse.ArgumentParser(description="Evalúa la recuperación con precision@k y recall@k.")
    parser.add_argument("--coleccion", default=COLECCION)
    parser.add_argument("-k", type=int, nargs="+", default=list(K_POR_DEFECTO), help="uno o más k, ej. -k 3 5")
    parser.add_argument("--no-guardar", action="store_true", help="solo imprime, no agrega al archivo")
    args = parser.parse_args(argv)
    if min(args.k) < 1:
        parser.error("cada k debe ser al menos 1")
    return args


if __name__ == "__main__":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    args = leer_argumentos()
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit('Falta GEMINI_API_KEY. Defínela en tu archivo ".env" y vuelve a ejecutar.')

    vectorstore = conectar(GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS), CARPETA_VECTORSTORE, args.coleccion)
    total = vectorstore._collection.count()
    if total == 0:
        raise SystemExit(f"La colección '{args.coleccion}' está vacía: ejecuta primero src/indexar_corpus.py.")
    print(f"Colección '{args.coleccion}': {total} chunks. Gasto: {len(CASOS_EVALUACION)} embeddings de consulta.")

    try:
        resultados = evaluar(vectorstore, CASOS_EVALUACION, args.k)
    except EtiquetaInvalida as error:
        raise SystemExit(f"Etiquetado inválido: {error}")
    except Exception as error:  # borde del programa: diagnóstico claro de la API
        raise SystemExit(describir_error_api(error))

    texto = formatear_resultados(resultados, args.coleccion, total)
    print(texto)
    if not args.no_guardar:
        print(f"Corrida agregada a {guardar(texto)}")
