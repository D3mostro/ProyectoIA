"""Bloque 28 (Core): registra la latencia y el costo real del flujo RAG completo.

Corre preguntas del conjunto del B27 de punta a punta (embed de la pregunta, búsqueda en
Chroma, armado del prompt y generación con Gemini), mide cada etapa por separado y agrega
una línea por consulta a `registros_latencia_costo.jsonl`.

Diferencias con el código de la página y de la profesora (fallos verificados):
- Conecta con `conectar()` (colección explícita). Sin `collection_name`, Chroma abre la
  colección "langchain", vacía, y mediría un prompt sin contexto sin dar ningún error.
- Mide embed y búsqueda por separado, como pide la lección 1 (la actividad Core los junta).
  Para eso calcula el vector una vez y busca con `similarity_search_by_vector`.
- Usa los tokens reales de la API (`usageMetadata`, incluidos los de razonamiento) y calcula
  el costo con `calcular_costo` del cliente. La Core de la página no calcula costo.
- Los reintentos ante 429 o timeouts no inflan la latencia: el tiempo de cada etapa es el
  del intento que funcionó, y la espera entre intentos se guarda aparte.
- Costo, tokens y factor de crecimiento se guardan como números (la profesora los guarda como
  texto y no se pueden promediar). Si la API no devuelve el uso, quedan en null sin caerse.
- Guarda la respuesta y sus citas válidas e inventadas (idea de la profesora): costo y
  calidad quedan en la misma línea.
- Una consulta que falla también deja su línea, con el error y la etapa donde falló.
- Con varios k, la pregunta se embebe una sola vez: comparar k=3 y k=6 no duplica el gasto
  de embeddings.
- Bloque 30: cada línea guarda también `finish_reason` y `partes_texto`, y la respuesta se arma
  con todas las partes de texto (antes solo con la primera, y una respuesta quedó cortada).
"""
import argparse
import json
import os
import statistics
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv

from cliente import PRECIO_ENTRADA, PRECIO_SALIDA, con_reintento
from consumir_gemini import (
    MODELO,
    armar_peticion,
    calcular_costo,
    enviar_peticion,
    extraer_texto,
    motivo_de_termino,
    partes_de_texto,
)
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
from recuperacion import SinContexto, formatear_contexto, validar_citas
from registro_prompts import prompt_activo

ARCHIVO_REGISTROS = RAIZ / "registros_latencia_costo.jsonl"
KS_POR_DEFECTO = (3, 6)

# Tres preguntas del conjunto del B27, de distinto largo y tipo (directa, parafraseada, amplia).
PREGUNTAS_POR_DEFECTO = [
    "¿Qué responde la API cuando se pasa el límite de peticiones?",
    "¿Conviene salir a mirar galaxias cuando la luna está redonda?",
    "¿Qué modelo usa la aplicación y cuánto cuesta cada consulta?",
]


@contextmanager
def medir(tiempos, etapa):
    """Guarda en tiempos[etapa] los segundos que tardó el bloque `with`, aunque falle."""
    inicio = time.perf_counter()
    try:
        yield
    finally:
        tiempos[etapa] = time.perf_counter() - inicio


def con_cronometro(reintentar, operacion):
    """Ejecuta operacion() a través de `reintentar` y separa trabajo útil de espera.

    Devuelve (resultado, info). info["s"] es la duración del intento que funcionó;
    info["espera_s"] es todo lo demás (intentos fallidos y pausas entre ellos).
    """
    info = {"intentos": 0, "s": 0.0}

    def intento():
        info["intentos"] += 1
        inicio = time.perf_counter()
        resultado = operacion()
        info["s"] = time.perf_counter() - inicio
        return resultado

    inicio_total = time.perf_counter()
    resultado = reintentar(intento)
    info["espera_s"] = max(0.0, time.perf_counter() - inicio_total - info["s"])
    return resultado, info


def aprox_tokens(texto):
    """Heurística de la lección 1 (~4 caracteres por token). Solo para compararla con la real."""
    return len(texto) // 4


def leer_tokens(datos):
    """Tokens reales de la respuesta de Gemini, o None si no trae usageMetadata."""
    uso = datos.get("usageMetadata")
    if not uso or "promptTokenCount" not in uso:
        return None
    return {
        "entrada": uso.get("promptTokenCount", 0),
        "salida": uso.get("candidatesTokenCount", 0),
        "razonamiento": uso.get("thoughtsTokenCount", 0),
        "total": uso.get("totalTokenCount", 0),
    }


def _ms(segundos):
    return round(segundos * 1000, 2)


def registrar_pregunta(
    pregunta,
    vectorstore,
    embeddings,
    ks=KS_POR_DEFECTO,
    coleccion=COLECCION,
    tipo="real",
    enviar=enviar_peticion,
    modelo=MODELO,
    reintentar_embed=con_reintentos,
    reintentar_generacion=con_reintento,
):
    """Corre la pregunta una vez por cada k y devuelve un registro (dict) por k.

    El embedding de la pregunta se calcula una sola vez y se reusa en cada k
    ("embed_compartido": true cuando hay más de un k).
    """
    if not pregunta or not pregunta.strip():
        raise ValueError("La pregunta no puede estar vacía.")
    if not ks or min(ks) < 1:
        raise ValueError(f"Cada k debe ser al menos 1 (recibido: {ks}).")

    base = {
        "pregunta": pregunta,
        "coleccion": coleccion,
        "chunk_size": SPLITTER._chunk_size,
        "chunk_overlap": SPLITTER._chunk_overlap,
        "modelo": modelo,
        "tipo": tipo,
    }
    try:
        vector, info_embed = con_cronometro(reintentar_embed, lambda: embeddings.embed_query(pregunta))
    except Exception as error:
        return [_registro_error(base, k, "embed_query", error) for k in ks]

    return [
        _registrar_k(pregunta, vector, info_embed, k, vectorstore, base, enviar, modelo,
                     reintentar_generacion, compartido=len(ks) > 1)
        for k in ks
    ]


def _registrar_k(pregunta, vector, info_embed, k, vectorstore, base, enviar, modelo,
                 reintentar_generacion, compartido):
    tiempos = {"embed_query": info_embed["s"]}
    etapa = "busqueda_vectorial"
    try:
        with medir(tiempos, "busqueda_vectorial"):
            chunks = vectorstore.similarity_search_by_vector(vector, k=k)
        if not chunks:
            raise SinContexto("La búsqueda no trajo chunks: revisa colección y carpeta del índice.")

        etapa = "armado_prompt"
        with medir(tiempos, "armado_prompt"):
            activo = prompt_activo()  # B32: la versión aprobada, no una constante
            contexto, citas = formatear_contexto(chunks)
            prompt = activo.plantilla.format(contexto=contexto, pregunta=pregunta)

        etapa = "generacion"
        url, body = armar_peticion(prompt, modelo=modelo, system_instruction=activo.sistema or None)
        prompt = activo.sistema + prompt  # para medir tokens y factor con el sistema incluido
        datos, info_gen = con_cronometro(reintentar_generacion, lambda: enviar(url, body))
        tiempos["generacion"] = info_gen["s"]
        respuesta = extraer_texto(datos)
    except Exception as error:
        return _registro_error(base, k, etapa, error, tiempos)

    tokens = leer_tokens(datos)
    costo = calcular_costo(datos, PRECIO_ENTRADA, PRECIO_SALIDA) if tokens else None
    validas, inventadas = validar_citas(respuesta, citas)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **base,
        "k": k,
        "prompt_huella": activo.huella,
        "tiempos_ms": {nombre: _ms(s) for nombre, s in tiempos.items()},
        "tiempo_total_ms": _ms(sum(tiempos.values())),
        "espera_reintentos_ms": _ms(info_embed["espera_s"] + info_gen["espera_s"]),
        "intentos": {"embed_query": info_embed["intentos"], "generacion": info_gen["intentos"]},
        "embed_compartido": compartido,
        "tokens": tokens,
        "tokens_aprox_prompt": aprox_tokens(prompt),
        "factor_crecimiento": round(len(prompt) / len(pregunta), 1),
        "costo_usd": round(costo, 8) if costo is not None else None,
        "respuesta": respuesta,
        # Bloque 30: para detectar respuestas cortadas (MAX_TOKENS) o divididas en varias partes.
        "finish_reason": motivo_de_termino(datos),
        "partes_texto": len(partes_de_texto(datos)),
        "citas": citas,
        "citas_validas": validas,
        "citas_inventadas": inventadas,
    }


def _registro_error(base, k, etapa, error, tiempos=None):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **base,
        "k": k,
        "tiempos_ms": {nombre: _ms(s) for nombre, s in (tiempos or {}).items()},
        "error": describir_error_api(error) if etapa == "embed_query" else f"{type(error).__name__}: {error}",
        "etapa_error": etapa,
    }


def guardar_registro(registro, archivo=ARCHIVO_REGISTROS):
    """Agrega el registro como una línea JSON (JSON Lines): nunca sobrescribe lo anterior."""
    with open(archivo, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def leer_registros(archivo=ARCHIVO_REGISTROS):
    """Lee todas las líneas del .jsonl. Una línea mal formada detiene la lectura con su número."""
    registros = []
    if not os.path.exists(archivo):
        return registros
    with open(archivo, encoding="utf-8") as f:
        for numero, linea in enumerate(f, start=1):
            if not linea.strip():
                continue
            try:
                registros.append(json.loads(linea))
            except json.JSONDecodeError as error:
                raise ValueError(f"{archivo}, línea {numero}: no es JSON válido ({error.msg}).") from None
    return registros


def resumir(registros, tipo="real"):
    """Por k: cantidad, mediana de cada etapa (ms), promedio de tokens de entrada y de costo.

    Usa mediana para el tiempo porque la red mete valores sueltos muy altos.
    Excluye los registros con error y los de otro tipo (por ejemplo, "prueba").
    """
    por_k = {}
    for r in registros:
        if r.get("error") or r.get("tipo", "real") != tipo:
            continue
        por_k.setdefault(r["k"], []).append(r)

    resumen = {}
    for k, filas in sorted(por_k.items()):
        etapas = {}
        for fila in filas:
            for etapa, ms in fila["tiempos_ms"].items():
                etapas.setdefault(etapa, []).append(ms)
        con_tokens = [f for f in filas if f.get("tokens")]
        resumen[k] = {
            "consultas": len(filas),
            "mediana_ms": {etapa: round(statistics.median(v), 2) for etapa, v in etapas.items()},
            "mediana_total_ms": round(statistics.median(f["tiempo_total_ms"] for f in filas), 2),
            "tokens_entrada_prom": (round(statistics.mean(f["tokens"]["entrada"] for f in con_tokens), 1)
                                    if con_tokens else None),
            "costo_prom_usd": (round(statistics.mean(f["costo_usd"] for f in con_tokens), 8)
                               if con_tokens else None),
            "etapa_mas_lenta": max(etapas, key=lambda e: statistics.median(etapas[e])) if etapas else None,
        }
    return resumen


def formatear_resumen(resumen):
    if not resumen:
        return "No hay registros válidos para resumir."
    lineas = []
    for k, datos in resumen.items():
        etapas = ", ".join(f"{e} {ms} ms" for e, ms in datos["mediana_ms"].items())
        costo = f"${datos['costo_prom_usd']:.6f}" if datos["costo_prom_usd"] is not None else "sin datos"
        lineas.append(
            f"k={k}: {datos['consultas']} consultas | mediana total {datos['mediana_total_ms']} ms "
            f"({etapas}) | más lenta: {datos['etapa_mas_lenta']} | "
            f"tokens de entrada prom. {datos['tokens_entrada_prom']} | costo prom. {costo}"
        )
    return "\n".join(lineas)


def leer_argumentos(argv=None):
    parser = argparse.ArgumentParser(description="Registra latencia y costo del flujo RAG completo.")
    parser.add_argument("preguntas", nargs="*", help="preguntas (por defecto, 3 del conjunto del B27)")
    parser.add_argument("-k", type=int, nargs="+", default=list(KS_POR_DEFECTO), help="uno o más k, ej. -k 3 6")
    parser.add_argument("--coleccion", default=COLECCION)
    parser.add_argument("--tipo", default="real", choices=["real", "prueba"],
                        help='marca cada línea; el resumen solo usa "real"')
    parser.add_argument("--resumen", action="store_true", help="resume el .jsonl sin llamar a la API")
    args = parser.parse_args(argv)
    if min(args.k) < 1:
        parser.error("cada k debe ser al menos 1")
    return args


if __name__ == "__main__":
    args = leer_argumentos()
    if args.resumen:
        print(formatear_resumen(resumir(leer_registros(), args.tipo)))
        raise SystemExit(0)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit('Falta GEMINI_API_KEY. Defínela en tu archivo ".env" y vuelve a ejecutar.')

    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS)
    vectorstore = conectar(embeddings, CARPETA_VECTORSTORE, args.coleccion)
    total = vectorstore._collection.count()
    if total == 0:
        raise SystemExit(f"La colección '{args.coleccion}' está vacía: ejecuta primero src/indexar_corpus.py.")

    preguntas = args.preguntas or PREGUNTAS_POR_DEFECTO
    print(f"Colección '{args.coleccion}': {total} chunks. Gasto: {len(preguntas)} embeddings "
          f"+ {len(preguntas) * len(args.k)} llamadas a {MODELO}.")

    for pregunta in preguntas:
        for registro in registrar_pregunta(pregunta, vectorstore, embeddings, args.k, args.coleccion, args.tipo):
            guardar_registro(registro)
            if registro.get("error"):
                print(f"k={registro['k']} | ERROR en {registro['etapa_error']}: {registro['error']} | {pregunta}")
                continue
            costo = registro["costo_usd"]
            print(f"k={registro['k']} | {registro['tiempos_ms']} | total {registro['tiempo_total_ms']} ms | "
                  f"tokens {registro['tokens']} | costo {f'${costo:.6f}' if costo is not None else 'sin datos'} | "
                  f"{pregunta[:40]}")
            if registro["citas_inventadas"]:
                print(f"   ATENCIÓN, citas inventadas: {registro['citas_inventadas']}")

    print("\nResumen de todo el archivo:")
    print(formatear_resumen(resumir(leer_registros(), args.tipo)))
