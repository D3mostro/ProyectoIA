"""Bloques 24-25 (Core): indexa el corpus de astroturismo en Chroma, una sola vez.

Las consultas (Bloque 26) se reconectan con `conectar()`; nunca vuelven a indexar.
B25: lotes con reintentos ante 429, reemplazo por documento sin huérfanos,
validación local antes de gastar cuota y corpus/colección configurables por CLI.
"""
import argparse
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_CORPUS = RAIZ / "corpus"
CARPETA_VECTORSTORE = RAIZ / "vectorstore"
COLECCION = "astroturismo"
MODELO_EMBEDDINGS = "models/gemini-embedding-001"

# Recursivo porque el corpus tiene párrafos con una idea cada uno (L2 del B24).
# 300 caracteres ≈ un párrafo; 50 de solape (~17 %) para no perder una idea que cruce el corte.
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

# Reglas de nombre de colección de Chroma, validadas aquí para dar un mensaje claro.
PATRON_COLECCION = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")
# Formato gRPC antiguo ("retry_delay { seconds: 33 }") y el del SDK google-genai ("'retryDelay': '33s'").
PATRON_RETRY = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)|retryDelay['\"]?\s*:\s*['\"]?(\d+)")


def chunks_de_documento(nombre_archivo, texto):
    """Chunks de un archivo, con metadata "fuente" e ids deterministas "<ruta sin extensión>-<i>"."""
    partes = SPLITTER.create_documents([texto], metadatas=[{"fuente": nombre_archivo}])
    base = Path(nombre_archivo).with_suffix("").as_posix()
    return partes, [f"{base}-{j}" for j in range(len(partes))]


def partir_en_chunks(carpeta=CARPETA_CORPUS, patron="*.md"):
    """Devuelve (chunks, ids). "fuente" es la ruta relativa a la carpeta (busca en subcarpetas),
    así dos archivos con el mismo nombre en carpetas distintas no chocan."""
    carpeta = Path(carpeta)
    chunks, ids = [], []
    for archivo in sorted(carpeta.rglob(patron)):
        relativa = archivo.relative_to(carpeta).as_posix()
        partes, ids_doc = chunks_de_documento(relativa, archivo.read_text(encoding="utf-8"))
        chunks += partes
        ids += ids_doc
    return chunks, ids


def deduplicar_chunks(chunks, ids):
    """Quita chunks con texto idéntico (menús o pies de página repetidos al scrapear).

    Se queda con la primera aparición. Devuelve (chunks, ids, cantidad_saltada).
    """
    vistos, unicos, ids_unicos = set(), [], []
    for chunk, id_ in zip(chunks, ids):
        texto = chunk.page_content.strip()
        if texto not in vistos:
            vistos.add(texto)
            unicos.append(chunk)
            ids_unicos.append(id_)
    return unicos, ids_unicos, len(chunks) - len(unicos)


def verificar_chunks(chunks):
    """Falla antes de llamar a la API si no hay chunks, si hay vacíos o textos duplicados."""
    if not chunks:
        raise ValueError("No hay chunks para indexar: revisa que la carpeta tenga archivos .md.")
    vistos = set()
    for chunk in chunks:
        texto = chunk.page_content.strip()
        if not texto:
            raise ValueError(f"Chunk vacío en {chunk.metadata.get('fuente', '?')}.")
        if texto in vistos:
            raise ValueError(f"Chunk duplicado en {chunk.metadata.get('fuente', '?')}: {texto[:60]!r}")
        vistos.add(texto)


def validar_coleccion(nombre):
    if not PATRON_COLECCION.match(nombre):
        raise ValueError(
            f"Nombre de colección inválido: {nombre!r}. Usa de 3 a 512 caracteres "
            "(letras, números, '.', '_' o '-') que empiecen y terminen en letra o número."
        )
    return nombre


def conectar(embeddings, carpeta=CARPETA_VECTORSTORE, coleccion=COLECCION):
    return Chroma(
        collection_name=validar_coleccion(coleccion),
        embedding_function=embeddings,
        persist_directory=str(carpeta),
    )


class CuotaAgotada(RuntimeError):
    """429 que no vale la pena esperar: límite diario o espera sugerida mayor al tope."""


def con_reintentos(operacion, max_intentos=4, espera_inicial=5, espera_maxima=120, dormir=time.sleep):
    """Ejecuta operacion(); ante 429 espera y reintenta. Otros errores (o el último 429) se re-lanzan.

    Hace falta porque GoogleGenerativeAIEmbeddings no reintenta solo. Si la cuota es diaria
    o la API pide esperar más de `espera_maxima`, falla de inmediato en vez de dormir horas.
    """
    for intento in range(max_intentos):
        try:
            return operacion()
        except Exception as error:
            mensaje = str(error)
            if "429" not in mensaje and "RESOURCE_EXHAUSTED" not in mensaje:
                raise
            if "perday" in mensaje.lower().replace(" ", ""):
                raise CuotaAgotada(
                    "Cuota diaria de la API agotada. Lo ya guardado se conserva: "
                    "vuelve a ejecutar mañana y se completa lo que falta."
                ) from error
            if intento == max_intentos - 1:
                raise
            sugerida = PATRON_RETRY.search(mensaje)
            espera = int(sugerida.group(1) or sugerida.group(2)) if sugerida else espera_inicial * 2**intento
            if espera > espera_maxima:
                raise CuotaAgotada(
                    f"La API pide esperar {espera} s (tope {espera_maxima} s). "
                    "Vuelve a ejecutar más tarde; lo ya guardado se conserva."
                ) from error
            print(f"429 (intento {intento + 1}/{max_intentos}): reintento en {espera} s", flush=True)
            dormir(espera)


def agregar_en_lotes(vectorstore, chunks, ids, tam_lote=20, **reintentos):
    """Agrega por lotes (upsert por id): un 429 que agote reintentos no borra los lotes ya guardados."""
    total = -(-len(chunks) // tam_lote)
    for n, i in enumerate(range(0, len(chunks), tam_lote), start=1):
        lote, ids_lote = chunks[i : i + tam_lote], ids[i : i + tam_lote]
        con_reintentos(lambda: vectorstore.add_documents(lote, ids=ids_lote), **reintentos)
        print(f"lote {n}/{total}: {i + len(lote)} chunks guardados", flush=True)


def reemplazar_documento(vectorstore, nombre_archivo, texto, **reintentos):
    """Re-indexa un archivo: borra todos sus chunks y agrega los nuevos.

    Solo con upsert, un documento que se acorta dejaría chunks huérfanos con texto viejo.
    """
    chunks, ids = chunks_de_documento(nombre_archivo, texto)
    verificar_chunks(chunks)
    vectorstore.delete(where={"fuente": nombre_archivo})
    agregar_en_lotes(vectorstore, chunks, ids, **reintentos)


def indexar(chunks, ids, embeddings, carpeta=CARPETA_VECTORSTORE, coleccion=COLECCION, **reintentos):
    """Deja la colección igual a (chunks, ids): upsert por lotes y borra ids que ya no existen.

    Re-ejecutar no duplica. A diferencia de borrar la colección y usar from_documents,
    si falla a mitad el índice anterior sigue usable y los lotes guardados no se pierden.
    Salta los chunks ya guardados con el mismo id y el mismo texto (no gasta cuota en ellos);
    si el texto cambió, se vuelve a embeber para no dejar contenido viejo.
    """
    verificar_chunks(chunks)
    vectorstore = conectar(embeddings, carpeta, coleccion)
    guardado = vectorstore.get(include=["documents"])
    texto_guardado = dict(zip(guardado["ids"], guardado["documents"]))
    viejos = set(texto_guardado) - set(ids)
    if viejos:
        vectorstore.delete(ids=list(viejos))
    pendientes = [(c, i) for c, i in zip(chunks, ids) if texto_guardado.get(i) != c.page_content]
    if len(pendientes) < len(chunks):
        print(f"Ya guardados sin cambios: {len(chunks) - len(pendientes)} chunks (se saltan)", flush=True)
    if pendientes:
        agregar_en_lotes(vectorstore, *map(list, zip(*pendientes)), **reintentos)
    return vectorstore


def describir_error_api(error):
    mensaje = str(error)
    if isinstance(error, CuotaAgotada):
        return mensaje
    if "429" in mensaje or "RESOURCE_EXHAUSTED" in mensaje:
        return "Límite de peticiones alcanzado. Espera un minuto y vuelve a ejecutar."
    if "API key" in mensaje or "API_KEY_INVALID" in mensaje or "401" in mensaje or "403" in mensaje:
        return 'La API rechazó tu clave. Revisa GEMINI_API_KEY en tu ".env".'
    return f"No se pudo generar el embedding ({type(error).__name__}): {mensaje}"


def leer_argumentos(argv=None):
    parser = argparse.ArgumentParser(description="Indexa un corpus .md en Chroma.")
    parser.add_argument("--corpus", type=Path, default=CARPETA_CORPUS, help="carpeta con archivos .md")
    parser.add_argument("--coleccion", default=COLECCION, help="nombre de la colección de Chroma")
    parser.add_argument("--patron", default="*.md", help='archivos a indexar, ej. "*.txt"')
    parser.add_argument("--seco", action="store_true", help="solo parte en chunks y cuenta, sin API")
    args = parser.parse_args(argv)
    try:
        validar_coleccion(args.coleccion)
    except ValueError as error:
        parser.error(str(error))
    if not args.corpus.is_dir():
        parser.error(f"No existe la carpeta de corpus: {args.corpus}")
    return args


if __name__ == "__main__":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    args = leer_argumentos()
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit('Falta GEMINI_API_KEY. Defínela en tu archivo ".env" y vuelve a ejecutar.')

    chunks, ids = partir_en_chunks(args.corpus, args.patron)
    chunks, ids, saltados = deduplicar_chunks(chunks, ids)
    if saltados:
        print(f"Se saltaron {saltados} chunks duplicados (texto idéntico).")
    try:
        verificar_chunks(chunks)  # antes de crear el cliente: sin gastar cuota
    except ValueError as error:
        raise SystemExit(str(error))
    fuentes = len({c.metadata["fuente"] for c in chunks})
    if args.seco:
        raise SystemExit(f"[seco] {len(chunks)} chunks de {fuentes} archivos; no se llamó a la API.")

    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS)
    try:
        indexar(chunks, ids, embeddings, coleccion=args.coleccion)
    except Exception as error:  # borde del programa: diagnóstico claro y código de salida 1
        print(describir_error_api(error))
        raise SystemExit(1)
    print(f"Indexados {len(chunks)} chunks de {fuentes} documentos en la colección '{args.coleccion}'")

    # Verificación: una conexión nueva ve lo mismo sin volver a indexar.
    recargado = conectar(embeddings, coleccion=args.coleccion)
    print("Chunks al reconectar:", recargado._collection.count())

    # Recuperación de prueba (adelanto del B26, idea del código de la profesora): 1 llamada de embedding.
    consulta = "¿Qué pasa si nadie aprueba mi reserva a tiempo?"
    try:
        for doc in con_reintentos(lambda: recargado.similarity_search(consulta, k=2)):
            print(f"[{doc.metadata['fuente']}] {doc.page_content[:100]}...")
    except Exception as error:
        print(describir_error_api(error))
        raise SystemExit(1)
