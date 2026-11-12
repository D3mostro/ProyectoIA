"""Bloque 24 (Core): indexa el corpus de astroturismo en Chroma, una sola vez.

Las consultas (Bloque 26) se reconectan con `conectar()`; nunca vuelven a indexar.
"""
import os
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


def partir_en_chunks(carpeta=CARPETA_CORPUS):
    """Devuelve (chunks, ids). Cada chunk lleva metadata "fuente" con el nombre del archivo."""
    chunks, ids = [], []
    for archivo in sorted(carpeta.glob("*.md")):
        partes = SPLITTER.create_documents(
            [archivo.read_text(encoding="utf-8")], metadatas=[{"fuente": archivo.name}]
        )
        chunks += partes
        ids += [f"{archivo.stem}-{j}" for j in range(len(partes))]
    return chunks, ids


def conectar(embeddings, carpeta=CARPETA_VECTORSTORE):
    return Chroma(
        collection_name=COLECCION, embedding_function=embeddings, persist_directory=str(carpeta)
    )


def indexar(chunks, ids, embeddings, carpeta=CARPETA_VECTORSTORE):
    # Se borra la colección antes de indexar: sin esto, correr el script dos veces duplica
    # los chunks (el ejemplo del curso), y un documento que se acorta deja chunks viejos.
    conectar(embeddings, carpeta).delete_collection()
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=ids,
        collection_name=COLECCION,
        persist_directory=str(carpeta),
    )


def describir_error_api(error):
    mensaje = str(error)
    if "429" in mensaje or "RESOURCE_EXHAUSTED" in mensaje:
        return "Límite de peticiones alcanzado. Espera un minuto y vuelve a ejecutar."
    if "API key" in mensaje or "API_KEY_INVALID" in mensaje or "401" in mensaje or "403" in mensaje:
        return 'La API rechazó tu clave. Revisa GEMINI_API_KEY en tu ".env".'
    return f"No se pudo generar el embedding ({type(error).__name__}): {mensaje}"


if __name__ == "__main__":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit('Falta GEMINI_API_KEY. Defínela en tu archivo ".env" y vuelve a ejecutar.')

    embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS)
    chunks, ids = partir_en_chunks()
    try:
        indexar(chunks, ids, embeddings)
    except Exception as error:  # borde del programa: diagnóstico claro y código de salida 1
        print(describir_error_api(error))
        raise SystemExit(1)
    print(f"Indexados {len(chunks)} chunks de {len(set(c.metadata['fuente'] for c in chunks))} documentos en vectorstore/")

    # Verificación: una conexión nueva ve lo mismo sin volver a indexar.
    recargado = conectar(embeddings)
    print("Chunks al reconectar:", recargado._collection.count())

    # Recuperación de prueba (adelanto del B26, idea del código de la profesora): 1 llamada de embedding.
    consulta = "¿Qué pasa si nadie aprueba mi reserva a tiempo?"
    try:
        for doc in recargado.similarity_search(consulta, k=2):
            print(f"[{doc.metadata['fuente']}] {doc.page_content[:100]}...")
    except Exception as error:
        print(describir_error_api(error))
        raise SystemExit(1)
