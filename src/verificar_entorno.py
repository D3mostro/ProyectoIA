"""Gate 2 (lección 20449): revisa el entorno antes de la corrida final. No usa la API.

Muestra la versión de cada paquete, si GEMINI_API_KEY está definida (nunca la imprime), si
.env está ignorado por Git y cuántos chunks tiene cada colección del índice persistido.
Termina con código 1 si falta algo.

    python src/verificar_entorno.py

Diferencias con el script de la lección: la carpeta del índice es `vectorstore/` (no
`indice_gate2/`), se cuentan los chunks por colección en vez de solo mirar si la carpeta
tiene archivos (una carpeta con una colección vacía pasaba como "ok"), y se revisa que `.env`
esté en `.gitignore`.
"""
import os
import sys
from importlib.metadata import PackageNotFoundError, version

from dotenv import load_dotenv

from indexar_corpus import CARPETA_VECTORSTORE, COLECCIONES_PREFERIDAS, RAIZ
from preflight import contar_chunks, entradas_gitignore

PAQUETES = ["langchain-core", "langchain-chroma", "langchain-google-genai", "langgraph",
            "chromadb", "python-dotenv", "skyfield"]
VARIABLES = ["GEMINI_API_KEY"]


def chunks_por_coleccion(carpeta=CARPETA_VECTORSTORE):
    """{nombre: chunks} de las colecciones preferidas que existen en el índice. Sin API."""
    if not carpeta.is_dir():
        return {}
    conteos = {n: contar_chunks(carpeta, n) for n in COLECCIONES_PREFERIDAS}
    return {n: c for n, c in conteos.items() if c is not None}


def revisar(carpeta=CARPETA_VECTORSTORE, raiz=RAIZ, entorno=os.environ):
    """Devuelve (lineas, problemas)."""
    lineas, problemas = [f"Python: {sys.version.split()[0]}"], []
    if sys.version_info < (3, 10):
        problemas.append("usar Python 3.10 o superior")
    for paquete in PAQUETES:
        try:
            lineas.append(f"{paquete}: {version(paquete)}")
        except PackageNotFoundError:
            lineas.append(f"{paquete}: NO instalado")
            problemas.append(f"instalar {paquete} (pip install -r requirements.txt)")
    for nombre in VARIABLES:
        definida = bool(entorno.get(nombre))
        lineas.append(f"{nombre}: {'definida' if definida else 'FALTA'}")
        if not definida:
            problemas.append(f"definir {nombre} en el archivo .env")
    gitignore = raiz / ".gitignore"
    if gitignore.exists() and ".env" in entradas_gitignore(gitignore):
        lineas.append(".env en .gitignore: ok")
    else:
        lineas.append(".env en .gitignore: NO")
        problemas.append('agregar ".env" a .gitignore')
    conteos = chunks_por_coleccion(carpeta)
    if any(conteos.values()):
        lineas += [f"índice {carpeta.name}/, colección {n}: {c} chunks" for n, c in conteos.items()]
    else:
        lineas.append(f"índice {carpeta.name}/: NO existe o no tiene chunks")
        problemas.append("indexar el corpus con python src/indexar_corpus.py")
    return lineas, problemas


def main():
    load_dotenv()
    lineas, problemas = revisar()
    print("\n".join(lineas))
    print()
    if problemas:
        print("Pendiente: " + "; ".join(problemas))
        return 1
    print("Entorno listo para la corrida final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
