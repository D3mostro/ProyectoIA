"""Bloque 28, lección 2: chequeo previo del flujo RAG, sin llamar a ninguna API.

Diferencias con el `preflight_check` de la página (fallos verificados):
- Compara líneas de .gitignore, no texto suelto: la página da OK si solo está
  "vectorstore_ejemplo/" o si la línea "# vectorstore/" está comentada.
- También revisa que .env esté ignorado (la clave nunca se sube).
- Revisa que la colección exista y tenga chunks (count() > 0), la causa más común de
  una recuperación vacía, y que el .jsonl de registros sea JSON Lines válido.
- Usa rutas absolutas desde la raíz del proyecto: funciona sin importar desde dónde se ejecute.
- Bloque 32 (matriz de riesgos): falla si los .txt de observatorios llevan más de 30 días sin
  revisión humana, o si un .jsonl versionado tiene correo, RUT o celular (src/revision_corpus.py).
"""
import argparse
import json
from pathlib import Path

import chromadb

from indexar_corpus import CARPETA_VECTORSTORE, COLECCION, RAIZ
from revision_corpus import problema_vigencia, problemas_pii

ARCHIVO_REGISTROS = RAIZ / "registros_latencia_costo.jsonl"


def entradas_gitignore(ruta):
    """Patrones activos de un .gitignore, sin comentarios, sin líneas vacías y sin '/' en los bordes."""
    patrones = set()
    for linea in Path(ruta).read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#"):
            patrones.add(linea.strip("/"))
    return patrones


def contar_chunks(carpeta, coleccion):
    """Chunks de la colección, leyendo Chroma directo (sin embeddings, sin API). None si no existe."""
    cliente = chromadb.PersistentClient(path=str(carpeta))
    # chromadb 0.6 devolvía nombres; 1.x devuelve objetos Collection
    if coleccion not in [getattr(c, "name", c) for c in cliente.list_collections()]:
        return None
    return cliente.get_collection(coleccion).count()


def validar_jsonl(ruta):
    """Números de línea que no son JSON válido (lista vacía si el archivo está bien o no existe)."""
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    malas = []
    for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), start=1):
        if not linea.strip():
            continue
        try:
            json.loads(linea)
        except json.JSONDecodeError:
            malas.append(numero)
    return malas


def preflight_check(raiz=RAIZ, carpeta=CARPETA_VECTORSTORE, coleccion=COLECCION, registros=ARCHIVO_REGISTROS,
                    observatorios=None, hoy=None):
    """Devuelve la lista de problemas encontrados (vacía si todo está en orden).

    `observatorios`: carpeta de los .txt de observatorios; por defecto, la del corpus real al
    lado de la carpeta del curso. Si no existe (clon limpio), no se revisa la vigencia.
    """
    raiz, carpeta = Path(raiz), Path(carpeta)
    if observatorios is None:
        observatorios = raiz.parent.parent / "Corpus RAG astroturismo" / "originales" / "observatorios_turisticos"
    problemas = []

    gitignore = raiz / ".gitignore"
    if gitignore.exists():
        patrones = entradas_gitignore(gitignore)
        try:
            nombre = carpeta.resolve().relative_to(raiz.resolve()).as_posix()
        except ValueError:
            nombre = carpeta.name
        if nombre not in patrones:
            problemas.append(f"{nombre}/ no está en .gitignore (línea activa, sin comentar)")
        if ".env" not in patrones:
            problemas.append(".env no está en .gitignore: tu clave podría subirse a GitHub")
    else:
        problemas.append(f"no existe {gitignore}")

    if not carpeta.is_dir():
        problemas.append(f"{carpeta} no existe: ¿corriste la indexación (src/indexar_corpus.py)?")
    else:
        total = contar_chunks(carpeta, coleccion)
        if total is None:
            problemas.append(f"la colección '{coleccion}' no existe en {carpeta}")
        elif total == 0:
            problemas.append(f"la colección '{coleccion}' está vacía")

    malas = validar_jsonl(registros)
    if malas:
        problemas.append(f"{Path(registros).name}: líneas que no son JSON válido: {malas}")

    vigencia = problema_vigencia(observatorios, raiz / "revision_observatorios.json", hoy)
    if vigencia:
        problemas.append(vigencia)
    problemas += problemas_pii(raiz)
    return problemas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chequeo previo del flujo RAG (sin API).")
    parser.add_argument("--coleccion", default=COLECCION)
    args = parser.parse_args()

    problemas = preflight_check(coleccion=args.coleccion)
    if problemas:
        print("Preflight con problemas:")
        for problema in problemas:
            print(" -", problema)
        raise SystemExit(1)
    print(f"Preflight OK: índice con la colección '{args.coleccion}', ignorado en Git, y registros válidos.")
