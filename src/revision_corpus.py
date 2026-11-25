"""Bloque 32: dos controles de la matriz de riesgos que antes eran solo intenciones.

1. Vigencia de los .txt de observatorios (riesgo ALTO "horarios o cupos vencidos").
   Cada .txt trae "Descargado: AAAA-MM-DD" en el encabezado. Si alguno tiene más de 30 días,
   `indexar_corpus.py` no reindexa y `preflight.py` falla hasta que una persona revise los
   sitios y lo registre con:
       python src/revision_corpus.py --confirmar German --nota "revisé ESO, ALMA y Mamalluca"
   La confirmación queda en `revision_observatorios.json` (se versiona: es la evidencia) y vale
   30 días. Volver a descargar los .txt también renueva la vigencia.

2. Datos personales en los .jsonl versionados (riesgo ALTO "texto del usuario en GitHub").
   `trazas_agente.jsonl`, `registros_latencia_costo.jsonl` y `muestra_formato_b30.jsonl` son
   evidencia de entregas (B23, B28, B30), así que no se sacan del repo. En cambio, `preflight.py`
   falla si alguno contiene un correo, un RUT o un celular chileno. No detecta nombres propios.

Uso:
    python src/revision_corpus.py               # estado de los dos controles (sale 1 si falla)
    python src/revision_corpus.py --confirmar German --nota "..."
Solo librería estándar, sin API.
"""
import argparse
import json
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_OBSERVATORIOS = (RAIZ.parent.parent / "Corpus RAG astroturismo" / "originales"
                         / "observatorios_turisticos")
ARCHIVO_REVISION = RAIZ / "revision_observatorios.json"
DIAS_VIGENCIA = 30
JSONL_VERSIONADOS = ("trazas_agente.jsonl", "registros_latencia_costo.jsonl",
                     "muestra_formato_b30.jsonl")

PATRON_DESCARGA = re.compile(r"Descargado:\s*(\d{4}-\d{2}-\d{2})")
PATRONES_PII = {
    "correo": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    "rut": re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b"),
    "celular": re.compile(r"(?:\+?56\s?)?\b9\s?\d{4}\s?\d{4}\b"),
}


# ---------------------------------------------------------------- 1. vigencia

def fecha_descarga(archivo):
    """Fecha de "Descargado:" en los primeros 600 caracteres, o None si no está."""
    with open(archivo, encoding="utf-8", errors="replace") as f:
        coincidencia = PATRON_DESCARGA.search(f.read(600))
    return date.fromisoformat(coincidencia.group(1)) if coincidencia else None


def leer_revisiones(archivo=ARCHIVO_REVISION):
    archivo = Path(archivo)
    if not archivo.exists():
        return []
    return json.loads(archivo.read_text(encoding="utf-8"))


def ultima_revision(archivo=ARCHIVO_REVISION):
    revisiones = leer_revisiones(archivo)
    return date.fromisoformat(revisiones[-1]["fecha"][:10]) if revisiones else None


def observatorios_vencidos(carpeta=CARPETA_OBSERVATORIOS, revision=ARCHIVO_REVISION,
                           hoy=None, dias=DIAS_VIGENCIA):
    """Lista de (nombre, fecha_descarga o None) que necesitan revisión humana.

    Un .txt está vigente si se descargó hace <= `dias` días o si hay una confirmación humana
    de hace <= `dias` días. Sin fecha en el encabezado cuenta como vencido.
    Si la carpeta no existe (por ejemplo, un clon limpio del repo), devuelve [].
    """
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        return []
    hoy = hoy or date.today()
    revisado = ultima_revision(revision)
    if revisado is not None and (hoy - revisado).days <= dias:
        return []
    vencidos = []
    for archivo in sorted(carpeta.glob("*.txt")):
        descargado = fecha_descarga(archivo)
        if descargado is None or (hoy - descargado).days > dias:
            vencidos.append((archivo.name, descargado))
    return vencidos


def problema_vigencia(carpeta=CARPETA_OBSERVATORIOS, revision=ARCHIVO_REVISION, hoy=None):
    """Texto del problema para preflight/indexar, o None si está todo vigente."""
    vencidos = observatorios_vencidos(carpeta, revision, hoy)
    if not vencidos:
        return None
    mas_viejo = min((d for _, d in vencidos if d), default=None)
    return (f"{len(vencidos)} .txt de observatorios sin revisar hace más de {DIAS_VIGENCIA} días"
            f" (el más antiguo: {mas_viejo or 'sin fecha'}). Revisa los sitios y confirma con: "
            f'python src/revision_corpus.py --confirmar <tu nombre> --nota "qué revisaste"')


def carpetas_observatorios(corpus):
    """Carpetas `observatorios_turisticos` dentro de `corpus` (o el corpus mismo si se llama así).

    El reindexado real usa --corpus "Corpus RAG astroturismo" y los .txt están en
    originales/observatorios_turisticos: por eso se busca en subcarpetas y no en una ruta fija.
    """
    corpus = Path(corpus)
    candidatas = [corpus] + [d for d in corpus.rglob("observatorios_turisticos")]
    return [d for d in candidatas if d.is_dir() and d.name == "observatorios_turisticos"]


def revisar_vigencia_corpus(corpus, patron="*.md", revision=ARCHIVO_REVISION, hoy=None):
    """Control para indexar_corpus.py. Devuelve (problema, aviso); cada uno None o un texto.

    - problema: hay .txt de observatorios vencidos en lo que se va a indexar (no reindexar).
    - aviso: se indexan .txt y no se encontró ninguna carpeta de observatorios, así que el
      control no revisó nada. Se avisa en vez de pasar en silencio (bug del 25-11: se buscaba
      en una ruta fija que no existía y el control nunca corría).
    """
    carpetas = carpetas_observatorios(corpus)
    if not carpetas:
        aviso = None
        if patron != "*.md":
            aviso = (f"no se encontró ninguna carpeta observatorios_turisticos en {corpus}: "
                     "no se revisó la vigencia de los horarios")
        return None, aviso
    problemas = [p for p in (problema_vigencia(c, revision, hoy) for c in carpetas) if p]
    return (" | ".join(problemas) or None), None


def confirmar_revision(revisor, nota, carpeta=CARPETA_OBSERVATORIOS, archivo=ARCHIVO_REVISION):
    if not revisor or not revisor.strip():
        raise ValueError("Falta quién revisó.")
    if not nota or not nota.strip():
        raise ValueError("La nota es obligatoria: qué sitios revisaste y qué encontraste.")
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise ValueError(f"No existe la carpeta de observatorios: {carpeta}")
    archivos = {a.name: (str(fecha_descarga(a)) if fecha_descarga(a) else None)
                for a in sorted(carpeta.glob("*.txt"))}
    revisiones = leer_revisiones(archivo)
    revision = {"fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "revisor": revisor.strip(), "nota": nota.strip(), "archivos": archivos}
    revisiones.append(revision)
    Path(archivo).write_text(json.dumps(revisiones, ensure_ascii=False, indent=2), encoding="utf-8")
    return revision


# ---------------------------------------------------------------- 2. datos personales

def buscar_pii(ruta):
    """Lista de (n_linea, tipo) con posibles datos personales. [] si no existe."""
    ruta = Path(ruta)
    if not ruta.exists():
        return []
    hallazgos = []
    for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), start=1):
        for tipo, patron in PATRONES_PII.items():
            if patron.search(linea):
                hallazgos.append((numero, tipo))
    return hallazgos


def problemas_pii(raiz=RAIZ, archivos=JSONL_VERSIONADOS):
    problemas = []
    for nombre in archivos:
        hallazgos = buscar_pii(Path(raiz) / nombre)
        if hallazgos:
            detalle = ", ".join(f"línea {n} ({t})" for n, t in hallazgos[:5])
            problemas.append(f"{nombre} tiene posibles datos personales: {detalle}. "
                             "Bórralos o reemplázalos antes de hacer push.")
    return problemas


def main(argv=None):
    parser = argparse.ArgumentParser(description="Controles de vigencia y datos personales (B32).")
    parser.add_argument("--confirmar", metavar="REVISOR", help="registra la revisión humana")
    parser.add_argument("--nota", default="", help="qué se revisó (obligatoria con --confirmar)")
    args = parser.parse_args(argv)

    if args.confirmar:
        try:
            revision = confirmar_revision(args.confirmar, args.nota)
        except ValueError as error:
            parser.error(str(error))
        print(f"Revisión registrada por {revision['revisor']} ({len(revision['archivos'])} archivos). "
              f"Vale {DIAS_VIGENCIA} días.")
        return 0

    problemas = [p for p in [problema_vigencia()] if p] + problemas_pii()
    for problema in problemas:
        print("FALLA:", problema)
    if not CARPETA_OBSERVATORIOS.is_dir():
        print("Aviso: no está la carpeta de observatorios; no se revisó la vigencia.")
    print("Controles OK" if not problemas else "Controles con fallas")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
