"""Auditoría de los archivos .txt del corpus turístico antes de indexar. No usa Gemini ni internet.

Revisa archivo por archivo lo que la limpieza de chunks no ve:
- ERROR (bloquea el indexado): sin fecha "Descargado" (el control de vigencia del B32 no puede
  revisarlo), sin fuente ("Fuente:" o "Sitio:"), texto roto por mala codificación (Ã©, â€, �)
  o el mismo texto en dos archivos.
- AVISO (no bloquea): sin Título o Licencia, correo/RUT/celular (si el RAG lo cita, queda en los
  .jsonl y preflight.py bloquea la subida; los de contacto de un observatorio pueden ser
  intencionales), cuerpo muy corto, casi-duplicado entre dos archivos o texto en inglés.

Uso (desde la raíz del repo):
    python src/auditar_corpus.py
    python src/auditar_corpus.py --corpus "../../Corpus RAG astroturismo"

indexar_corpus.py la corre sola con el patrón "*.txt" y no indexa si hay errores.
"""
import argparse
import hashlib
import re
import sys
from datetime import date
from itertools import combinations
from pathlib import Path

from revision_corpus import PATRON_DESCARGA, PATRONES_PII

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_CORPUS = RAIZ.parent.parent / "Corpus RAG astroturismo"
CAMPOS_RECOMENDADOS = ("Título", "Licencia")
LINEAS_DE_ENCABEZADO = 8
MINIMO_CUERPO = 200            # caracteres; menos suele ser una página que no cargó
UMBRAL_CASI_DUPLICADO = 0.8    # proporción de frases de 5 palabras compartidas
PATRON_TEXTO_ROTO = re.compile(r"Ã[\x80-\xbf¡-ÿ]|â€|Â[\xa0-\xbf]|�")
PALABRAS_EN = {"the", "and", "of", "to", "is", "in", "with", "for", "are", "this", "that", "from"}
PALABRAS_ES = {"el", "la", "de", "que", "y", "en", "los", "las", "del", "por", "con", "una", "para"}


def separar_encabezado(texto):
    """Devuelve ({campo: valor}, cuerpo). El encabezado son las primeras líneas "Campo: valor"."""
    lineas = texto.splitlines()
    campos, inicio = {}, 0
    for i, linea in enumerate(lineas[:LINEAS_DE_ENCABEZADO]):
        m = re.match(r"^([A-Za-zÁÉÍÓÚáéíóúñ ]{3,30}):\s*(.*)$", linea)
        if not m:
            break
        campos[m.group(1).strip()] = m.group(2).strip()
        inicio = i + 1
    return campos, "\n".join(lineas[inicio:]).strip()


def _frases(texto, n=5):
    palabras = re.findall(r"\w+", texto.lower())
    return {" ".join(palabras[i:i + n]) for i in range(len(palabras) - n + 1)}


def _parece_ingles(texto):
    palabras = re.findall(r"[a-záéíóúñ]+", texto.lower())
    en = sum(p in PALABRAS_EN for p in palabras)
    es = sum(p in PALABRAS_ES for p in palabras)
    return en > 20 and en > es


def _enmascarar(valor):
    return valor[:2] + "***" + valor[-4:] if len(valor) > 8 else "***"


def problemas_de_archivo(texto):
    """[(nivel, mensaje)] de un archivo, sin compararlo con los demás."""
    problemas = []
    campos, cuerpo = separar_encabezado(texto)
    # Misma regla que el control de vigencia (revision_corpus.py): fecha en los primeros 600 caracteres.
    fecha = PATRON_DESCARGA.search(texto[:600])
    if not fecha:
        problemas.append(("ERROR", 'sin "Descargado: AAAA-MM-DD" al inicio (la vigencia no se puede revisar)'))
    else:
        try:
            date.fromisoformat(fecha.group(1))
        except ValueError:
            problemas.append(("ERROR", f"fecha Descargado inválida: {fecha.group(1)!r}"))
    if not (campos.get("Fuente") or campos.get("Sitio")):
        problemas.append(("ERROR", 'sin "Fuente:" ni "Sitio:" en el encabezado'))
    faltan = [c for c in CAMPOS_RECOMENDADOS if not campos.get(c)]
    if faltan:
        problemas.append(("AVISO", "falta en el encabezado: " + ", ".join(faltan)))
    roto = PATRON_TEXTO_ROTO.search(texto)
    if roto:
        problemas.append(("ERROR", f"texto roto por codificación cerca de {texto[max(0, roto.start() - 20):roto.end() + 20]!r}"))
    for tipo, patron in PATRONES_PII.items():
        encontrados = patron.findall(texto)
        if encontrados:
            problemas.append(("AVISO", f"{len(encontrados)} {tipo}(s), ej. {_enmascarar(encontrados[0])}: "
                                       "si el RAG lo cita, preflight bloquea los .jsonl"))
    if len(cuerpo) < MINIMO_CUERPO:
        problemas.append(("AVISO", f"cuerpo muy corto ({len(cuerpo)} caracteres)"))
    if _parece_ingles(cuerpo):
        problemas.append(("AVISO", "parece estar en inglés"))
    return problemas


def auditar(carpeta=CARPETA_CORPUS, patron="*.txt"):
    """{ruta relativa: [(nivel, mensaje)]} solo de los archivos con problemas."""
    carpeta = Path(carpeta)
    hallazgos, cuerpos = {}, {}
    for archivo in sorted(carpeta.rglob(patron)):
        relativa = archivo.relative_to(carpeta).as_posix()
        texto = archivo.read_text(encoding="utf-8", errors="replace")
        problemas = problemas_de_archivo(texto)
        if problemas:
            hallazgos[relativa] = problemas
        cuerpos[relativa] = separar_encabezado(texto)[1]

    por_huella = {}
    for relativa, cuerpo in cuerpos.items():
        por_huella.setdefault(hashlib.sha256(cuerpo.encode()).hexdigest(), []).append(relativa)
    exactos = set()
    for iguales in por_huella.values():
        for relativa in iguales[1:]:
            exactos.add(relativa)
            hallazgos.setdefault(relativa, []).append(("ERROR", f"mismo texto que {iguales[0]}"))

    frases = {r: _frases(c) for r, c in cuerpos.items() if r not in exactos}
    for a, b in combinations(sorted(frases), 2):
        fa, fb = frases[a], frases[b]
        if fa and fb and len(fa & fb) / min(len(fa), len(fb)) >= UMBRAL_CASI_DUPLICADO:
            hallazgos.setdefault(b, []).append(("AVISO", f"casi igual a {a}"))
    return hallazgos


def hay_errores(hallazgos):
    return any(nivel == "ERROR" for problemas in hallazgos.values() for nivel, _ in problemas)


def formatear(hallazgos, total=None):
    if not hallazgos:
        return "Auditoría del corpus: sin problemas."
    errores = sum(n == "ERROR" for p in hallazgos.values() for n, _ in p)
    avisos = sum(n == "AVISO" for p in hallazgos.values() for n, _ in p)
    lineas = [f"Auditoría del corpus: {errores} errores y {avisos} avisos en {len(hallazgos)} archivos"
              + (f" (de {total})" if total else "") + "."]
    for relativa, problemas in sorted(hallazgos.items(), key=lambda x: (min(n for n, _ in x[1]), x[0])):
        for nivel, mensaje in problemas:
            lineas.append(f"  {nivel:5} {relativa}: {mensaje}")
    return "\n".join(lineas)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Revisa los .txt del corpus antes de indexar.")
    parser.add_argument("--corpus", type=Path, default=CARPETA_CORPUS)
    parser.add_argument("--patron", default="*.txt")
    args = parser.parse_args(argv)
    if not args.corpus.is_dir():
        raise SystemExit(f"No existe la carpeta de corpus: {args.corpus}")
    hallazgos = auditar(args.corpus, args.patron)
    print(formatear(hallazgos, len(list(args.corpus.rglob(args.patron)))))
    return 1 if hay_errores(hallazgos) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
