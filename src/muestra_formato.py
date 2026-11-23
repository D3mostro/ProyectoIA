"""Bloque 30: muestra de 20 respuestas reales para la pregunta "problema_de_formato_o_tono".

La tabla del Paso 1 (lección 20440) pide: "toma 20 respuestas reales, revísalas con una lista
fija de criterios y cuenta los incumplimientos". La respuesta es True si fallan 3 o más (15 %).

Corre las 20 consultas de `casos_evaluacion.py` por el flujo RAG completo del Bloque 28
(`registrar_pregunta`, k=3, colección real) y guarda una línea por respuesta en
`muestra_formato_b30.jsonl`, aparte de `registros_latencia_costo.jsonl` para no mezclar la
muestra con las mediciones de latencia del Bloque 28.

Criterios fijos (una respuesta falla si no cumple alguno):
1. sin_error: la consulta no terminó en error.
2. completa: finishReason es "STOP" y el texto termina en un signo de cierre (. ! ? ) ] » ").
3. cita_o_reconoce: trae al menos una cita válida, o dice explícitamente que el contexto
   no alcanza (lo que pide el prompt del Bloque 26).
4. sin_citas_inventadas: no cita números que no estaban en el contexto.

Uso (desde la raíz del repo):
    python src/muestra_formato.py --revisar      # solo lee el .jsonl, sin API
    python src/muestra_formato.py --confirmar    # corre lo que falte de las 20 (gasta cuota)
    python src/muestra_formato.py --confirmar --rehacer-sin-contexto
        # además vuelve a correr las que dijeron "el contexto no alcanza" (después de ampliar
        # el corpus). Las líneas viejas quedan en el .jsonl; el resumen usa la última.

Gasto con el archivo vacío: 20 embeddings + 20 llamadas de generación. Si se corta (por
ejemplo, por un 429), al volver a ejecutarlo salta las consultas que ya tienen respuesta.
"""
import argparse
import os
import re
import sys

from casos_evaluacion import CASOS_EVALUACION
from indexar_corpus import RAIZ
from registro_rag import guardar_registro, leer_registros

ARCHIVO_MUESTRA = RAIZ / "muestra_formato_b30.jsonl"
COLECCION_MUESTRA = "astroturismo_real"
K_MUESTRA = 3
TAMANO_MUESTRA = 20
UMBRAL_FALLOS = 3  # 15 % de 20, el umbral de la tabla del Paso 1

CIERRES = tuple(".!?)]»\"")
# "el contexto no menciona", "no hay información", "la información proporcionada no incluye"...
PATRON_RECONOCE = re.compile(
    r"(contexto|informaci[oó]n|texto|fuentes?)[^.]{0,60}\bno\b"
    r"|\bno\b[^.]{0,40}(contexto|informaci[oó]n|datos)",
    re.IGNORECASE,
)


def consultas_de_la_muestra():
    return [caso["consulta"] for caso in CASOS_EVALUACION[:TAMANO_MUESTRA]]


def revisar_respuesta(registro):
    """Aplica los 4 criterios a una línea del .jsonl. Devuelve {criterio: bool}."""
    if registro.get("error"):
        return {"sin_error": False, "completa": False, "cita_o_reconoce": False,
                "sin_citas_inventadas": False}
    texto = (registro.get("respuesta") or "").strip()
    motivo = registro.get("finish_reason")
    return {
        "sin_error": True,
        "completa": bool(texto) and texto.endswith(CIERRES) and motivo in (None, "STOP"),
        "cita_o_reconoce": bool(registro.get("citas_validas")) or bool(PATRON_RECONOCE.search(texto)),
        "sin_citas_inventadas": not registro.get("citas_inventadas"),
    }


def ultima_por_consulta(registros):
    """Se queda con la última línea de cada consulta (una re-ejecución reemplaza a un error)."""
    por_consulta = {}
    for registro in registros:
        por_consulta[registro.get("pregunta")] = registro
    return por_consulta


PATRON_NO_ALCANZA = re.compile(r"contexto no alcanza", re.IGNORECASE)


def sin_contexto(registros, consultas):
    """Consultas cuya última respuesta dice que el contexto no alcanza."""
    por_consulta = ultima_por_consulta(registros)
    return [c for c in consultas
            if c in por_consulta and PATRON_NO_ALCANZA.search(por_consulta[c].get("respuesta") or "")]


def pendientes(registros, consultas):
    hechas = {p for p, r in ultima_por_consulta(registros).items() if not r.get("error")}
    return [c for c in consultas if c not in hechas]


def resumir_muestra(registros, consultas):
    """Revisa la última respuesta de cada consulta. Devuelve (filas, fallos, faltan, veredicto)."""
    por_consulta = ultima_por_consulta(registros)
    filas = []
    for consulta in consultas:
        if consulta not in por_consulta:
            continue
        criterios = revisar_respuesta(por_consulta[consulta])
        filas.append((consulta, criterios, all(criterios.values())))
    fallos = sum(1 for _, _, ok in filas if not ok)
    faltan = len(consultas) - len(filas)
    veredicto = None if faltan else fallos >= UMBRAL_FALLOS
    return filas, fallos, faltan, veredicto


def formatear_resumen(filas, fallos, faltan, veredicto):
    lineas = []
    for consulta, criterios, ok in filas:
        malos = [c for c, v in criterios.items() if not v]
        lineas.append(f"{'OK   ' if ok else 'FALLA'} {consulta[:60]}" + (f"  -> {', '.join(malos)}" if malos else ""))
    lineas.append("")
    lineas.append(f"revisadas: {len(filas)}  fallos: {fallos}  faltan: {faltan}  (umbral: {UMBRAL_FALLOS})")
    if veredicto is None:
        lineas.append("problema_de_formato_o_tono: sin veredicto (la muestra no está completa)")
    else:
        lineas.append(f"problema_de_formato_o_tono: {veredicto}")
    return "\n".join(lineas)


def leer_argumentos(argv=None):
    parser = argparse.ArgumentParser(description="Muestra de 20 respuestas RAG para el Bloque 30.")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--revisar", action="store_true", help="solo revisa el .jsonl (sin API)")
    grupo.add_argument("--confirmar", action="store_true", help="corre las consultas que faltan (gasta cuota)")
    parser.add_argument("--coleccion", default=COLECCION_MUESTRA)
    parser.add_argument("--rehacer-sin-contexto", action="store_true",
                        help='con --confirmar: repite las que dijeron "el contexto no alcanza"')
    args = parser.parse_args(argv)
    if args.rehacer_sin_contexto and not args.confirmar:
        parser.error("--rehacer-sin-contexto solo sirve junto con --confirmar")
    return args


def main(argv=None):
    args = leer_argumentos(argv)
    consultas = consultas_de_la_muestra()

    if args.confirmar:
        registros = leer_registros(ARCHIVO_MUESTRA)
        faltantes = pendientes(registros, consultas)
        if args.rehacer_sin_contexto:
            faltantes += [c for c in sin_contexto(registros, consultas) if c not in faltantes]
        if faltantes:
            from dotenv import load_dotenv
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            from indexar_corpus import CARPETA_VECTORSTORE, MODELO_EMBEDDINGS, conectar
            from registro_rag import registrar_pregunta

            load_dotenv()
            if not os.environ.get("GEMINI_API_KEY"):
                raise SystemExit('Falta GEMINI_API_KEY. Defínela en tu archivo ".env" y vuelve a ejecutar.')
            embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS)
            vectorstore = conectar(embeddings, CARPETA_VECTORSTORE, args.coleccion)
            if vectorstore._collection.count() == 0:
                raise SystemExit(f"La colección '{args.coleccion}' está vacía.")
            print(f"Gasto: {len(faltantes)} embeddings + {len(faltantes)} llamadas de generación.")
            for numero, consulta in enumerate(faltantes, start=1):
                registro = registrar_pregunta(consulta, vectorstore, embeddings, [K_MUESTRA],
                                              args.coleccion, tipo="muestra_b30")[0]
                guardar_registro(registro, ARCHIVO_MUESTRA)
                estado = f"ERROR: {registro['error']}" if registro.get("error") else registro.get("finish_reason")
                print(f"{numero}/{len(faltantes)} {estado} | {consulta[:50]}")
        else:
            print("La muestra ya está completa: no se llama a la API.")

    registros = leer_registros(ARCHIVO_MUESTRA)
    print(formatear_resumen(*resumir_muestra(registros, consultas)))
    print(f'respuestas que dicen "el contexto no alcanza": {len(sin_contexto(registros, consultas))}')


if __name__ == "__main__":
    main(sys.argv[1:])
