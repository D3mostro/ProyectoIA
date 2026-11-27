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
        # además vuelve a correr las que dijeron que el contexto no alcanza (después de ampliar
        # el corpus). Las líneas viejas quedan en el .jsonl; el resumen usa la última.
    python src/muestra_formato.py --confirmar --rehacer-todas
        # vuelve a correr las 20 (después de cambiar el corpus, para ver si algo empeoró).
        # Gasto: 20 embeddings + 20 llamadas. El resumen usa la última respuesta de cada una.
    python src/muestra_formato.py --confirmar --solo-sin-contexto --k 5
        # experimento: repite con otro k solo las que hoy no tienen contexto. Con k distinto
        # de 3 escribe en muestra_formato_b30_k5.jsonl y no toca la muestra oficial (k=3).
    python src/muestra_formato.py --confirmar --familia
        # 27-11: 6 preguntas del perfil turista/familia (mascotas, accesibilidad, reservas,
        # alojamiento, qué llevar). Van a muestra_familia.jsonl y no tocan la muestra oficial.
        # Miden cobertura del corpus para ese perfil; el veredicto de formato del B30 sigue
        # siendo el de las 20. Gasto: 6 embeddings + 6 llamadas.

Gasto con el archivo vacío: 20 embeddings + 20 llamadas de generación. Si se corta (por
ejemplo, por un 429), al volver a ejecutarlo salta las consultas que ya tienen respuesta.
"""
import argparse
import os
import re
import sys

from casos_evaluacion import CASOS_EVALUACION, PREGUNTAS_FAMILIA
from indexar_corpus import RAIZ
from registro_rag import guardar_registro, leer_registros

ARCHIVO_MUESTRA = RAIZ / "muestra_formato_b30.jsonl"
COLECCION_MUESTRA = "astroturismo_real"
K_MUESTRA = 3
TAMANO_MUESTRA = 20
UMBRAL_FALLOS = 3  # 15 % de 20, el umbral de la tabla del Paso 1
# 27-11: el perfil elegido es turista/familia. Las 20 de casos_evaluacion son casi todas de
# astronomía general (solo 1 de "dónde"), así que no miden ese perfil: PREGUNTAS_FAMILIA
# (en casos_evaluacion.py) sí. Van a un archivo aparte.
ARCHIVO_FAMILIA = RAIZ / "muestra_familia.jsonl"

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


# El prompt pide decir "el contexto no alcanza", pero el modelo lo parafrasea ("El contexto
# recuperado no contiene información...", "no es posible responder..."). Frases concretas,
# no comodines: "el texto no menciona X, pero [1] dice Y" sigue contando como sin contexto,
# y por eso conviene mirar a mano las que marque.
PATRON_NO_ALCANZA = re.compile(
    r"contexto no alcanza"
    r"|(contexto|informaci[oó]n|fragmentos?|textos?)( \w+){0,3} no (contiene|menciona|incluye|entrega"
    r"|aporta|tiene|permite|especifica|indica|dice|alcanza)"
    r"|no (es posible|puedo|se puede) (responder|determinar|precisar)"
    r"|no (hay|tengo) (suficiente )?informaci[oó]n",
    re.IGNORECASE,
)


def sin_contexto(registros, consultas):
    """Consultas cuya última respuesta dice que el contexto no alcanza."""
    por_consulta = ultima_por_consulta(registros)
    return [c for c in consultas
            if c in por_consulta and PATRON_NO_ALCANZA.search(por_consulta[c].get("respuesta") or "")]


def archivo_para_k(k, base=ARCHIVO_MUESTRA):
    """La muestra oficial (k=3) va a muestra_formato_b30.jsonl; otro k, a un archivo aparte."""
    return base if k == K_MUESTRA else base.with_name(f"{base.stem}_k{k}.jsonl")


def fuentes_citadas(registro):
    """Archivos que la respuesta citó (sin repetir), para ver qué recuperó."""
    vistos = []
    for ruta in (registro.get("citas_validas") or {}).values():
        nombre = str(ruta).replace("\\", "/").rsplit("/", 1)[-1]
        if nombre not in vistos:
            vistos.append(nombre)
    return vistos


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
        lineas.append(f"{'OK   ' if ok else 'FALLA'} {consulta}" + (f"  -> {', '.join(malos)}" if malos else ""))
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
                        help="con --confirmar: repite las que dijeron que el contexto no alcanza")
    parser.add_argument("--rehacer-todas", action="store_true",
                        help="con --confirmar: vuelve a correr todas las consultas de la muestra")
    parser.add_argument("--solo-sin-contexto", action="store_true",
                        help="limita la muestra a las que hoy no tienen contexto en la muestra oficial")
    parser.add_argument("--k", type=int, default=K_MUESTRA,
                        help=f"chunks recuperados por consulta (oficial: {K_MUESTRA}; otro valor va a un archivo aparte)")
    parser.add_argument("--familia", action="store_true",
                        help=f"usa las {len(PREGUNTAS_FAMILIA)} preguntas del perfil turista/familia ({ARCHIVO_FAMILIA.name})")
    args = parser.parse_args(argv)
    if args.familia and args.solo_sin_contexto:
        parser.error("--solo-sin-contexto se refiere a la muestra oficial; no va con --familia")
    if args.rehacer_sin_contexto and not args.confirmar:
        parser.error("--rehacer-sin-contexto solo sirve junto con --confirmar")
    if args.rehacer_todas and not args.confirmar:
        parser.error("--rehacer-todas solo sirve junto con --confirmar")
    if args.k < 1:
        parser.error("--k debe ser al menos 1")
    return args


def main(argv=None):
    args = leer_argumentos(argv)
    consultas = list(PREGUNTAS_FAMILIA) if args.familia else consultas_de_la_muestra()
    if args.solo_sin_contexto:
        consultas = sin_contexto(leer_registros(ARCHIVO_MUESTRA), consultas)
        print(f"Solo las {len(consultas)} sin contexto en la muestra oficial (k={K_MUESTRA}).")
    archivo = archivo_para_k(args.k, ARCHIVO_FAMILIA if args.familia else ARCHIVO_MUESTRA)
    tipo = "muestra_familia" if args.familia else "muestra_b30"

    if args.confirmar:
        registros = leer_registros(archivo)
        faltantes = list(consultas) if args.rehacer_todas else pendientes(registros, consultas)
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
                registro = registrar_pregunta(consulta, vectorstore, embeddings, [args.k],
                                              args.coleccion, tipo=tipo)[0]
                guardar_registro(registro, archivo)
                estado = f"ERROR: {registro['error']}" if registro.get("error") else registro.get("finish_reason")
                print(f"{numero}/{len(faltantes)} {estado} | {consulta}")
                if not registro.get("error"):
                    print(f"      citó: {', '.join(fuentes_citadas(registro)) or 'nada'}")
        else:
            print("La muestra ya está completa: no se llama a la API.")

    registros = leer_registros(archivo)
    print(f"Archivo: {archivo.name} (k={args.k})")
    print(formatear_resumen(*resumir_muestra(registros, consultas)))
    if args.solo_sin_contexto or args.k != K_MUESTRA or args.familia:
        print(f"(Experimento: el veredicto de formato solo vale para la muestra oficial de "
              f"{TAMANO_MUESTRA} con k={K_MUESTRA}.)")
    faltas = sin_contexto(registros, consultas)
    print(f"respuestas que dicen que el contexto no alcanza: {len(faltas)} de {len(consultas)}")
    for consulta in faltas:
        print(f"  - {consulta}")


if __name__ == "__main__":
    main(sys.argv[1:])
