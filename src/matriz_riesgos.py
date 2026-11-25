"""Bloque 32 (Core), Paso 3: matriz de riesgos del asistente de astroturismo.

Formato de la lección "Cómo completar una matriz de riesgos de IA generativa" (id 20445):
riesgo, probabilidad (1 a 5), impacto (1 a 5), mitigación y punto de supervisión asociado.
Puntaje = probabilidad x impacto; cortes fijos 15 / 9 / 4 (convención del curso).

Cada fila sale de algo verificado en este repo, no de la lista de ejemplo de la lección:
- La lección pone la inyección de prompt en 4x4 (CRITICO). Aquí queda en 2x3: el corpus no
  tiene comentarios de usuarios (solo Wikipedia y 8 sitios oficiales de observatorios), el flujo
  RAG no tiene herramientas que ejecuten acciones y el prompt de sistema ya es público en GitHub,
  así que filtrarlo no cuesta nada. Lo que sí puede pasar es una respuesta engañosa al turista.
- "Dataset de tuning desbalanceado" (B29) no aplica: el dataset tiene 5 intenciones x 3 ejemplos
  y la decisión del B30 fue no hacer PEFT. Queda como riesgo de enrutador (fila 9).
- Las filas ALTO o CRITICO deben tener un punto de supervisión humana que exista en el código o
  en un proceso con evidencia. `--validar` falla si una de esas filas dice "PENDIENTE".
- Probabilidad e impacto son riesgo residual: se califican con los controles que existen hoy
  en el código, no con un escenario futuro. Si una fila cambia al cambiar el sistema, lo dice
  en "nota" (por ejemplo, la reserva simulada).
- Ideas del repo del ayudante: cada fila tiene id, justificación, evidencia (los tests que la
  prueban) y estado (qué falta), y `--exportar` escribe docs/matriz_riesgos.md para leerla sin
  correr código. `--validar` exige esos campos y que los tests citados existan.

Uso:
    python src/matriz_riesgos.py           # tabla ordenada + filas al límite
    python src/matriz_riesgos.py --validar # además, revisa las reglas de la entrega (sale 1 si falla)
    python src/matriz_riesgos.py --exportar # escribe docs/matriz_riesgos.md
No usa API ni dependencias fuera de la librería estándar.
"""
import argparse
import re
import sys
from pathlib import Path

CARPETA_SRC = Path(__file__).resolve().parent
ARCHIVO_DOC = CARPETA_SRC.parent / "docs" / "matriz_riesgos.md"

CORTES = ((15, "CRITICO"), (9, "ALTO"), (4, "MEDIO"))
CATEGORIAS_SEVERAS = ("CRITICO", "ALTO")


def puntaje_riesgo(probabilidad, impacto):
    for nombre, valor in (("probabilidad", probabilidad), ("impacto", impacto)):
        # bool es subclase de int: True pasaría como 1 sin este chequeo.
        if isinstance(valor, bool) or not isinstance(valor, int) or not 1 <= valor <= 5:
            raise ValueError(f"{nombre} debe ser un entero de 1 a 5, llegó {valor!r}")
    return probabilidad * impacto


def categoria_riesgo(puntaje):
    for minimo, nombre in CORTES:
        if puntaje >= minimo:
            return nombre
    return "BAJO"


def esta_en_el_limite(probabilidad, impacto):
    """True si mover probabilidad o impacto un punto (dentro de 1..5) cambia la categoría."""
    categoria_actual = categoria_riesgo(puntaje_riesgo(probabilidad, impacto))
    vecinos = [
        (probabilidad + 1, impacto), (probabilidad - 1, impacto),
        (probabilidad, impacto + 1), (probabilidad, impacto - 1),
    ]
    for p2, i2 in vecinos:
        if not (1 <= p2 <= 5 and 1 <= i2 <= 5):
            continue
        if categoria_riesgo(puntaje_riesgo(p2, i2)) != categoria_actual:
            return True
    return False


mis_riesgos = [
    {"id": "R1",
     "riesgo": "Respuesta con horarios o cupos de visita vencidos porque los .txt de "
               "observatorios no se vuelven a descargar",
     "probabilidad": 4,  # ESO abre reservas por trimestre y ALMA publica visitas mes a mes
     "impacto": 3,       # el turista viaja de noche a un observatorio cerrado o sin cupo
     "justificacion": "Los 8 .txt de observatorios son del 2026-10-10 y los sitios cambian "
                      "horarios por temporada; el asistente responde con lo indexado.",
     "mitigacion": "Respuesta obligada a citar la fuente [n] (Bloque 26) y reindexado "
                   "incremental que salta chunks sin cambios (Bloque 25)",
     "supervision": "indexar_corpus.py y preflight.py se detienen si un .txt de observatorio "
                    "tiene más de 30 días hasta que una persona revisa los sitios y lo registra "
                    "con revision_corpus.py --confirmar (revision_observatorios.json, Bloque 32)",
     "evidencia": "test_vencidos_y_vigentes; test_confirmacion_humana_libera_y_caduca; "
                  "test_preflight_b32_frena_observatorios_vencidos_y_pii",
     "estado": "Control integrado. Falta la primera revisión real de los 8 sitios.",
     "nota": "El control frena el reindexado, no las respuestas: si nunca se reindexa, el "
             "índice viejo sigue respondiendo. Por eso la probabilidad queda en 4"},
    {"id": "R2",
     "riesgo": "Texto libre del usuario (pregunta y respuesta) en trazas_agente.jsonl y "
               "registros_latencia_costo.jsonl, que están versionados y se suben a GitHub",
     "probabilidad": 3,  # *.log está en .gitignore, pero los .jsonl no
     "impacto": 3,       # un turista que escribe su nombre o correo queda publicado
     "justificacion": "Esos .jsonl guardan la pregunta y la respuesta completas y son evidencia "
                      "de las entregas B23, B28 y B30, así que se suben al repo.",
     "mitigacion": "logging_rag.py registra solo largo y huella, nunca el texto (Bloque 31); "
                   "preflight.py falla si un .jsonl versionado tiene correo, RUT o celular "
                   "(revision_corpus.py, Bloque 32)",
     "supervision": "German corre preflight.py antes de cada push (él hace siempre el push a "
                    "mano) y limpia la línea que marque",
     "evidencia": "test_detecta_pii; test_no_marca_datos_normales; test_jsonl_reales_del_repo_sin_pii; "
                  "test_logs_no_guardan_texto_de_pregunta_ni_respuesta",
     "estado": "Chequeo integrado; los 3 .jsonl actuales pasan. No detecta nombres propios.",
     "nota": "Los .jsonl no se sacan del repo: son evidencia de las entregas B23, B28 y B30. "
             "El chequeo no detecta nombres propios: por eso sigue en probabilidad 3"},
    {"id": "R3",
     "riesgo": "Se activa una versión del prompt sin revisar su diff, o con un "
               "registro_prompts.json editado a mano",
     "probabilidad": 2,  # todo flujo RAG lee el prompt activo; activar exige el grafo
     "impacto": 3,       # una versión sin capas anti-inyección vuelve a los tres flujos a la vez
     "justificacion": "Desde el Bloque 32 los tres flujos (responder, registro_rag y "
                      "procesar_consulta) usan la misma versión aprobada, así que un cambio "
                      "malo afecta a todos.",
     "mitigacion": "Registro con huella, diff y verificación de huellas al cargar "
                   "(registro_prompts.py, Bloque 32); prompt_huella en logs_rag.jsonl y en "
                   "registros_latencia_costo.jsonl",
     "supervision": "interrupt() con el diff; resolver exige la huella revisada y no activa si "
                    "la versión activa cambió durante la revisión (grafo_supervision.py, Bloque 32)",
     "evidencia": "test_resume_que_no_es_true_no_activa; test_resolver_rechaza_una_huella_distinta_a_la_revisada; "
                  "test_no_activa_si_la_version_activa_cambio_durante_la_revision; "
                  "test_registro_editado_a_mano_no_se_carga; test_b32_usa_el_prompt_activo_con_system_instruction",
     "estado": "Integrado. El checkpointer es InMemorySaver: una solicitud pendiente se pierde al "
               "cerrar el proceso.",
     "nota": "Antes de unificar los flujos este riesgo era 'dos prompts activos a la vez' (4x3, "
             "ALTO); se cerró adoptando la idea del repo del ayudante"},
    {"id": "R4",
     "riesgo": "Reserva de cupo en observatorio ejecutada sin una aprobación humana real",
     "probabilidad": 2,  # dos corridas reales y los tests: timeout y rechazo no reservan
     "impacto": 2,       # hoy es simulada: solo devuelve un texto, no reserva nada
     "justificacion": "Es la única acción del sistema con efecto fuera de él; el agente podría "
                      "invocarla sin que el turista lo pida.",
     "mitigacion": "Validación Pydantic antes de pedir aprobación, esperar_aprobacion con "
                   "timeout = no aprobada y registrar_aprobacion (Bloques 22 y 23)",
     "supervision": "esperar_aprobacion en reservar_cupo_observatorio (Bloque 23)",
     "evidencia": "test_reserva_pide_aprobacion_y_la_registra; test_aprobacion_por_timeout_devuelve_false; "
                  "test_reserva_invalida_no_pide_aprobacion",
     "estado": "Integrado y probado en 2 corridas reales (2026-11-09). Aprobación por consola.",
     "nota": "Es la acción de mayor riesgo del sistema. Conectada a un sistema real de "
             "reservas sube a impacto 5 (2x5 = 10, ALTO); la aprobación humana ya está"},
    {"id": "R5",
     "riesgo": "Inyección de prompt vía texto de Wikipedia descargado al corpus con "
               "descargar_wikipedia.py",
     "probabilidad": 2,  # 0 falsos positivos y 0 detecciones en los 49 .txt del corpus real
     "impacto": 3,       # respuesta engañosa; el flujo RAG no ejecuta acciones
     "justificacion": "Wikipedia es editable por cualquiera y los artículos entran al corpus "
                      "sin revisión línea por línea.",
     "mitigacion": "Detector sobre texto normalizado + etiquetas <contexto> + "
                   "systemInstruction aparte (Bloque 31), hoy en los tres flujos por el prompt "
                   "activo (Bloque 32); el filtro de chunks solo corre en procesar_consulta",
     "supervision": "Revisar los eventos documento_descartado_por_inyeccion de "
                    "logs_rag.jsonl después de cada descarga nueva",
     "evidencia": "test_detector_no_se_esquiva_con_tildes_ni_invisibles_ni_etiquetas; "
                  "test_escanear_coleccion_del_corpus_del_repo_sin_falsos_positivos",
     "estado": "Filtro de chunks pendiente en responder() y registro_rag."},
    {"id": "R6",
     "riesgo": "Cuota de Gemini agotada (error 429) porque las dos carpetas del proyecto "
               "usan la misma clave",
     "probabilidad": 4,
     "impacto": 2,       # se pierde una corrida, no hay daño a terceros
     "justificacion": "Pasó en la indexación del B25 (2374 chunks) y las dos carpetas comparten "
                      "la cuota gratuita.",
     "mitigacion": "Reintentos con espera tope de 120 s y salto de chunks ya indexados "
                   "(Bloque 25); tokens y costo reales por consulta (Bloque 28)",
     "supervision": "",
     "evidencia": "test_con_reintentos_reintenta_429_con_backoff_exponencial; "
                  "test_indexar_salta_chunks_ya_guardados_y_reembebe_los_cambiados",
     "estado": "Integrado."},
    {"id": "R7",
     "riesgo": "Retiro de gemini-3.5-flash-lite o de gemini-embedding-001 sin aviso, que "
               "obliga a cambiar el modelo y reindexar los 2488 chunks",
     "probabilidad": 2,
     "impacto": 4,       # sin embeddings compatibles la recuperación deja de funcionar
     "justificacion": "El curso ya vio deprecarse text-embedding-004 en el Bloque 28.",
     "mitigacion": "preflight.py revisa que la colección exista y tenga chunks (Bloque 28); "
                   "no detecta un modelo retirado (falta)",
     "supervision": "",
     "evidencia": "test_preflight_ok_con_indice_real",
     "estado": "Sin control del modelo: revisar la página de modelos de Gemini a mano."},
    {"id": "R8",
     "riesgo": "El asistente responde 'el contexto no alcanza' en preguntas comunes de "
               "turistas (5 de 20 en la muestra del Bloque 30)",
     "probabilidad": 4,
     "impacto": 2,       # no inventa: el turista se queda sin respuesta
     "justificacion": "Medido: 5 de 20 en la muestra del B30 después de ampliar el corpus "
                      "(2 temas faltan, 1 por k=3, 2 el modelo no infiere).",
     "mitigacion": "Corpus ampliado con descargar_wikipedia.py (Bloque 30) y evaluación de "
                   "k=3 contra k=5 (Bloque 27)",
     "supervision": "",
     "evidencia": "muestra_formato_b30.jsonl; test_trade_off_de_k_y_f1_maximo_en_k3",
     "estado": "Brecha conocida; sin corrección todavía."},
    {"id": "R9",
     "riesgo": "El enrutador elige mal la intención con mensajes distintos a las 4 consultas "
               "ya probadas",
     "probabilidad": 3,  # 16/16 aciertos, pero son 4 consultas repetidas 4 veces
     "impacto": 2,       # se usa una herramienta de lectura equivocada, sin acción real
     "justificacion": "La muestra demuestra consistencia, no cobertura de las 5 intenciones.",
     "mitigacion": "dataset_tuning.py con 15 ejemplos validados como conjunto de regresión "
                   "(Bloque 29) y decisión de no personalizar (Bloque 30)",
     "supervision": "",
     "evidencia": "test_dataset_propio_valido; test_dataset_roto_detecta_cada_error",
     "estado": "Falta correr el enrutador sobre los 15 ejemplos."},
]

CAMPOS_OBLIGATORIOS = ("id", "riesgo", "justificacion", "mitigacion", "evidencia", "estado")


def calcular(riesgos):
    for r in riesgos:
        r["puntaje"] = puntaje_riesgo(r["probabilidad"], r["impacto"])
        r["categoria"] = categoria_riesgo(r["puntaje"])
        r["al_limite"] = esta_en_el_limite(r["probabilidad"], r["impacto"])
    return sorted(riesgos, key=lambda x: -x["puntaje"])


def problemas_de_entrega(riesgos):
    """Reglas de la actividad core. Devuelve una lista de textos; vacía = cumple."""
    problemas = []
    if len(riesgos) < 5:
        problemas.append(f"Hay {len(riesgos)} riesgos; la entrega pide al menos 5.")
    if not any(r["categoria"] in CATEGORIAS_SEVERAS for r in riesgos):
        problemas.append("Ningún riesgo es ALTO o CRITICO: revisa si estás subestimando.")
    for r in riesgos:
        supervision = r.get("supervision", "").strip()
        if r["categoria"] in CATEGORIAS_SEVERAS and not supervision:
            problemas.append(f"{r['categoria']} sin punto de supervisión: {r['riesgo']}")
        elif r["categoria"] in CATEGORIAS_SEVERAS and "PENDIENTE" in supervision.upper():
            problemas.append(f"{r['categoria']} con supervisión pendiente de construir: {r['riesgo']}")
    for r in riesgos:
        faltan = [c for c in CAMPOS_OBLIGATORIOS if not str(r.get(c, "")).strip()]
        if faltan:
            problemas.append(f"{r.get('id', '?')} sin {', '.join(faltan)}")
    citados = tests_citados(riesgos)
    existentes = tests_existentes()
    for nombre in sorted(citados - existentes):
        problemas.append(f"La evidencia cita {nombre}, que no existe en src/test_*.py")
    con_bloque = sum("Bloque" in r.get("mitigacion", "") for r in riesgos)
    if con_bloque < 2:
        problemas.append("Menos de 2 mitigaciones apuntan a algo construido (con su bloque).")
    if len({r["categoria"] for r in riesgos}) == 1:
        problemas.append("Todos los riesgos tienen la misma categoría.")
    return problemas


def tests_citados(riesgos):
    return {n for r in riesgos for n in re.findall(r"\btest_\w+", r.get("evidencia", ""))}


def tests_existentes(carpeta=CARPETA_SRC):
    nombres = set()
    for archivo in Path(carpeta).glob("test_*.py"):
        nombres |= set(re.findall(r"^def (test_\w+)", archivo.read_text(encoding="utf-8"), re.M))
    return nombres


def exportar_markdown(riesgos, archivo=ARCHIVO_DOC):
    lineas = [
        "# Matriz de riesgos del asistente de astroturismo", "",
        "Generado por `python src/matriz_riesgos.py --exportar`; no editar a mano.", "",
        "Probabilidad (P) e impacto (I) de 1 a 5, como **riesgo residual**: con los controles que "
        "existen hoy en el código. Puntaje = P x I. BAJO 1-3, MEDIO 4-8, ALTO 9-14, CRITICO 15-25 "
        "(cortes del curso). AL LIMITE: mover P o I un punto cambia la categoría; esas filas "
        "conviene revisarlas con otra persona.", "",
        "| ID | Riesgo | P | I | Puntaje | Categoría | Al límite |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in riesgos:
        lineas.append(f"| {r['id']} | {r['riesgo']} | {r['probabilidad']} | {r['impacto']} | "
                      f"{r['puntaje']} | {r['categoria']} | {'Sí' if r['al_limite'] else 'No'} |")
    for r in riesgos:
        lineas += ["", f"## {r['id']} ({r['categoria']}, {r['puntaje']}). {r['riesgo']}", ""]
        for campo, titulo in (("justificacion", "Justificación"), ("mitigacion", "Mitigación"),
                              ("supervision", "Punto de supervisión"), ("evidencia", "Evidencia"),
                              ("estado", "Estado"), ("nota", "Nota")):
            valor = r.get(campo) or ("no corresponde (MEDIO)" if campo == "supervision" else "")
            if valor:
                lineas.append(f"- **{titulo}:** {valor}")
    archivo = Path(archivo)
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return archivo


def pendientes(riesgos):
    return [r for r in riesgos if "PENDIENTE" in r.get("supervision", "")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--validar", action="store_true", help="revisa las reglas de la entrega")
    parser.add_argument("--exportar", action="store_true", help="escribe docs/matriz_riesgos.md")
    args = parser.parse_args(argv)

    ordenados = calcular(mis_riesgos)
    for r in ordenados:
        marca = " [AL LIMITE]" if r["al_limite"] else ""
        print(f"{r['id']} {r['categoria']:8} ({r['puntaje']:2}) - {r['riesgo']}{marca}")

    print()
    for r in ordenados:
        if r["al_limite"]:
            print("revisar con otra persona:", r["riesgo"][:70])

    print()
    for r in ordenados:
        if r["categoria"] in CATEGORIAS_SEVERAS:
            print(f"supervisión ({r['categoria']}): {r['supervision']}")

    if args.exportar:
        print("\nEscrito:", exportar_markdown(ordenados))
    if not args.validar:
        return 0
    print()
    problemas = problemas_de_entrega(ordenados)
    for p in problemas:
        print("FALLA:", p)
    for r in pendientes(ordenados):
        print("AVISO: supervisión pendiente de construir:", r["riesgo"][:70])
    print("Entrega OK" if not problemas else "Entrega con fallas")
    return 1 if problemas else 0


if __name__ == "__main__":
    sys.exit(main())
