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
- Las filas ALTO o CRITICO deben tener un punto de supervisión humana. Si ese punto todavía no
  existe en el código, la fila lo dice con "PENDIENTE".

Uso:
    python src/matriz_riesgos.py           # tabla ordenada + filas al límite
    python src/matriz_riesgos.py --validar # además, revisa las reglas de la entrega (sale 1 si falla)
No usa API ni dependencias fuera de la librería estándar.
"""
import argparse
import sys

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
    {"riesgo": "Respuesta con horarios o cupos de visita vencidos porque los .txt de "
               "observatorios no se vuelven a descargar",
     "probabilidad": 4,  # ESO abre reservas por trimestre y ALMA publica visitas mes a mes
     "impacto": 3,       # el turista viaja de noche a un observatorio cerrado o sin cupo
     "mitigacion": "Respuesta obligada a citar la fuente [n] (Bloque 26) y reindexado "
                   "incremental que salta chunks sin cambios (Bloque 25)",
     "supervision": "Revisión humana mensual de los 8 .txt de observatorios antes de "
                    "reindexar (PENDIENTE: no hay recordatorio ni script que la exija)"},
    {"riesgo": "Dos prompts de sistema activos a la vez y sin versión: PROMPT_TEMPLATE "
               "(recuperacion.py, lo usan responder y registro_rag) y PROMPT_SISTEMA "
               "(logging_rag.py, el único con systemInstruction y etiquetas <contexto>)",
     "probabilidad": 4,  # ya pasó: el B31 cambió el prompt y el B26/B28 siguen con el viejo
     "impacto": 3,       # las capas anti-inyección no cubren responder() ni registro_rag
     "mitigacion": "Registro de versiones con huella y diff (registro_prompts.py, Bloque 32) "
                   "y prompt_huella en el evento llm_start de logs_rag.jsonl (Bloque 32); "
                   "falta que recuperacion.py lea la versión activa del registro",
     "supervision": "interrupt() que pide aprobación antes de activar una huella nueva "
                    "(grafo_supervision.py, Bloque 32)"},
    {"riesgo": "Texto libre del usuario (pregunta y respuesta) en trazas_agente.jsonl y "
               "registros_latencia_costo.jsonl, que están versionados y se suben a GitHub",
     "probabilidad": 3,  # *.log está en .gitignore, pero los .jsonl no
     "impacto": 3,       # un turista que escribe su nombre o correo queda publicado
     "mitigacion": "logging_rag.py registra solo largo y huella, nunca el texto (Bloque 31)",
     "supervision": "German revisa git diff --cached de los .jsonl antes de cada push "
                    "(él hace siempre el push a mano)"},
    {"riesgo": "Reserva de cupo en observatorio ejecutada sin una aprobación humana real",
     "probabilidad": 2,  # dos corridas reales y los tests: timeout y rechazo no reservan
     "impacto": 4,       # hoy es simulada; conectada, compromete cupos y dinero del turista
     "mitigacion": "Validación Pydantic antes de pedir aprobación, esperar_aprobacion con "
                   "timeout = no aprobada y registrar_aprobacion (Bloques 22 y 23)",
     "supervision": "esperar_aprobacion en reservar_cupo_observatorio (Bloque 23)"},
    {"riesgo": "Inyección de prompt vía texto de Wikipedia descargado al corpus con "
               "descargar_wikipedia.py",
     "probabilidad": 2,  # 0 falsos positivos y 0 detecciones en los 49 .txt del corpus real
     "impacto": 3,       # respuesta engañosa; el flujo RAG no ejecuta acciones
     "mitigacion": "Detector sobre texto normalizado + etiquetas <contexto> + "
                   "systemInstruction aparte (Bloque 31); solo en procesar_consulta",
     "supervision": "Revisar los eventos documento_descartado_por_inyeccion de "
                    "logs_rag.jsonl después de cada descarga nueva"},
    {"riesgo": "Cuota de Gemini agotada (error 429) porque las dos carpetas del proyecto "
               "usan la misma clave",
     "probabilidad": 4,
     "impacto": 2,       # se pierde una corrida, no hay daño a terceros
     "mitigacion": "Reintentos con espera tope de 120 s y salto de chunks ya indexados "
                   "(Bloque 25); tokens y costo reales por consulta (Bloque 28)",
     "supervision": ""},
    {"riesgo": "Retiro de gemini-3.5-flash-lite o de gemini-embedding-001 sin aviso, que "
               "obliga a cambiar el modelo y reindexar los 2488 chunks",
     "probabilidad": 2,
     "impacto": 4,       # sin embeddings compatibles la recuperación deja de funcionar
     "mitigacion": "preflight.py revisa que la colección exista y tenga chunks (Bloque 28); "
                   "no detecta un modelo retirado (falta)",
     "supervision": ""},
    {"riesgo": "El asistente responde 'el contexto no alcanza' en preguntas comunes de "
               "turistas (5 de 20 en la muestra del Bloque 30)",
     "probabilidad": 4,
     "impacto": 2,       # no inventa: el turista se queda sin respuesta
     "mitigacion": "Corpus ampliado con descargar_wikipedia.py (Bloque 30) y evaluación de "
                   "k=3 contra k=5 (Bloque 27)",
     "supervision": ""},
    {"riesgo": "El enrutador elige mal la intención con mensajes distintos a las 4 consultas "
               "ya probadas",
     "probabilidad": 3,  # 16/16 aciertos, pero son 4 consultas repetidas 4 veces
     "impacto": 2,       # se usa una herramienta de lectura equivocada, sin acción real
     "mitigacion": "dataset_tuning.py con 15 ejemplos validados como conjunto de regresión "
                   "(Bloque 29) y decisión de no personalizar (Bloque 30)",
     "supervision": ""},
]


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
        if r["categoria"] in CATEGORIAS_SEVERAS and not r.get("supervision", "").strip():
            problemas.append(f"{r['categoria']} sin punto de supervisión: {r['riesgo']}")
    con_bloque = sum("Bloque" in r.get("mitigacion", "") for r in riesgos)
    if con_bloque < 2:
        problemas.append("Menos de 2 mitigaciones apuntan a algo construido (con su bloque).")
    if len({r["categoria"] for r in riesgos}) == 1:
        problemas.append("Todos los riesgos tienen la misma categoría.")
    return problemas


def pendientes(riesgos):
    return [r for r in riesgos if "PENDIENTE" in r.get("supervision", "")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--validar", action="store_true", help="revisa las reglas de la entrega")
    args = parser.parse_args(argv)

    ordenados = calcular(mis_riesgos)
    for r in ordenados:
        marca = " [AL LIMITE]" if r["al_limite"] else ""
        print(f"{r['categoria']:8} ({r['puntaje']:2}) - {r['riesgo']}{marca}")

    print()
    for r in ordenados:
        if r["al_limite"]:
            print("revisar con otra persona:", r["riesgo"][:70])

    print()
    for r in ordenados:
        if r["categoria"] in CATEGORIAS_SEVERAS:
            print(f"supervisión ({r['categoria']}): {r['supervision']}")

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
