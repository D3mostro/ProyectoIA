"""Bloque 30 (actividad Core): método de decisión de personalización aplicado al proyecto.

Corre el método de cinco preguntas de la lección 20439 sobre dos casos reales del asistente de
astroturismo, muestra el análisis de sensibilidad (una respuesta cambiada a la vez) y genera el
esqueleto de docs/decision_personalizacion.md. El documento final se completa a mano.

Uso (desde la raíz del repo):
    python src/decision.py              # imprime puntajes y sensibilidad
    python src/decision.py --esqueleto  # además escribe docs/decision_personalizacion_esqueleto.md

Solo Python estándar: no llama a ninguna API ni gasta cuota.

Diferencias con el código de la página:
- La página solo revisa que no quede ningún None. Un "si" o un "no" (string) pasa ese control y
  el "if" lo toma como verdadero sin avisar (la propia página lo advierte en "Si algo no sale
  como esperabas"). Aquí cada respuesta tiene que ser bool, y falta o sobra una clave da error.
- La página escribe decision_personalizacion.md con open(..., "w") cada vez que se ejecuta:
  correr el script de nuevo (la página lo pide en "Cómo se prueba") borra el documento que ya
  completaste a mano. Aquí el esqueleto va a un archivo aparte y nunca se sobrescribe.
- El análisis de sensibilidad es una función reutilizable, no un bucle suelto.
- Cada respuesta va acompañada de la evidencia que la sustenta, como pide la tabla del Paso 1.
"""
import os
import sys

PREGUNTAS = (
    "informacion_cambia_seguido",
    "problema_de_formato_o_tono",
    "dataset_grande_y_de_calidad",
    "presupuesto_limitado_o_iterar_rapido",
    "necesita_multiples_variantes",
)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_ESQUELETO = os.path.join(RAIZ, "docs", "decision_personalizacion_esqueleto.md")


def validar_respuestas(respuestas):
    """Exige exactamente las cinco claves y que cada valor sea True o False."""
    faltan = [p for p in PREGUNTAS if p not in respuestas]
    sobran = [c for c in respuestas if c not in PREGUNTAS]
    if faltan or sobran:
        raise KeyError(f"claves faltantes: {faltan}, claves desconocidas: {sobran}")
    no_bool = {c: v for c, v in respuestas.items() if not isinstance(v, bool)}
    if no_bool:
        raise TypeError(f"cada respuesta debe ser True o False, no: {no_bool}")


def recomendar_estrategia(respuestas):
    """Versión corregida de la lección 20439: devuelve todas las técnicas empatadas."""
    validar_respuestas(respuestas)
    puntos = {"RAG": 0, "Fine-tuning completo": 0, "PEFT": 0}

    if respuestas["informacion_cambia_seguido"]:
        puntos["RAG"] += 3
    if respuestas["problema_de_formato_o_tono"]:
        puntos["PEFT"] += 2
        puntos["Fine-tuning completo"] += 1
    if respuestas["dataset_grande_y_de_calidad"]:
        puntos["Fine-tuning completo"] += 2
        puntos["PEFT"] += 1
    if respuestas["presupuesto_limitado_o_iterar_rapido"]:
        puntos["RAG"] += 2
        puntos["PEFT"] += 1
    if respuestas["necesita_multiples_variantes"]:
        puntos["PEFT"] += 2

    maximo = max(puntos.values())
    ganadoras = [tecnica for tecnica, p in puntos.items() if p == maximo]
    return puntos, ganadoras


def analizar_sensibilidad(respuestas):
    """Invierte una respuesta a la vez. Devuelve [(clave, nuevo_valor, ganadoras, cambia)]."""
    _, ganadoras_base = recomendar_estrategia(respuestas)
    filas = []
    for clave in PREGUNTAS:
        variante = dict(respuestas)
        variante[clave] = not variante[clave]
        _, ganadoras_v = recomendar_estrategia(variante)
        filas.append((clave, variante[clave], ganadoras_v, set(ganadoras_v) != set(ganadoras_base)))
    return filas


def estimar_costo_de_etiquetar(num_ejemplos, minutos_por_ejemplo, costo_por_hora):
    """Lección 20438: horas y costo del etiquetado humano."""
    horas = (num_ejemplos * minutos_por_ejemplo) / 60
    return horas, horas * costo_por_hora


def generar_esqueleto_decision(problema, puntos, ganadoras):
    """Lección 20440, Paso 2: esqueleto de las cuatro secciones."""
    empate = len(ganadoras) > 1
    lineas = ["# Decision de personalizacion", "", "## 1. El problema", problema, "",
              "## 2. Resultado del metodo de decision"]
    for tecnica, p in sorted(puntos.items(), key=lambda x: -x[1]):
        lineas.append(f"- {tecnica}: {p} puntos")
    if empate:
        lineas += ["", f"**Empate entre: {', '.join(ganadoras)}**"]
    lineas += ["", "## 3. Tecnica elegida y por que"]
    if empate:
        lineas.append("[COMPLETAR: el puntaje no resolvio el empate solo -- "
                      "justifica aqui con criterio propio]")
    else:
        lineas.append(f"[COMPLETAR: por que {ganadoras[0]} responde mejor a este caso]")
    lineas += ["", "## 4. Que necesitarias para ejecutarla",
               "[COMPLETAR: corpus faltante, o cantidad de ejemplos etiquetados que faltan]"]
    return "\n".join(lineas)


def guardar_esqueleto(texto, ruta=RUTA_ESQUELETO):
    """Escribe el esqueleto solo si el archivo no existe. Devuelve True si lo escribió."""
    if os.path.exists(ruta):
        return False
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "x", encoding="utf-8") as f:
        f.write(texto + "\n")
    return True


# Respuestas con su evidencia (revisada el 2026-11-23). Detalle en docs/decision_personalizacion.md.
CASOS = {
    "asistente": {
        "problema": ("El asistente de astroturismo tiene que responder con datos que el modelo no "
                     "trae o trae desactualizados (horarios, precios y estado de visitas de los "
                     "observatorios) sin inventarlos."),
        "respuestas": {
            # Fuentes oficiales revisadas el 23-11-2026: ESO abrió reservas de oct-dic el 02-09 (ciclo
            # trimestral); ALMA abrió visitas el 17 y 24-10 (anuncio del 02-10). Cumple el umbral
            # mensual en 2 de los últimos 3 meses (sin cambios encontrados en noviembre).
            "informacion_cambia_seguido": True,
            # Muestra de 20 (muestra_formato_b30.jsonl, 23-11-2026): 0 fallos, umbral 3.
            # Ver src/muestra_formato.py.
            "problema_de_formato_o_tono": False,
            # 15 ejemplos etiquetados en src/dataset_tuning.py (< 200).
            "dataset_grande_y_de_calidad": False,
            # 0 entrenamientos costeables: sin Google Cloud no hay tuning en la Gemini API.
            "presupuesto_limitado_o_iterar_rapido": True,
            # Un solo asistente, en español, con el mismo comportamiento para todos los usuarios.
            "necesita_multiples_variantes": False,
        },
    },
    "enrutador": {
        "problema": ("El agente tiene que elegir siempre la herramienta correcta (o ninguna) para "
                     "cada mensaje del usuario, con las mismas etiquetas."),
        "respuestas": {
            # Las 5 intenciones y las 4 herramientas son fijas desde el Bloque 21.
            "informacion_cambia_seguido": False,
            # trazas_agente.jsonl: 16 turnos (4 consultas x 4 corridas), 16 elecciones correctas.
            "problema_de_formato_o_tono": False,
            "dataset_grande_y_de_calidad": False,
            "presupuesto_limitado_o_iterar_rapido": True,
            "necesita_multiples_variantes": False,
        },
    },
}


def imprimir_caso(nombre, caso):
    puntos, ganadoras = recomendar_estrategia(caso["respuestas"])
    print(f"== caso {nombre} ==")
    print("puntos:", puntos)
    print("candidatas:", ganadoras, "(EMPATE)" if len(ganadoras) > 1 else "")
    print("sensibilidad (una respuesta invertida a la vez):")
    for clave, valor, ganadoras_v, cambia in analizar_sensibilidad(caso["respuestas"]):
        print(f"  {clave} -> {valor}: {ganadoras_v}  [{'CAMBIA' if cambia else 'igual'}]")
    print()
    return puntos, ganadoras


def main(argv):
    resultados = {nombre: imprimir_caso(nombre, caso) for nombre, caso in CASOS.items()}

    horas, costo = estimar_costo_de_etiquetar(200 - 15, 5, 25)
    print(f"etiquetar los 185 ejemplos que faltan para 200, a 5 min: {horas:.2f} h, aprox ${costo:.2f}")

    if "--esqueleto" in argv:
        puntos, ganadoras = resultados["asistente"]
        texto = generar_esqueleto_decision(CASOS["asistente"]["problema"], puntos, ganadoras)
        if guardar_esqueleto(texto):
            print(f"esqueleto escrito en {RUTA_ESQUELETO}")
        else:
            print(f"{RUTA_ESQUELETO} ya existe: no se sobrescribe")


if __name__ == "__main__":
    main(sys.argv[1:])
