import json
from pathlib import Path

# Revisión a mano (Bloque 17) de los 5 casos con F1 más bajo y los 5 con F1 más alto
# de resultados_metricas.json (20 respuestas generadas el 2026-10-30 con gemini-3.5-flash-lite).
# Rúbrica: correccion_factual (si/no/parcialmente), completitud (si/no/parcialmente), contradiccion (si/no).
revision_manual = [
    {
        "consulta": "¿Necesito un telescopio para empezar a observar el cielo?",
        "f1": 0.267,
        "correccion_factual": "si",
        "completitud": "parcialmente",
        "contradiccion": "no",
        "nota": (
            "dice lo mismo que la referencia (no hace falta telescopio, sirven binoculares) "
            "pero omite la linterna roja y el mapa del cielo; el F1 bajo se debe a que lo dice con otras palabras"
        ),
    },
    {
        "consulta": "¿Qué es el Telescopio Extremadamente Grande (ELT)?",
        "f1": 0.28,
        "correccion_factual": "si",
        "completitud": "no",
        "contradiccion": "no",
        "nota": (
            "lo que afirma es correcto (óptico e infrarrojo, en construcción, el más grande), "
            "pero no dice dónde está (cerro Armazones, Antofagasta) ni el espejo de 39 m, que es lo útil para el proyecto"
        ),
    },
    {
        "consulta": "¿Cuál es el mejor momento del mes para observar estrellas?",
        "f1": 0.367,
        "correccion_factual": "si",
        "completitud": "si",
        "contradiccion": "no",
        "nota": (
            "respuesta correcta y completa (luna nueva, cielo más oscuro); "
            "F1 bajo por paráfrasis: 'la Luna no opaca' frente a 'la oscuridad del cielo permite ver'"
        ),
    },
    {
        "consulta": "¿Por qué el norte de Chile es bueno para observar estrellas?",
        "f1": 0.383,
        "correccion_factual": "si",
        "completitud": "parcialmente",
        "contradiccion": "no",
        "nota": (
            "cubre cielos despejados, poca humedad y poca contaminación lumínica, pero no menciona la altura; "
            "'baja humedad' equivale a 'aire seco' y la métrica no lo reconoce"
        ),
    },
    {
        "consulta": "¿Qué es la contaminación lumínica?",
        "f1": 0.412,
        "correccion_factual": "si",
        "completitud": "parcialmente",
        "contradiccion": "no",
        "nota": (
            "define bien el fenómeno ('brillo artificial del cielo') y agrega la causa, "
            "pero no dice que oculta las estrellas, que es lo que importa al astroturista"
        ),
    },
    {
        "consulta": "¿Cuándo fue el último eclipse total de Sol visible en Chile?",
        "f1": 0.732,
        "correccion_factual": "parcialmente",
        "completitud": "no",
        "contradiccion": "si",
        "nota": (
            "FALSO POSITIVO: da otra fecha y otro lugar (4 de diciembre de 2021, Antártida) en vez de "
            "14 de diciembre de 2020, La Araucanía. Ese eclipse existió y pasó por la Antártida que Chile reclama, "
            "por eso 'parcialmente', pero para un turista en Chile continental la respuesta es inútil. "
            "F1 alto porque el resto de la oración copia la pregunta"
        ),
    },
    {
        "consulta": "¿Se puede ver la Vía Láctea desde una ciudad grande?",
        "f1": 0.733,
        "correccion_factual": "si",
        "completitud": "si",
        "contradiccion": "no",
        "nota": "misma respuesta que la referencia, incluida la negación; el F1 alto es merecido",
    },
    {
        "consulta": "¿Cuándo es el máximo de la lluvia de meteoros Gemínidas?",
        "f1": 0.743,
        "correccion_factual": "si",
        "completitud": "si",
        "contradiccion": "no",
        "nota": "fecha correcta (13 al 14 de diciembre); el F1 alto es merecido",
    },
    {
        "consulta": "¿Por qué se ven pocas estrellas en Santiago?",
        "f1": 0.743,
        "correccion_factual": "si",
        "completitud": "si",
        "contradiccion": "no",
        "nota": "nombra las dos causas de la referencia (contaminación lumínica y smog); F1 alto merecido",
    },
    {
        "consulta": "¿Qué es un año luz?",
        "f1": 0.878,
        "correccion_factual": "si",
        "completitud": "si",
        "contradiccion": "no",
        "nota": "definición y cifra correctas (9,46 billones de km); agrega 'en el vacío', que es correcto",
    },
]

if __name__ == "__main__":
    archivo = Path(__file__).parent / "revision_manual.json"
    with open(archivo, "w", encoding="utf-8") as salida:
        json.dump(revision_manual, salida, indent=2, ensure_ascii=False)
    print(f"{len(revision_manual)} casos revisados guardados en {archivo.name}")
