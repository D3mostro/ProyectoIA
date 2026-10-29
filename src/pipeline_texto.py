import json
import re
from collections import Counter
from pathlib import Path
from cliente import consultar_modelo


def limpiar_texto(texto: str) -> str:
    "Normaliza el texto: minúsculas y sin puntuación."
    if not isinstance(texto, str):
        tipo = type(texto).__name__
        raise TypeError(f"limpiar_texto espera str, no {tipo}")

    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)  # \w en Python 3 incluye tildes, ñ y dígitos
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar(texto_limpio: str) -> list[str]:
    """Convierte texto limpio en una lista de tokens."""
    return texto_limpio.split()


def procesar_texto(texto: str) -> list[str]:
    "Limpia y tokeniza en un solo paso, para no olvidar ninguno de los dos."
    return tokenizar(limpiar_texto(texto))


def vectorizar(tokens: list[str]) -> Counter:
    "Cuenta cuántas veces aparece cada token (bolsa de palabras)."
    return Counter(tokens)


def calcular_coincidencia(tokens_referencia: list[str], tokens_respuesta: list[str]) -> float:
    "Proporción de palabras distintas de la referencia que también aparecen en la respuesta."
    if not tokens_referencia:
        return 0.0  # referencia vacía (o solo puntuación): evita la división por cero
    comunes = set(tokens_referencia) & set(tokens_respuesta)
    return len(comunes) / len(set(tokens_referencia))


# Referencias escritas a mano para preguntas del proyecto de astroturismo
CASOS = [
    {
        "consulta": "¿Por qué el norte de Chile es bueno para observar estrellas?",
        "referencia": "El norte de Chile tiene cielos despejados, aire seco, altura y poca contaminación lumínica.",
    },
    {
        "consulta": "¿Qué es la contaminación lumínica?",
        "referencia": "La contaminación lumínica es el exceso de luz artificial que ilumina el cielo nocturno y oculta las estrellas.",
    },
    {
        "consulta": "¿Qué es una lluvia de meteoros?",
        "referencia": "Una lluvia de meteoros ocurre cuando la Tierra cruza los restos que deja un cometa y esas partículas se queman en la atmósfera.",
    },
]

INSTRUCCION = "Responde en español, en una sola oración breve y directa."
ARCHIVO_REGISTRO = Path(__file__).parent / "registro_pipeline_texto.json"


if __name__ == "__main__":
    registro = []
    for numero, caso in enumerate(CASOS, start=1):
        respuesta = consultar_modelo(caso["consulta"], system_instruction=INSTRUCCION)

        # Antes: texto crudo separado por espacios. Después: texto pasado por el pipeline.
        pct_crudo = calcular_coincidencia(caso["referencia"].split(), respuesta.split())
        pct_proc = calcular_coincidencia(procesar_texto(caso["referencia"]), procesar_texto(respuesta))

        print(f"[Caso {numero}] {caso['consulta']}")
        print(f"[Caso {numero}] Respuesta: {respuesta.strip()}")
        print(f"[Caso {numero}] Coincidencia texto crudo: {pct_crudo:.1%}")
        print(f"[Caso {numero}] Coincidencia texto procesado: {pct_proc:.1%}\n")

        registro.append({
            "consulta": caso["consulta"],
            "referencia": caso["referencia"],
            "respuesta": respuesta.strip(),
            "coincidencia_crudo": round(pct_crudo, 3),
            "coincidencia_procesado": round(pct_proc, 3),
        })

    ARCHIVO_REGISTRO.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Registro guardado en {ARCHIVO_REGISTRO.name}")
