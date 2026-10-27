import time
import random
from typing import Callable, Any

from consumir_gemini import armar_peticion, enviar_peticion, extraer_texto, calcular_costo


def con_reintento(
    funcion: Callable[[], Any],
    intentos: int = 3,
    espera_base: float = 2,
    espera_maxima: float = 30,
) -> Any:
    for intento in range(1, intentos + 1):
        try:
            return funcion()
        except TimeoutError:
            if intento == intentos:
                raise
            # Backoff exponencial (2, 4, 8... s) con techo, más un jitter de hasta 10 %
            espera = min(espera_base * (2 ** (intento - 1)), espera_maxima)
            espera_con_jitter = espera + random.uniform(0, espera * 0.1)
            print(f"Intento {intento} de {intentos} falló. Reintentando en {espera_con_jitter:.1f}s...")
            time.sleep(espera_con_jitter)


# Precios por token de gemini-3.5-flash-lite (entrada y salida por separado, en vez de un precio plano)
PRECIO_ENTRADA = 0.30 / 1_000_000
PRECIO_SALIDA = 2.50 / 1_000_000
costo_total_sesion = 0.0


def registrar_costo(respuesta: dict) -> float:
    global costo_total_sesion
    uso = respuesta.get("usageMetadata", {})
    total_tokens = uso.get("totalTokenCount", 0)
    costo_estimado = calcular_costo(respuesta, PRECIO_ENTRADA, PRECIO_SALIDA)
    costo_total_sesion += costo_estimado
    print(f"Operación: {total_tokens} tokens, costo estimado: ${costo_estimado:.5f} (acumulado: ${costo_total_sesion:.5f})")
    return costo_estimado


def consultar_modelo(
    pregunta: str,
    system_instruction: str = "",
) -> str:
    # armar_peticion devuelve (url, body); "" equivale a no mandar system_instruction
    url, body = armar_peticion(pregunta, system_instruction=system_instruction)
    try:
        respuesta = con_reintento(lambda: enviar_peticion(url, body))
    except TimeoutError:
        raise RuntimeError("El servicio no respondió tras varios intentos. Intenta más tarde.")
    registrar_costo(respuesta)  # solo después de una respuesta exitosa
    return extraer_texto(respuesta)


if __name__ == "__main__":  # solo corre al ejecutar este archivo, no al importarlo
    # Ensayo del Bloque 13: muchas consultas seguidas para provocar un rate limit (429) a propósito.
    # Si no aparece ningún reintento, subir el número de peticiones.
    for i in range(10):
        try:
            respuesta = consultar_modelo(f"Pregunta de prueba número {i}")
            print(f"Respuesta {i}: {respuesta[:40]}...")
        except RuntimeError as error:
            print(f"Respuesta {i} falló definitivamente: {error}")

    print(f"Costo total del ensayo: ${costo_total_sesion:.5f}")
