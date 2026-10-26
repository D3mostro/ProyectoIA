"""Bloque 12 - Cliente modular de consumo del modelo, con reintentos.

consultar_modelo es el único punto de entrada: arma la petición, la envía con
reintento y devuelve el texto. Quien lo usa solo maneja dos errores:
  - ValueError: credencial inválida o sin permiso (no se reintenta).
  - RuntimeError: el servicio no respondió tras varios intentos.

Ejecutar desde la carpeta src:  python cliente.py
"""
import time
from typing import Callable, Any

from consumir_gemini import armar_peticion, enviar_peticion, extraer_texto


def con_reintento(
    funcion: Callable[[], Any],
    intentos: int = 3,
    espera_segundos: int = 2,
) -> Any:
    """Llama a funcion; si lanza TimeoutError (429, 5xx, timeout o conexión), espera y reintenta.

    Cualquier otra excepción (por ejemplo ValueError por credencial) se propaga sin reintentar.
    En el último intento vuelve a lanzar el TimeoutError original.
    """
    for intento in range(1, intentos + 1):
        try:
            return funcion()
        except TimeoutError:
            if intento == intentos:
                raise
            print(f"Intento {intento} de {intentos} falló por límite de uso. Reintentando...")
            time.sleep(espera_segundos)


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
    return extraer_texto(respuesta)


if __name__ == "__main__":  # solo corre al ejecutar este archivo, no al importarlo
    try:
        respuesta = consultar_modelo("¿Cuál es la capital de Chile?")
        print(respuesta)
    except ValueError as error:
        print(f"Error de configuración: {error}")
    except RuntimeError as error:
        print(f"El servicio no respondió tras varios intentos: {error}")
