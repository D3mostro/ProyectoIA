import time
from datetime import date

from consumir_gemini import armar_peticion, enviar_peticion, extraer_texto, calcular_costo

# Precio POR TOKEN (nivel pagado). Verificar el día que armes la tabla en
# https://ai.google.dev/gemini-api/docs/pricing  ->  precio por 1M / 1_000_000
# Confirma también el nombre exacto de cada modelo en AI Studio.
CANDIDATOS = {
    "gemini-3.1-flash-lite": (0.25 / 1_000_000, 1.50 / 1_000_000),  # el más barato (se retira el 07-05-2027)
    "gemini-3.5-flash-lite": (0.30 / 1_000_000, 2.50 / 1_000_000),  # A: económico
    "gemini-3.8-flash":      (0.75 / 1_000_000, 3.75 / 1_000_000),  # B: más capaz
}

PREGUNTA_LATENCIA = "Confirma la conexion"
REPETICIONES = 4
PAUSA = 3  # segundos entre peticiones, para no gatillar el error 429
MAX_REINTENTOS = 4  # si Google responde 429 o 503, reintenta con backoff exponencial

# Preguntas reales de la miniaplicación (astroturismo en Chile)
PREGUNTAS_CALIDAD = [
    "Voy a Vicuña la noche del 13 al 14 de diciembre de 2026. ¿Qué puedo ver?",
    "¿Puedo ver las Perseidas desde el Valle del Elqui?",
    "¿Puedo visitar ALMA este fin de semana?",
]


def enviar_con_reintentos(url, body):
    """Envía la petición; si el servicio está saturado (429/503), espera 1, 2, 4, 8 s y reintenta.

    Devuelve (datos, latencia). La latencia es solo la del intento exitoso,
    así las esperas del reintento no inflan la medición.
    """
    espera = 1
    for intento in range(MAX_REINTENTOS + 1):
        try:
            inicio = time.perf_counter()
            datos = enviar_peticion(url, body)
            fin = time.perf_counter()
            return datos, fin - inicio
        except TimeoutError as error:
            # Desde el Bloque 12, enviar_peticion lanza TimeoutError ante 429, 5xx o falla de red
            if intento == MAX_REINTENTOS:
                raise
            print(f"  ({error}) -> reintentando en {espera} s...")
            time.sleep(espera)
            espera *= 2


def medir_modelo(modelo, precio_entrada, precio_salida):
    print(f"\n========== {modelo} ==========")

    # 1) Latencia: repetir, guardar cada resultado, promediar
    latencias = []
    for i in range(REPETICIONES):
        url, body = armar_peticion(PREGUNTA_LATENCIA, modelo)
        datos, latencia = enviar_con_reintentos(url, body)
        latencias.append(latencia)
        print(f"Intento {i + 1}: {latencias[-1]:.2f} segundos")
        time.sleep(PAUSA)
    promedio = sum(latencias) / len(latencias)
    print(f"Latencia promedio: {promedio:.2f} segundos")
    costo = calcular_costo(datos, precio_entrada, precio_salida)
    print(f"Costo de la última petición: ${costo:.6f} USD")

    # 2) Calidad: respuestas a las preguntas reales (las evalúas tú)
    for n, pregunta in enumerate(PREGUNTAS_CALIDAD, start=1):
        url, body = armar_peticion(pregunta, modelo)
        datos, _ = enviar_con_reintentos(url, body)
        respuesta = extraer_texto(datos)
        print(f"\n--- Pregunta {n}: {pregunta}\n{respuesta}")
        time.sleep(PAUSA)

    return promedio


def imprimir_tabla(latencias):
    modelos = list(CANDIDATOS)  # funciona con 2, 3 o más modelos

    def por_mil(precio):
        return f"{precio * 1000:.5f}".rstrip("0") + " USD"

    def fila(criterio, celdas):
        print(f"| {criterio} | " + " | ".join(celdas) + " |")

    print(f"\n\n## Comparación de candidatos (medido el {date.today():%d-%m-%Y})\n")
    fila("Criterio", modelos)
    print("|---" * (len(modelos) + 1) + "|")
    fila("Costo por 1000 tokens de entrada", [por_mil(CANDIDATOS[m][0]) for m in modelos])
    fila("Costo por 1000 tokens de salida", [por_mil(CANDIDATOS[m][1]) for m in modelos])
    fila("Latencia promedio observada",
         [f"{latencias[m]:.2f} segundos" if latencias[m] is not None else "ERROR: volver a medir" for m in modelos])
    fila("Calidad de respuesta para tu caso concreto", ["COMPLETAR (x/6)" for m in modelos])


if __name__ == "__main__":
    latencias = {}
    for modelo, (precio_entrada, precio_salida) in CANDIDATOS.items():
        try:
            latencias[modelo] = medir_modelo(modelo, precio_entrada, precio_salida)
        except (ValueError, RuntimeError, TimeoutError) as error:
            # Si un modelo falla, se anota y se sigue con los demás
            print(f"No se pudo medir {modelo}: {error}")
            latencias[modelo] = None
    imprimir_tabla(latencias)
