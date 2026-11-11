import json
import logging
import queue
import sys
import threading
import unicodedata
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from math import cos, radians
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from skyfield import almanac
from pydantic import BaseModel, Field, field_validator
from skyfield.api import Loader

from consumir_gemini import MODELO
from enrutamiento import procesar_consulta_segura

URL_OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
CARPETA_EFEMERIDES = Path(__file__).resolve().parent.parent / "data"
ARCHIVO_TRAZAS = Path(__file__).resolve().parent.parent / "trazas_agente.jsonl"
SEGUNDOS_PARA_APROBAR = 300
# Open-Meteo acepta hasta hoy + 16, pero ese último día llega sin datos de noche (probado 2026-11-10): hoy + 15.
DIAS_PRONOSTICO = 16
MAXIMO_PERSONAS = 20
AVISO_SIMULADA = "RESERVA SIMULADA (no se reservó nada real): "
# Sale de la revisión de trazas del B23: con Open-Meteo en error, el agente afirmó un clima "histórico".
REGLA_OBSERVACIONES = (
    "Si una herramienta devuelve un error o no trae datos, dilo tal cual y no lo completes "
    "con conocimiento general ni con datos históricos. Cuando uses herramientas, afirma solo "
    "lo que digan sus observaciones. Las reservas son simuladas: dilo siempre que hables de una."
)

# shortcut: 5 destinos escritos a mano (lat, lon de lugares.csv), pasar al CSV completo cuando el agente use el corpus.
LUGARES = {
    "vicuña": (-30.03405, -70.71267),
    "pisco elqui": (-30.12312, -70.493),
    "combarbalá": (-31.17815, -71.00244),
    "ovalle": (-30.60308, -71.20299),
    "san pedro de atacama": (-23.35455, -67.90266),
}

# Dos sesiones de dos turnos: el segundo turno solo se entiende con la memoria del primero.
CONVERSACIONES_EJEMPLO = [
    ("turista_1", "¿Cuánta luna habrá el 14 de diciembre de 2026 para ver las Gemínidas?"),
    ("turista_1", "¿Y esa misma noche estará despejado en Vicuña?"),
    ("turista_2", "Quiero reservar 2 cupos en el Observatorio Mamalluca para el 14 de diciembre de 2026."),
    ("turista_2", "¿Para qué fecha te pedí la reserva?"),
]

# shortcut: observatorios escritos a mano (sección E del corpus + Collowara y Del Pangue), pasar al corpus cuando el agente lo use.
OBSERVATORIOS = {
    "mamalluca": "Observatorio Mamalluca",
    "cancana": "Observatorio Cancana",
    "alfa aldea": "Observatorio Alfa Aldea",
    "cruz del sur": "Observatorio Cruz del Sur",
    "collowara": "Observatorio Collowara",
    "del pangue": "Observatorio Del Pangue",
}

registro_aprobaciones = []
historiales = {}


def normalizar(texto):
    # Minúsculas y sin tildes: el modelo puede escribir "Mamalluca", "MAMALLUCA" o "Combarbalá".
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.lower().split())


@lru_cache(maxsize=1)
def cargar_efemerides():
    # La primera vez Skyfield descarga de421.bsp (unos 17 MB) en data/, que queda fuera de Git.
    carga = Loader(str(CARPETA_EFEMERIDES), verbose=False)
    return carga.timescale(), carga("de421.bsp")


def angulo_de_fase(dia):
    escala, efemerides = cargar_efemerides()
    # Hora 27 UTC = 03:00 UTC del día siguiente, la noche chilena de esa fecha (Skyfield normaliza la hora).
    instante = escala.utc(dia.year, dia.month, dia.day, 27)
    return almanac.moon_phase(efemerides, instante).degrees


def porcentaje_iluminado(angulo):
    return (1 - cos(radians(angulo))) / 2 * 100


@tool
def consultar_fase_lunar(fecha: str) -> str:
    """Indica la fase de la Luna y su porcentaje iluminado en la noche de una
    fecha AAAA-MM-DD. Usar cuando preguntan cuánta luna habrá o qué tan oscuro
    estará el cielo por la Luna. No sirve para saber si habrá nubes."""
    try:
        angulo = angulo_de_fase(date.fromisoformat(fecha))
    except ValueError:
        return f"Fecha no válida: {fecha}. Usa el formato AAAA-MM-DD."
    except OSError as error:
        return f"No se pudieron cargar las efemérides: {error}"
    sentido = "creciente" if angulo < 180 else "menguante"
    return f"Luna {sentido}, {porcentaje_iluminado(angulo):.0f} % iluminada la noche del {fecha}."


class ArgumentosPronostico(BaseModel):
    """Esquema que el modelo lee campo por campo antes de llamar a pronosticar_nubes."""

    lugar: str = Field(description=f"Localidad de Chile. Disponibles: {', '.join(LUGARES)}.")
    fecha: str = Field(description="Fecha de la noche a consultar, en formato AAAA-MM-DD.")


@tool(args_schema=ArgumentosPronostico)
def pronosticar_nubes(lugar: str, fecha: str) -> str:
    """Entrega la nubosidad promedio pronosticada entre las 21:00 y las 23:59 de
    un lugar de Chile en una fecha AAAA-MM-DD. Usar cuando preguntan si estará
    despejado o nublado. Sirve solo para los próximos días, no para fechas lejanas."""
    coordenadas = LUGARES.get(lugar.strip().lower())
    if coordenadas is None:
        return f"No tengo coordenadas de {lugar}. Lugares disponibles: {', '.join(LUGARES)}."
    try:
        dia = date.fromisoformat(fecha)
    except ValueError:
        return f"Fecha no válida: {fecha}. Usa el formato AAAA-MM-DD."
    hoy = date.today()
    # Se corta antes de llamar a la API: una fecha lejana daba 400 y el modelo terminaba inventando el clima.
    if not hoy <= dia < hoy + timedelta(days=DIAS_PRONOSTICO):
        return f"{fecha} está fuera del rango de pronóstico (hasta {DIAS_PRONOSTICO} días desde hoy, {hoy})."
    latitud, longitud = coordenadas
    try:
        respuesta = requests.get(
            URL_OPEN_METEO,
            params={
                "latitude": latitud,
                "longitude": longitud,
                "hourly": "cloud_cover",
                "start_date": fecha,
                "end_date": fecha,
                "timezone": "America/Santiago",
            },
            timeout=15,
        )
        respuesta.raise_for_status()
        horas_noche = respuesta.json()["hourly"]["cloud_cover"][21:24]
    except (requests.RequestException, KeyError, ValueError) as error:
        return f"No se pudo obtener el pronóstico: {error}"
    valores = [valor for valor in horas_noche if valor is not None]
    if not valores:
        return f"No hay pronóstico de nubes para el {fecha}."
    promedio = sum(valores) / len(valores)
    return f"Nubosidad promedio de {promedio:.0f} % en {lugar} la noche del {fecha} (datos: Open-Meteo)."


def esperar_aprobacion(mensaje, tiempo_limite_segundos=None):
    # None: se lee SEGUNDOS_PARA_APROBAR al llamar, así el __main__ puede bajarlo para la demo.
    if tiempo_limite_segundos is None:
        tiempo_limite_segundos = SEGUNDOS_PARA_APROBAR
    # Encabezado separado con líneas en blanco: en la 1ª corrida real el aviso quedó tapado por el verbose.
    print(f"\n\n=== APROBACIÓN REQUERIDA (tienes {tiempo_limite_segundos} s; sin respuesta = no aprobada) ===")
    # queue + hilo en vez del select() del curso: select() no funciona sobre stdin en Windows.
    respuestas = queue.Queue()

    def leer_respuesta():
        try:
            respuestas.put(input(mensaje))
        except (EOFError, KeyboardInterrupt):
            respuestas.put("")

    # daemon=True: si se agota el tiempo, el hilo que sigue esperando input() no impide cerrar el programa.
    # shortcut: ese hilo se come la próxima línea tecleada; pasar a un panel de aprobaciones si hay supervisión real.
    threading.Thread(target=leer_respuesta, daemon=True).start()
    try:
        return respuestas.get(timeout=tiempo_limite_segundos).strip().lower() == "s"
    except queue.Empty:
        print("Sin respuesta a tiempo: la acción queda no aprobada.")
        return False


def registrar_aprobacion(herramienta, argumentos, aprobado):
    registro_aprobaciones.append({
        "herramienta": herramienta,
        "argumentos": argumentos,
        "aprobado": aprobado,
        "momento": datetime.now(timezone.utc).isoformat(),  # UTC, igual que la traza, para poder cruzarlas
    })


class ArgumentosReserva(BaseModel):
    """Se valida antes de pedir aprobación: el supervisor no debe ver una reserva imposible."""

    observatorio: str = Field(description=f"Observatorio turístico. Disponibles: {', '.join(OBSERVATORIOS.values())}.")
    fecha: str = Field(description="Fecha de la visita, en formato AAAA-MM-DD.")
    personas: int = Field(ge=1, le=MAXIMO_PERSONAS, description=f"Cantidad de cupos, de 1 a {MAXIMO_PERSONAS}.")

    @field_validator("observatorio")
    @classmethod
    def observatorio_conocido(cls, valor):
        clave = normalizar(valor).removeprefix("observatorio ").strip()
        if clave not in OBSERVATORIOS:
            raise ValueError(f"observatorio desconocido '{valor}'; disponibles: {', '.join(OBSERVATORIOS.values())}")
        return OBSERVATORIOS[clave]

    @field_validator("fecha")
    @classmethod
    def fecha_iso(cls, valor):
        # len == 10: fromisoformat también acepta "20261214", que no es el formato pedido.
        try:
            if len(valor) != 10:
                raise ValueError
            date.fromisoformat(valor)
        except ValueError:
            raise ValueError(f"fecha no válida '{valor}', usa AAAA-MM-DD") from None
        return valor


def avisar_datos_invalidos(error):
    detalle = "; ".join(f"{e['loc'][0]}: {e['msg'].removeprefix('Value error, ')}" for e in error.errors())
    return f"Datos de reserva no válidos, no se pidió aprobación: {detalle}."


@tool(args_schema=ArgumentosReserva)
def reservar_cupo_observatorio(observatorio: str, fecha: str, personas: int) -> str:
    """Reserva cupos para una visita guiada a un observatorio turístico de Chile en
    una fecha AAAA-MM-DD. Usar SOLO cuando el usuario pide reservar de forma explícita.
    Acción de riesgo alto: requiere aprobación humana antes de ejecutarse."""
    argumentos = {"observatorio": observatorio, "fecha": fecha, "personas": personas}
    aprobado = esperar_aprobacion(
        f"El agente quiere reservar {personas} cupos en {observatorio} para el {fecha}. ¿Aprobar? (s/n): "
    )
    registrar_aprobacion("reservar_cupo_observatorio", argumentos, aprobado)
    if not aprobado:
        return f"{AVISO_SIMULADA}no aprobada por el supervisor humano."
    # Simulada: no hay sistema de reservas real detrás; aquí iría la llamada al observatorio.
    return f"{AVISO_SIMULADA}aprobada, {personas} cupos en {observatorio} para el {fecha}."


# El esquema falla antes de entrar a la función: sin esto el ValidationError tumbaría al AgentExecutor.
reservar_cupo_observatorio.handle_validation_error = avisar_datos_invalidos

HERRAMIENTAS = [consultar_fase_lunar, pronosticar_nubes, reservar_cupo_observatorio]


def construir_prompt_agente():
    # El modelo no conoce la fecha actual: sin ella no puede resolver "esta noche".
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres asesor de astroturismo en Chile. Hoy es {hoy}. Usa las herramientas "
            "disponibles cuando la consulta necesite datos de una fecha o un lugar concretos. "
            "Si es una pregunta general, responde directo. Reserva solo si el usuario lo pide "
            "de forma explícita. " + REGLA_OBSERVACIONES + " Responde en máximo 4 líneas.",
        ),
        # optional=True: el ejecutor sin memoria (y sus tests del B21) sigue funcionando sin chat_history.
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{consulta}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]).partial(hoy=date.today().isoformat())


def construir_agente(modelo):
    agente = create_tool_calling_agent(modelo, HERRAMIENTAS, construir_prompt_agente())
    # max_iterations=5 en vez de los 15 por defecto: cada vuelta es una llamada al modelo y gasta cuota.
    return AgentExecutor(
        agent=agente,
        tools=HERRAMIENTAS,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=5,
    )


def extraer_texto_final(salida):
    # gemini-3.5-flash-lite entrega el contenido como lista de bloques, no como un str.
    if isinstance(salida, str):
        return salida
    return "".join(bloque.get("text", "") for bloque in salida if isinstance(bloque, dict))


def obtener_historial(session_id):
    # shortcut: historial en memoria, se pierde al cerrar el programa; pasar a una base de datos con usuarios reales.
    if session_id not in historiales:
        historiales[session_id] = InMemoryChatMessageHistory()
    return historiales[session_id]


def construir_agente_con_memoria(modelo):
    # Se limpia la salida antes de guardarla: el historial debe recibir texto, no la lista de bloques de Gemini.
    limpiar_salida = RunnableLambda(lambda d: {**d, "output": extraer_texto_final(d["output"])})
    return RunnableWithMessageHistory(
        construir_agente(modelo) | limpiar_salida,
        obtener_historial,
        input_messages_key="consulta",
        history_messages_key="chat_history",
        output_messages_key="output",
    )


def registrar_traza(session_id, entrada, resultado, archivo=ARCHIVO_TRAZAS):
    registro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "entrada": entrada,
        "acciones": [
            {"herramienta": accion.tool, "argumentos": accion.tool_input, "observacion": observacion}
            for accion, observacion in resultado.get("intermediate_steps", [])
        ],
        "respuesta_final": extraer_texto_final(resultado.get("output", "")),
    }
    try:
        with open(archivo, "a", encoding="utf-8") as destino:
            destino.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except OSError as error:
        # Perder una traza no debe impedir mostrarle la respuesta al turista.
        print(f"No se pudo guardar la traza en {archivo}: {error}", file=sys.stderr)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="error.log",
        filemode="a",
        encoding="utf-8",
    )
    load_dotenv()
    SEGUNDOS_PARA_APROBAR = 60  # demo en consola: 300 s es demasiado esperar si no se ve el aviso
    agente = construir_agente_con_memoria(ChatGoogleGenerativeAI(model=MODELO, timeout=30))
    for session_id, consulta in CONVERSACIONES_EJEMPLO:
        resultado = procesar_consulta_segura(agente, consulta, session_id=session_id)
        if isinstance(resultado, str):  # procesar_consulta_segura devuelve el mensaje de error
            print(f"[{session_id}] > {consulta}\n{resultado}\n")
            continue
        registrar_traza(session_id, consulta, resultado)
        for accion, observacion in resultado["intermediate_steps"]:
            print(accion.tool, accion.tool_input, "->", observacion)
        print(f"[{session_id}] > {consulta}\n{resultado['output']}\n")
    print("Aprobaciones:", registro_aprobaciones)
