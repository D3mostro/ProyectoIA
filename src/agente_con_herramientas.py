import logging
from datetime import date
from functools import lru_cache
from math import cos, radians
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from skyfield import almanac
from skyfield.api import Loader

from consumir_gemini import MODELO
from enrutamiento import procesar_consulta_segura

URL_OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
CARPETA_EFEMERIDES = Path(__file__).resolve().parent.parent / "data"

# shortcut: 5 destinos escritos a mano (lat, lon de lugares.csv), pasar al CSV completo cuando el agente use el corpus.
LUGARES = {
    "vicuña": (-30.03405, -70.71267),
    "pisco elqui": (-30.12312, -70.493),
    "combarbalá": (-31.17815, -71.00244),
    "ovalle": (-30.60308, -71.20299),
    "san pedro de atacama": (-23.35455, -67.90266),
}

CONSULTAS_EJEMPLO = [
    "¿Cuánta luna habrá el 14 de diciembre de 2026 para ver las Gemínidas?",
    "¿Estará despejado esta noche en Vicuña?",
    "¿Qué es una nebulosa?",
]


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


@tool
def pronosticar_nubes(lugar: str, fecha: str) -> str:
    """Entrega la nubosidad promedio pronosticada entre las 21:00 y las 23:59 de
    un lugar de Chile en una fecha AAAA-MM-DD. Usar cuando preguntan si estará
    despejado o nublado. Sirve solo para los próximos días, no para fechas lejanas."""
    coordenadas = LUGARES.get(lugar.strip().lower())
    if coordenadas is None:
        return f"No tengo coordenadas de {lugar}. Lugares disponibles: {', '.join(LUGARES)}."
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


HERRAMIENTAS = [consultar_fase_lunar, pronosticar_nubes]


def construir_prompt_agente():
    # El modelo no conoce la fecha actual: sin ella no puede resolver "esta noche".
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres asesor de astroturismo en Chile. Hoy es {hoy}. Usa las herramientas "
            "disponibles cuando la consulta necesite datos de una fecha o un lugar concretos. "
            "Si es una pregunta general, responde directo. Responde en máximo 4 líneas.",
        ),
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
    ejecutor_agente = construir_agente(ChatGoogleGenerativeAI(model=MODELO, timeout=30))
    for consulta in CONSULTAS_EJEMPLO:
        resultado = procesar_consulta_segura(ejecutor_agente, consulta)
        if isinstance(resultado, str):  # procesar_consulta_segura devuelve el mensaje de error
            print(f"> {consulta}\n{resultado}\n")
            continue
        for accion, observacion in resultado["intermediate_steps"]:
            print(accion.tool, accion.tool_input, "->", observacion)
        print(f"> {consulta}\n{extraer_texto_final(resultado['output'])}\n")
