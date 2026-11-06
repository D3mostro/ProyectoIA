from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import agente_con_herramientas as agente_mod
from agente_con_herramientas import (
    construir_agente,
    consultar_fase_lunar,
    extraer_texto_final,
    porcentaje_iluminado,
    pronosticar_nubes,
)


def modelo_falso(*respuestas):
    # create_tool_calling_agent llama a bind_tools: el Runnable devuelto entrega las respuestas en orden.
    pendientes = iter(respuestas)
    modelo = MagicMock()
    modelo.bind_tools.return_value = RunnableLambda(lambda _: next(pendientes))
    return modelo


def pide_herramienta(nombre, **argumentos):
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": argumentos, "id": "1"}])


def respuesta_open_meteo(nubes_por_hora):
    respuesta = MagicMock()
    respuesta.json.return_value = {"hourly": {"cloud_cover": nubes_por_hora}}
    return respuesta


def test_agente_usa_fase_lunar():
    ejecutor = construir_agente(modelo_falso(
        pide_herramienta("consultar_fase_lunar", fecha="2026-12-14"),
        AIMessage(content="Habrá poca luna."),
    ))
    with patch.object(agente_mod, "angulo_de_fase", return_value=45.0):
        resultado = ejecutor.invoke({"consulta": "¿Cuánta luna habrá el 14 de diciembre?"})

    accion, observacion = resultado["intermediate_steps"][0]
    assert accion.tool == "consultar_fase_lunar"
    assert "creciente, 15 %" in observacion
    assert resultado["output"] == "Habrá poca luna."


def test_agente_usa_pronostico_de_nubes():
    ejecutor = construir_agente(modelo_falso(
        pide_herramienta("pronosticar_nubes", lugar="Vicuña", fecha="2026-11-05"),
        AIMessage(content="Estará despejado."),
    ))
    with patch("agente_con_herramientas.requests.get", return_value=respuesta_open_meteo([0] * 21 + [10, 20, 30])):
        resultado = ejecutor.invoke({"consulta": "¿Estará despejado esta noche en Vicuña?"})

    accion, observacion = resultado["intermediate_steps"][0]
    assert accion.tool == "pronosticar_nubes"
    assert "20 %" in observacion


def test_agente_encadena_las_dos_herramientas():
    ejecutor = construir_agente(modelo_falso(
        pide_herramienta("consultar_fase_lunar", fecha="2026-11-05"),
        pide_herramienta("pronosticar_nubes", lugar="Vicuña", fecha="2026-11-05"),
        AIMessage(content="Buena noche."),
    ))
    with patch.object(agente_mod, "angulo_de_fase", return_value=200.0), patch(
        "agente_con_herramientas.requests.get", return_value=respuesta_open_meteo([0] * 24)
    ):
        resultado = ejecutor.invoke({"consulta": "¿Es buena noche para observar en Vicuña?"})

    assert [accion.tool for accion, _ in resultado["intermediate_steps"]] == [
        "consultar_fase_lunar",
        "pronosticar_nubes",
    ]


def test_agente_no_usa_herramienta_para_pregunta_general():
    ejecutor = construir_agente(modelo_falso(AIMessage(content="Es una nube de gas y polvo.")))
    resultado = ejecutor.invoke({"consulta": "¿Qué es una nebulosa?"})
    assert resultado["intermediate_steps"] == []


def test_agente_se_detiene_en_el_limite_de_iteraciones():
    ejecutor = construir_agente(modelo_falso(
        *[pide_herramienta("consultar_fase_lunar", fecha="fecha-mala") for _ in range(5)]
    ))
    resultado = ejecutor.invoke({"consulta": "¿Cuánta luna habrá?"})
    assert len(resultado["intermediate_steps"]) == 5
    assert "max iterations" in resultado["output"]


@pytest.mark.parametrize("angulo, esperado", [(0, 0), (90, 50), (180, 100), (270, 50)])
def test_porcentaje_iluminado(angulo, esperado):
    assert porcentaje_iluminado(angulo) == pytest.approx(esperado)


def test_fase_lunar_devuelve_texto_ante_fecha_invalida():
    assert "Fecha no válida" in consultar_fase_lunar.invoke({"fecha": "mañana"})


def test_fase_lunar_devuelve_texto_si_faltan_efemerides():
    with patch.object(agente_mod, "cargar_efemerides", side_effect=OSError("sin red")):
        assert "sin red" in consultar_fase_lunar.invoke({"fecha": "2026-12-14"})


def test_nubes_devuelve_texto_ante_lugar_desconocido():
    observacion = pronosticar_nubes.invoke({"lugar": "Marte", "fecha": "2026-11-05"})
    assert "No tengo coordenadas de Marte" in observacion


def test_nubes_devuelve_texto_si_falla_la_red():
    with patch("agente_con_herramientas.requests.get", side_effect=requests.ConnectionError("sin red")):
        observacion = pronosticar_nubes.invoke({"lugar": "Vicuña", "fecha": "2026-11-05"})
    assert "No se pudo obtener el pronóstico" in observacion


def test_nubes_ignora_horas_sin_dato():
    with patch("agente_con_herramientas.requests.get", return_value=respuesta_open_meteo([None] * 24)):
        observacion = pronosticar_nubes.invoke({"lugar": "Vicuña", "fecha": "2026-11-05"})
    assert "No hay pronóstico" in observacion


def test_extraer_texto_final_acepta_str_y_bloques():
    assert extraer_texto_final("hola") == "hola"
    assert extraer_texto_final([{"type": "text", "text": "ho"}, {"type": "text", "text": "la"}]) == "hola"
