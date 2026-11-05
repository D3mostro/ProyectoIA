import logging
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from enrutamiento import (
    construir_cadena_con_enrutamiento,
    construir_paso_clasificacion,
    construir_rutas,
    procesar_consulta_segura,
)


CONSULTA_LUGAR = "Quiero ver la Vía Láctea con mi familia en diciembre. ¿Qué lugar de Chile me recomiendan?"
CONSULTA_AMBIGUA = "Cuéntame algo sobre el cielo de Chile."


def modelo_falso(respuesta):
    # LangChain llama al modelo como función: el valor va en return_value del propio mock.
    modelo = MagicMock()
    modelo.return_value = AIMessage(content=respuesta)
    return modelo


def modelo_roto():
    # side_effect con una excepción: el mock la lanza como lo haría una llamada real a la API que falla.
    return MagicMock(side_effect=ConnectionError("simulado"))


def test_clasificacion_vacia_va_a_la_ruta_general():
    rutas = {f"cadena_{r}": (lambda x, r=r: f"ruta {r}") for r in ["lugar", "evento", "equipo", "logistica", "general"]}
    paso = construir_paso_clasificacion(modelo_falso("  "))
    cadena = construir_cadena_con_enrutamiento(paso, **rutas)
    assert cadena.invoke({"consulta": CONSULTA_AMBIGUA}) == "ruta general"


def test_cada_ruta_usa_un_prompt_distinto():
    modelo = modelo_falso("ok")
    for ruta in construir_rutas(modelo).values():
        ruta.invoke({"consulta": CONSULTA_LUGAR})
    prompts = [llamada.args[0].to_string() for llamada in modelo.call_args_list]
    assert len(prompts) == 5 and len(set(prompts)) == 5


def test_ruta_usa_fallback_ante_fallo_del_modelo():
    resultado = construir_rutas(modelo_roto())["cadena_lugar"].invoke({"consulta": CONSULTA_LUGAR})
    assert "equipo" in resultado.lower()


def test_fallback_registra_el_error_real(caplog):
    caplog.set_level(logging.WARNING)
    construir_rutas(modelo_roto())["cadena_lugar"].invoke({"consulta": CONSULTA_LUGAR})
    assert "simulado" in caplog.text


def test_fallback_no_se_ejecuta_si_la_ruta_funciona(caplog):
    caplog.set_level(logging.WARNING)
    resultado = construir_rutas(modelo_falso("respuesta real"))["cadena_lugar"].invoke({"consulta": CONSULTA_LUGAR})
    assert resultado == "respuesta real"
    assert caplog.records == []


def test_procesar_consulta_segura_no_propaga_excepciones(caplog):
    caplog.set_level(logging.WARNING)
    cadena = construir_cadena_con_enrutamiento(paso_clasificacion=modelo_roto())
    resultado = procesar_consulta_segura(cadena, CONSULTA_LUGAR)
    assert "no pudimos procesar" in resultado.lower()
    assert "simulado" in caplog.text


def test_procesar_consulta_segura_devuelve_el_resultado_normal():
    cadena = construir_cadena_con_enrutamiento(
        lambda x: "lugar", cadena_lugar=lambda x: "respuesta del lugar"
    )
    assert procesar_consulta_segura(cadena, CONSULTA_LUGAR) == "respuesta del lugar"


def test_flujo_completo_responde_si_falla_la_ruta_tras_clasificar():
    # 1.ª llamada (clasificación) OK; 2.ª (la ruta) falla.
    modelo = MagicMock(side_effect=[AIMessage(content="lugar"), ConnectionError("simulado")])
    cadena = construir_cadena_con_enrutamiento(construir_paso_clasificacion(modelo), **construir_rutas(modelo))
    resultado = procesar_consulta_segura(cadena, CONSULTA_LUGAR)
    assert "equipo" in resultado.lower()
    assert "no pudimos procesar" not in resultado.lower()
