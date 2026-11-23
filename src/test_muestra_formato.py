"""Tests del Bloque 30: extracción de texto con varias partes y revisión de la muestra (sin red)."""
import pytest

from consumir_gemini import extraer_texto, motivo_de_termino, partes_de_texto
from muestra_formato import (
    TAMANO_MUESTRA,
    UMBRAL_FALLOS,
    consultas_de_la_muestra,
    pendientes,
    resumir_muestra,
    revisar_respuesta,
)


def _datos(*partes, motivo="STOP"):
    return {"candidates": [{"content": {"parts": list(partes)}, "finishReason": motivo}]}


def test_extraer_texto_une_todas_las_partes():
    datos = _datos({"text": "No, no conviene, ya"}, {"text": " que la luna llena aclara el cielo [1]."})
    assert extraer_texto(datos) == "No, no conviene, ya que la luna llena aclara el cielo [1]."
    assert len(partes_de_texto(datos)) == 2


def test_extraer_texto_ignora_partes_de_razonamiento():
    datos = _datos({"text": "pienso...", "thought": True}, {"text": "Respuesta."})
    assert extraer_texto(datos) == "Respuesta."


def test_extraer_texto_una_parte_igual_que_antes():
    assert extraer_texto(_datos({"text": "Santiago"})) == "Santiago"


@pytest.mark.parametrize("datos", [{}, {"candidates": []}, _datos(), _datos({"functionCall": {}})])
def test_extraer_texto_sin_texto_da_error_claro(datos):
    with pytest.raises(RuntimeError, match="formato esperado"):
        extraer_texto(datos)


def test_motivo_de_termino():
    assert motivo_de_termino(_datos({"text": "x"}, motivo="MAX_TOKENS")) == "MAX_TOKENS"
    assert motivo_de_termino({}) is None


def _registro(respuesta="El cielo es oscuro [1].", motivo="STOP", validas=None, inventadas=None, **extra):
    return {"pregunta": extra.pop("pregunta", "p"), "respuesta": respuesta, "finish_reason": motivo,
            "citas_validas": {"1": "a.txt"} if validas is None else validas,
            "citas_inventadas": inventadas or [], **extra}


def test_respuesta_buena_cumple_todo():
    assert all(revisar_respuesta(_registro()).values())


def test_respuesta_cortada_falla_completa():
    assert revisar_respuesta(_registro("No, no conviene, ya", validas={}))["completa"] is False
    assert revisar_respuesta(_registro("Texto [1].", motivo="MAX_TOKENS"))["completa"] is False


def test_sin_cita_pero_reconoce_limite_cumple():
    r = revisar_respuesta(_registro("El contexto no menciona la altura de ALMA.", validas={}))
    assert r["cita_o_reconoce"] is True


def test_sin_cita_y_sin_reconocer_falla():
    assert revisar_respuesta(_registro("Venus es el más brillante.", validas={}))["cita_o_reconoce"] is False


def test_cita_inventada_falla():
    assert revisar_respuesta(_registro(inventadas=[4]))["sin_citas_inventadas"] is False


def test_error_falla_todo():
    assert not any(revisar_respuesta({"pregunta": "p", "error": "429"}).values())


def test_muestra_son_20_consultas_distintas():
    consultas = consultas_de_la_muestra()
    assert len(consultas) == TAMANO_MUESTRA == len(set(consultas))


def test_pendientes_salta_hechas_y_reintenta_errores():
    registros = [_registro(pregunta="a"), {"pregunta": "b", "error": "429"}]
    assert pendientes(registros, ["a", "b", "c"]) == ["b", "c"]


def test_resumen_umbral_y_ultima_linea_gana():
    consultas = [f"q{i}" for i in range(TAMANO_MUESTRA)]
    registros = [_registro(pregunta=q) for q in consultas]
    registros += [_registro("cortada", pregunta=q) for q in consultas[:UMBRAL_FALLOS - 1]]
    _, fallos, faltan, veredicto = resumir_muestra(registros, consultas)
    assert (fallos, faltan, veredicto) == (UMBRAL_FALLOS - 1, 0, False)
    registros.append(_registro("cortada", pregunta=consultas[-1]))
    assert resumir_muestra(registros, consultas)[3] is True


def test_resumen_incompleto_no_da_veredicto():
    assert resumir_muestra([_registro(pregunta="q0")], ["q0", "q1"])[3] is None


def test_sin_contexto_usa_la_ultima_respuesta():
    from muestra_formato import sin_contexto
    registros = [_registro("El contexto no alcanza para responder.", pregunta="a", validas={}),
                 _registro("El contexto no alcanza.", pregunta="b", validas={}),
                 _registro("Ahora sí: es X [1].", pregunta="b")]
    assert sin_contexto(registros, ["a", "b", "c"]) == ["a"]


def test_rehacer_exige_confirmar():
    from muestra_formato import leer_argumentos
    with pytest.raises(SystemExit):
        leer_argumentos(["--revisar", "--rehacer-sin-contexto"])
    assert leer_argumentos(["--confirmar", "--rehacer-sin-contexto"]).rehacer_sin_contexto
