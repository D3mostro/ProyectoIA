from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from cadena_tres_pasos import construir_cadena


def modelo_falso(*respuestas):
    # El side_effect va en el mock mismo (LangChain lo llama como modelo(...), no modelo.invoke(...))
    modelo = MagicMock()
    modelo.side_effect = [AIMessage(content=r) for r in respuestas]
    return modelo


def test_cadena_llama_al_modelo_tres_veces():
    modelo = modelo_falso("lugar", "borrador de respuesta", "respuesta final")
    resultado = construir_cadena(modelo).invoke({"consulta": "¿Dónde veo la Vía Láctea?"})
    assert modelo.call_count == 3
    assert resultado == "respuesta final"


def test_categoria_del_paso_1_llega_al_paso_2():
    modelo = modelo_falso("  Equipo\n", "borrador", "final")
    construir_cadena(modelo).invoke({"consulta": "¿Qué telescopio compro?"})
    prompt_paso_2 = modelo.call_args_list[1].args[0].to_string()
    assert "'equipo'" in prompt_paso_2
    assert "¿Qué telescopio compro?" in prompt_paso_2


def test_borrador_y_consulta_llegan_al_paso_3():
    modelo = modelo_falso("lugar", "BORRADOR-UNICO", "final")
    construir_cadena(modelo).invoke({"consulta": "consulta original"})
    prompt_paso_3 = modelo.call_args_list[2].args[0].to_string()
    assert "BORRADOR-UNICO" in prompt_paso_3
    assert "consulta original" in prompt_paso_3


def test_falta_la_consulta_lanza_error():
    modelo = modelo_falso("lugar", "borrador", "final")
    with pytest.raises(KeyError):
        construir_cadena(modelo).invoke({"texto": "clave equivocada"})
    assert modelo.call_count == 0
