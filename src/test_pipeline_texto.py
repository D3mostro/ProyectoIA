from collections import Counter

import pytest

from pipeline_texto import (
    limpiar_texto,
    tokenizar,
    procesar_texto,
    vectorizar,
    calcular_coincidencia,
)


def test_procesar_texto_normaliza_mayusculas_y_puntuacion():
    resultado = procesar_texto("El Cielo, es AZUL!")
    assert resultado == ["el", "cielo", "es", "azul"]


def test_procesar_texto_solo_puntuacion_devuelve_lista_vacia():
    resultado = procesar_texto("¡¡¡...!!!")
    assert resultado == []


def test_limpiar_texto_conserva_tildes_enie_y_numeros():
    assert limpiar_texto("¿Año 2027? ¡Eclipse en Ñuble!") == "año 2027 eclipse en ñuble"


def test_limpiar_texto_colapsa_espacios():
    assert limpiar_texto("  cielo   oscuro \n\t despejado  ") == "cielo oscuro despejado"


def test_limpiar_texto_rechaza_lo_que_no_es_texto():
    with pytest.raises(TypeError, match="espera str"):
        limpiar_texto(None)


def test_tokenizar_texto_vacio():
    assert tokenizar("") == []


def test_vectorizar_cuenta_repeticiones():
    assert vectorizar(["el", "gato", "el"]) == Counter({"el": 2, "gato": 1})


def test_coincidencia_mejora_con_el_pipeline():
    referencia, respuesta = "El cielo es azul.", "el cielo es azul"
    assert calcular_coincidencia(referencia.split(), respuesta.split()) == 0.5
    assert calcular_coincidencia(procesar_texto(referencia), procesar_texto(respuesta)) == 1.0


def test_coincidencia_ignora_el_orden():
    # Límite conocido de la métrica: mismas palabras con sentido opuesto dan 100 %
    referencia = procesar_texto("El perro persiguió al gato.")
    respuesta = procesar_texto("El gato persiguió al perro.")
    assert calcular_coincidencia(referencia, respuesta) == 1.0


def test_coincidencia_con_referencia_vacia_no_divide_por_cero():
    assert calcular_coincidencia([], ["algo"]) == 0.0
