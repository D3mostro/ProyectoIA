import pytest

from metricas import (
    contar_coincidencias,
    calcular_precision_cobertura_f1,
    calcular_metricas,
    resumir_metricas,
    peores_casos,
    muestra_para_revisar,
)


def test_contar_coincidencias_no_cuenta_repeticiones_de_mas():
    # Con set() del Bloque 15 esto coincidía 100 %; Counter toma el mínimo de cada palabra
    assert contar_coincidencias(["el", "gato"], ["el", "gato", "el", "gato"]) == 2


def test_ejemplo_calculado_a_mano_del_curso():
    resultado = calcular_metricas("El gato duerme en el sofá.", "El gato duerme en el sillón todo el día.")
    assert resultado["precision"] == pytest.approx(5 / 9)
    assert resultado["cobertura"] == pytest.approx(5 / 6)
    assert resultado["f1"] == pytest.approx(2 / 3)


def test_respuesta_vacia_no_divide_por_cero():
    assert calcular_precision_cobertura_f1(["el", "cielo"], []) == {"precision": 0.0, "cobertura": 0.0, "f1": 0.0}
    assert calcular_metricas("", "algo") == {"precision": 0.0, "cobertura": 0.0, "f1": 0.0}


def test_sin_palabras_en_comun_f1_cero():
    assert calcular_precision_cobertura_f1(["luna"], ["sol"])["f1"] == 0.0


def test_respuesta_identica_f1_uno():
    assert calcular_metricas("Cielo despejado en Atacama.", "cielo despejado en atacama")["f1"] == 1.0


# Límites de la métrica (lección 2): se prueban para dejar documentado que existen
def test_limite_sinonimo_perfecto_castiga_respuesta_correcta():
    assert calcular_metricas("El auto es veloz.", "El vehículo es rápido.")["f1"] == pytest.approx(0.5)


def test_limite_dato_equivocado_premia_respuesta_incorrecta():
    assert calcular_metricas("Python fue creado en 1991.", "Python fue creado en 1989.")["f1"] == pytest.approx(0.8)


def test_limite_negacion_casi_no_mueve_el_numero():
    resultado = calcular_metricas("Python no es un lenguaje compilado.", "Python es un lenguaje compilado.")
    assert resultado["precision"] == 1.0
    assert resultado["f1"] == pytest.approx(10 / 11)


def test_resumir_metricas_promedia():
    resultados = [
        {"precision": 0.83, "cobertura": 0.71, "f1": 0.77},
        {"precision": 0.40, "cobertura": 0.50, "f1": 0.44},
        {"precision": 0.62, "cobertura": 0.62, "f1": 0.62},
    ]
    resumen = resumir_metricas(resultados)
    assert resumen["f1_promedio"] == pytest.approx(0.61)
    assert resumen["precision_promedio"] == pytest.approx(0.6166, abs=1e-3)


def test_resumir_metricas_lista_vacia():
    assert resumir_metricas([]) == {"precision_promedio": 0.0, "cobertura_promedio": 0.0, "f1_promedio": 0.0}


def test_peores_casos_ordena_por_f1_ascendente():
    resultados = [{"consulta": "a", "f1": 0.77}, {"consulta": "b", "f1": 0.44}, {"consulta": "c", "f1": 0.62}]
    assert [r["consulta"] for r in peores_casos(resultados, cantidad=2)] == ["b", "c"]


def test_muestra_para_revisar_junta_peores_y_mejores():
    resultados = [{"consulta": f"c{i}", "f1": i / 10} for i in range(12)]
    muestra = [r["consulta"] for r in muestra_para_revisar(resultados)]
    assert muestra == ["c0", "c1", "c2", "c3", "c4", "c7", "c8", "c9", "c10", "c11"]


def test_muestra_para_revisar_no_duplica_con_pocos_resultados():
    resultados = [{"consulta": f"c{i}", "f1": i / 10} for i in range(6)]
    muestra = muestra_para_revisar(resultados)
    assert len(muestra) == 6
    assert len({r["consulta"] for r in muestra}) == 6
