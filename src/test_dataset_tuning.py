"""Bloque 29: pruebas del dataset de tuning y de su validador (sin red ni API)."""
import json
from collections import Counter

import pytest
from google.genai import types

from dataset_tuning import (
    CLAVES_ESPERADAS,
    HERRAMIENTA_POR_INTENCION,
    MINIMO_EJEMPLOS,
    a_texto,
    chequear_balance,
    construir_examples,
    dataset_roto,
    ejemplos_raw,
    razon_desbalance,
    validar_conjunto,
    validar_dataset,
)

E = types.TuningExample
BUENO = '{"intencion": "fase_lunar", "herramienta": "consultar_fase_lunar"}'


@pytest.fixture
def examples():
    return construir_examples(ejemplos_raw)


def test_dataset_propio_valido(examples):
    assert validar_dataset(examples) == []
    assert validar_conjunto(examples, chequear_balance(ejemplos_raw)) == []


def test_minimo_y_todas_las_etiquetas(examples):
    assert len(examples) >= MINIMO_EJEMPLOS
    balance = chequear_balance(ejemplos_raw)
    assert set(balance) == set(HERRAMIENTA_POR_INTENCION)
    assert razon_desbalance(balance) <= 2


def test_outputs_canonicos(examples):
    for ej in examples:
        salida = json.loads(ej.output)
        assert list(salida) == list(CLAVES_ESPERADAS)
        assert ej.output == a_texto(salida)


def test_tildes_sin_escapar():
    assert "\\u" not in a_texto({"intencion": "información", "herramienta": None})


@pytest.mark.parametrize(
    "output, esperado",
    [
        ("", "output vacio"),
        ("no es json", "no es JSON valido"),
        ('"intencion herramienta"', "no es un objeto JSON"),
        ("5", "no es un objeto JSON"),
        ("[1, 2]", "no es un objeto JSON"),
        ('{"intencion": "fase_lunar"}', 'no tiene la clave "herramienta"'),
        ('{"intencion": "fase_lunar", "herramienta": "consultar_fase_lunar", "x": 1}', 'clave no esperada "x"'),
        ('{"intencion": "rde", "herramienta": null}', 'intencion no permitida "rde"'),
        ('{"intencion": "", "herramienta": null}', 'intencion no permitida ""'),
        ('{"intencion": "fase_lunar", "herramienta": "pronosticar_nubes"}', "no corresponde"),
        ('{"intencion": "fuera_de_dominio", "herramienta": "buscar_en_corpus"}', "no corresponde"),
        ('{"herramienta": "consultar_fase_lunar", "intencion": "fase_lunar"}', "forma canonica"),
        ('{"intencion":"fase_lunar","herramienta":"consultar_fase_lunar"}', "forma canonica"),
    ],
)
def test_validador_detecta_output_malo(output, esperado):
    problemas = validar_dataset([E(text_input="pregunta", output=output)])
    assert any(esperado in p for p in problemas), problemas


def test_validador_no_se_cae_con_numero():
    # En la página, json.loads("5") -> int y `clave in 5` lanza TypeError.
    validar_dataset([E(text_input="a", output="5")])


def test_duplicado_ignora_mayusculas_y_espacios():
    problemas = validar_dataset([E(text_input="Hola  Luna", output=BUENO), E(text_input="hola luna ", output=BUENO)])
    assert problemas == ["ejemplo 1: text_input duplicado"]


def test_duplicado_con_output_vacio_tambien_se_detecta():
    problemas = validar_dataset([E(text_input="a", output=BUENO), E(text_input="a", output="")])
    assert "ejemplo 1: text_input duplicado" in problemas
    assert "ejemplo 1: output vacio" in problemas


def test_reporta_todos_los_problemas_de_un_ejemplo():
    problemas = validar_dataset([E(text_input="", output='{"intencion": "fase_lunar"}')])
    assert len(problemas) == 2


def test_dataset_roto_detecta_cada_error(examples):
    n = len(examples)
    problemas = validar_dataset(dataset_roto(examples))
    indices = {int(p.split()[1].rstrip(":")) for p in problemas}
    assert indices == set(range(n, n + 8))


def test_balance_no_se_cae_sin_clave():
    balance = chequear_balance([("a", {"otra": "x"}), ("b", "no es dict")])
    assert balance["<sin clave>"] == 1
    assert balance["<no es dict>"] == 1


def test_balance_muestra_etiquetas_sin_ejemplos():
    balance = chequear_balance([("a", {"intencion": "fase_lunar"})])
    assert balance["pronostico_nubes"] == 0
    assert razon_desbalance(balance) is None


def test_conjunto_chico_o_de_una_categoria():
    raw = [(f"luna {i}", {"intencion": "fase_lunar", "herramienta": "consultar_fase_lunar"}) for i in range(3)]
    problemas = validar_conjunto(construir_examples(raw), chequear_balance(raw))
    assert any("solo hay 3" in p for p in problemas)
    assert any("una sola categoria" in p for p in problemas)
    assert any("sin ejemplos" in p for p in problemas)


def test_razon_desbalance():
    assert razon_desbalance(Counter({"a": 3, "b": 1})) == 3.0
