from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from enrutamiento import construir_cadena_con_enrutamiento, construir_paso_clasificacion

RUTAS = ["lugar", "evento", "equipo", "logistica"]


def modelo_falso(respuesta):
    # LangChain llama al modelo como función: el valor va en return_value del propio mock.
    modelo = MagicMock()
    modelo.return_value = AIMessage(content=respuesta)
    return modelo


def clasificador_falso(categoria):
    return lambda entrada: categoria


def rutas_falsas():
    return {f"cadena_{r}": (lambda x, r=r: f"ruta {r}") for r in RUTAS + ["general"]}


@pytest.mark.parametrize("categoria", RUTAS)
def test_cada_categoria_va_a_su_ruta(categoria):
    cadena = construir_cadena_con_enrutamiento(clasificador_falso(categoria), **rutas_falsas())
    assert cadena.invoke({"consulta": "hola"}) == f"ruta {categoria}"


def test_categoria_inesperada_va_a_la_ruta_general():
    cadena = construir_cadena_con_enrutamiento(clasificador_falso("queja"), **rutas_falsas())
    assert cadena.invoke({"consulta": "hola"}) == "ruta general"


def test_la_ruta_recibe_la_consulta_y_la_categoria():
    recibido = {}

    def ruta_lugar(entrada):
        recibido.update(entrada)
        return "ok"

    rutas = rutas_falsas()
    rutas["cadena_lugar"] = ruta_lugar
    construir_cadena_con_enrutamiento(clasificador_falso("lugar"), **rutas).invoke({"consulta": "hola"})
    assert recibido == {"consulta": "hola", "categoria": "lugar"}


@pytest.mark.parametrize("salida", ["  EQUIPO  ", "equipo.", "Equipo.\n"])
def test_clasificacion_normaliza_la_salida(salida):
    paso = construir_paso_clasificacion(modelo_falso(salida))
    assert paso.invoke({"consulta": "¿Qué llevo?"}) == "equipo"
