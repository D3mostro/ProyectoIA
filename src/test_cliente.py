"""Bloque 14 - Pruebas unitarias del cliente modular (cliente.py).

Todas usan mocks: no llaman a la API real ni esperan con time.sleep real.

Ejecutar desde la carpeta src, con el .venv activo:
    python -m pytest test_cliente.py -v
"""
from unittest.mock import patch, Mock

import pytest
import requests

import cliente
from cliente import enviar_peticion, con_reintento, registrar_costo, consultar_modelo

URL = "https://url-de-prueba"
CUERPO = {"contents": [{"parts": [{"text": "test"}]}]}


@pytest.fixture(autouse=True)
def reiniciar_costo_total():
    # costo_total_sesion es global: se reinicia antes de cada prueba para que no se contaminen entre sí
    cliente.costo_total_sesion = 0
    yield


# --- enviar_peticion (vive en consumir_gemini.py, por eso el patch apunta a ese módulo) ---

def test_enviar_peticion_credencial_invalida():
    respuesta_falsa = Mock(status_code=401)
    with patch("consumir_gemini.requests.post", return_value=respuesta_falsa):
        with pytest.raises(ValueError):
            enviar_peticion(URL, CUERPO)


def test_enviar_peticion_sin_permiso():
    respuesta_falsa = Mock(status_code=403)
    with patch("consumir_gemini.requests.post", return_value=respuesta_falsa):
        with pytest.raises(ValueError):
            enviar_peticion(URL, CUERPO)


def test_enviar_peticion_api_key_invalida_400():
    # Gemini responde una key inválida con 400 + "API_KEY_INVALID", no con 401/403
    respuesta_falsa = Mock(status_code=400, text='{"reason": "API_KEY_INVALID"}')
    with patch("consumir_gemini.requests.post", return_value=respuesta_falsa):
        with pytest.raises(ValueError):
            enviar_peticion(URL, CUERPO)


def test_enviar_peticion_limite_de_uso():
    respuesta_falsa = Mock(status_code=429)
    with patch("consumir_gemini.requests.post", return_value=respuesta_falsa):
        with pytest.raises(TimeoutError):
            enviar_peticion(URL, CUERPO)


def test_enviar_peticion_error_del_servidor():
    respuesta_falsa = Mock(status_code=503)
    with patch("consumir_gemini.requests.post", return_value=respuesta_falsa):
        with pytest.raises(TimeoutError):
            enviar_peticion(URL, CUERPO)


def test_enviar_peticion_timeout_de_red():
    with patch("consumir_gemini.requests.post", side_effect=requests.exceptions.Timeout):
        with pytest.raises(TimeoutError):
            enviar_peticion(URL, CUERPO)


def test_enviar_peticion_exito():
    respuesta_falsa = Mock(status_code=200)
    respuesta_falsa.json.return_value = {"texto": "respuesta simulada"}
    with patch("consumir_gemini.requests.post", return_value=respuesta_falsa):
        resultado = enviar_peticion(URL, CUERPO)
    assert resultado == {"texto": "respuesta simulada"}


# --- con_reintento ---

def test_con_reintento_reintenta_ante_timeout():
    intentos = {"n": 0}

    def falla_una_vez():
        intentos["n"] += 1
        if intentos["n"] == 1:
            raise TimeoutError("simulado")
        return "ok"

    with patch("cliente.time.sleep") as mock_sleep:
        resultado = con_reintento(falla_una_vez)

    assert resultado == "ok"
    mock_sleep.assert_called_once()  # no se compara cuántos segundos: el jitter es aleatorio


def test_con_reintento_no_reintenta_value_error():
    funcion = Mock(side_effect=ValueError("credencial inválida"))
    with patch("cliente.time.sleep") as mock_sleep:
        with pytest.raises(ValueError):
            con_reintento(funcion, intentos=3)
    assert funcion.call_count == 1
    mock_sleep.assert_not_called()


def test_con_reintento_agota_intentos():
    def siempre_falla():
        raise TimeoutError("simulado")

    with patch("cliente.time.sleep") as mock_sleep:
        with pytest.raises(TimeoutError):
            con_reintento(siempre_falla, intentos=3)
    assert mock_sleep.call_count == 2  # espera entre intentos, no después del último


# --- registrar_costo (no toca la red: no necesita mocks) ---

def test_registrar_costo_calcula_y_acumula():
    respuesta_falsa = {"usageMetadata": {
        "promptTokenCount": 1000,
        "candidatesTokenCount": 1000,
        "totalTokenCount": 2000,
    }}
    esperado = 1000 * cliente.PRECIO_ENTRADA + 1000 * cliente.PRECIO_SALIDA

    costo = registrar_costo(respuesta_falsa)
    assert costo == pytest.approx(esperado)
    assert cliente.costo_total_sesion == pytest.approx(esperado)

    registrar_costo(respuesta_falsa)
    assert cliente.costo_total_sesion == pytest.approx(2 * esperado)


# --- consultar_modelo ---

def test_consultar_modelo_convierte_error():
    with patch("cliente.con_reintento", side_effect=TimeoutError("simulado")):
        with pytest.raises(RuntimeError):
            consultar_modelo("pregunta de prueba")


def test_consultar_modelo_propaga_value_error():
    with patch("cliente.con_reintento", side_effect=ValueError("credencial inválida")):
        with pytest.raises(ValueError, match="credencial inválida"):
            consultar_modelo("pregunta de prueba")


def test_consultar_modelo_exito_registra_costo():
    respuesta_falsa = {
        "candidates": [{"content": {"parts": [{"text": "Santiago"}]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
    }
    with patch("cliente.enviar_peticion", return_value=respuesta_falsa):
        resultado = consultar_modelo("¿Capital de Chile?")
    assert resultado == "Santiago"
    assert cliente.costo_total_sesion > 0
