"""Gate 2 (lección 20447): los cuatro criterios no compensables, sobre el servicio real.

Sin red ni Gemini: índice temporal de corpus/*.md con embeddings falsos y un modelo falso.
    python -m pytest src/test_gate2.py -v
"""
import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

import logging_rag as lr
from agente_con_herramientas import registrar_traza
from grafo_supervision import construir_grafo, pendiente, proponer_despliegue, resolver
from indexar_corpus import conectar, elegir_coleccion, indexar, partir_en_chunks
from registro_prompts import NOMBRE_PROMPT, RegistroDePrompts, registrar_actuales
from registro_rag import guardar_registro
from servicio_rag import AVISO_SIN_CITA, ejecutar_servicio, main, preparar_simulado

CORREO = "ana.perez@example.com"
PREGUNTA = "¿Qué pasa si nadie aprueba mi reserva a tiempo? Mi correo es " + CORREO


def modelo_falso(texto):
    def enviar(url, body):
        return {"candidates": [{"content": {"parts": [{"text": texto}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 20,
                                  "thoughtsTokenCount": 0, "totalTokenCount": 220}}
    return enviar


def sin_reintentos(funcion):
    return funcion()


@pytest.fixture(scope="module")
def vectorstore():
    return preparar_simulado()[0]


@pytest.fixture
def logs(tmp_path):
    ruta = tmp_path / "logs_rag.jsonl"
    lr.configurar_logger(ruta, consola=False)
    yield ruta
    lr.configurar_logger(None, consola=False)


def consultar(vectorstore, texto="La reserva queda pendiente si nadie la aprueba [1]."):
    return ejecutar_servicio(PREGUNTA, vectorstore, 3, modelo_falso(texto),
                             reintentar_embed=sin_reintentos, reintentar_generacion=sin_reintentos)


# 1. pipeline_corre_de_punta_a_punta
def test_1_servicio_corre_de_punta_a_punta(tmp_path, capsys):
    assert main(["--simulado"], archivo_logs=tmp_path / "logs.jsonl") == 0
    salida = capsys.readouterr().out
    assert "estado: ok" in salida and "Proceso terminado sin errores." in salida


def test_1b_clon_limpio_usa_la_coleccion_del_repo(tmp_path):
    # Sin astroturismo_real (corpus solo en el PC de German) se usa corpus/*.md.
    embeddings = DeterministicFakeEmbedding(size=16)
    indexar(*partir_en_chunks(), embeddings, tmp_path)
    vs, nombre = elegir_coleccion(embeddings, tmp_path)
    assert nombre == "astroturismo" and vs._collection.count() > 0


def test_1c_sin_indice_explica_que_hacer(tmp_path):
    with pytest.raises(SystemExit, match="indexar_corpus.py"):
        elegir_coleccion(DeterministicFakeEmbedding(size=16), tmp_path)


# 2. respuestas_citan_fuente
def test_2_respuesta_trae_el_nombre_del_archivo(vectorstore, logs):
    r = consultar(vectorstore)
    assert r["estado"] == "ok" and r["fuentes"]
    assert all(f.endswith(".md") and f in r["texto"] for f in r["fuentes"])


def test_2b_sin_cita_no_queda_ok(vectorstore, logs):
    r = consultar(vectorstore, "La reserva queda pendiente.")
    assert r["estado"] == "sin_cita" and AVISO_SIN_CITA in r["texto"]


def test_2c_reconocer_el_limite_no_es_un_fallo(vectorstore, logs):
    r = consultar(vectorstore, "El contexto no alcanza para responder eso.")
    assert r["estado"] == "reconoce_limite" and AVISO_SIN_CITA not in r["texto"]


# 3. logs_sin_contenido_crudo
def test_3_trazas_del_servicio_sin_texto_crudo(vectorstore, logs):
    r = consultar(vectorstore)
    texto = logs.read_text(encoding="utf-8")
    assert '"evento": "consulta_recibida"' in texto  # hubo logs: la prueba no pasa vacía
    assert CORREO not in texto and PREGUNTA not in texto and r["respuesta"] not in texto


def test_3b_registros_versionados_sin_texto_libre(tmp_path):
    archivo = tmp_path / "registros.jsonl"
    guardar_registro({"pregunta": PREGUNTA, "respuesta": "Escríbele a " + CORREO, "k": 3}, archivo)
    registrar_traza("turista_1", PREGUNTA, {"output": "Te escribo a " + CORREO}, archivo=tmp_path / "trazas.jsonl")
    for ruta in (archivo, tmp_path / "trazas.jsonl"):
        texto = ruta.read_text(encoding="utf-8")
        assert CORREO not in texto and PREGUNTA not in texto
    assert json.loads(archivo.read_text(encoding="utf-8"))["texto_omitido"] is True


# 4. supervision_humana_funciona
@pytest.fixture
def registro_temporal(tmp_path):
    registro = RegistroDePrompts(tmp_path / "registro.json")
    v1, v2 = registrar_actuales(registro)
    registro.activar(NOMBRE_PROMPT, v1["huella"], "German", "estado inicial")
    return registro, v1, v2


def test_4_supervision_pausa_y_solo_una_persona_activa(registro_temporal):
    registro, v1, v2 = registro_temporal
    app = construir_grafo(registro)
    config, resultado = proponer_despliegue(app, v2["huella"])
    assert "__interrupt__" in resultado and pendiente(app, config)["huella_propuesta"] == v2["huella"]
    assert registro.activa(NOMBRE_PROMPT)["huella"] == v1["huella"]  # nada cambió mientras espera
    resolver(app, config, True, v2["huella"])
    assert registro.activa(NOMBRE_PROMPT)["huella"] == v2["huella"]


def test_4b_rechazo_mantiene_la_version(registro_temporal):
    registro, v1, v2 = registro_temporal
    app = construir_grafo(registro)
    config, _ = proponer_despliegue(app, v2["huella"])
    resolver(app, config, False, v2["huella"])
    assert registro.activa(NOMBRE_PROMPT)["huella"] == v1["huella"]
