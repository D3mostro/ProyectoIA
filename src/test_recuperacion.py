"""Bloque 26: recuperación y citas, sin red (embeddings falsos, Chroma en tmp_path, modelo simulado)."""
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from indexar_corpus import conectar, indexar, partir_en_chunks
from recuperacion import (
    PROMPT_TEMPLATE,
    SinContexto,
    prompt_activo,
    construir_prompt,
    extraer_citas,
    formatear_contexto,
    inspeccionar_chunks,
    leer_argumentos,
    responder,
    validar_citas,
)

EMBEDDINGS = DeterministicFakeEmbedding(size=16)  # sin llamadas a Gemini


@pytest.fixture
def vectorstore(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks, ids, EMBEDDINGS, tmp_path)
    return conectar(EMBEDDINGS, tmp_path)


def test_formatear_contexto_numera_desde_1_y_mapea_fuentes():
    docs = [
        Document(page_content="El cliente reintenta hasta 3 veces.", metadata={"fuente": "doc0.md"}),
        Document(page_content="El costo usa totalTokenCount.", metadata={"fuente": "doc2.md"}),
    ]
    contexto, citas = formatear_contexto(docs)
    assert citas == {1: "doc0.md", 2: "doc2.md"}  # resultado esperado de la página
    assert contexto == "[1] El cliente reintenta hasta 3 veces.\n\n[2] El costo usa totalTokenCount."


def test_formatear_contexto_sin_fuente_da_desconocida():
    _, citas = formatear_contexto([Document(page_content="x", metadata={})])
    assert citas == {1: "desconocida"}


def test_construir_prompt_cumple_lo_que_exige_la_actividad(vectorstore):
    pregunta = "¿Qué pasa si nadie aprueba mi reserva a tiempo?"
    prompt, citas = construir_prompt(pregunta, vectorstore, k=3)
    assert list(citas) == [1, 2, 3]  # exactamente k entradas, claves 1..k
    assert "desconocida" not in citas.values()
    assert all(f.endswith(".md") for f in citas.values())  # archivos reales del corpus
    assert f"Pregunta: {pregunta}" in prompt  # pregunta exacta, sin recortar
    for n in citas:
        assert f"[{n}] " in prompt
    assert "[4] " not in prompt
    # B32: usa la plantilla de la versión aprobada, no la constante del B26
    assert prompt.startswith(prompt_activo().plantilla.split("{contexto}")[0])


def test_construir_prompt_usa_la_version_activa_y_no_la_constante(vectorstore):
    activo = prompt_activo()
    falso = activo._replace(plantilla="PROMPT DE PRUEBA {contexto} -- {pregunta}")
    prompt, _ = construir_prompt("reserva", vectorstore, k=1, activo=falso)
    assert prompt.startswith("PROMPT DE PRUEBA [1] ") and prompt.endswith("-- reserva")
    assert PROMPT_TEMPLATE.split("{contexto}")[0] not in prompt


def test_construir_prompt_preguntas_distintas_traen_contexto_distinto(vectorstore):
    p1, _ = construir_prompt("reserva del observatorio", vectorstore)
    p2, _ = construir_prompt("fase lunar y luna llena", vectorstore)
    # El bloque de contexto va entre "[1] " y la línea "Pregunta:" en cualquier versión del prompt.
    assert p1.split("[1] ", 1)[1].split("Pregunta:")[0] != p2.split("[1] ", 1)[1].split("Pregunta:")[0]


def test_coleccion_equivocada_falla_claro_en_vez_de_devolver_vacio(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks, ids, EMBEDDINGS, tmp_path)  # colección "astroturismo"
    otra = conectar(EMBEDDINGS, tmp_path, "langchain")  # lo que hace la página sin collection_name
    with pytest.raises(SinContexto):
        construir_prompt("reserva", otra)


def test_pregunta_vacia_y_k_invalido(vectorstore):
    with pytest.raises(ValueError):
        construir_prompt("   ", vectorstore)
    with pytest.raises(ValueError):
        construir_prompt("reserva", vectorstore, k=0)


def test_k_mayor_que_el_indice_devuelve_los_que_hay(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks[:2], ids[:2], EMBEDDINGS, tmp_path)
    _, citas = construir_prompt("reserva", conectar(EMBEDDINGS, tmp_path), k=10)
    assert list(citas) == [1, 2]


def test_reintenta_la_recuperacion_ante_429(vectorstore, monkeypatch):
    import indexar_corpus

    monkeypatch.setattr(indexar_corpus.time, "sleep", lambda s: None)
    original = vectorstore.similarity_search
    llamadas = []

    def falla_una_vez(*args, **kwargs):
        llamadas.append(1)
        if len(llamadas) == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return original(*args, **kwargs)

    monkeypatch.setattr(vectorstore, "similarity_search", falla_una_vez)
    _, citas = construir_prompt("reserva", vectorstore, k=2)
    assert len(llamadas) == 2 and len(citas) == 2


def test_extraer_citas_reconoce_los_formatos_habituales():
    respuesta = "Reintenta [1], falla [2]. Costo [3] y [99]. También [1, 4] y [5][2]."
    assert extraer_citas(respuesta) == [1, 2, 3, 99, 4, 5]
    assert extraer_citas("Sin citas.") == []
    assert extraer_citas("Año [2026-11]") == []  # no es una cita


def test_validar_citas_separa_las_inventadas():
    validas, invalidas = validar_citas("Dato [1] y otro [99].", {1: "a.md", 2: "b.md"})
    assert validas == {1: "a.md"}
    assert invalidas == [99]


def test_responder_usa_el_prompt_y_valida_citas(vectorstore):
    recibido = []

    def modelo_falso(prompt, system_instruction=""):
        recibido.append((prompt, system_instruction))
        return "La reserva se cancela si nadie aprueba [1]. Dato extra [7]."

    resultado = responder("¿Qué pasa si nadie aprueba mi reserva?", vectorstore, consultar=modelo_falso, k=3)
    assert "Pregunta: ¿Qué pasa si nadie aprueba mi reserva?" in recibido[0][0]
    activo = prompt_activo()
    assert recibido[0][1] == activo.sistema  # el texto de sistema va aparte, no dentro del prompt
    assert resultado["prompt_huella"] == activo.huella
    assert resultado["citas_validas"] == {1: resultado["citas"][1]}
    assert resultado["citas_invalidas"] == [7]
    assert resultado["sin_citas"] is False


def test_responder_marca_respuesta_sin_citas(vectorstore):
    resultado = responder("¿Capital de Australia?", vectorstore, consultar=lambda p, **_: "El contexto no alcanza.")
    assert resultado["sin_citas"] is True and resultado["citas_invalidas"] == []


def test_inspeccionar_chunks_lista_id_fuente_y_resumen(vectorstore):
    filas = inspeccionar_chunks(vectorstore)
    chunks, ids = partir_en_chunks()
    assert [f[0] for f in filas] == sorted(ids)
    assert all(len(resumen) <= 100 and "\n" not in resumen for _, _, resumen in filas)
    assert {f[1] for f in filas} == {c.metadata["fuente"] for c in chunks}


def test_argumentos():
    args = leer_argumentos(["¿pregunta?", "-k", "4", "--solo-prompt"])
    assert args.preguntas == ["¿pregunta?"] and args.k == 4 and args.solo_prompt
    with pytest.raises(SystemExit):
        leer_argumentos(["-k", "0"])
