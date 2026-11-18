"""Bloque 27: métricas y evaluación de la recuperación, sin red (embeddings falsos, Chroma en tmp_path)."""
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_text_splitters import RecursiveCharacterTextSplitter

import indexar_corpus
from evaluacion_recuperacion import (
    CASOS_EVALUACION,
    EtiquetaInvalida,
    evaluar,
    f1,
    formatear_resultados,
    guardar,
    leer_argumentos,
    posicion_primer_relevante,
    precision_at_k,
    promediar,
    rango_reciproco,
    recall_at_k,
    resolver_relevantes,
)
from indexar_corpus import conectar, indexar, partir_en_chunks

EMBEDDINGS = DeterministicFakeEmbedding(size=16)  # sin llamadas a Gemini


@pytest.fixture
def vectorstore(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks, ids, EMBEDDINGS, tmp_path)
    return conectar(EMBEDDINGS, tmp_path)


# --- Métricas: resultados esperados de la página (L1) ---

def test_precision_y_recall_del_ejemplo_de_la_pagina():
    assert precision_at_k(["doc2", "doc0", "doc4"], {"doc0"}) == pytest.approx(1 / 3)
    assert recall_at_k(["doc2", "doc0", "doc4"], {"doc0"}) == 1.0


def test_trade_off_de_k_y_f1_maximo_en_k3():
    ranked = ["doc2", "doc4", "doc0", "doc1", "doc3"]
    f1s = [f1(precision_at_k(ranked[:k], {"doc0"}), recall_at_k(ranked[:k], {"doc0"})) for k in range(1, 6)]
    assert [round(x, 3) for x in f1s] == [0.0, 0.0, 0.5, 0.4, 0.333]


def test_casos_borde_no_dividen_por_cero():
    assert precision_at_k([], {"doc0"}) == 0.0
    assert recall_at_k(["doc0"], set()) is None
    assert precision_at_k(["doc0"], set()) is None  # pregunta sin respuesta: no castiga la precisión
    assert f1(None, 1.0) is None
    assert f1(0.0, 0.0) == 0.0


def test_recall_no_supera_1_con_ids_repetidos():
    assert recall_at_k(["doc0", "doc0"], {"doc0"}) == 1.0


def test_promediar_ignora_none_y_no_inventa_cero():
    assert promediar([1.0, None, 0.0]) == 0.5
    assert promediar([None, None]) is None


# --- Etiquetado ---

def test_todas_las_evidencias_calzan_con_el_corpus_real(vectorstore):
    relevantes = resolver_relevantes(vectorstore, CASOS_EVALUACION)
    con_respuesta = [r for r, c in zip(relevantes, CASOS_EVALUACION) if c["evidencias"]]
    assert all(con_respuesta)
    assert sum(1 for c in CASOS_EVALUACION if not c["evidencias"]) == 2
    assert 8 <= len(CASOS_EVALUACION) - 2  # la página pide al menos 8 preguntas con respuesta


def test_evidencia_que_no_calza_detiene_la_evaluacion(vectorstore):
    casos = [{"pregunta": "x", "evidencias": [("fase_lunar.md", "texto que no existe")]}]
    with pytest.raises(EtiquetaInvalida, match="no aparece"):
        resolver_relevantes(vectorstore, casos)


def test_evidencia_con_fuente_equivocada_detiene_la_evaluacion(vectorstore):
    casos = [{"pregunta": "x", "evidencias": [("modelo_y_costo.md", "17 MB")]}]
    with pytest.raises(EtiquetaInvalida):
        resolver_relevantes(vectorstore, casos)


def test_etiquetas_sobreviven_a_cambiar_chunk_size(tmp_path, monkeypatch):
    """Con ids fijos (como en la página) la etiqueta apuntaría a otro texto; con evidencias, no."""
    caso = [{"pregunta": "x", "evidencias": [("fase_lunar.md", "iluminada en un 30 %")]}]
    ids_por_config = []
    for i, (size, overlap) in enumerate([(300, 50), (120, 20)]):
        monkeypatch.setattr(indexar_corpus, "SPLITTER", RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap))
        chunks, ids = partir_en_chunks()
        carpeta = tmp_path / str(i)
        indexar(chunks, ids, EMBEDDINGS, carpeta)
        vs = conectar(EMBEDDINGS, carpeta)
        (rel,) = resolver_relevantes(vs, caso)
        textos = vs.get(ids=list(rel))["documents"]
        assert all("iluminada en un 30 %" in t for t in textos)
        ids_por_config.append(rel)
    assert ids_por_config[0] != ids_por_config[1]  # el id cambió, la etiqueta siguió siendo correcta


# --- Evaluación completa ---

def test_evaluar_es_determinista_y_respeta_el_trade_off(vectorstore):
    primera = evaluar(vectorstore, CASOS_EVALUACION, ks=[3, 5])
    segunda = evaluar(vectorstore, CASOS_EVALUACION, ks=[3, 5])
    assert primera == segunda  # mismo índice, mismos números
    assert primera[5]["recall"] >= primera[3]["recall"]
    for f3, f5 in zip(primera[3]["casos"], primera[5]["casos"]):
        assert f5["recuperados"][:3] == f3["recuperados"]
    assert primera[3]["sin_respuesta"] == 2


def test_un_solo_retrieval_por_pregunta_aunque_se_comparen_varios_k(vectorstore, monkeypatch):
    llamadas = []
    original = vectorstore.as_retriever

    def contar(**kwargs):
        retriever = original(**kwargs)
        llamadas.append(kwargs["search_kwargs"]["k"])
        return retriever

    monkeypatch.setattr(vectorstore, "as_retriever", contar)
    evaluar(vectorstore, CASOS_EVALUACION, ks=[3, 5, 1])
    assert llamadas == [5]  # un retriever con el k mayor, no uno por k


def test_retrieval_de_top_k_es_prefijo_del_top_k_mayor(vectorstore):
    """Justifica recortar el ranking de k=5 para obtener k=3 en vez de recuperar dos veces."""
    pregunta = CASOS_EVALUACION[0]["pregunta"]
    k3 = [d.id for d in vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(pregunta)]
    k5 = [d.id for d in vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(pregunta)]
    assert k5[:3] == k3


def test_k_invalido():
    with pytest.raises(ValueError):
        evaluar(None, CASOS_EVALUACION, ks=[0])


def test_formato_y_guardado_agrega_sin_sobrescribir(vectorstore, tmp_path):
    resultados = evaluar(vectorstore, CASOS_EVALUACION, ks=[3])
    texto = formatear_resultados(resultados, "astroturismo", 19, fecha="2026-11-16 21:00:00")
    assert "chunk_size=300, chunk_overlap=50" in texto  # configuración real, no escrita a mano
    assert "promedio k=3 -> precision:" in texto
    assert "mrr:" in texto and "parafraseada" in texto
    assert texto.count("n/a ") >= 6  # 2 preguntas sin respuesta x (precision, recall, f1)
    archivo = tmp_path / "resultados_evaluacion.txt"
    guardar(texto, archivo)
    guardar(texto, archivo)
    assert archivo.read_text(encoding="utf-8").count("Fecha: 2026-11-16 21:00:00") == 2


def test_cli():
    assert leer_argumentos([]).k == [3]
    assert leer_argumentos(["-k", "3", "5", "--no-guardar"]).k == [3, 5]
    with pytest.raises(SystemExit):
        leer_argumentos(["-k", "0"])


# --- Posición del relevante, MRR y preguntas parafraseadas ---

def test_posicion_y_rango_reciproco():
    assert posicion_primer_relevante(["a", "b", "c"], {"b"}) == 2
    assert posicion_primer_relevante(["a", "b"], {"z"}) is None
    assert rango_reciproco(2, {"b"}) == 0.5
    assert rango_reciproco(None, {"b"}) == 0.0  # no llegó: cuenta como 0, no se ignora
    assert rango_reciproco(None, set()) is None  # sin respuesta: fuera del promedio


def test_mrr_distingue_lo_que_precision_no(vectorstore):
    """Dos rankings con la misma precision@3 (1/3) pero el relevante en 1.º vs 3.º lugar."""
    assert precision_at_k(["r", "x", "y"], {"r"}) == precision_at_k(["x", "y", "r"], {"r"})
    assert rango_reciproco(posicion_primer_relevante(["r", "x", "y"], {"r"}), {"r"}) == 1.0
    assert rango_reciproco(posicion_primer_relevante(["x", "y", "r"], {"r"}), {"r"}) == pytest.approx(1 / 3)


def test_hay_parafraseadas_y_sus_evidencias_calzan(vectorstore):
    parafraseadas = [c for c in CASOS_EVALUACION if c.get("parafrasis")]
    assert len(parafraseadas) >= 5
    assert all(resolver_relevantes(vectorstore, parafraseadas))


def test_evaluar_separa_grupos_y_calcula_mrr(vectorstore):
    res = evaluar(vectorstore, CASOS_EVALUACION, ks=[3])[3]
    assert res["por_grupo"]["directa"]["n"] == 14
    assert res["por_grupo"]["parafraseada"]["n"] == 6
    assert res["por_grupo"]["amplia"]["n"] == 5
    assert 0.0 <= res["mrr"] <= 1.0
    for fila in res["casos"]:
        if fila["posicion"] is not None:
            assert 1 <= fila["posicion"] <= 3
            assert fila["recall"] > 0


def test_preguntas_amplias_tienen_varios_relevantes(vectorstore):
    """Idea de la profesora: con 2-3 relevantes, precision@3 puede pasar de 0.33 y distingue configuraciones."""
    amplias = [c for c in CASOS_EVALUACION if c.get("amplia")]
    assert len(amplias) == 5
    assert all(len(rel) >= 2 for rel in resolver_relevantes(vectorstore, amplias))


def test_clave_invalida_marca_los_dos_chunks_que_la_responden(vectorstore):
    (caso,) = [c for c in CASOS_EVALUACION if "clave de la API es inválida" in c["pregunta"]]
    (rel,) = resolver_relevantes(vectorstore, [caso])
    assert rel == {"errores_y_cuota-1", "errores_y_cuota-2"}


def test_precision_puede_superar_un_tercio_con_varios_relevantes():
    assert precision_at_k(["a", "b", "x"], {"a", "b"}) == pytest.approx(2 / 3)
