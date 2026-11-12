from langchain_core.embeddings import DeterministicFakeEmbedding

from indexar_corpus import SPLITTER, conectar, indexar, partir_en_chunks

EMBEDDINGS = DeterministicFakeEmbedding(size=16)  # sin llamadas a Gemini


def test_corpus_tiene_al_menos_5_documentos_y_se_parte():
    chunks, ids = partir_en_chunks()
    fuentes = {c.metadata["fuente"] for c in chunks}
    assert len(fuentes) >= 5
    assert len(chunks) > len(fuentes)  # hubo chunking real, no un chunk por documento
    assert len(ids) == len(set(ids)) == len(chunks)


def test_chunks_no_superan_chunk_size_y_tienen_fuente():
    chunks, _ = partir_en_chunks()
    assert all(len(c.page_content) <= SPLITTER._chunk_size for c in chunks)
    assert all(c.metadata["fuente"].endswith(".md") for c in chunks)


def test_indexar_dos_veces_no_duplica_y_reconexion_ve_lo_mismo(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks, ids, EMBEDDINGS, tmp_path)
    indexar(chunks, ids, EMBEDDINGS, tmp_path)
    assert conectar(EMBEDDINGS, tmp_path)._collection.count() == len(chunks)


def test_reindexar_con_menos_chunks_borra_los_viejos(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks, ids, EMBEDDINGS, tmp_path)
    indexar(chunks[:3], ids[:3], EMBEDDINGS, tmp_path)
    assert conectar(EMBEDDINGS, tmp_path)._collection.count() == 3


def test_recuperacion_devuelve_la_fuente(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks, ids, EMBEDDINGS, tmp_path)
    resultado = conectar(EMBEDDINGS, tmp_path).similarity_search("reserva", k=2)
    assert len(resultado) == 2
    assert all("fuente" in doc.metadata for doc in resultado)
