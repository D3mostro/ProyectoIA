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


# --- Bloque 25: robustez de la indexación (sin red: embeddings falsos y Chroma en tmp_path) ---
import pytest
from langchain_core.documents import Document

from indexar_corpus import (
    agregar_en_lotes,
    con_reintentos,
    leer_argumentos,
    reemplazar_documento,
    validar_coleccion,
    verificar_chunks,
)

RESERVA_LARGA = "\n\n".join(
    f"Paso {i} de la reserva del observatorio: el operador revisa la solicitud número {i} "
    "antes de la noche de observación y confirma el telescopio asignado." * 2
    for i in range(6)
)


class EmbeddingsQueFallan(DeterministicFakeEmbedding):
    """Falla con 429 a partir de la llamada número `falla_desde` (1-based)."""

    llamadas: int = 0
    falla_desde: int = 2

    def embed_documents(self, texts):
        self.llamadas += 1
        if self.llamadas >= self.falla_desde:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: cuota de embeddings agotada")
        return super().embed_documents(texts)


def test_reemplazar_documento_acortado_no_deja_huerfanos(tmp_path):
    vs = conectar(EMBEDDINGS, tmp_path)
    reemplazar_documento(vs, "reserva_observatorio.md", RESERVA_LARGA)
    reemplazar_documento(vs, "fase_lunar.md", "La luna llena arruina la observación de cielo profundo.")
    antes = vs.get(where={"fuente": "reserva_observatorio.md"})["ids"]
    assert len(antes) > 1

    reemplazar_documento(vs, "reserva_observatorio.md", "Las reservas se aprueban 24 h antes.")
    textos = vs.get(where={"fuente": "reserva_observatorio.md"})["documents"]
    assert textos == ["Las reservas se aprueban 24 h antes."]
    assert len(vs.get(where={"fuente": "fase_lunar.md"})["ids"]) == 1  # otro documento intacto


def test_reindexar_documentos_cambiados_no_duplica_ni_deja_ids_viejos(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks, ids, EMBEDDINGS, tmp_path)
    vs = indexar(chunks, ids, EMBEDDINGS, tmp_path)
    assert sorted(vs.get(include=[])["ids"]) == sorted(ids)


def test_con_reintentos_reintenta_429_con_backoff_exponencial():
    esperas, intentos = [], []

    def operacion():
        intentos.append(1)
        if len(intentos) < 3:
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    assert con_reintentos(operacion, dormir=esperas.append) == "ok"
    assert esperas == [5, 10]


def test_con_reintentos_respeta_retry_delay():
    esperas = []
    errores = iter([
        RuntimeError("429 RESOURCE_EXHAUSTED retry_delay {\n  seconds: 33\n}"),
        RuntimeError("429 RESOURCE_EXHAUSTED. {'error': {'details': [{'retryDelay': '17s'}]}}"),
    ])

    def operacion():
        error = next(errores, None)
        if error:
            raise error
        return "ok"

    assert con_reintentos(operacion, dormir=esperas.append) == "ok"
    assert esperas == [33, 17]


def test_con_reintentos_relanza_otros_errores_sin_esperar():
    esperas = []

    def operacion():
        raise ValueError("API_KEY_INVALID")

    with pytest.raises(ValueError, match="API_KEY_INVALID"):
        con_reintentos(operacion, dormir=esperas.append)
    assert esperas == []


def test_con_reintentos_relanza_al_agotar_intentos():
    esperas = []

    def operacion():
        raise RuntimeError("RESOURCE_EXHAUSTED")

    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        con_reintentos(operacion, max_intentos=4, dormir=esperas.append)
    assert esperas == [5, 10, 20]


def test_agregar_en_lotes_conserva_lotes_guardados_si_falla_uno_posterior(tmp_path):
    chunks, ids = partir_en_chunks()
    vs = conectar(EmbeddingsQueFallan(size=16, falla_desde=3), tmp_path)
    with pytest.raises(RuntimeError, match="429"):
        agregar_en_lotes(vs, chunks, ids, tam_lote=5, max_intentos=2, dormir=lambda s: None)
    assert vs._collection.count() == 10  # los 2 lotes previos al 429 siguen guardados


def test_verificar_chunks_detecta_vacios_duplicados_y_lista_vacia():
    with pytest.raises(ValueError, match="vacío"):
        verificar_chunks([Document(page_content="  ", metadata={"fuente": "fase_lunar.md"})])
    repetido = Document(page_content="Cielo despejado en el observatorio.", metadata={"fuente": "a.md"})
    with pytest.raises(ValueError, match="duplicado"):
        verificar_chunks([repetido, repetido])
    with pytest.raises(ValueError, match="No hay chunks"):
        verificar_chunks([])
    verificar_chunks(partir_en_chunks()[0])  # el corpus real pasa


@pytest.mark.parametrize("nombre", ["ab", "-astro", "astro_", "astro turismo", "a" * 513])
def test_nombre_de_coleccion_invalido(nombre):
    with pytest.raises(ValueError, match="colección inválido"):
        validar_coleccion(nombre)


def test_coleccion_valida_y_cli_con_corpus_y_coleccion(tmp_path):
    assert validar_coleccion("corpus_real-2026.v1") == "corpus_real-2026.v1"
    args = leer_argumentos(["--corpus", str(tmp_path), "--coleccion", "corpus_real"])
    assert (args.corpus, args.coleccion) == (tmp_path, "corpus_real")
    with pytest.raises(SystemExit):
        leer_argumentos(["--coleccion", "x"])


def test_colecciones_distintas_no_se_mezclan(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks, ids, EMBEDDINGS, tmp_path)
    indexar(chunks[:2], ids[:2], EMBEDDINGS, tmp_path, coleccion="corpus_real")
    assert conectar(EMBEDDINGS, tmp_path)._collection.count() == len(chunks)
    assert conectar(EMBEDDINGS, tmp_path, "corpus_real")._collection.count() == 2


def test_corpus_recursivo_txt_con_mismo_nombre_en_subcarpetas_no_choca(tmp_path):
    for sub, texto in [("wikipedia_es", "El observatorio Mamalluca está en el valle de Elqui."),
                       ("observatorios_turisticos", "Tour nocturno en Mamalluca con telescopios.")]:
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "mamalluca.txt").write_text(texto, encoding="utf-8")
    chunks, ids = partir_en_chunks(tmp_path, "*.txt")
    assert [c.metadata["fuente"] for c in chunks] == [
        "observatorios_turisticos/mamalluca.txt", "wikipedia_es/mamalluca.txt"]
    assert ids == ["observatorios_turisticos/mamalluca-0", "wikipedia_es/mamalluca-0"]


def test_deduplicar_chunks_conserva_la_primera_aparicion():
    from indexar_corpus import deduplicar_chunks

    menu = "Inicio | Tours nocturnos | Contacto"
    chunks = [Document(page_content=texto, metadata={"fuente": fuente}) for texto, fuente in [
        (menu, "observatorios_turisticos/mamalluca.txt"),
        ("Tour nocturno en Mamalluca con telescopios.", "observatorios_turisticos/mamalluca.txt"),
        (menu + "  ", "observatorios_turisticos/alfa_aldea.txt"),
    ]]
    unicos, ids, saltados = deduplicar_chunks(chunks, ["m-0", "m-1", "a-0"])
    assert (ids, saltados) == (["m-0", "m-1"], 1)
    assert unicos[0].metadata["fuente"] == "observatorios_turisticos/mamalluca.txt"
    verificar_chunks(unicos)  # ya no frena por duplicados


from indexar_corpus import CuotaAgotada


class EmbeddingsQueCuentan(DeterministicFakeEmbedding):
    """Cuenta cuántos textos se embeben (cada uno gastaría cuota real)."""

    textos: int = 0

    def embed_documents(self, texts):
        self.textos += len(texts)
        return super().embed_documents(texts)


def test_con_reintentos_cuota_diaria_falla_sin_esperar():
    esperas = []

    def operacion():
        raise RuntimeError("429 RESOURCE_EXHAUSTED. quotaId: 'EmbedContentRequestsPerDayPerProjectPerModel-FreeTier'")

    with pytest.raises(CuotaAgotada, match="diaria"):
        con_reintentos(operacion, dormir=esperas.append)
    assert esperas == []


def test_con_reintentos_espera_sugerida_mayor_al_tope_falla_sin_esperar():
    esperas = []

    def operacion():
        raise RuntimeError("429 RESOURCE_EXHAUSTED. {'retryDelay': '3600s'}")

    with pytest.raises(CuotaAgotada, match="3600 s"):
        con_reintentos(operacion, espera_maxima=120, dormir=esperas.append)
    assert esperas == []


def test_indexar_salta_chunks_ya_guardados_y_reembebe_los_cambiados(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks[:10], ids[:10], EMBEDDINGS, tmp_path)  # simula una corrida cortada por cuota

    cuenta = EmbeddingsQueCuentan(size=16)
    indexar(chunks, ids, cuenta, tmp_path)
    assert cuenta.textos == len(chunks) - 10  # solo los que faltaban

    cambiado = Document(page_content="Mamalluca ahora cobra $12.000 el tour nocturno.", metadata=chunks[0].metadata)
    cuenta.textos = 0
    vs = indexar([cambiado] + chunks[1:], ids, cuenta, tmp_path)
    assert cuenta.textos == 1
    assert vs.get(ids=[ids[0]])["documents"] == [cambiado.page_content]
    assert vs._collection.count() == len(chunks)
