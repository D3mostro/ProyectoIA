"""Gate 2: verificar_entorno.py sin red ni API."""
from langchain_core.embeddings import DeterministicFakeEmbedding

from indexar_corpus import indexar, partir_en_chunks
from verificar_entorno import revisar


def _gitignore(tmp_path, texto=".env\nvectorstore/\n"):
    (tmp_path / ".gitignore").write_text(texto, encoding="utf-8")


def test_sin_indice_ni_clave_queda_pendiente(tmp_path):
    _gitignore(tmp_path)
    lineas, problemas = revisar(tmp_path / "vectorstore", tmp_path, entorno={})
    assert any("GEMINI_API_KEY" in p for p in problemas)
    assert any("indexar_corpus.py" in p for p in problemas)
    assert "GEMINI_API_KEY: FALTA" in lineas


def test_con_indice_clave_y_gitignore_no_hay_pendientes(tmp_path):
    _gitignore(tmp_path)
    indexar(*partir_en_chunks(), DeterministicFakeEmbedding(size=8), tmp_path / "vectorstore")
    lineas, problemas = revisar(tmp_path / "vectorstore", tmp_path, entorno={"GEMINI_API_KEY": "x" * 30})
    assert not [p for p in problemas if "instalar" not in p]  # en la nube puede faltar algún paquete
    assert any("colección astroturismo:" in l for l in lineas)
    assert not any("x" * 30 in l for l in lineas)  # la clave nunca se imprime


def test_env_fuera_de_gitignore_es_un_problema(tmp_path):
    _gitignore(tmp_path, "# .env\nvectorstore/\n")
    _, problemas = revisar(tmp_path / "vectorstore", tmp_path, entorno={"GEMINI_API_KEY": "x"})
    assert any(".gitignore" in p for p in problemas)
