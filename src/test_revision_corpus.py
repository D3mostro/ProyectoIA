import json
from datetime import date

import pytest

import revision_corpus as rc

HOY = date(2026, 11, 25)


def crear_txt(carpeta, nombre, fecha):
    carpeta.mkdir(parents=True, exist_ok=True)
    encabezado = f"Fuente: https://ejemplo.cl\nDescargado: {fecha}\n\n" if fecha else "Sin encabezado\n"
    (carpeta / nombre).write_text(encabezado + "Horario visitas 21:00 hrs", encoding="utf-8")


def test_fecha_descarga_con_texto_extra_en_la_linea(tmp_path):
    (tmp_path / "a.txt").write_text("Observatorio municipal de Vicuña. Descargado: 2026-10-10\n",
                                    encoding="utf-8")
    assert rc.fecha_descarga(tmp_path / "a.txt") == date(2026, 10, 10)


def test_vencidos_y_vigentes(tmp_path):
    obs = tmp_path / "obs"
    crear_txt(obs, "viejo.txt", "2026-10-10")
    crear_txt(obs, "nuevo.txt", "2026-11-20")
    crear_txt(obs, "sin_fecha.txt", None)
    vencidos = rc.observatorios_vencidos(obs, tmp_path / "rev.json", hoy=HOY)
    assert vencidos == [("sin_fecha.txt", None), ("viejo.txt", date(2026, 10, 10))]
    assert "2026-10-10" in rc.problema_vigencia(obs, tmp_path / "rev.json", hoy=HOY)


def test_limite_exacto_de_30_dias(tmp_path):
    obs = tmp_path / "obs"
    crear_txt(obs, "a.txt", "2026-10-26")  # 30 días antes de HOY
    assert rc.observatorios_vencidos(obs, tmp_path / "rev.json", hoy=HOY) == []
    crear_txt(obs, "a.txt", "2026-10-25")  # 31 días
    assert len(rc.observatorios_vencidos(obs, tmp_path / "rev.json", hoy=HOY)) == 1


def test_confirmacion_humana_libera_y_caduca(tmp_path):
    obs, rev = tmp_path / "obs", tmp_path / "rev.json"
    crear_txt(obs, "viejo.txt", "2026-10-10")
    revision = rc.confirmar_revision("German", "revisé ESO y ALMA", obs, rev)
    assert revision["archivos"] == {"viejo.txt": "2026-10-10"}
    hoy_real = date.fromisoformat(revision["fecha"][:10])
    assert rc.observatorios_vencidos(obs, rev, hoy=hoy_real) == []
    assert len(rc.observatorios_vencidos(obs, rev, hoy=date(hoy_real.year + 1, 1, 1))) == 1
    assert len(json.loads(rev.read_text(encoding="utf-8"))) == 1


@pytest.mark.parametrize("revisor, nota", [("", "n"), ("German", ""), ("  ", "n")])
def test_confirmacion_exige_revisor_y_nota(tmp_path, revisor, nota):
    crear_txt(tmp_path / "obs", "a.txt", "2026-10-10")
    with pytest.raises(ValueError):
        rc.confirmar_revision(revisor, nota, tmp_path / "obs", tmp_path / "rev.json")


def test_sin_carpeta_no_falla(tmp_path):
    assert rc.observatorios_vencidos(tmp_path / "no_existe", tmp_path / "rev.json", hoy=HOY) == []
    assert rc.problema_vigencia(tmp_path / "no_existe", tmp_path / "rev.json", hoy=HOY) is None


@pytest.mark.parametrize("texto, tipo", [
    ("escribe a ana.perez@example.com", "correo"),
    ("mi rut es 12.345.678-9", "rut"),
    ("rut 12345678-K", "rut"),
    ("llámame al +56 9 9579 7816", "celular"),
    ("o al 912345678", "celular"),
])
def test_detecta_pii(tmp_path, texto, tipo):
    (tmp_path / "x.jsonl").write_text(json.dumps({"entrada": texto}) + "\n", encoding="utf-8")
    assert rc.buscar_pii(tmp_path / "x.jsonl") == [(1, tipo)]


@pytest.mark.parametrize("texto", [
    "Luna creciente, 30 % iluminada la noche del 2026-12-14.",
    '"tiempo_total_ms": 5925.55, "k": 3',
    "Horario visitas 21:00 hrs | 23:00 hrs",
    '"timestamp": "2026-11-10T00:47:56.682707+00:00"',
])
def test_no_marca_datos_normales(tmp_path, texto):
    (tmp_path / "x.jsonl").write_text(texto + "\n", encoding="utf-8")
    assert rc.buscar_pii(tmp_path / "x.jsonl") == []


def test_problemas_pii_por_archivo(tmp_path):
    (tmp_path / "trazas_agente.jsonl").write_text('{"entrada": "a@b.cl"}\n', encoding="utf-8")
    problemas = rc.problemas_pii(tmp_path)
    assert len(problemas) == 1 and "trazas_agente.jsonl" in problemas[0]


def test_jsonl_reales_del_repo_sin_pii():
    for nombre in rc.JSONL_VERSIONADOS:
        assert rc.buscar_pii(rc.RAIZ / nombre) == [], nombre


def test_vigencia_con_la_estructura_real_del_corpus(tmp_path):
    # Bug 25-11: indexar_corpus buscaba <corpus>/observatorios_turisticos, pero los .txt están en
    # <corpus>/originales/observatorios_turisticos; el control no encontraba nada y dejaba pasar.
    corpus = tmp_path / "Corpus RAG astroturismo"
    crear_txt(corpus / "originales" / "observatorios_turisticos", "eso.txt", "2026-10-10")
    crear_txt(corpus / "originales" / "wikipedia_es", "luna.txt", "2026-10-10")
    problema, aviso = rc.revisar_vigencia_corpus(corpus, "*.txt", tmp_path / "rev.json", hoy=HOY)
    assert problema and "1 .txt de observatorios" in problema and aviso is None
    # también si --corpus apunta a originales/ o a la carpeta misma
    for raiz in (corpus / "originales", corpus / "originales" / "observatorios_turisticos"):
        assert rc.revisar_vigencia_corpus(raiz, "*.txt", tmp_path / "rev.json", hoy=HOY)[0]


def test_sin_carpeta_de_observatorios_avisa_en_vez_de_pasar_en_silencio(tmp_path):
    crear_txt(tmp_path / "corpus" / "wikipedia_es", "luna.txt", "2026-10-10")
    problema, aviso = rc.revisar_vigencia_corpus(tmp_path / "corpus", "*.txt", tmp_path / "rev.json", hoy=HOY)
    assert problema is None
    assert aviso and "no se revisó la vigencia" in aviso


def test_corpus_md_del_repo_no_avisa(tmp_path):
    (tmp_path / "corpus").mkdir()
    assert rc.revisar_vigencia_corpus(tmp_path / "corpus", "*.md", tmp_path / "rev.json", hoy=HOY) == (None, None)
