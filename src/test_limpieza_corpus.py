"""Tests de limpieza_corpus.py: sin red ni Gemini (embeddings falsos que cuentan llamadas)."""
from langchain_core.embeddings import DeterministicFakeEmbedding

import limpieza_corpus as lc
from indexar_corpus import chunks_de_documento, conectar, indexar, partir_en_chunks

ENCABEZADO = ("Título: Vicuña (Chile)\nFuente: https://es.wikipedia.org/wiki/Vicu%C3%B1a_(Chile)\n"
              "Licencia: CC BY-SA 4.0 (Wikipedia en español)\nDescargado: 2026-10-10\n\n")
PARRAFO_TURISTA = ("La comuna es conocida internacionalmente como uno de los mejores lugares para la "
                   "observación astronómica, por la limpieza de sus cielos durante casi todo el año.")
PARRAFO_ALCALDE = ("La Municipalidad está dirigida por el alcalde y un concejo municipal elegido cada "
                   "cuatro años, que se reúne en el edificio consistorial frente a la plaza.")


class EmbeddingsContados(DeterministicFakeEmbedding):
    llamadas: int = 0

    def embed_documents(self, textos):
        self.llamadas += len(textos)
        return super().embed_documents(textos)


def articulo(nombre="Vicuña_(Chile)"):
    texto = (ENCABEZADO + PARRAFO_TURISTA + "\n\n== Administración ==\n" + PARRAFO_ALCALDE + "\n\n"
             "== Lugares de interés ==\n" + PARRAFO_TURISTA.replace("comuna", "ciudad") + "\n\n"
             "== Referencias ==\n" + "Instituto Nacional de Estadísticas (2017), censo de población y vivienda. " * 3
             + "\n\n== Enlaces externos ==\nSitio oficial de la municipalidad de Vicuña.\n")
    return f"wikipedia_es/{nombre}.txt", texto


def motivos(fuente, texto):
    partes, ids = chunks_de_documento(fuente, texto, limpiar=False)
    return dict(zip(ids, lc.motivos_de_descarte(texto, [p.page_content for p in partes], fuente)))


def test_quita_encabezado_solo_seccion_de_localidad_y_cola():
    # (un chunk que empieza en una sección útil y termina en "Referencias" se conserva:
    # la sección se decide por dónde empieza el chunk)
    fuente, texto = articulo()
    descartes = []
    partes, ids = chunks_de_documento(fuente, texto, descartes=descartes)
    textos = " ".join(p.page_content for p in partes)
    assert "mejores lugares para la observación" in textos and "La ciudad es conocida" in textos
    assert "alcalde" not in textos and "Sitio oficial" not in textos
    assert {m for _, m, _ in descartes} == {"solo_encabezado", "localidad_fuera_de_tema", "cola_wikipedia"}


def test_los_que_quedan_conservan_id_y_texto():
    fuente, texto = articulo()
    todos, ids_todos = chunks_de_documento(fuente, texto, limpiar=False)
    limpios, ids_limpios = chunks_de_documento(fuente, texto)
    originales = dict(zip(ids_todos, (p.page_content for p in todos)))
    assert len(limpios) < len(todos)
    assert all(originales[i] == p.page_content for i, p in zip(ids_limpios, limpios))


def test_historia_se_conserva_fuera_de_las_localidades():
    fuente, texto = articulo("Astronomía_en_Chile")
    assert "localidad_fuera_de_tema" not in motivos(fuente, texto).values()


def test_seccion_de_localidad_que_habla_de_astronomia_se_conserva():
    texto = (ENCABEZADO + "== Economía ==\nEl astroturismo es hoy una de las actividades económicas "
             "principales, con varios observatorios turísticos abiertos al público durante el año.\n")
    assert set(motivos("wikipedia_es/Vicuña_(Chile).txt", texto).values()) == {"solo_encabezado", None}


def test_ruido_pero_no_campos_cortos_de_las_fichas():
    assert lc._es_ruido("== Morfología ==")
    assert lc._es_ruido("=== https://alarkapin.cl/contacto/")
    assert lc._es_ruido("m\n  −\n  2.5\n  log\n  10\n  {\\displaystyle m}")
    assert not lc._es_ruido("Cómo llegar: De forma particular")
    assert not lc._es_ruido("geográficas son Latitud 29° 15,3' Sur y Longitud 70° 44,3' Oeste.")


def test_solo_encabezado():
    assert lc._es_solo_encabezado(ENCABEZADO)
    assert lc._es_solo_encabezado("Sitio: https://alarkapin.cl/\nDescargado: 2026-10-10")
    assert not lc._es_solo_encabezado(ENCABEZADO + PARRAFO_TURISTA)


def test_fuera_de_tema_general():
    texto = (ENCABEZADO + PARRAFO_TURISTA + "\n\n== En ciencia ficción ==\nEn la saga de videojuegos "
             "una colonia humana entra en guerra con la Tierra en un planeta ficticio de este sistema.\n")
    assert "fuera_de_tema" in motivos("wikipedia_es/Alfa_Centauri.txt", texto).values()


def test_corpus_de_la_app_no_pierde_chunks():
    descartes = []
    partir_en_chunks(descartes=descartes)
    assert descartes == []


def test_reindexar_con_limpieza_solo_borra_y_no_gasta_embeddings(tmp_path):
    fuente, texto = articulo()
    todos, ids_todos = chunks_de_documento(fuente, texto, limpiar=False)
    embeddings = EmbeddingsContados(size=8)
    indexar(todos, ids_todos, embeddings, tmp_path)
    antes = embeddings.llamadas
    limpios, ids_limpios = chunks_de_documento(fuente, texto)
    indexar(limpios, ids_limpios, embeddings, tmp_path)
    assert embeddings.llamadas == antes  # 0 embeddings nuevos
    assert sorted(conectar(embeddings, tmp_path).get()["ids"]) == sorted(ids_limpios)


def test_resumen():
    assert "no se descarta" in lc.resumir_descartes([], 10)
    resumen = lc.resumir_descartes([("wikipedia_es/A-0", "ruido", "x"), ("wikipedia_es/A-3", "ruido", "y")], 10)
    assert "2 de 10 chunks (20 %)" in resumen and "ruido: 2" in resumen and "A (2)" in resumen


def test_otro_idioma():
    assert lc._es_otro_idioma("We use cookies that are essential for accessing our websites and using our services.")
    assert not lc._es_otro_idioma(PARRAFO_TURISTA)
    assert not lc._es_otro_idioma("Dirección Camino Pisco Elqui a Horcón Km 3,5. An oasis of peace where "
                                  "you can rest and enjoy the stars with your family in the valley.")
    assert not lc._es_otro_idioma("International Dark Sky Communities (Comunidades internacionales de cielo oscuro)")
