"""Tests de convertir_originales.py: sin red, sin Gemini y sin pypdf (los PDF se simulan)."""
import os
import time

import pytest

import convertir_originales as co

HTML_CON_RUIDO = """<html><head><link rel="canonical" href="https://ejemplo.cl/visitas/">
<style>.x{color:red}</style><script>var a = 1;</script></head><body>
<nav>Inicio Español English Deutsch</nav><header>Menú</header>
<main><h1>Visitas</h1><p>Las visitas son los sábados.</p><p>Reserva con un mes de anticipación.</p>
<form><select><option>Idioma</option></select><button>Enviar</button></form></main>
<footer>© 2026 Todos los derechos</footer></body></html>"""


def test_html_quita_menus_scripts_y_pie():
    texto, canonica = co.texto_de_html(HTML_CON_RUIDO.replace("<main>", "<div>").replace("</main>", "</div>"))
    assert "Las visitas son los sábados." in texto
    assert "Reserva con un mes de anticipación." in texto
    for ruido in ("Deutsch", "Menú", "var a", "color:red", "Idioma", "Enviar", "derechos"):
        assert ruido not in texto
    assert canonica == "https://ejemplo.cl/visitas/"


def test_html_prefiere_main_si_tiene_texto_suficiente():
    relleno = "<p>" + "Dato útil del observatorio. " * 30 + "</p>"
    html = f"<body><div>Texto fuera de main</div><main>{relleno}</main></body>"
    texto, _ = co.texto_de_html(html)
    assert "Texto fuera de main" not in texto
    assert "Dato útil" in texto


def test_html_usa_todo_el_cuerpo_si_main_es_corto():
    texto, _ = co.texto_de_html("<body><div>Contenido real</div><main>poco</main></body>")
    assert "Contenido real" in texto


def test_html_og_url_si_no_hay_canonica():
    _, canonica = co.texto_de_html('<head><meta property="og:url" content="https://gob.cl/a/"></head>')
    assert canonica == "https://gob.cl/a/"


def test_limpiar_quita_cabeceras_repetidas_numeros_de_pagina_y_duplicados():
    texto = "\n".join(["DIARIO OFICIAL", "Texto uno", "1", "DIARIO OFICIAL", "Texto dos",
                       "Página 2 de 8", "DIARIO OFICIAL", "Texto  tres", "Texto  tres"])
    assert co.limpiar(texto) == "Texto uno\nTexto dos\nTexto tres"


def test_limpiar_no_quita_lineas_largas_repetidas():
    larga = "Esta es una línea larga de contenido real que podría repetirse en un documento técnico serio."
    assert co.limpiar("\n".join([larga, "a", larga, "b", larga])).count(larga) == 3


def test_encabezado_igual_al_de_wikipedia_mas_convertido():
    doc = co.armar_documento("T", "https://f", "L", "2026-10-10", "cuerpo", hoy=co.date(2026, 11, 25))
    assert doc.startswith("Título: T\nFuente: https://f\nLicencia: L\nDescargado: 2026-10-10\n"
                          "Convertido: 2026-11-25\n\ncuerpo")


def test_convertir_html_usa_canonica_y_fecha_del_archivo(tmp_path):
    (tmp_path / "pagina.html").write_text(HTML_CON_RUIDO, encoding="utf-8")
    momento = time.mktime((2026, 10, 10, 12, 0, 0, 0, 0, -1))
    os.utime(tmp_path / "pagina.html", (momento, momento))
    curados = {"pagina.html": ("Visitas", None, "L")}
    contenido, caracteres = co.convertir("pagina.html", tmp_path, curados)
    assert "Fuente: https://ejemplo.cl/visitas/" in contenido
    assert "Descargado: 2026-10-10" in contenido
    assert caracteres > 0


def test_convertir_pdf_sin_fuente_cita_el_archivo_local(tmp_path, monkeypatch):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-falso")
    monkeypatch.setattr(co, "texto_de_pdf", lambda ruta: "Texto del PDF")
    contenido, _ = co.convertir("doc.pdf", tmp_path, {"doc.pdf": ("Doc", None, "L")})
    assert "Fuente: archivo local originales/doc.pdf" in contenido
    assert contenido.rstrip().endswith("Texto del PDF")


def test_main_no_sobrescribe_y_forzar_si(tmp_path, monkeypatch, capsys):
    (tmp_path / "pagina.html").write_text(HTML_CON_RUIDO, encoding="utf-8")
    monkeypatch.setattr(co, "CURADOS", {"pagina.html": ("Visitas", None, "L")})
    co.main(["--originales", str(tmp_path)])
    destino = tmp_path / co.SUBCARPETA_DESTINO / "pagina.txt"
    assert destino.exists()
    destino.write_text("editado a mano", encoding="utf-8")
    co.main(["--originales", str(tmp_path)])
    assert destino.read_text(encoding="utf-8") == "editado a mano"
    co.main(["--originales", str(tmp_path), "--forzar"])
    assert destino.read_text(encoding="utf-8").startswith("Título: Visitas")
    assert "ya existe" in capsys.readouterr().out


def test_main_seco_no_escribe(tmp_path, monkeypatch):
    (tmp_path / "pagina.html").write_text(HTML_CON_RUIDO, encoding="utf-8")
    monkeypatch.setattr(co, "CURADOS", {"pagina.html": ("Visitas", None, "L")})
    co.main(["--originales", str(tmp_path), "--seco"])
    assert not (tmp_path / co.SUBCARPETA_DESTINO).exists()


def test_main_avisa_original_faltante(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(co, "CURADOS", {"no_existe.pdf": ("X", None, "L")})
    co.main(["--originales", str(tmp_path)])
    assert "NO ENCONTRADO no_existe.pdf" in capsys.readouterr().out


def test_main_falla_si_no_existe_la_carpeta(tmp_path):
    with pytest.raises(SystemExit):
        co.main(["--originales", str(tmp_path / "nada")])


def test_curados_no_incluye_lo_descartado():
    descartados = ("estrategia_turismo_sostenible_2035.pdf", "aura_idss_solicitud_santuario_2015.pdf",
                   "eso_blog_astroturismo.html", "noirlab_visitas_cerro_tololo.html", "cielos_chile_portal.html")
    assert not set(descartados) & set(co.CURADOS)
    assert not any(r.startswith("licencias_y_terminos/") for r in co.CURADOS)
