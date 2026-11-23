"""Tests del Bloque 30: descarga de artículos de Wikipedia (sin red: sesión falsa)."""
from datetime import date

import pytest

from descargar_wikipedia import (
    TituloNoEncontrado,
    armar_documento,
    bajar_articulo,
    guardar,
    main,
    nombre_archivo,
)


class _Respuesta:
    def __init__(self, datos):
        self._datos = datos

    def raise_for_status(self):
        pass

    def json(self):
        return self._datos


class _Sesion:
    """Devuelve una página para "Año luz" (con redirección) y nada para lo demás."""

    def __init__(self):
        self.llamadas = []

    def get(self, url, params, headers, timeout):
        self.llamadas.append(params)
        assert "User-Agent" in headers
        if params.get("list") == "search":
            return _Respuesta({"query": {"search": [{"title": "Parecido 1"}, {"title": "Parecido 2"}]}})
        if params["titles"] in ("Año luz", "año-luz"):
            return _Respuesta({"query": {"pages": [{"title": "Año luz", "extract": " Un año luz es... "}]}})
        return _Respuesta({"query": {"pages": [{"title": params["titles"], "missing": True}]}})


def test_bajar_articulo_existente():
    assert bajar_articulo("año-luz", _Sesion()) == ("Año luz", "Un año luz es...")


def test_titulo_inexistente_sugiere_parecidos_y_no_inventa():
    with pytest.raises(TituloNoEncontrado) as error:
        bajar_articulo("No existe", _Sesion())
    assert error.value.sugerencias == ["Parecido 1", "Parecido 2"]


def test_documento_con_el_mismo_encabezado_del_corpus():
    texto = armar_documento("Año luz", "cuerpo", hoy=date(2026, 11, 23))
    assert texto.splitlines()[:5] == [
        "Título: Año luz",
        "Fuente: https://es.wikipedia.org/wiki/A%C3%B1o_luz",
        "Licencia: CC BY-SA 4.0 (Wikipedia en español)",
        "Descargado: 2026-11-23",
        "",
    ]
    assert nombre_archivo("Año luz") == "Año_luz.txt"


def test_guardar_no_sobrescribe_sin_forzar(tmp_path):
    assert guardar("Año luz", "v1", tmp_path).name == "Año_luz.txt"
    assert guardar("Año luz", "v2", tmp_path) is None
    assert "v1" in (tmp_path / "Año_luz.txt").read_text(encoding="utf-8")
    guardar("Año luz", "v2", tmp_path, forzar=True)
    assert "v2" in (tmp_path / "Año_luz.txt").read_text(encoding="utf-8")


def test_main_sin_carpeta_del_corpus_se_detiene(tmp_path):
    with pytest.raises(SystemExit):
        main(["--destino", str(tmp_path / "no" / "existe"), "Año luz"])
