"""Tests de descargar_oferentes.py: sin red (la API de ArcGIS se simula)."""
import pytest

import descargar_oferentes as do

REGISTRO = {
    "ObjectId": 1, "Nombre": "Canopus astroturismo y educación", "Categoría_principal": "Tour operador",
    "Región": "Tarapacá", "Comuna": "Pica ", "Dirección": "Bajo Miraflores, Sitio g8",
    "Servicios_ofrecidos": "Observación con telescopios, Astrofotografía",
    "Temporada_de_funcionamiento": "Operativo todo el año", "Permiten_mascotas": "Sí",
    "Reestricciones": "N/A", "Procedimiento_para_agendar": "Escribir a reservas@canopus.cl o al +56 9 5755 5679",
    "Página_web": "https://www.canopus.cl/redes/index.php?fbclid=AbC123https://www.canopus.cl/redes/index.php?fbclid=AbC123",
    "Número_celular": "+56957555679", "Correo": "canopusdetarapaca@gmail.com", "Instagram": "@canopus",
}


class Respuesta:
    def __init__(self, datos):
        self.datos = datos

    def raise_for_status(self):
        pass

    def json(self):
        return self.datos


class SesionFalsa:
    """Devuelve la metadata de la capa y luego las páginas indicadas."""

    def __init__(self, paginas, meta=None):
        self.paginas = list(paginas)
        self.meta = meta or {"editingInfo": {"lastEditDate": 1782417668000}}  # 2026-06-25
        self.consultas = []

    def get(self, url, params=None, **_):
        if url.endswith("/query"):
            self.consultas.append(params)
            return Respuesta(self.paginas.pop(0))
        return Respuesta(self.meta)


def test_documento_no_trae_celular_ni_correo():
    doc = do.armar_documento(REGISTRO, "2026-06-25", hoy=do.date(2026, 11, 26))
    assert "57555679" not in doc and "5755 5679" not in doc
    assert "@gmail.com" not in doc and "reservas@canopus.cl" not in doc
    assert "[dato de contacto omitido]" in doc
    for patron in do.PATRONES_PII.values():
        assert not patron.search(doc)


def test_documento_encabezado_y_campos():
    doc = do.armar_documento(REGISTRO, "2026-06-25", hoy=do.date(2026, 11, 26))
    assert doc.startswith("Título: Canopus astroturismo y educación (oferente de astroturismo)\nFuente: ")
    assert "Descargado: 2026-11-26\nActualizado en la fuente: 2026-06-25\n" in doc
    assert "Comuna: Pica\n" in doc
    assert "Permiten mascotas: Sí" in doc
    assert "Restricciones" not in doc  # "N/A" cuenta como vacío
    assert "horarios y cupos no están en esta ficha" in doc


def test_web_sin_rastreo_ni_url_repetida():
    assert do.limpiar_web(REGISTRO["Página_web"]) == "https://www.canopus.cl/redes/index.php"
    assert do.limpiar_web("https://a.cl/?id=3&utm_source=fb") == "https://a.cl/?id=3"
    assert do.limpiar_web("www.sin-protocolo.cl") == "www.sin-protocolo.cl"
    assert do.limpiar_web("  ") is None


@pytest.mark.parametrize("nombre,esperado", [
    ("Observatorio Cruz del Sur", "001_Observatorio_Cruz_del_Sur.txt"),
    ("Paniri Caur - Chiuchiu / Ñandú", "001_Paniri_Caur_Chiuchiu_Nandu.txt"),
    (None, "001_sin_nombre.txt"),
])
def test_nombre_archivo(nombre, esperado):
    assert do.nombre_archivo({"ObjectId": 1, "Nombre": nombre}) == esperado


def test_bajar_pagina_hasta_terminar():
    sesion = SesionFalsa([
        {"features": [{"attributes": {"ObjectId": 1}}, {"attributes": {"ObjectId": 2}}], "exceededTransferLimit": True},
        {"features": [{"attributes": {"ObjectId": 3}}]},
    ])
    registros, actualizado = do.bajar_oferentes(sesion, tam_pagina=2)
    assert [r["ObjectId"] for r in registros] == [1, 2, 3]
    assert actualizado == "2026-06-25"
    assert [c["resultOffset"] for c in sesion.consultas] == [0, 2]
    pedidos = sesion.consultas[0]["outFields"].split(",")
    assert "Número_celular" not in pedidos and "Correo" not in pedidos


def test_error_de_arcgis_se_informa():
    with pytest.raises(RuntimeError, match="error"):
        do.bajar_oferentes(SesionFalsa([], meta={"error": {"code": 400}}))


def test_main_escribe_y_no_sobrescribe(tmp_path, capsys):
    destino = tmp_path / "oferentes"
    do.main(["--destino", str(destino)], sesion=SesionFalsa([{"features": [{"attributes": REGISTRO}]}]))
    archivo = destino / "001_Canopus_astroturismo_y_educacion.txt"
    assert archivo.exists()
    archivo.write_text("editado", encoding="utf-8")
    do.main(["--destino", str(destino)], sesion=SesionFalsa([{"features": [{"attributes": REGISTRO}]}]))
    assert archivo.read_text(encoding="utf-8") == "editado"
    do.main(["--destino", str(destino), "--forzar"], sesion=SesionFalsa([{"features": [{"attributes": REGISTRO}]}]))
    assert archivo.read_text(encoding="utf-8").startswith("Título: Canopus")
    assert "ya existían: 1" in capsys.readouterr().out


def test_main_seco_no_escribe(tmp_path):
    destino = tmp_path / "oferentes"
    do.main(["--destino", str(destino), "--seco"], sesion=SesionFalsa([{"features": [{"attributes": REGISTRO}]}]))
    assert not destino.exists()


def test_main_falla_si_no_existe_el_corpus(tmp_path):
    with pytest.raises(SystemExit):
        do.main(["--destino", str(tmp_path / "no" / "existe")], sesion=SesionFalsa([]))
