"""Tests de auditar_corpus.py: archivos falsos en tmp_path, sin red."""
import auditar_corpus as ac

ENCABEZADO = "Título: T\nFuente: https://f.cl\nLicencia: L\nDescargado: 2026-11-25\n\n"
CUERPO = "El observatorio abre de martes a domingo y las visitas se reservan con una semana de anticipación. " * 4


def escribir(carpeta, nombre, texto):
    ruta = carpeta / "originales" / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")


def mensajes(hallazgos, nombre):
    return [f"{n} {m}" for n, m in hallazgos.get(f"originales/{nombre}", [])]


def test_archivo_correcto_no_tiene_hallazgos(tmp_path):
    escribir(tmp_path, "a.txt", ENCABEZADO + CUERPO)
    assert ac.auditar(tmp_path) == {}
    assert "sin problemas" in ac.formatear({})


def test_sin_fecha_o_sin_fuente_es_error(tmp_path):
    escribir(tmp_path, "sin_fecha.txt", "Título: T\nFuente: https://f.cl\nLicencia: L\n\n" + CUERPO)
    escribir(tmp_path, "sin_fuente.txt", "Título: T\nLicencia: L\nDescargado: 2026-11-25\n\n" + CUERPO)
    hallazgos = ac.auditar(tmp_path)
    assert any("Descargado" in m for m in mensajes(hallazgos, "sin_fecha.txt"))
    assert any(m.startswith("ERROR") and "Fuente" in m for m in mensajes(hallazgos, "sin_fuente.txt"))
    assert ac.hay_errores(hallazgos)


def test_formato_de_observatorios_con_sitio_no_es_error(tmp_path):
    escribir(tmp_path, "obs.txt", "Sitio: https://obs.cl/\nDescargado: 2026-11-25 (revisado en vivo)\n\n" + CUERPO)
    hallazgos = ac.auditar(tmp_path)
    assert not ac.hay_errores(hallazgos)
    assert mensajes(hallazgos, "obs.txt") == ["AVISO falta en el encabezado: Título, Licencia"]


def test_texto_roto_y_duplicado_exacto_son_error(tmp_path):
    escribir(tmp_path, "roto.txt", ENCABEZADO + CUERPO + " ObservaciÃ³n astronÃ³mica.")
    escribir(tmp_path, "a.txt", ENCABEZADO + CUERPO)
    escribir(tmp_path, "b.txt", ENCABEZADO.replace("T\n", "Otro\n") + CUERPO)
    hallazgos = ac.auditar(tmp_path)
    assert any("codificación" in m for m in mensajes(hallazgos, "roto.txt"))
    assert mensajes(hallazgos, "b.txt") == ["ERROR mismo texto que originales/a.txt"]


def test_pii_es_aviso_y_se_muestra_enmascarada(tmp_path):
    escribir(tmp_path, "c.txt", ENCABEZADO + CUERPO + " Escribe a reservas@observatorio.cl o al +56 9 5755 5679.")
    hallazgos = ac.auditar(tmp_path)
    texto = " ".join(mensajes(hallazgos, "c.txt"))
    assert not ac.hay_errores(hallazgos)
    assert "correo" in texto and "celular" in texto and "reservas@observatorio.cl" not in texto


def test_corto_ingles_y_casi_duplicado_son_aviso(tmp_path):
    escribir(tmp_path, "corto.txt", ENCABEZADO + "Poco texto.")
    escribir(tmp_path, "en.txt", ENCABEZADO + "We use cookies that are essential for the use of this website and of our services. " * 5)
    escribir(tmp_path, "a.txt", ENCABEZADO + CUERPO)
    escribir(tmp_path, "casi.txt", ENCABEZADO + CUERPO + " Además hay cafetería.")
    hallazgos = ac.auditar(tmp_path)
    assert not ac.hay_errores(hallazgos)
    assert any("corto" in m for m in mensajes(hallazgos, "corto.txt"))
    assert any("inglés" in m for m in mensajes(hallazgos, "en.txt"))
    assert any("casi igual a originales/a.txt" in m for m in mensajes(hallazgos, "casi.txt"))


def test_main_devuelve_1_con_errores(tmp_path, capsys):
    escribir(tmp_path, "sin_fecha.txt", "Título: T\nFuente: f\nLicencia: L\n\n" + CUERPO)
    assert ac.main(["--corpus", str(tmp_path)]) == 1
    assert "1 errores" in capsys.readouterr().out
