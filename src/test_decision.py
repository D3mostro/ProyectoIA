"""Tests del Bloque 30: método de decisión de personalización (solo Python estándar)."""
import pytest

from decision import (
    CASOS,
    PREGUNTAS,
    analizar_sensibilidad,
    estimar_costo_de_etiquetar,
    generar_esqueleto_decision,
    guardar_esqueleto,
    recomendar_estrategia,
)

CASO2_PAGINA = {
    "informacion_cambia_seguido": False,
    "problema_de_formato_o_tono": True,
    "dataset_grande_y_de_calidad": True,
    "presupuesto_limitado_o_iterar_rapido": False,
    "necesita_multiples_variantes": False,
}
CASO1_PAGINA = {
    "informacion_cambia_seguido": True,
    "problema_de_formato_o_tono": True,
    "dataset_grande_y_de_calidad": False,
    "presupuesto_limitado_o_iterar_rapido": True,
    "necesita_multiples_variantes": False,
}


def test_empate_de_la_pagina_se_reporta():
    puntos, ganadoras = recomendar_estrategia(CASO2_PAGINA)
    assert puntos == {"RAG": 0, "Fine-tuning completo": 3, "PEFT": 3}
    assert ganadoras == ["Fine-tuning completo", "PEFT"]


def test_ganador_claro_de_la_pagina():
    puntos, ganadoras = recomendar_estrategia(CASO1_PAGINA)
    assert puntos == {"RAG": 5, "Fine-tuning completo": 1, "PEFT": 3}
    assert ganadoras == ["RAG"]


def test_sensibilidad_de_la_pagina_cambia_las_cinco():
    filas = analizar_sensibilidad(CASO2_PAGINA)
    assert [f[0] for f in filas] == list(PREGUNTAS)
    assert all(f[3] for f in filas)
    assert filas[0][2] == ["RAG", "Fine-tuning completo", "PEFT"]


@pytest.mark.parametrize("valor", ["si", "no", 1, 0, None])
def test_respuesta_no_bool_se_rechaza(valor):
    respuestas = dict(CASO1_PAGINA, problema_de_formato_o_tono=valor)
    with pytest.raises(TypeError):
        recomendar_estrategia(respuestas)


def test_clave_faltante_o_desconocida_se_rechaza():
    incompleto = dict(CASO1_PAGINA)
    del incompleto["necesita_multiples_variantes"]
    with pytest.raises(KeyError):
        recomendar_estrategia(incompleto)
    with pytest.raises(KeyError):
        recomendar_estrategia(dict(CASO1_PAGINA, informacion_cambia=True))


def test_caso_asistente_rag_estable():
    puntos, ganadoras = recomendar_estrategia(CASOS["asistente"]["respuestas"])
    assert puntos == {"RAG": 5, "Fine-tuning completo": 0, "PEFT": 1}
    assert ganadoras == ["RAG"]
    assert not any(f[3] for f in analizar_sensibilidad(CASOS["asistente"]["respuestas"]))


def test_caso_enrutador_y_sus_puntos_fragiles():
    respuestas = CASOS["enrutador"]["respuestas"]
    puntos, ganadoras = recomendar_estrategia(respuestas)
    assert puntos == {"RAG": 2, "Fine-tuning completo": 0, "PEFT": 1}
    assert ganadoras == ["RAG"]
    cambian = {f[0]: f[2] for f in analizar_sensibilidad(respuestas) if f[3]}
    assert cambian == {
        "problema_de_formato_o_tono": ["PEFT"],
        "dataset_grande_y_de_calidad": ["RAG", "Fine-tuning completo", "PEFT"],
        # todo en 0: el método no tiene la opción "no personalizar" y lo muestra como empate
        "presupuesto_limitado_o_iterar_rapido": ["RAG", "Fine-tuning completo", "PEFT"],
        "necesita_multiples_variantes": ["PEFT"],
    }


def test_costo_de_etiquetar_como_la_pagina():
    horas, costo = estimar_costo_de_etiquetar(8, 5, 25)
    assert round(horas, 2) == 0.67 and round(costo, 2) == 16.67
    horas, costo = estimar_costo_de_etiquetar(500, 4, 20)
    assert round(horas, 2) == 33.33 and round(costo, 2) == 666.67


def test_esqueleto_tiene_cuatro_secciones_y_marca_empate():
    texto = generar_esqueleto_decision("problema", *recomendar_estrategia(CASO2_PAGINA))
    for seccion in ("## 1.", "## 2.", "## 3.", "## 4."):
        assert seccion in texto
    assert "**Empate entre: Fine-tuning completo, PEFT**" in texto
    assert "el puntaje no resolvio el empate solo" in texto


def test_esqueleto_sin_empate_nombra_ganadora():
    texto = generar_esqueleto_decision("problema", *recomendar_estrategia(CASO1_PAGINA))
    assert "Empate" not in texto
    assert "por que RAG responde mejor" in texto


def test_guardar_esqueleto_no_sobrescribe(tmp_path):
    ruta = tmp_path / "docs" / "esqueleto.md"
    assert guardar_esqueleto("primero", str(ruta)) is True
    assert guardar_esqueleto("segundo", str(ruta)) is False
    assert ruta.read_text(encoding="utf-8") == "primero\n"
