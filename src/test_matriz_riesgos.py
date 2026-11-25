import copy

import pytest

import matriz_riesgos as m


def test_cortes_de_categoria():
    assert m.categoria_riesgo(15) == "CRITICO"
    assert m.categoria_riesgo(14) == "ALTO"
    assert m.categoria_riesgo(9) == "ALTO"
    assert m.categoria_riesgo(8) == "MEDIO"
    assert m.categoria_riesgo(4) == "MEDIO"
    assert m.categoria_riesgo(3) == "BAJO"


@pytest.mark.parametrize("p, i", [(0, 3), (6, 3), (3, 0), (2.5, 3), (True, 3), ("4", 3)])
def test_puntaje_rechaza_valores_fuera_de_rango(p, i):
    with pytest.raises(ValueError):
        m.puntaje_riesgo(p, i)


def test_al_limite_igual_que_la_leccion():
    # Los 5 ejemplos de la lección salen [AL LIMITE] y el contraste 1x1 no.
    for p, i in [(4, 4), (3, 5), (3, 3), (2, 4), (3, 2)]:
        assert m.esta_en_el_limite(p, i)
    assert not m.esta_en_el_limite(1, 1)


def test_matriz_propia_cumple_la_entrega():
    riesgos = m.calcular(copy.deepcopy(m.mis_riesgos))
    assert m.problemas_de_entrega(riesgos) == []
    assert len(riesgos) >= 5
    assert riesgos[0]["puntaje"] >= riesgos[-1]["puntaje"]


def test_detecta_alto_sin_supervision():
    riesgos = m.calcular(copy.deepcopy(m.mis_riesgos))
    alto = next(r for r in riesgos if r["categoria"] == "ALTO")
    alto["supervision"] = "  "
    assert any("sin punto de supervisión" in p for p in m.problemas_de_entrega(riesgos))


def test_detecta_matriz_sin_riesgos_severos_y_corta():
    riesgos = m.calcular([
        {"riesgo": "x", "probabilidad": 1, "impacto": 1, "mitigacion": "Bloque 1"},
        {"riesgo": "y", "probabilidad": 2, "impacto": 1, "mitigacion": "nada"},
    ])
    problemas = m.problemas_de_entrega(riesgos)
    assert any("al menos 5" in p for p in problemas)
    assert any("ALTO o CRITICO" in p for p in problemas)
    assert any("Menos de 2" in p for p in problemas)
    assert any("misma categoría" in p for p in problemas)


def test_main_validar_sale_cero(capsys):
    assert m.main(["--validar"]) == 0
    salida = capsys.readouterr().out
    assert "Entrega OK" in salida
    assert "AVISO: supervisión pendiente" in salida
