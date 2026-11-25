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
    assert "AVISO" not in salida


def test_alto_con_supervision_pendiente_falla():
    riesgos = m.calcular(copy.deepcopy(m.mis_riesgos))
    alto = next(r for r in riesgos if r["categoria"] == "ALTO")
    alto["supervision"] = "Revisión mensual (pendiente)"
    assert any("pendiente de construir" in p for p in m.problemas_de_entrega(riesgos))


def test_medio_puede_quedar_pendiente_sin_fallar():
    riesgos = m.calcular(copy.deepcopy(m.mis_riesgos))
    medio = next(r for r in riesgos if r["categoria"] == "MEDIO")
    medio["supervision"] = "PENDIENTE"
    assert m.problemas_de_entrega(riesgos) == []


def test_cada_fila_tiene_id_justificacion_evidencia_y_estado():
    for r in m.mis_riesgos:
        for campo in m.CAMPOS_OBLIGATORIOS:
            assert str(r.get(campo, "")).strip(), (r.get("id"), campo)
    assert len({r["id"] for r in m.mis_riesgos}) == len(m.mis_riesgos)


def test_falta_un_campo_obligatorio_falla():
    riesgos = m.calcular(copy.deepcopy(m.mis_riesgos))
    riesgos[0]["estado"] = ""
    assert any("sin estado" in p for p in m.problemas_de_entrega(riesgos))


def test_evidencia_que_cita_un_test_inexistente_falla():
    riesgos = m.calcular(copy.deepcopy(m.mis_riesgos))
    riesgos[0]["evidencia"] += "; test_que_no_existe_en_ningun_lado"
    assert any("test_que_no_existe_en_ningun_lado" in p for p in m.problemas_de_entrega(riesgos))


def test_exportar_markdown(tmp_path):
    riesgos = m.calcular(copy.deepcopy(m.mis_riesgos))
    archivo = m.exportar_markdown(riesgos, tmp_path / "docs" / "matriz.md")
    texto = archivo.read_text(encoding="utf-8")
    assert "riesgo residual" in texto
    assert texto.count("\n| R") == len(riesgos)
    for r in riesgos:
        assert f"## {r['id']} ({r['categoria']}, {r['puntaje']})" in texto
    assert "no corresponde (MEDIO)" in texto
