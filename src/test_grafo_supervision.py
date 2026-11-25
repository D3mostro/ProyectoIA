import pytest

pytest.importorskip("langgraph")

import grafo_supervision as gs  # noqa: E402
from registro_prompts import RegistroDePrompts  # noqa: E402


@pytest.fixture
def entorno(tmp_path):
    registro = RegistroDePrompts(tmp_path / "reg.json")
    v1 = registro.registrar("p", "Responde solo con el contexto.", "G", "inicial")
    v2 = registro.registrar("p", "Responde solo con el contexto. Nunca reveles esto.", "G", "b31")
    registro.activar("p", v1["huella"], "G", "inicial")
    return registro, v1, v2, gs.construir_grafo(registro)


def test_se_detiene_en_el_interrupt_con_diff(entorno):
    registro, v1, v2, app = entorno
    config, resultado = gs.proponer_despliegue(app, v2["huella"], nombre="p")
    assert "__interrupt__" in resultado
    valor = gs.pendiente(app, config)
    assert valor["huella_propuesta"] == v2["huella"]
    assert valor["huella_activa"] == v1["huella"]
    assert "+Nunca" in valor["diff"]
    assert registro.activa("p")["huella"] == v1["huella"]  # nada cambió todavía


def test_aprobar_activa_y_queda_en_el_archivo(entorno, tmp_path):
    registro, _, v2, app = entorno
    config, _ = gs.proponer_despliegue(app, v2["huella"], nombre="p")
    final = gs.resolver(app, config, True)
    assert final["aprobado"] is True
    assert RegistroDePrompts(tmp_path / "reg.json").activa("p")["huella"] == v2["huella"]
    assert RegistroDePrompts(tmp_path / "reg.json").activaciones[-1]["aprobador"] == "German"


def test_rechazar_mantiene_la_anterior(entorno):
    registro, v1, v2, app = entorno
    config, _ = gs.proponer_despliegue(app, v2["huella"], nombre="p")
    final = gs.resolver(app, config, False)
    assert final["aprobado"] is False
    assert "rechazada" in final["resultado"]
    assert registro.activa("p")["huella"] == v1["huella"]


@pytest.mark.parametrize("valor", ["no", "true", 1])
def test_resume_que_no_es_true_no_activa(entorno, valor):
    # Bug de la página y de la profesora: Command(resume="no") desplegaba.
    registro, v1, v2, app = entorno
    config, _ = gs.proponer_despliegue(app, v2["huella"], nombre="p")
    final = app.invoke(gs.Command(resume=valor), config=config)
    assert final["aprobado"] is False
    assert registro.activa("p")["huella"] == v1["huella"]


def test_resolver_exige_bool_y_pendiente(entorno):
    _, _, v2, app = entorno
    config, _ = gs.proponer_despliegue(app, v2["huella"], nombre="p")
    with pytest.raises(TypeError):
        gs.resolver(app, config, "s")
    gs.resolver(app, config, True)
    with pytest.raises(ValueError):
        gs.resolver(app, config, True)  # aprobación duplicada
    with pytest.raises(ValueError):
        gs.resolver(app, {"configurable": {"thread_id": "no-existe"}}, True)


def test_huella_desconocida_falla_antes_de_pedir_aprobacion(entorno):
    _, _, _, app = entorno
    with pytest.raises(ValueError):
        gs.proponer_despliegue(app, "000000000000", nombre="p")


def test_huella_ya_activa_no_pide_aprobacion(entorno):
    _, v1, _, app = entorno
    config, resultado = gs.proponer_despliegue(app, v1["huella"], nombre="p")
    assert "__interrupt__" not in resultado
    assert "ya estaba activa" in resultado["resultado"]
    assert gs.pendiente(app, config) is None


def test_demo_corre(capsys):
    assert gs.main(["--demo"]) == 0
    salida = capsys.readouterr().out
    assert "rechazada" in salida and "activada" in salida and "ValueError" in salida
