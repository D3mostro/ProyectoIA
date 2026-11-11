import json
import threading
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import agente_con_herramientas as agente_mod
from agente_con_herramientas import (
    ArgumentosPronostico,
    construir_agente,
    construir_agente_con_memoria,
    construir_prompt_agente,
    consultar_fase_lunar,
    esperar_aprobacion,
    extraer_texto_final,
    porcentaje_iluminado,
    pronosticar_nubes,
    registrar_traza,
    reservar_cupo_observatorio,
)
from enrutamiento import procesar_consulta_segura


# Open-Meteo solo pronostica 16 días: los tests usan fechas relativas a hoy.
MANANA = (date.today() + timedelta(days=1)).isoformat()


def modelo_falso(*respuestas):
    # create_tool_calling_agent llama a bind_tools: el Runnable devuelto entrega las respuestas en orden.
    pendientes = iter(respuestas)
    modelo = MagicMock()
    modelo.bind_tools.return_value = RunnableLambda(lambda _: next(pendientes))
    return modelo


def pide_herramienta(nombre, **argumentos):
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": argumentos, "id": "1"}])


def respuesta_open_meteo(nubes_por_hora):
    respuesta = MagicMock()
    respuesta.json.return_value = {"hourly": {"cloud_cover": nubes_por_hora}}
    return respuesta


def test_agente_usa_fase_lunar():
    ejecutor = construir_agente(modelo_falso(
        pide_herramienta("consultar_fase_lunar", fecha="2026-12-14"),
        AIMessage(content="Habrá poca luna."),
    ))
    with patch.object(agente_mod, "angulo_de_fase", return_value=45.0):
        resultado = ejecutor.invoke({"consulta": "¿Cuánta luna habrá el 14 de diciembre?"})

    accion, observacion = resultado["intermediate_steps"][0]
    assert accion.tool == "consultar_fase_lunar"
    assert "creciente, 15 %" in observacion
    assert resultado["output"] == "Habrá poca luna."


def test_agente_usa_pronostico_de_nubes():
    ejecutor = construir_agente(modelo_falso(
        pide_herramienta("pronosticar_nubes", lugar="Vicuña", fecha=MANANA),
        AIMessage(content="Estará despejado."),
    ))
    with patch("agente_con_herramientas.requests.get", return_value=respuesta_open_meteo([0] * 21 + [10, 20, 30])):
        resultado = ejecutor.invoke({"consulta": "¿Estará despejado esta noche en Vicuña?"})

    accion, observacion = resultado["intermediate_steps"][0]
    assert accion.tool == "pronosticar_nubes"
    assert "20 %" in observacion


def test_agente_encadena_las_dos_herramientas():
    ejecutor = construir_agente(modelo_falso(
        pide_herramienta("consultar_fase_lunar", fecha=MANANA),
        pide_herramienta("pronosticar_nubes", lugar="Vicuña", fecha=MANANA),
        AIMessage(content="Buena noche."),
    ))
    with patch.object(agente_mod, "angulo_de_fase", return_value=200.0), patch(
        "agente_con_herramientas.requests.get", return_value=respuesta_open_meteo([0] * 24)
    ):
        resultado = ejecutor.invoke({"consulta": "¿Es buena noche para observar en Vicuña?"})

    assert [accion.tool for accion, _ in resultado["intermediate_steps"]] == [
        "consultar_fase_lunar",
        "pronosticar_nubes",
    ]


def test_agente_no_usa_herramienta_para_pregunta_general():
    ejecutor = construir_agente(modelo_falso(AIMessage(content="Es una nube de gas y polvo.")))
    resultado = ejecutor.invoke({"consulta": "¿Qué es una nebulosa?"})
    assert resultado["intermediate_steps"] == []


def test_agente_se_detiene_en_el_limite_de_iteraciones():
    ejecutor = construir_agente(modelo_falso(
        *[pide_herramienta("consultar_fase_lunar", fecha="fecha-mala") for _ in range(5)]
    ))
    resultado = ejecutor.invoke({"consulta": "¿Cuánta luna habrá?"})
    assert len(resultado["intermediate_steps"]) == 5
    assert "max iterations" in resultado["output"]


@pytest.mark.parametrize("angulo, esperado", [(0, 0), (90, 50), (180, 100), (270, 50)])
def test_porcentaje_iluminado(angulo, esperado):
    assert porcentaje_iluminado(angulo) == pytest.approx(esperado)


def test_fase_lunar_devuelve_texto_ante_fecha_invalida():
    assert "Fecha no válida" in consultar_fase_lunar.invoke({"fecha": "mañana"})


def test_fase_lunar_devuelve_texto_si_faltan_efemerides():
    with patch.object(agente_mod, "cargar_efemerides", side_effect=OSError("sin red")):
        assert "sin red" in consultar_fase_lunar.invoke({"fecha": "2026-12-14"})


def test_nubes_devuelve_texto_ante_lugar_desconocido():
    observacion = pronosticar_nubes.invoke({"lugar": "Marte", "fecha": MANANA})
    assert "No tengo coordenadas de Marte" in observacion


def test_nubes_devuelve_texto_si_falla_la_red():
    with patch("agente_con_herramientas.requests.get", side_effect=requests.ConnectionError("sin red")):
        observacion = pronosticar_nubes.invoke({"lugar": "Vicuña", "fecha": MANANA})
    assert "No se pudo obtener el pronóstico" in observacion


def test_nubes_ignora_horas_sin_dato():
    with patch("agente_con_herramientas.requests.get", return_value=respuesta_open_meteo([None] * 24)):
        observacion = pronosticar_nubes.invoke({"lugar": "Vicuña", "fecha": MANANA})
    assert "No hay pronóstico" in observacion


def test_extraer_texto_final_acepta_str_y_bloques():
    assert extraer_texto_final("hola") == "hola"
    assert extraer_texto_final([{"type": "text", "text": "ho"}, {"type": "text", "text": "la"}]) == "hola"


# --- Bloque 22: args_schema y memoria ---

@pytest.fixture(autouse=True)
def estado_limpio():
    agente_mod.historiales.clear()
    agente_mod.registro_aprobaciones.clear()


def modelo_que_registra(*respuestas):
    # Igual que modelo_falso, pero guarda el texto de cada prompt para revisar qué historial recibió el modelo.
    pendientes = iter(respuestas)
    prompts = []

    def responder(prompt):
        prompts.append(" | ".join(str(m.content) for m in prompt.to_messages()))
        return next(pendientes)

    modelo = MagicMock()
    modelo.bind_tools.return_value = RunnableLambda(responder)
    return modelo, prompts


def test_nubes_expone_su_esquema_pydantic():
    assert pronosticar_nubes.args_schema is ArgumentosPronostico
    assert "AAAA-MM-DD" in pronosticar_nubes.args["fecha"]["description"]


def test_memoria_recuerda_el_turno_anterior_como_texto():
    # El primer turno responde en bloques, como Gemini: el historial debe guardar el texto unido.
    modelo, prompts = modelo_que_registra(
        AIMessage(content=[{"type": "text", "text": "Habrá "}, {"type": "text", "text": "poca luna."}]),
        AIMessage(content="Sí, esa noche."),
    )
    agente = construir_agente_con_memoria(modelo)
    procesar_consulta_segura(agente, "¿Cuánta luna habrá el 14 de diciembre?", session_id="turista_1")
    resultado = procesar_consulta_segura(agente, "¿Y esa noche estará despejado?", session_id="turista_1")

    assert resultado["output"] == "Sí, esa noche."
    assert "¿Cuánta luna habrá el 14 de diciembre?" in prompts[1]
    assert "Habrá poca luna." in prompts[1]
    assert [m.content for m in agente_mod.historiales["turista_1"].messages][1] == "Habrá poca luna."


def test_memoria_separa_las_sesiones():
    modelo, prompts = modelo_que_registra(AIMessage(content="Luna llena."), AIMessage(content="Hola."))
    agente = construir_agente_con_memoria(modelo)
    procesar_consulta_segura(agente, "¿Cuánta luna habrá?", session_id="turista_1")
    procesar_consulta_segura(agente, "¿Qué me preguntaste antes?", session_id="turista_2")

    assert "Luna llena." not in prompts[1]
    assert len(agente_mod.historiales["turista_2"].messages) == 2


# --- Bloque 23: aprobación humana y traza ---

def test_aprobacion_por_timeout_devuelve_false():
    bloqueo = threading.Event()
    with patch("builtins.input", side_effect=lambda _: bloqueo.wait() or "s"):
        assert esperar_aprobacion("¿Aprobar?", tiempo_limite_segundos=0.1) is False
    bloqueo.set()  # libera el hilo que quedó esperando


@pytest.mark.parametrize("tecleado, esperado", [(" S \n", True), ("n", False), ("", False)])
def test_aprobacion_lee_la_respuesta(tecleado, esperado):
    with patch("builtins.input", return_value=tecleado):
        assert esperar_aprobacion("¿Aprobar?", tiempo_limite_segundos=1) is esperado


def test_aprobacion_sin_entrada_estandar_no_aprueba():
    with patch("builtins.input", side_effect=EOFError):
        assert esperar_aprobacion("¿Aprobar?", tiempo_limite_segundos=1) is False


def test_aprobacion_muestra_encabezado_y_usa_el_limite_global(capsys):
    bloqueo = threading.Event()
    with patch.object(agente_mod, "SEGUNDOS_PARA_APROBAR", 0.1), patch(
        "builtins.input", side_effect=lambda _: bloqueo.wait() or "s"
    ):
        assert esperar_aprobacion("¿Aprobar?") is False
    bloqueo.set()
    assert "=== APROBACIÓN REQUERIDA (tienes 0.1 s" in capsys.readouterr().out


@pytest.mark.parametrize("aprobado, esperado", [(True, "aprobada, 2 cupos"), (False, "no aprobada")])
def test_reserva_pide_aprobacion_y_la_registra(aprobado, esperado):
    argumentos = {"observatorio": "Mamalluca", "fecha": "2026-12-14", "personas": 2}
    with patch.object(agente_mod, "esperar_aprobacion", return_value=aprobado):
        observacion = reservar_cupo_observatorio.invoke(argumentos)

    assert observacion.startswith("RESERVA SIMULADA (no se reservó nada real): ")
    assert esperado in observacion
    assert agente_mod.registro_aprobaciones[0]["argumentos"] == {**argumentos, "observatorio": "Observatorio Mamalluca"}
    assert agente_mod.registro_aprobaciones[0]["aprobado"] is aprobado


def test_agente_usa_la_reserva_con_aprobacion():
    ejecutor = construir_agente(modelo_falso(
        pide_herramienta("reservar_cupo_observatorio", observatorio="Mamalluca", fecha="2026-12-14", personas=2),
        AIMessage(content="Reserva lista."),
    ))
    with patch.object(agente_mod, "esperar_aprobacion", return_value=True) as aprobar:
        resultado = ejecutor.invoke({"consulta": "Reserva 2 cupos en Mamalluca el 14 de diciembre"})

    aprobar.assert_called_once()
    assert resultado["intermediate_steps"][0][0].tool == "reservar_cupo_observatorio"


def test_traza_guarda_una_linea_json_con_texto(tmp_path):
    archivo = tmp_path / "trazas.jsonl"
    ejecutor = construir_agente(modelo_falso(
        pide_herramienta("consultar_fase_lunar", fecha="2026-12-14"),
        AIMessage(content=[{"type": "text", "text": "Poca luna."}]),
    ))
    with patch.object(agente_mod, "angulo_de_fase", return_value=45.0):
        resultado = ejecutor.invoke({"consulta": "¿Cuánta luna habrá?"})
    registrar_traza("turista_1", "¿Cuánta luna habrá?", resultado, archivo=archivo)
    registrar_traza("turista_2", "Hola", {"output": "Hola."}, archivo=archivo)  # sin intermediate_steps

    lineas = [json.loads(linea) for linea in archivo.read_text(encoding="utf-8").splitlines()]
    assert lineas[0]["acciones"][0] == {
        "herramienta": "consultar_fase_lunar",
        "argumentos": {"fecha": "2026-12-14"},
        "observacion": "Luna creciente, 15 % iluminada la noche del 2026-12-14.",
    }
    assert lineas[0]["respuesta_final"] == "Poca luna."
    assert lineas[1]["acciones"] == []


def test_prompt_incluye_la_regla_de_observaciones():
    sistema = construir_prompt_agente().format_messages(consulta="hola", agent_scratchpad=[])[0].content
    assert "no lo completes con conocimiento general ni con datos históricos" in sistema
    assert "Las reservas son simuladas" in sistema


# --- Fix posterior al B23: validaciones antes de aprobar, rango de pronóstico, traza y hora UTC ---

RESERVA_VALIDA = {"observatorio": "Mamalluca", "fecha": "2026-12-14", "personas": 2}


@pytest.mark.parametrize("cambio, esperado", [
    ({"personas": 0}, "personas"),
    ({"personas": 21}, "personas"),
    ({"personas": "muchos"}, "personas"),
    ({"observatorio": "Observatorio de Marte"}, "observatorio desconocido"),
    ({"fecha": "14-12-2026"}, "fecha no válida"),
    ({"fecha": "20261214"}, "fecha no válida"),
])
def test_reserva_invalida_no_pide_aprobacion(cambio, esperado):
    with patch.object(agente_mod, "esperar_aprobacion") as aprobar:
        observacion = reservar_cupo_observatorio.invoke({**RESERVA_VALIDA, **cambio})

    aprobar.assert_not_called()
    assert observacion.startswith("Datos de reserva no válidos, no se pidió aprobación")
    assert esperado in observacion
    assert agente_mod.registro_aprobaciones == []


def test_reserva_normaliza_mayusculas_y_tildes():
    with patch.object(agente_mod, "esperar_aprobacion", return_value=True):
        observacion = reservar_cupo_observatorio.invoke(
            {**RESERVA_VALIDA, "observatorio": "OBSERVATORIO COLLOWÁRA", "personas": "2"}
        )
    assert "2 cupos en Observatorio Collowara" in observacion
    assert agente_mod.registro_aprobaciones[0]["argumentos"]["personas"] == 2


def test_agente_no_se_cae_con_reserva_invalida():
    ejecutor = construir_agente(modelo_falso(
        pide_herramienta("reservar_cupo_observatorio", observatorio="Mamalluca", fecha="2026-12-14", personas=0),
        AIMessage(content="No pude reservar."),
    ))
    with patch.object(agente_mod, "esperar_aprobacion") as aprobar:
        resultado = ejecutor.invoke({"consulta": "Reserva 0 cupos en Mamalluca"})

    aprobar.assert_not_called()
    assert "no se pidió aprobación" in resultado["intermediate_steps"][0][1]
    assert resultado["output"] == "No pude reservar."


@pytest.mark.parametrize("dias", [-1, 16, 40])
def test_nubes_fuera_de_rango_no_llama_a_la_api(dias):
    fecha = (date.today() + timedelta(days=dias)).isoformat()
    with patch("agente_con_herramientas.requests.get") as llamada:
        observacion = pronosticar_nubes.invoke({"lugar": "Vicuña", "fecha": fecha})

    llamada.assert_not_called()
    assert "fuera del rango de pronóstico (hasta 16 días" in observacion


def test_nubes_acepta_el_ultimo_dia_del_rango():
    fecha = (date.today() + timedelta(days=15)).isoformat()
    with patch("agente_con_herramientas.requests.get", return_value=respuesta_open_meteo([0] * 24)) as llamada:
        observacion = pronosticar_nubes.invoke({"lugar": "Vicuña", "fecha": fecha})
    llamada.assert_called_once()
    assert "0 %" in observacion


def test_nubes_fecha_mal_formada_no_llama_a_la_api():
    with patch("agente_con_herramientas.requests.get") as llamada:
        observacion = pronosticar_nubes.invoke({"lugar": "Vicuña", "fecha": "mañana"})
    llamada.assert_not_called()
    assert "Fecha no válida" in observacion


def test_traza_no_rompe_si_falla_el_disco(tmp_path, capsys):
    with patch("builtins.open", side_effect=OSError("disco lleno")):
        registrar_traza("turista_1", "Hola", {"output": "Hola."}, archivo=tmp_path / "trazas.jsonl")
    assert "disco lleno" in capsys.readouterr().err


def test_traza_sin_output_no_falla(tmp_path):
    archivo = tmp_path / "trazas.jsonl"
    registrar_traza("turista_1", "Hola", {}, archivo=archivo)
    assert json.loads(archivo.read_text(encoding="utf-8"))["respuesta_final"] == ""


def test_momento_de_aprobacion_en_utc():
    agente_mod.registrar_aprobacion("reservar_cupo_observatorio", RESERVA_VALIDA, True)
    momento = datetime.fromisoformat(agente_mod.registro_aprobaciones[0]["momento"])
    assert momento.utcoffset() == timedelta(0)
