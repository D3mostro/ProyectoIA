"""Bloque 28: registro de latencia y costo, sin red (embeddings falsos, Chroma en tmp_path, API simulada)."""
import json

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from evaluacion_recuperacion import CASOS_EVALUACION
from indexar_corpus import conectar, indexar, partir_en_chunks
from preflight import entradas_gitignore, preflight_check
from registro_rag import (
    PREGUNTAS_POR_DEFECTO,
    con_cronometro,
    guardar_registro,
    leer_argumentos,
    leer_registros,
    leer_tokens,
    medir,
    registrar_pregunta,
    resumir,
)

EMBEDDINGS = DeterministicFakeEmbedding(size=16)  # sin llamadas a Gemini
CLAVES_PAGINA = {"timestamp", "pregunta", "k", "tiempos_ms", "tiempo_total_ms", "tokens"}


def sin_reintentos(funcion):
    return funcion()


def api_falsa(respuesta="Según el contexto [1], sí.", con_uso=True):
    """Imita enviar_peticion: promptTokenCount crece con el largo del prompt."""
    def enviar(url, body):
        prompt = body["contents"][0]["parts"][0]["text"]
        datos = {"candidates": [{"content": {"parts": [{"text": respuesta}]}}]}
        if con_uso:
            datos["usageMetadata"] = {"promptTokenCount": len(prompt) // 4, "candidatesTokenCount": 20,
                                      "thoughtsTokenCount": 5, "totalTokenCount": len(prompt) // 4 + 25}
        return datos
    return enviar


@pytest.fixture
def vectorstore(tmp_path):
    chunks, ids = partir_en_chunks()
    indexar(chunks, ids, EMBEDDINGS, tmp_path)
    return conectar(EMBEDDINGS, tmp_path)


def registrar(vectorstore, ks=(3,), **kwargs):
    kwargs.setdefault("enviar", api_falsa())
    kwargs.setdefault("reintentar_embed", sin_reintentos)
    kwargs.setdefault("reintentar_generacion", sin_reintentos)
    return registrar_pregunta("¿Qué pasa si nadie aprueba la reserva a tiempo?", vectorstore, EMBEDDINGS, ks, **kwargs)


def test_medir_guarda_el_tiempo_aunque_el_bloque_falle():
    tiempos = {}
    with pytest.raises(RuntimeError):
        with medir(tiempos, "etapa"):
            raise RuntimeError("falla")
    assert tiempos["etapa"] >= 0


def test_registro_tiene_las_claves_de_la_pagina_y_etapas_separadas(vectorstore):
    (r,) = registrar(vectorstore)
    assert CLAVES_PAGINA <= set(r)
    assert list(r["tiempos_ms"]) == ["embed_query", "busqueda_vectorial", "armado_prompt", "generacion"]
    assert r["tiempo_total_ms"] == pytest.approx(sum(r["tiempos_ms"].values()), abs=0.05)
    assert r["coleccion"] == "astroturismo" and r["chunk_size"] == 300 and r["chunk_overlap"] == 50


def test_costo_y_tokens_son_numeros_e_incluyen_razonamiento(vectorstore):
    (r,) = registrar(vectorstore)
    assert r["tokens"]["razonamiento"] == 5
    entrada, salida = r["tokens"]["entrada"], 20 + 5
    assert r["costo_usd"] == pytest.approx(entrada * 0.30e-6 + salida * 2.50e-6, rel=1e-6)
    assert isinstance(r["factor_crecimiento"], float) and r["factor_crecimiento"] > 1


def test_subir_k_sube_los_tokens_de_entrada_con_un_solo_embedding(vectorstore):
    llamadas = []

    class EmbeddingsContados(DeterministicFakeEmbedding):
        def embed_query(self, texto):
            llamadas.append(texto)
            return super().embed_query(texto)

    r3, r6 = registrar_pregunta("reserva del observatorio", vectorstore, EmbeddingsContados(size=16), (3, 6),
                                enviar=api_falsa(), reintentar_embed=sin_reintentos,
                                reintentar_generacion=sin_reintentos)
    assert len(llamadas) == 1  # comparar k no duplica el gasto de embeddings
    assert r6["tokens"]["entrada"] > r3["tokens"]["entrada"]
    assert len(r6["citas"]) == 6 and r3["embed_compartido"] is True


def test_sin_usage_metadata_no_se_cae(vectorstore):
    (r,) = registrar(vectorstore, enviar=api_falsa(con_uso=False))
    assert r["tokens"] is None and r["costo_usd"] is None
    assert leer_tokens({}) is None


def test_citas_inventadas_quedan_en_el_registro(vectorstore):
    (r,) = registrar(vectorstore, enviar=api_falsa("Dato [1] y otro [9]."))
    assert list(r["citas_validas"]) == [1] and r["citas_inventadas"] == [9]


def test_reintento_no_infla_la_latencia_de_generacion(vectorstore):
    intentos = {"n": 0}
    enviar_ok = api_falsa()

    def enviar_con_429(url, body):
        intentos["n"] += 1
        if intentos["n"] == 1:
            raise TimeoutError("HTTP 429")
        return enviar_ok(url, body)

    def reintentar_lento(funcion):
        try:
            return funcion()
        except TimeoutError:
            import time
            time.sleep(0.2)  # la espera entre intentos
            return funcion()

    (r,) = registrar(vectorstore, enviar=enviar_con_429, reintentar_generacion=reintentar_lento)
    assert r["intentos"]["generacion"] == 2
    assert r["espera_reintentos_ms"] >= 200
    assert r["tiempos_ms"]["generacion"] < 100  # solo el intento que funcionó


def test_con_cronometro_separa_trabajo_y_espera():
    resultado, info = con_cronometro(sin_reintentos, lambda: 42)
    assert resultado == 42 and info["intentos"] == 1 and info["espera_s"] >= 0


def test_error_de_generacion_deja_su_linea(vectorstore):
    def enviar_roto(url, body):
        raise ValueError("Credencial inválida")

    (r,) = registrar(vectorstore, enviar=enviar_roto)
    assert r["etapa_error"] == "generacion" and "Credencial" in r["error"]
    assert "busqueda_vectorial" in r["tiempos_ms"]


def test_coleccion_vacia_es_error_de_busqueda(tmp_path):
    vacio = conectar(EMBEDDINGS, tmp_path, "otra_coleccion")
    (r,) = registrar(vacio)
    assert r["etapa_error"] == "busqueda_vectorial" and "SinContexto" in r["error"]


def test_guardar_agrega_una_linea_por_registro_y_se_lee(vectorstore, tmp_path):
    archivo = tmp_path / "registros.jsonl"
    for r in registrar(vectorstore, ks=(3, 6)):
        guardar_registro(r, archivo)
    lineas = archivo.read_text(encoding="utf-8").splitlines()
    assert len(lineas) == 2 and all(CLAVES_PAGINA <= set(json.loads(l)) for l in lineas)
    assert [r["k"] for r in leer_registros(archivo)] == [3, 6]


def test_leer_registros_avisa_linea_rota(tmp_path):
    archivo = tmp_path / "r.jsonl"
    archivo.write_text('{"k": 3}\nno es json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="línea 2"):
        leer_registros(archivo)


def test_resumen_por_k_ignora_errores_y_pruebas(vectorstore):
    registros = registrar(vectorstore, ks=(3, 6)) + registrar(vectorstore, ks=(3,), tipo="prueba")
    registros.append({"k": 3, "error": "x", "tipo": "real"})
    resumen = resumir(registros)
    assert resumen[3]["consultas"] == 1 and resumen[6]["consultas"] == 1
    assert resumen[6]["tokens_entrada_prom"] > resumen[3]["tokens_entrada_prom"]


def test_preguntas_por_defecto_son_del_conjunto_b27():
    preguntas_b27 = {c["pregunta"] for c in CASOS_EVALUACION}
    assert set(PREGUNTAS_POR_DEFECTO) <= preguntas_b27 and len(PREGUNTAS_POR_DEFECTO) >= 3


def test_argumentos():
    args = leer_argumentos(["-k", "3", "6"])
    assert args.k == [3, 6] and args.tipo == "real" and not args.resumen
    with pytest.raises(SystemExit):
        leer_argumentos(["-k", "0"])


# --- preflight ---

def test_gitignore_ignora_comentarios_y_nombres_parecidos(tmp_path):
    gi = tmp_path / ".gitignore"
    gi.write_text("# vectorstore/\nvectorstore_ejemplo/\n.env\n", encoding="utf-8")
    assert "vectorstore" not in entradas_gitignore(gi)  # la página aquí daba OK


def test_preflight_detecta_falsos_ok_de_la_pagina(tmp_path):
    (tmp_path / ".gitignore").write_text("# vectorstore/\nvectorstore_ejemplo/\n", encoding="utf-8")
    (tmp_path / "vectorstore").mkdir()
    problemas = preflight_check(tmp_path, tmp_path / "vectorstore", "astroturismo", tmp_path / "r.jsonl")
    assert any("vectorstore/ no está" in p for p in problemas)
    assert any(".env" in p for p in problemas)
    assert any("no existe" in p for p in problemas)  # la colección


def test_preflight_ok_con_indice_real(tmp_path):
    (tmp_path / ".gitignore").write_text(".env\n/vectorstore/\n", encoding="utf-8")
    chunks, ids = partir_en_chunks()
    indexar(chunks, ids, EMBEDDINGS, tmp_path / "vectorstore")
    (tmp_path / "r.jsonl").write_text('{"k": 3}\n', encoding="utf-8")
    assert preflight_check(tmp_path, tmp_path / "vectorstore", "astroturismo", tmp_path / "r.jsonl") == []


def test_preflight_marca_jsonl_roto(tmp_path):
    (tmp_path / ".gitignore").write_text(".env\nvectorstore/\n", encoding="utf-8")
    (tmp_path / "r.jsonl").write_text('{"k": 3}\n{roto\n', encoding="utf-8")
    problemas = preflight_check(tmp_path, tmp_path / "vectorstore", "astroturismo", tmp_path / "r.jsonl")
    assert any("[2]" in p for p in problemas)


def test_preflight_b32_frena_observatorios_vencidos_y_pii(tmp_path):
    from datetime import date
    (tmp_path / ".gitignore").write_text(".env\nvectorstore/\n", encoding="utf-8")
    obs = tmp_path / "obs"
    obs.mkdir()
    (obs / "eso.txt").write_text("Descargado: 2026-10-10\n\nVisitas los sábados", encoding="utf-8")
    (tmp_path / "trazas_agente.jsonl").write_text('{"entrada": "soy ana@correo.cl"}\n', encoding="utf-8")
    problemas = preflight_check(tmp_path, tmp_path / "vectorstore", "astroturismo", tmp_path / "r.jsonl",
                                observatorios=obs, hoy=date(2026, 11, 25))
    assert any("observatorios sin revisar" in p for p in problemas)
    assert any("trazas_agente.jsonl" in p and "correo" in p for p in problemas)


def test_b32_usa_el_prompt_activo_con_system_instruction(vectorstore):
    from registro_prompts import prompt_activo
    enviados = []
    base = api_falsa()

    def enviar(url, body):
        enviados.append(body)
        return base(url, body)

    (r,) = registrar(vectorstore, enviar=enviar)
    activo = prompt_activo()
    assert r["prompt_huella"] == activo.huella
    assert enviados[0]["systemInstruction"]["parts"][0]["text"] == activo.sistema
    assert enviados[0]["contents"][0]["parts"][0]["text"].startswith(activo.plantilla.split("{contexto}")[0])


def test_gate2_pregunta_libre_no_se_guarda_en_texto(tmp_path):
    from registro_rag import para_guardar, preguntas_de_evaluacion
    libre = {"pregunta": "Soy Ana, ana@correo.cl", "respuesta": "Hola Ana", "k": 3, "costo_usd": 0.1}
    limpio = para_guardar(libre)
    assert "pregunta" not in limpio and "respuesta" not in limpio and limpio["texto_omitido"]
    assert limpio["k"] == 3 and limpio["costo_usd"] == 0.1 and len(limpio["pregunta_huella"]) == 12
    conocida = sorted(preguntas_de_evaluacion())[0]
    assert para_guardar({"pregunta": conocida, "respuesta": "x"})["pregunta"] == conocida


def test_gate2_las_preguntas_por_defecto_y_de_muestra_se_conservan():
    from casos_evaluacion import PREGUNTAS_FAMILIA
    from muestra_formato import consultas_de_la_muestra
    from registro_rag import preguntas_de_evaluacion
    conocidas = preguntas_de_evaluacion()
    assert set(PREGUNTAS_POR_DEFECTO) | set(consultas_de_la_muestra()) | set(PREGUNTAS_FAMILIA) <= conocidas
