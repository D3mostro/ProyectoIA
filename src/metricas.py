import json
import time
from pathlib import Path

from casos_evaluacion import CASOS_EVALUACION
from cliente import consultar_modelo
from pipeline_texto import INSTRUCCION, procesar_texto, vectorizar


def contar_coincidencias(tokens_referencia: list[str], tokens_respuesta: list[str]) -> int:
    """Cuenta las coincidencias sin perder repeticiones (mínimo de cada palabra entre ambos lados)."""
    interseccion = vectorizar(tokens_referencia) & vectorizar(tokens_respuesta)
    return sum(interseccion.values())


def calcular_precision_cobertura_f1(tokens_referencia: list[str], tokens_respuesta: list[str]) -> dict[str, float]:
    """Calcula precisión, cobertura y F1 de dos listas de tokens."""
    if not tokens_respuesta or not tokens_referencia:
        return {"precision": 0.0, "cobertura": 0.0, "f1": 0.0}  # texto vacío: evita la división por cero

    coincidencias = contar_coincidencias(tokens_referencia, tokens_respuesta)
    precision = coincidencias / len(tokens_respuesta)
    cobertura = coincidencias / len(tokens_referencia)

    if precision + cobertura == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * cobertura / (precision + cobertura)

    return {"precision": precision, "cobertura": cobertura, "f1": f1}


def calcular_metricas(texto_referencia: str, texto_respuesta: str) -> dict[str, float]:
    """Procesa los dos textos y calcula sus métricas, para no olvidar procesar_texto en ningún lado."""
    return calcular_precision_cobertura_f1(procesar_texto(texto_referencia), procesar_texto(texto_respuesta))


def resumir_metricas(resultados: list[dict]) -> dict[str, float]:
    """Promedia precisión, cobertura y F1 de una lista de resultados."""
    if not resultados:
        return {"precision_promedio": 0.0, "cobertura_promedio": 0.0, "f1_promedio": 0.0}

    n = len(resultados)
    return {
        "precision_promedio": sum(r["precision"] for r in resultados) / n,
        "cobertura_promedio": sum(r["cobertura"] for r in resultados) / n,
        "f1_promedio": sum(r["f1"] for r in resultados) / n,
    }


def peores_casos(resultados: list[dict], cantidad: int = 3) -> list[dict]:
    """Devuelve los casos con F1 más bajo primero: son los que vale la pena leer antes."""
    return sorted(resultados, key=lambda r: r["f1"])[:cantidad]


def muestra_para_revisar(resultados: list[dict], cantidad: int = 5) -> list[dict]:
    """Junta los casos de F1 más bajo y más alto para leerlos a mano (Bloque 17), sin repetir ninguno."""
    ordenados = sorted(resultados, key=lambda r: r["f1"])
    if len(ordenados) <= 2 * cantidad:
        return ordenados  # con pocos resultados, [:5] + [-5:] contaría dos veces los del medio
    return ordenados[:cantidad] + ordenados[-cantidad:]


ARCHIVO_RESULTADOS = Path(__file__).parent / "resultados_metricas.json"


PAUSA_ENTRE_CONSULTAS = 4  # segundos: con 20 consultas seguidas el nivel gratuito de Gemini responde 429


def guardar_resultados(resultados: list[dict]) -> None:
    ARCHIVO_RESULTADOS.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")


def cargar_resultados() -> list[dict]:
    """Lee lo ya medido para no volver a pagar consultas si una corrida anterior se cortó."""
    if not ARCHIVO_RESULTADOS.exists():
        return []
    return json.loads(ARCHIVO_RESULTADOS.read_text(encoding="utf-8"))


if __name__ == "__main__":
    # 20 consultas con referencia (Bloque 17): las 3 del Bloque 15 más 17 nuevas
    pendientes = {caso["consulta"] for caso in CASOS_EVALUACION}
    resultados = [r for r in cargar_resultados() if r["consulta"] in pendientes]
    ya_medidas = {r["consulta"] for r in resultados}

    for numero, caso in enumerate(CASOS_EVALUACION, start=1):
        if caso["consulta"] in ya_medidas:
            continue
        try:
            respuesta = consultar_modelo(caso["consulta"], system_instruction=INSTRUCCION)
        except RuntimeError as error:
            print(f"[Caso {numero}] Sin respuesta ({error}). Vuelve a correr el script para completarlo.\n")
            continue
        finally:
            time.sleep(PAUSA_ENTRE_CONSULTAS)
        metricas = calcular_metricas(caso["referencia"], respuesta)

        print(f"[Caso {numero}] {caso['consulta']}")
        print(f"[Caso {numero}] Respuesta: {respuesta.strip()}")
        print(f"[Caso {numero}] Precisión {metricas['precision']:.2f} · Cobertura {metricas['cobertura']:.2f} · F1 {metricas['f1']:.2f}\n")

        resultados.append({
            "consulta": caso["consulta"],
            "referencia": caso["referencia"],
            "respuesta": respuesta.strip(),
            **{clave: round(valor, 3) for clave, valor in metricas.items()},
        })
        guardar_resultados(resultados)  # se guarda en cada vuelta: si la API falla a mitad, no se pierde lo ya medido

    orden = [caso["consulta"] for caso in CASOS_EVALUACION]
    resultados.sort(key=lambda r: orden.index(r["consulta"]))
    guardar_resultados(resultados)
    print(f"Casos medidos: {len(resultados)} de {len(CASOS_EVALUACION)}")

    resumen = resumir_metricas(resultados)
    print(f"Promedios: precisión {resumen['precision_promedio']:.2f} · cobertura {resumen['cobertura_promedio']:.2f} · F1 {resumen['f1_promedio']:.2f}")

    print("\nPeores casos primero:")
    for caso in peores_casos(resultados):
        print(f"  F1 {caso['f1']:.2f}  {caso['consulta']}")

    print("\nMuestra para revisar a mano (5 peores y 5 mejores):")
    for caso in muestra_para_revisar(resultados):
        print(f"  F1 {caso['f1']:.2f}  {caso['consulta']}")

    print(f"\nResultados guardados en {ARCHIVO_RESULTADOS.name}")
