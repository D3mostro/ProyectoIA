import json
from pathlib import Path

from cliente import consultar_modelo
from pipeline_texto import CASOS, INSTRUCCION, procesar_texto, vectorizar


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


ARCHIVO_RESULTADOS = Path(__file__).parent / "resultados_metricas.json"


def guardar_resultados(resultados: list[dict]) -> None:
    ARCHIVO_RESULTADOS.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    # Mismas consultas y referencias del Bloque 15, ahora con precisión, cobertura y F1
    resultados = []
    for numero, caso in enumerate(CASOS, start=1):
        respuesta = consultar_modelo(caso["consulta"], system_instruction=INSTRUCCION)
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

    resumen = resumir_metricas(resultados)
    print(f"Promedios: precisión {resumen['precision_promedio']:.2f} · cobertura {resumen['cobertura_promedio']:.2f} · F1 {resumen['f1_promedio']:.2f}")

    print("\nPeores casos primero:")
    for caso in peores_casos(resultados):
        print(f"  F1 {caso['f1']:.2f}  {caso['consulta']}")

    print(f"\nResultados guardados en {ARCHIVO_RESULTADOS.name}")
