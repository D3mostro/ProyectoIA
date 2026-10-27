import json
import sys
import time
from datetime import date
from pathlib import Path

# El código del proyecto vive en src/: se agrega al path para poder importarlo
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from consumir_gemini import MODELO, armar_peticion, extraer_texto, parsear_respuesta_json
from comparar_modelos import enviar_con_reintentos  # reintenta con backoff si Google responde 429 o 503
from banco import BANCO_DE_PROMPTS

INTENTOS = 3  # mínimo pedido por la actividad; más repeticiones = medida más confiable
PAUSA = 2  # segundos entre peticiones, para no gatillar el error 429
RUTA_REGISTRO = Path(__file__).resolve().parent / "registro_resultados.json"


def probar_prompt(prompt):
    exitos = 0
    errores = []
    for intento in range(1, INTENTOS + 1):
        try:
            url, body = armar_peticion(
                prompt["ejemplo_usuario"],
                system_instruction=prompt["system_instruction"],
                response_schema=prompt.get("response_schema"),
            )
            datos_respuesta, _ = enviar_con_reintentos(url, body)
            texto = extraer_texto(datos_respuesta)
            parsear_respuesta_json(texto, prompt["claves_esperadas"])
            exitos += 1
        except (ValueError, RuntimeError) as error:
            # ValueError: JSON inválido o sin las claves. RuntimeError: fallo de red/HTTP/formato de la API
            print(f"  intento {intento} falló: {error}")
            errores.append(f"intento {intento}: {error}")
        time.sleep(PAUSA)
    return exitos, errores


def main():
    registro = {"modelo": MODELO, "fecha": date.today().isoformat(), "prompts": {}}
    for prompt in BANCO_DE_PROMPTS:
        exitos, errores = probar_prompt(prompt)
        print(f"{prompt['nombre']}: {exitos}/{INTENTOS} respuestas válidas")
        registro["prompts"][prompt["nombre"]] = {
            "exitos": exitos,
            "intentos": INTENTOS,
            "errores": errores,
        }
        if exitos < 2:
            print(f"  ⚠ {prompt['nombre']} tiene menos de 2/{INTENTOS}: ajusta su prompt y vuelve a medir.")

    with open(RUTA_REGISTRO, "w", encoding="utf-8") as archivo:
        json.dump(registro, archivo, indent=2, ensure_ascii=False)
    print(f"\nRegistro guardado en {RUTA_REGISTRO}")


if __name__ == "__main__":
    main()
