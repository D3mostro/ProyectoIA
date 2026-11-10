import logging

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler

from consumir_gemini import MODELO

# basicConfig va en __main__ (no al importar) para no reconfigurar el log de quien importe este módulo.
logger = logging.getLogger(__name__)

MENSAJE_RESPALDO = (
    "Gracias por tu consulta. Un miembro de nuestro equipo te va a responder pronto."
)
MENSAJE_ERROR = "No pudimos procesar tu consulta en este momento. Intenta de nuevo más tarde."

CONSULTAS_EJEMPLO = [
    "Quiero ver la Vía Láctea con mi familia en diciembre. ¿Qué lugar de Chile me recomiendan?",
    "¿Qué telescopio o binoculares conviene llevar para ver Saturno?",
    "¿Cómo llego desde Santiago a San Pedro de Atacama y dónde me alojo?",
]


def construir_paso_clasificacion(modelo):
    # El modelo llega por parámetro (no se crea aquí) para poder reemplazarlo por un mock en las pruebas.
    plantilla = PromptTemplate.from_template(
        "Clasifica la consulta de un turista en UNA sola categoría: "
        "lugar, evento, equipo o logistica. Responde solo con la palabra.\n\n"
        "Consulta: {consulta}"
    )
    # strip("."): el curso solo usa strip().lower(), pero "equipo." (con punto) no coincidiría con ninguna ruta.
    return plantilla | modelo | StrOutputParser() | (lambda texto: texto.strip().strip(".").lower())


def respuesta_de_respaldo(entrada):
    # Con exception_key="error", with_fallbacks agrega la excepción real a la entrada: así queda registrada.
    logger.error("Falló una ruta: %s", entrada["error"], exc_info=entrada["error"])
    return MENSAJE_RESPALDO


def construir_ruta_segura(ruta):
    # Si la ruta lanza cualquier excepción, responde el respaldo en vez de tumbar todo el flujo.
    return ruta.with_fallbacks([RunnableLambda(respuesta_de_respaldo)], exception_key="error")


class CapturadorError(BaseCallbackHandler):
    # Adoptado del código de la profesora (Bloque 20): registra el error de cualquier paso, incluida la
    # clasificación, antes de que un fallback lo tape. Se dispara una vez por cada Runnable anidado que falla,
    # así que un solo error puede dejar varias líneas en el log.
    def on_chain_error(self, error: BaseException, **kwargs):
        logger.warning("Error capturado antes del fallback: [%s] -> %s", type(error).__name__, error)


manejador_error = CapturadorError()


def procesar_consulta_segura(cadena, consulta, session_id=None):
    # Red de seguridad final: cubre también lo que ningún fallback de ruta alcanza (p. ej. la clasificación).
    config = {"callbacks": [manejador_error]}
    if session_id is not None:  # solo el agente con memoria (B22) usa session_id
        config["configurable"] = {"session_id": session_id}
    try:
        return cadena.invoke({"consulta": consulta}, config=config)
    except Exception:
        logger.exception("Error procesando la consulta")
        return MENSAJE_ERROR


def construir_rutas(modelo):
    # Cada ruta es una mini-chain con su propio prompt (no la misma plantilla con otro nombre).
    def ruta(instruccion):
        plantilla = PromptTemplate.from_template(
            "Eres asesor de astroturismo en Chile. " + instruccion
            + " Responde en máximo 4 líneas.\n\nConsulta: {consulta}"
        )
        return construir_ruta_segura(plantilla | modelo | StrOutputParser())

    return {
        "cadena_lugar": ruta("Recomienda destinos y explica por qué sirven para observar el cielo."),
        "cadena_evento": ruta("Indica qué fenómenos o eventos astronómicos conviene mirar y cuándo (en general)."),
        "cadena_equipo": ruta("Aconseja qué equipo o abrigo llevar y cómo usarlo."),
        "cadena_logistica": ruta("Orienta sobre traslado, alojamiento y organización del viaje."),
        "cadena_general": ruta("Responde de forma general y pide un dato más si la consulta no es clara."),
    }


def construir_cadena_con_enrutamiento(
    paso_clasificacion,
    cadena_lugar=None, cadena_evento=None, cadena_equipo=None,
    cadena_logistica=None, cadena_general=None,
):
    # Las rutas que no se entregan quedan como RunnablePassthrough (devuelven la entrada tal cual);
    # así cada prueba puede reemplazar solo la ruta que le interesa.
    cadena_lugar = cadena_lugar or RunnablePassthrough()
    cadena_evento = cadena_evento or RunnablePassthrough()
    cadena_equipo = cadena_equipo or RunnablePassthrough()
    cadena_logistica = cadena_logistica or RunnablePassthrough()
    cadena_general = cadena_general or RunnablePassthrough()

    # assign agrega la clave "categoria" sin perder "consulta", que las rutas necesitan.
    ruteador = RunnableBranch(
        (lambda x: x["categoria"] == "lugar", cadena_lugar),
        (lambda x: x["categoria"] == "evento", cadena_evento),
        (lambda x: x["categoria"] == "equipo", cadena_equipo),
        (lambda x: x["categoria"] == "logistica", cadena_logistica),
        cadena_general,  # ruta por defecto: categoría inesperada
    )
    return RunnablePassthrough.assign(categoria=paso_clasificacion) | ruteador


if __name__ == "__main__":
    # Log a archivo con fecha (adoptado del código de la profesora). error.log queda fuera de Git por "*.log".
    # Sin force=True: basicConfig solo corre aquí, al ejecutar el script, no al importar el módulo.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename="error.log",
        filemode="a",
        encoding="utf-8",
    )
    load_dotenv()
    # Sin temperature: gemini-3.5-flash-lite la ignora (ver README, Bloque 18).
    # timeout: with_fallbacks solo reacciona a excepciones, no a esperas; así una demora se vuelve excepción.
    modelo = ChatGoogleGenerativeAI(model=MODELO, timeout=30)
    cadena = construir_cadena_con_enrutamiento(
        construir_paso_clasificacion(modelo), **construir_rutas(modelo)
    )
    for consulta in CONSULTAS_EJEMPLO:
        print(f"> {consulta}\n{procesar_consulta_segura(cadena, consulta)}\n")
