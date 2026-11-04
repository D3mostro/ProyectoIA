from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from consumir_gemini import MODELO

CONSULTA_EJEMPLO = (
    "Quiero ver la Vía Láctea con mi familia en diciembre. "
    "¿Qué lugar de Chile me recomiendan y qué debo llevar?"
)


def construir_cadena(modelo):
    # El modelo llega por parámetro (no se crea aquí) para poder reemplazarlo por un mock en las pruebas.
    # Paso 1: clasifica la consulta en una sola palabra (la categoría también servirá para el enrutamiento del Bloque 19).
    plantilla_clasificacion = PromptTemplate.from_template(
        "Clasifica la consulta de un turista en UNA sola categoría: "
        "lugar, evento, equipo o logistica. Responde solo con la palabra.\n\n"
        "Consulta: {consulta}"
    )
    # Paso 2: redacta la respuesta según la categoría del paso 1.
    plantilla_redaccion = PromptTemplate.from_template(
        "Eres asesor de astroturismo en Chile. La consulta se clasificó como "
        "'{categoria}'. Redacta una respuesta breve (máximo 4 líneas) enfocada en "
        "esa categoría.\n\nConsulta: {consulta}"
    )
    # Paso 3: revisa el borrador contra la consulta original, sin agregar datos nuevos.
    plantilla_revision = PromptTemplate.from_template(
        "Revisa este borrador. Elimina cualquier fecha, cifra o nombre propio que "
        "no aparezca en la consulta original o que no sea un dato general y seguro. "
        "Deja un tono cercano y claro. Devuelve solo el texto final.\n\n"
        "Consulta: {consulta}\n\nBorrador:\n{borrador}"
    )

    paso_clasificacion = (
        plantilla_clasificacion | modelo | StrOutputParser()
        | (lambda texto: texto.strip().lower())
    )
    paso_redaccion = plantilla_redaccion | modelo | StrOutputParser()
    paso_revision = plantilla_revision | modelo | StrOutputParser()

    # assign (Bloque 19) agrega una clave nueva al diccionario sin perder las anteriores: la consulta original
    # sigue viajando junto a la salida de cada paso. Dos diccionarios unidos con "|" no sirven: Python los mezcla.
    return (
        RunnablePassthrough.assign(categoria=paso_clasificacion)
        | RunnablePassthrough.assign(borrador=paso_redaccion)
        | paso_revision
    )


if __name__ == "__main__":
    load_dotenv()
    # Sin temperature: gemini-3.5-flash-lite la ignora y Google la elimina en los modelos nuevos.
    modelo = ChatGoogleGenerativeAI(model=MODELO)
    cadena = construir_cadena(modelo)
    print(cadena.invoke({"consulta": CONSULTA_EJEMPLO}))
