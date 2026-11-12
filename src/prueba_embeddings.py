"""Bloque 24, lección 1: primer embedding real con Gemini (demo, no es parte del Core)."""
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from indexar_corpus import MODELO_EMBEDDINGS, describir_error_api

load_dotenv()
if not os.environ.get("GEMINI_API_KEY"):
    raise SystemExit('Falta GEMINI_API_KEY. Defínela en tu archivo ".env" y vuelve a ejecutar.')

embeddings = GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS)

try:
    vector = embeddings.embed_query("¿Estará despejado esta noche en Vicuña?")
except Exception as error:
    print(describir_error_api(error))
    raise SystemExit(1)  # la versión de la profesora seguía y caía en NameError

print("Dimensión:", len(vector))
print("Primeros valores:", vector[:5])
