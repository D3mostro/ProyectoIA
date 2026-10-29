import os
import requests
import time
import json
import re
from dotenv import load_dotenv

load_dotenv()

#Constantes
MODELO = "gemini-3.5-flash-lite"
URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
API_KEY = os.getenv("GEMINI_API_KEY")

#FUNCION PARA CREAR la solicitud
#system_instruction y response_schema son opcionales: si no se pasan, la petición queda igual que antes (así comparar_modelos.py sigue funcionando)
def armar_peticion(pregunta, modelo=MODELO, system_instruction=None, response_schema=None):#pregunta="Confirma la xonexion con la api"
    if not pregunta:
        raise ValueError("La pregunta no puede estar vacía.")
    url = f"{URL_BASE}{modelo}:generateContent?key={API_KEY}"#url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key=API_KEY"
    body = {
        "contents": [
            {
                "parts": [{"text": pregunta}]}]
    }
    if system_instruction:
        # Ojo: el campo JSON es "systemInstruction" (camelCase); "system_instruction" da HTTP 400 "Unknown name"
        body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    if response_schema is not None:
        # Salida estructurada: la API aplica el esquema al generar (no es solo una sugerencia en el texto)
        body["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema,
        }
    return url, body

#Funcion envía petición
def enviar_peticion(url, body):
    try:
        respuesta = requests.post(url, json=body, timeout=30)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        # Fallas momentáneas de red: TimeoutError para que con_reintento (cliente.py) las reintente
        raise TimeoutError("No se pudo contactar al proveedor. Puede ser momentáneo.")

    # Gemini responde una key inválida con 400 + "API_KEY_INVALID" (no con 401/403)
    if respuesta.status_code in(401, 403) or (respuesta.status_code == 400 and "API_KEY_INVALID" in respuesta.text):
        raise ValueError("Credencial inválida o sin permiso. Revisa tu GEMINI_API_KEY")    
    if respuesta.status_code >= 500:
        raise TimeoutError("El proveedor del servicio no pudo procesar la peticion. Intenta de nuevo en unos momentos.")
    if respuesta.status_code == 429:
        segundos = respuesta.headers.get("retry-After")
        mensaje = f"Límite excedido. Reintentar en {segundos}s." if segundos else "Límite excedido."
        raise TimeoutError(mensaje)

    if respuesta.status_code != 200:
        raise RuntimeError(f"Error en la solicitud:Status Code: {respuesta.status_code}")

    try:
        return respuesta.json()
    except requests.exceptions.JSONDecodeError:
        raise RuntimeError("La respuesta no es un JSON válido. Intenta de nuevo en unos momentos.")

#Funcion procesa y extraae datos
def extraer_texto(datos):#Gemini devuelve un diccionario con la respuesta, de la cual se extrae el texto
    try:
        return datos["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError("La respuesta no contiene el formato esperado. Intenta de nuevo en unos momentos.")


#Funcion limpia y valida la respuesta JSON del modelo (Bloque 10)
def parsear_respuesta_json(texto, claves_requeridas=None):
    texto_limpio = texto.strip()
    texto_limpio = re.sub(r"^```(?:json)?\s*", "", texto_limpio)#quita ```json o ``` al inicio
    texto_limpio = re.sub(r"\s*```$", "", texto_limpio)#quita ``` al final
    try:
        datos = json.loads(texto_limpio)
    except json.JSONDecodeError:
        raise ValueError("La respuesta del modelo no es JSON válido.")
    if not isinstance(datos, dict):
        raise ValueError("La respuesta no es un objeto JSON.")
    if claves_requeridas:
        faltantes = [c for c in claves_requeridas if c not in datos]
        if faltantes:
            raise ValueError(f"La respuesta no tiene la estructura esperada. Faltan: {faltantes}")
    return datos


def obtener_dato_de_prueba(pregunta, system_instruction=None):
    url, body = armar_peticion(pregunta, system_instruction=system_instruction)#obtiene la url y el body de la petición

    inicio = time.perf_counter() #Empezar el contador
    datos = enviar_peticion(url, body)#obtiene la respuesta de la petición (lo que se mide)
    fin = time.perf_counter() #Termina el contador
    latencia = fin - inicio
    print(f"Latencia: {latencia:.2f} segundos")


    costo = calcular_costo(datos, 0.30 / 1_000_000, 2.50 / 1_000_000)#precios de gemini-3.5-flash-lite por token; calcula el costo de la petición
    print(f"Costo de la petición: ${costo:.6f} USD")
    return extraer_texto(datos)#obtiene el texto de la respuesta


#def calcular_costo(datos: dict, precio_entrada: float,):
def calcular_costo(datos, precio_entrada, precio_salida):# Implementar lógica para calcular el costo
    
    #datos = {dict... "usageMetadata"}
    uso = datos["usageMetadata"]
    tokens_entrada = uso["promptTokenCount"]
    tokens_salida = uso["candidatesTokenCount"]
    tokens_razonamiento = uso.get("thoughtsTokenCount", 0)
    costo = (tokens_entrada * precio_entrada) + ((tokens_salida + tokens_razonamiento) * precio_salida)
    
    return costo


if __name__ == "__main__":  # solo corre al ejecutar este archivo, no al importarlo
    try:
        system_instruction = (
            "Eres un asistente de astroturismo en Chile. "
            "Responde siempre en español y en un máximo de 3 oraciones."
        )
        print(obtener_dato_de_prueba(
            "Como estara el cielo en santiago mañana para ver las estrellas como astroturismo",
            system_instruction,
        ))
    except (ValueError, RuntimeError, TimeoutError) as error:
        print("no se pudo completar la operacion: ", error)