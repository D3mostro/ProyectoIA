import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

#Constantes
MODELO = "gemini-flash-latest"
URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
API_KEY = os.getenv("GEMINI_API_KEY")

#FUNCION PARA CREAR la solicitud
def armar_peticion(pregunta, modelo=MODELO):#pregunta="Confirma la xonexion con la api"
    if not pregunta:
        raise ValueError("La pregunta no puede estar vacía.")
    url = f"{URL_BASE}{modelo}:generateContent?key={API_KEY}"#url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=API_KEY"
    body = {
        "contents": [
            {
                "parts": [{"text": pregunta}]}]
    }
    return url, body

#Funcion envía petición
def enviar_peticion(url, body):
    try:
        respuesta = requests.post(url, json=body, timeout=30)
    except requests.Timeout:
        raise RuntimeError("La solicitud ha excedido el tiempo de espera. Intenta de nuevo en unos momentos.")
    except requests.RequestException:
        raise RuntimeError("No se pudo conectar con el servicio. Intenta de nuevo en unos momentos.")

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


def obtener_dato_de_prueba(pregunta):
    url, body = armar_peticion(pregunta)#obtiene la url y el body de la petición

    inicio = time.perf_counter() #Empezar el contador
    datos = enviar_peticion(url, body)#obtiene la respuesta de la petición (lo que se mide)
    fin = time.perf_counter() #Termina el contador
    latencia = fin - inicio
    print(f"Latencia: {latencia:.2f} segundos")


    costo = calcular_costo(datos, 0.00000075, 0.00000375)#calcula el costo de la petición
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
        print(obtener_dato_de_prueba("Como estara el cielo en santiago mañana para ver las estrellas como astroturismo"))
    except (ValueError, RuntimeError) as error:
        print("no se pudo completar la operacion: ", error)
