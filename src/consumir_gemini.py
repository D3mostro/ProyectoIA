import os
import requests
from dotenv import load_dotenv

load_dotenv()

#Constantes
MODELO = "gemini-flash-latest"
URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
API_KEY = os.getenv("GEMINI_API_KEY")

#FUNCION PARA CREAR la solicitud
def armar_peticion(pregunta):#pregunta="Confirma la xonexion con la api"
    if not pregunta:
        raise ValueError("La pregunta no puede estar vacía.")
    url = f"{URL_BASE}{MODELO}:generateContent?key={API_KEY}"#url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=API_KEY"
    body = {
        "contents": [
            {
                "parts": [{"text": pregunta}]}]
    }
    return url, body

#Funcion envía petición
def enviar_peticion(url, body):
    try:
        respuesta = requests.post(url, json=body)
    except requests.RequestException:
        raise RuntimeError("No se pudo conectar con el servicio. Intenta de nuevo en unos momentos.")   
    except requests.Timeout:
        raise RuntimeError("La solicitud ha excedido el tiempo de espera. Intenta de nuevo en unos momentos.")

    try:     
        return respuesta.json()
    except  requests.exceptions.JSONDecodeError:
        raise RuntimeError("La respuesta no es un JSON válido. Intenta de nuevo en unos momentos.")

    if respuesta.status_code != 200:
        raise RuntimeError(f"Error en la solicitud:Status Code: {respuesta.status_code}")        

#Funcion procesa y extraae datos
def extraer_texto(datos):#Gemini devuelve un diccionario con la respuesta, de la cual se extrae el texto
    try:
        return datos["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError("La respuesta no contiene el formato esperado. Intenta de nuevo en unos momentos.")    

def obtener_dato_de_prueba(pregunta):
    url, body = armar_peticion(pregunta)#obtiene la url y el body de la petición
    datos = enviar_peticion(url, body)#obtiene la respuesta de la petición
    return extraer_texto(datos)#obtiene el texto de la respuesta
try:
    print(obtener_dato_de_prueba("funcionas la api correctamente?"))
except (valueError, RuntimeError) as error:
    print("no se pudo completar la operacion: ", error)