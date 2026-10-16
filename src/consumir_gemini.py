import os
import requests
from dotenv import load_dotenv

load_dotenv()
MODELO = "gemini-flash-latest"
URL_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
API_KEY = os.getenv("GEMINI_API_KEY")
#FUNCION PARA CREAR la solicitud
def armar_peticion(pregunta):#pregunta="Confirma la xonexion con la api"
    url = f"{URL_BASE}{MODELO}:generateContent?key={API_KEY}"#url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key=API_KEY"
    body = {
        "contents": [
            {
                "parts": [{"text": pregunta}]}]
    }
    return url, body

#Funcion envvía petición
def enviar_peticion(url, body):
    respuesta = requests.post(url, json=body)
    return respuesta.json()

#Funcion procesa y extraae datos
def extraer_texto(datos):#Gemini devuelve un diccionario con la respuesta, de la cual se extrae el texto
    return datos["candidates"][0]["content"]["parts"][0]["text"]

def obtener_dato_de_prueba(pregunta):
    url, body = armar_peticion(pregunta)#obtiene la url y el body de la petición
    datos = enviar_peticion(url, body)#obtiene la respuesta de la petición
    return extraer_texto(datos)#obtiene el texto de la respuesta

print(obtener_dato_de_prueba("funcionas la api correctamente?"))

