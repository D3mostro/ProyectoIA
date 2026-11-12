# Modelo y costo

El modelo de la aplicación es gemini-3.5-flash-lite. Se eligió el 20 de octubre de 2026 por su latencia, de unos 1,5 segundos frente a más de 7 de las otras opciones, y por su costo: 0,30 USD por millón de tokens de entrada y 2,50 USD por millón de salida.

Cada respuesta trae el uso de tokens, y el cliente calcula el costo de cada llamada y lo acumula en la sesión. Un ensayo de 10 consultas seguidas costó menos de un milésimo de dólar.

Este modelo ignora el parámetro temperature, así que la consistencia se consigue con instrucciones de sistema claras y con salida en formato JSON.
