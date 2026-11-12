# Errores y cuota de la API

La aplicación usa el nivel gratuito de Gemini, que limita las peticiones por minuto. Cuando se pasa el límite, la API responde 429 (RESOURCE_EXHAUSTED).

El cliente reintenta solo los errores pasajeros, como un tiempo de espera agotado, hasta tres intentos con espera creciente (2 y luego 4 segundos, más un pequeño margen al azar). Una clave inválida no se reintenta, porque reintentar no la arregla: falla de inmediato con un mensaje que pide revisar GEMINI_API_KEY en el archivo .env.

Si un paso de la cadena falla, el resto sigue: cada ruta tiene un respaldo que devuelve un aviso en vez de botar todo el flujo, y el error queda en el log.
