## Registro del ensayo de rate limit (Bloque 13)

- Fecha: 2026-11-04 (hora de Chile)
- Comando: `python src/cliente.py` (10 consultas seguidas con `consultar_modelo`, modelo `gemini-3.5-flash-lite`, nivel gratuito)
- Peticiones enviadas: 10
- Petición donde apareció el primer 429: ninguna (las 10 respondieron al primer intento)
- Esperas observadas: ninguna; `con_reintento` no tuvo que reintentar
- Duración total: 12,4 s
- Tokens por operación: entre 29 y 58
- Costo total del ensayo: $0.00093 (estimado con `calcular_costo`, precios de entrada y salida por separado)
- Resultado: las 10 peticiones terminaron con éxito. Con 10 peticiones no se alcanza el límite por minuto; para ver el backoff en acción hay que subir el número de peticiones en el `__main__` de `src/cliente.py`. El backoff ante 503 se observó en la comparación de modelos (ver README, "Pruebas de reproducción del entorno"): `gemini-3.8-flash` respondió 503; el reintento con backoff de `enviar_con_reintentos` (`comparar_modelos.py`) actuó y el script terminó sin caerse.
