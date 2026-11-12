# Pronóstico de nubes

La herramienta pronosticar_nubes consulta el pronóstico de nubosidad de Open-Meteo para un lugar y una fecha. Los lugares aceptados están escritos a mano en el código (Vicuña, Pisco Elqui, Combarbalá, Ovalle y San Pedro de Atacama).

Open-Meteo solo pronostica hasta 16 días hacia adelante, y el último día llega sin datos de noche. Por eso la herramienta rechaza las fechas pasadas y las que están a 16 días o más de hoy, sin llamar a la API: responde que la fecha está fuera del rango de pronóstico.

El agente tiene prohibido completar un pronóstico que no existe con clima histórico o conocimiento general. Si la herramienta no trae datos, lo dice tal cual.
