# Memoria y trazas del agente

El agente recuerda la conversación por session_id: si un turista pregunta por la luna del 14 de diciembre y luego pregunta "¿y estará nublado ese día?", el segundo turno usa la fecha del primero. La memoria vive en el proceso y se pierde al cerrar el programa.

Cada consulta se guarda como una línea en trazas_agente.jsonl, con la entrada, las herramientas usadas, sus argumentos, lo que devolvieron y la respuesta final.

Las trazas se revisan cada semana: se busca una reserva sin aprobación registrada, una herramienta repetida sin avanzar, o una cifra de luna o nubes en la respuesta que no aparece en ninguna observación, porque eso significa que el agente la inventó.
