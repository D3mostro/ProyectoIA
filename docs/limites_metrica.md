# Límites de la métrica de evaluación

## Qué mide

Coincidencia de palabras, contando repeticiones, entre la respuesta de Gemini y una referencia escrita a mano, después de limpiar el texto y quitar palabras vacías (`calcular_metricas` en `src/metricas.py`: precisión, cobertura y F1).

## Modos de falla conocidos

Cada ejemplo sale de `src/resultados_metricas.json` y está calificado en `src/revision_manual.py`.

- **Fechas o lugares equivocados puntúan alto.** "¿Cuándo fue el último eclipse total de Sol visible en Chile?": la respuesta dice "4 de diciembre de 2021, en la Antártida" en vez de "14 de diciembre de 2020, en La Araucanía" y saca F1 = 0.73, el quinto mejor de 20. Las palabras que repiten la pregunta tapan el error.
- **Paráfrasis correctas puntúan bajo.** "¿Cuál es el mejor momento del mes para observar estrellas?" es correcta y completa, pero obtiene F1 = 0.37. En "¿Por qué el norte de Chile es bueno...?" la respuesta usa "baja humedad" en lugar de "aire seco" y "cielos extremadamente despejados" en lugar de "cielos despejados", y obtiene F1 = 0.38.
- **Omitir el dato clave no se distingue de decirlo de otra forma.** "¿Qué es el Telescopio Extremadamente Grande (ELT)?" no menciona el cerro Armazones ni el espejo de 39 m, que es lo útil para un viaje, y saca F1 = 0.28, casi lo mismo que la respuesta correcta sobre el telescopio (0.27). El número no dice cuál de las dos falla.
- **Negaciones.** En las dos consultas con "no" (Vía Láctea en la ciudad y telescopio para empezar), las respuestas mantuvieron la negación. En el Bloque 16, con un ejemplo armado a mano, quitar el "no" dejó un F1 de 0.91: la métrica casi no lo castiga, aunque esta muestra no tuvo un caso real.

## Muestra revisada

- 2026-10-30: 10 de 20 respuestas (`gemini-3.5-flash-lite`, una oración por respuesta), los 5 con menor F1 y los 5 con mayor F1 (`muestra_para_revisar`). Resultado: 1 falso positivo con F1 alto (eclipse), los 5 F1 bajos no tenían errores factuales (a 4 les faltaba algo y solo una omitía el dato clave, la del ELT), y 4 de los 5 F1 altos estaban bien de verdad.

## Recomendación de uso

Usar el F1 promedio solo para comparar versiones del cliente sobre los mismos 20 casos. Leer siempre a mano cualquier respuesta con fechas, lugares o cifras de observatorios, aunque su F1 sea alto, porque son los datos que el turista va a usar para planificar.
