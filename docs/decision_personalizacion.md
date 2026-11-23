# Decisión de personalización: asistente de astroturismo en Chile

Fecha: lunes 23 de noviembre de 2026
Método: las cinco preguntas de la lección 20439 (Bloque 30), corrido con [`src/decision.py`](../src/decision.py) antes de redactar este documento. Umbrales: los de la tabla del Paso 1 (lección 20440), salvo una desviación justificada en la pregunta 1.

El método se corrió sobre los dos casos candidatos del proyecto: el **asistente** que responde preguntas (flujo RAG de los Bloques 24 a 28) y el **enrutador**, es decir, la elección de herramienta del agente (Bloques 21 a 23), que es el caso que el Bloque 29 preparó para PEFT con `src/dataset_tuning.py`.

## 1. El problema

**Asistente:** la app tiene que dar horarios, precios y estado de visitas de los observatorios, datos que el modelo no trae o trae desactualizados. En la comparación de modelos del 20-10-2026 (`docs/decision-arquitectura.md`), `gemini-3.5-flash-lite` sin documentos sacó 3/6 en calidad: inventó nombres de observatorios y afirmó "cielo despejado garantizado casi al 100 %".

**Enrutador:** el agente tiene que elegir siempre la herramienta correcta (o ninguna) para cada mensaje, con las mismas etiquetas.

## 2. Resultado del método de decisión

### Respuestas y evidencia

| Pregunta | Asistente | Enrutador | Evidencia |
|---|---|---|---|
| `informacion_cambia_seguido` | **True** | False | Asistente: revisé las fuentes oficiales el 23-11-2026 (detalle abajo). Hubo cambios fechados en septiembre y en octubre de 2026, y ESO abre reservas por trimestre. Enrutador: las 5 intenciones y las 4 herramientas no han cambiado desde el Bloque 21. |
| `problema_de_formato_o_tono` | False | False | Asistente: muestra de 20 respuestas reales (`muestra_formato_b30.jsonl`, 23-11-2026, k = 3, colección `astroturismo_real`), revisada con `src/muestra_formato.py` y además a mano: 0 fallos de 20 (umbral: 3). Las 20 terminaron con `STOP`, ninguna salió cortada y hubo 0 citas inventadas. Las 12 que responden traen cita válida y las 8 restantes dicen que el contexto no alcanza. Enrutador: en `trazas_agente.jsonl` hay 16 turnos (4 consultas × 4 corridas) y los 16 eligen la herramienta correcta. |
| `dataset_grande_y_de_calidad` | False | False | `src/dataset_tuning.py` tiene 15 ejemplos etiquetados (5 intenciones × 3), menos de los 200 del umbral. |
| `presupuesto_limitado_o_iterar_rapido` | True | True | Sin Google Cloud puedo costear **0 entrenamientos**: la Gemini API con API key no ofrece modelos con tuning (documentación oficial, verificado en el Bloque 29), y la única vía, Vertex AI, exige un proyecto de Google Cloud con facturación, que descarté. |
| `necesita_multiples_variantes` | False | False | Un solo asistente, en español, con el mismo comportamiento para turistas, guías y operadores. |

### Verificación de la pregunta 1 con fuentes oficiales (23-11-2026)

| Fuente | Qué cambió y cuándo |
|---|---|
| [ESO, anuncio del 02-09-2026](https://eso.org/public/chile/announcements/annlang26009-es-cl/) | Abrió las reservas de Paranal para octubre a diciembre. Las reservas de cada trimestre abren la primera semana del mes anterior (la próxima, en diciembre). La última visita nocturna de 2026 fue el 10 de octubre. |
| [ALMA, anuncio del 02-10-2026](https://www.almaobservatory.org/en/announcements/during-science-month-alma-opens-its-doors-to-celebrate-15-years-of-observing-the-universe/) | Visitas públicas gratuitas los días 17 y 24 de octubre, con inscripción que abrió el 9 y el 19 de octubre. |
| [ESO, página de visitas a ALMA](https://www.eso.org/public/about-eso/visitors/alma/) | Dice que las visitas turísticas a ALMA no son posibles por ahora, pero más abajo mantiene el horario antiguo de fines de semana: la misma página se contradice. |
| [ALMA, visitas educativas](https://www.almaobservatory.org/en/outreach/alma-site-visit-educational-science-institutional/) | Los cupos de 2026 están completos. Las solicitudes para 2027 abren el 02-01-2027. |
| [ESO, visitas nocturnas (13-03-2026)](https://eso.org/public/chile/announcements/annlang26003-es-cl/) | Se programan cada año, en fechas de eventos, con al menos 8 por año. |

Resultado: hubo cambios fechados en septiembre y en octubre de 2026. En noviembre no encontré ninguno. La única cadencia que las fuentes declaran es trimestral (ESO). Por lo tanto, el umbral de "al menos una vez al mes" se cumple en 2 de los últimos 3 meses, no en los 3. Mantengo True por dos razones: la apertura de ALMA de octubre **no está en mi corpus**. Mi copia de la página de visitas de ALMA (descargada el 10-10) dice "suspendido", aunque ALMA ya había anunciado el 2 de octubre las jornadas del 17 y del 24: la página de visitas y los anuncios no coinciden, y mi corpus solo tiene la página de visitas. Además, la sensibilidad muestra que con False la decisión del asistente no cambia. Los precios de los observatorios privados (por ejemplo, Cancana: $20.000 adultos) no traen fecha, así que no pude medir con qué frecuencia cambian.

### Puntajes

| Técnica | Asistente | Enrutador |
|---|---|---|
| RAG | **5** | **2** |
| PEFT | 1 | 1 |
| Fine-tuning completo | 0 | 0 |
| Resultado | RAG, sin empate | RAG, sin empate |

### Sensibilidad (una respuesta invertida a la vez)

- **Asistente:** ninguna de las cinco inversiones cambia el resultado. Con `informacion_cambia_seguido` en False, RAG sigue ganando (2 contra 1). La decisión es estable.
- **Enrutador:** cuatro de cinco inversiones cambian el resultado. Si `problema_de_formato_o_tono` o `necesita_multiples_variantes` pasan a True, gana PEFT. Si `dataset_grande_y_de_calidad` pasa a True, hay empate a tres. Si `presupuesto_limitado_o_iterar_rapido` pasa a False, las tres técnicas quedan en 0 puntos: es un empate a tres en cero.

## 3. Técnica elegida y por qué

**Elijo RAG, el que ya está construido, y no aplico PEFT ni fine-tuning en este proyecto.**

- **Asistente:** RAG gana 5 a 1 y ninguna respuesta lo mueve. Lo que le falta al modelo es información actualizada, y eso solo lo resuelve una técnica que se actualiza sin reentrenar.
- **Enrutador:** aquí el puntaje no se puede leer de forma literal. RAG gana 2 a 1 solo por los puntos de la pregunta de presupuesto, pero el enrutador no tiene un problema de conocimiento que RAG resuelva. La lectura honesta es que **no hay nada que personalizar**: la llamada a herramientas (`create_tool_calling_agent`) ya acierta en 16 de 16 turnos. El empate a tres en cero de la sensibilidad lo confirma: cuando no hay problema, el método no tiene una opción "no personalizar" y lo muestra como empate. Por eso el dataset de 15 ejemplos del Bloque 29 queda como **conjunto de regresión**: sirve para detectar si un cambio de modelo o de prompt rompe la elección de herramienta, pero no como dataset de entrenamiento.

Por qué descarto PEFT y fine-tuning aunque el Bloque 29 dejó preparado el dataset:

1. No hay evidencia del problema que resolverían: 0 fallos de formato en 16 turnos del enrutador.
2. Aunque la hubiera, hoy no puedo ejecutarlos: 0 entrenamientos posibles sin Google Cloud.
3. Si el formato llegara a fallar, la alternativa más barata no está entre las tres opciones del método: un prompt con ejemplos (few-shot) usando los 15 ejemplos del Bloque 29, o la salida estructurada de la Gemini API (JSON con un esquema y valores permitidos). Ambas iteran en segundos y no requieren entrenar.

**Condición para revisar esta decisión:** volver a correr el método si, en una muestra de 20 mensajes nuevos del enrutador, 3 o más (15 %) eligen mal la herramienta o devuelven una salida inválida, y además la salida estructurada no lo corrige.

## 4. Qué necesitaría para ejecutarla

**RAG (elegido):**
- Hoy: 2374 chunks indexados en la colección `astroturismo_real` (46 archivos .txt, de los cuales 8 son de observatorios turísticos con horarios y precios) y 19 chunks en la colección `astroturismo` (6 archivos .md).
- Falta: un proceso de actualización. Los 8 archivos de observatorios se descargaron el 10-10-2026 y no se han vuelto a bajar en 6 semanas, aunque sus horarios cambian por temporada. Plan: volver a descargar esos 8 archivos una vez al mes (o antes de una demo) y reindexarlos con `src/indexar_corpus.py` (Bloque 25), que salta los chunks ya guardados con el mismo texto y solo vuelve a calcular embeddings de lo que cambió.
- Cobertura del corpus (trabajado el 23-11-2026). En la muestra de 20, el modelo dijo que el contexto no alcanza en 8 respuestas. No es un problema de formato: falta información o la recuperación no la encuentra, y eso se corrige en el RAG, no entrenando el modelo. Agregué 3 artículos de Wikipedia con `src/descargar_wikipedia.py` ("Año luz", "Eclipse solar del 14 de diciembre de 2020", "Astronomía amateur": 115 chunks nuevos, 2488 en total) y repetí esas 8 consultas:
  - Ahora sí responden con cita: el año luz y el eclipse de 2020.
  - Siguen sin respuesta, por falta de corpus: el primer santuario de cielo oscuro (el artículo "Reserva de cielo oscuro" no existe en es.wikipedia) y el mejor momento del mes (ningún archivo dice que la luna nueva sea lo mejor para observar).
  - Siguen sin respuesta, por la recuperación: "¿necesito un telescopio para empezar?". La respuesta está en `Astronomía_amateur.txt`, pero con k = 3 se recuperan chunks de observatorios turísticos.
  - Siguen sin respuesta, porque el modelo no infiere: la Vía Láctea desde una ciudad y Santiago. La recuperación trae `Contaminación_lumínica.txt`, pero el modelo no conecta "cielo urbano" con la pregunta. Con el prompt actual, que exige responder solo con el contexto, es lo esperado.
  - El formato se mantiene: 0 fallos de 20 después de la segunda corrida.

**PEFT (descartado, por si la condición de revisión se cumple):**
- Tengo 15 ejemplos etiquetados y el umbral de "dataset grande" es 200, así que me faltan 185.
- Etiquetarlos a 5 minutos por ejemplo son 15,4 horas, unos USD 385 a USD 25 la hora (`estimar_costo_de_etiquetar(185, 5, 25)`).
- Además haría falta Vertex AI: un proyecto de Google Cloud con facturación activa, la API habilitada, un bucket de Cloud Storage y credenciales.

## Riesgos y límites de esta decisión

- **La pregunta 1 cumple el umbral mensual solo en 2 de los últimos 3 meses.** Además, la tabla dice que se compruebe con la fecha de modificación de mis fuentes, y por ese criterio daría False, porque mi corpus no cambió desde el 10-10. Usé las fuentes oficiales en su lugar (tabla de arriba). La decisión del asistente no depende de esta respuesta.
- **La respuesta cortada era un bug y ya está corregido.** `extraer_texto` (`src/consumir_gemini.py`) leía solo `parts[0]`. En la muestra de 20, 4 respuestas llegaron en 2 partes: con el código anterior, esas 4 se habrían cortado, y la pregunta 2 habría dado True por un bug y no por el modelo. Ahora une todas las partes de texto y descarta las de razonamiento, y el registro guarda `finish_reason` y `partes_texto`.
- **Una respuesta se contradice.** En "¿Por qué el norte de Chile es bueno para observar estrellas?" el modelo dice que el contexto no alcanza y después responde con citas válidas. Cumple los 4 criterios, pero es un caso a vigilar.
- **La muestra del enrutador es chica.** Los 16 turnos son 4 consultas distintas repetidas 4 veces. Demuestran consistencia, pero no cobertura de las 5 intenciones con mensajes variados.
- **El método solo tiene tres opciones (RAG, fine-tuning completo y PEFT).** Le faltan "no personalizar" y "prompt con ejemplos o salida estructurada", que en este proyecto son las respuestas más probables para el enrutador.
