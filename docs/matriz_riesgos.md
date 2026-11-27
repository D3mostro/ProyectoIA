# Matriz de riesgos del asistente de astroturismo

Generado por `python src/matriz_riesgos.py --exportar`; no editar a mano.

Probabilidad (P) e impacto (I) de 1 a 5, como **riesgo residual**: con los controles que existen hoy en el código. Puntaje = P x I. BAJO 1-3, MEDIO 4-8, ALTO 9-14, CRITICO 15-25 (cortes del curso). AL LIMITE: mover P o I un punto cambia la categoría; esas filas conviene revisarlas con otra persona.

| ID | Riesgo | P | I | Puntaje | Categoría | Al límite |
|---|---|---:|---:|---:|---|---|
| R1 | Respuesta con horarios o cupos de visita vencidos porque los .txt de observatorios no se vuelven a descargar | 4 | 3 | 12 | ALTO | Sí |
| R2 | Texto libre del usuario (pregunta y respuesta) en trazas_agente.jsonl y registros_latencia_costo.jsonl, que están versionados y se suben a GitHub | 3 | 3 | 9 | ALTO | Sí |
| R6 | Cuota de Gemini agotada (error 429) porque las dos carpetas del proyecto usan la misma clave | 4 | 2 | 8 | MEDIO | Sí |
| R7 | Retiro de gemini-3.5-flash-lite o de gemini-embedding-001 sin aviso, que obliga a cambiar el modelo y reindexar los 3948 chunks | 2 | 4 | 8 | MEDIO | Sí |
| R8 | El asistente responde 'el contexto no alcanza' en preguntas comunes de turistas (5 de 20 en la muestra del Bloque 30) | 4 | 2 | 8 | MEDIO | Sí |
| R3 | Se activa una versión del prompt sin revisar su diff, o con un registro_prompts.json editado a mano | 2 | 3 | 6 | MEDIO | Sí |
| R5 | Inyección de prompt vía texto de Wikipedia descargado al corpus con descargar_wikipedia.py | 2 | 3 | 6 | MEDIO | Sí |
| R9 | El enrutador elige mal la intención con mensajes distintos a las 4 consultas ya probadas | 3 | 2 | 6 | MEDIO | Sí |
| R4 | Reserva de cupo en observatorio ejecutada sin una aprobación humana real | 2 | 2 | 4 | MEDIO | Sí |

## R1 (ALTO, 12). Respuesta con horarios o cupos de visita vencidos porque los .txt de observatorios no se vuelven a descargar

- **Justificación:** Los 8 .txt de observatorios son del 2026-10-10 y los sitios cambian horarios por temporada; el asistente responde con lo indexado.
- **Mitigación:** Respuesta obligada a citar la fuente [n] (Bloque 26) y reindexado incremental que salta chunks sin cambios (Bloque 25)
- **Punto de supervisión:** indexar_corpus.py y preflight.py se detienen si un .txt de observatorio tiene más de 30 días hasta que una persona revisa los sitios y lo registra con revision_corpus.py --confirmar (revision_observatorios.json, Bloque 32)
- **Evidencia:** test_vencidos_y_vigentes; test_confirmacion_humana_libera_y_caduca; test_preflight_b32_frena_observatorios_vencidos_y_pii
- **Estado:** Control integrado. Falta la primera revisión real de los 8 sitios.
- **Nota:** El control frena el reindexado, no las respuestas: si nunca se reindexa, el índice viejo sigue respondiendo. Por eso la probabilidad queda en 4

## R2 (ALTO, 9). Texto libre del usuario (pregunta y respuesta) en trazas_agente.jsonl y registros_latencia_costo.jsonl, que están versionados y se suben a GitHub

- **Justificación:** Esos .jsonl guardan la pregunta y la respuesta completas y son evidencia de las entregas B23, B28 y B30, así que se suben al repo.
- **Mitigación:** logging_rag.py registra solo largo y huella, nunca el texto (Bloque 31); desde el Gate 2, registro_rag.guardar_registro guarda en texto solo las preguntas fijas de evaluación y el agente guarda largo y huella del mensaje y de la respuesta; preflight.py falla si un .jsonl versionado tiene correo, RUT o celular (revision_corpus.py, Bloque 32)
- **Punto de supervisión:** German corre preflight.py antes de cada push (él hace siempre el push a mano) y limpia la línea que marque
- **Evidencia:** test_detecta_pii; test_no_marca_datos_normales; test_jsonl_reales_del_repo_sin_pii; test_logs_no_guardan_texto_de_pregunta_ni_respuesta; test_3b_registros_versionados_sin_texto_libre; test_gate2_pregunta_libre_no_se_guarda_en_texto
- **Estado:** Integrado en el código (Gate 2). Las líneas viejas de trazas_agente.jsonl siguen con texto (sin PII real según preflight). Las respuestas a preguntas de evaluación pueden citar contactos del corpus.
- **Nota:** Los .jsonl no se sacan del repo: son evidencia de las entregas B23, B28 y B30. El chequeo no detecta nombres propios: por eso sigue en probabilidad 3

## R6 (MEDIO, 8). Cuota de Gemini agotada (error 429) porque las dos carpetas del proyecto usan la misma clave

- **Justificación:** Pasó en la indexación del B25 (2374 chunks) y las dos carpetas comparten la cuota gratuita.
- **Mitigación:** Reintentos con espera tope de 120 s y salto de chunks ya indexados (Bloque 25); tokens y costo reales por consulta (Bloque 28)
- **Punto de supervisión:** no corresponde (MEDIO)
- **Evidencia:** test_con_reintentos_reintenta_429_con_backoff_exponencial; test_indexar_salta_chunks_ya_guardados_y_reembebe_los_cambiados
- **Estado:** Integrado.

## R7 (MEDIO, 8). Retiro de gemini-3.5-flash-lite o de gemini-embedding-001 sin aviso, que obliga a cambiar el modelo y reindexar los 3948 chunks

- **Justificación:** El curso ya vio deprecarse text-embedding-004 en el Bloque 28.
- **Mitigación:** preflight.py revisa que la colección exista y tenga chunks (Bloque 28); no detecta un modelo retirado (falta)
- **Punto de supervisión:** no corresponde (MEDIO)
- **Evidencia:** test_preflight_ok_con_indice_real
- **Estado:** Sin control del modelo: revisar la página de modelos de Gemini a mano.

## R8 (MEDIO, 8). El asistente responde 'el contexto no alcanza' en preguntas comunes de turistas (5 de 20 en la muestra del Bloque 30)

- **Justificación:** Medido: 5 de 20 en la muestra del B30 después de ampliar el corpus (2 temas faltan, 1 por k=3, 2 el modelo no infiere).
- **Mitigación:** Corpus ampliado con descargar_wikipedia.py (Bloque 30) y evaluación de k=3 contra k=5 (Bloque 27)
- **Punto de supervisión:** no corresponde (MEDIO)
- **Evidencia:** muestra_formato_b30.jsonl; test_trade_off_de_k_y_f1_maximo_en_k3
- **Estado:** Medido de nuevo el 27-11 con el corpus limpio (3948 chunks): 5 de 20 con k=3 y también con k=5, así que k se queda en 3. Perfil familia (muestra_familia.jsonl): 2 de 6; 'silla de ruedas' no recupera las fichas que dicen 'movilidad reducida'.

## R3 (MEDIO, 6). Se activa una versión del prompt sin revisar su diff, o con un registro_prompts.json editado a mano

- **Justificación:** Desde el Bloque 32 los tres flujos (responder, registro_rag y procesar_consulta) usan la misma versión aprobada, así que un cambio malo afecta a todos.
- **Mitigación:** Registro con huella, diff y verificación de huellas al cargar (registro_prompts.py, Bloque 32); prompt_huella en logs_rag.jsonl y en registros_latencia_costo.jsonl
- **Punto de supervisión:** interrupt() con el diff; resolver exige la huella revisada y no activa si la versión activa cambió durante la revisión (grafo_supervision.py, Bloque 32)
- **Evidencia:** test_resume_que_no_es_true_no_activa; test_resolver_rechaza_una_huella_distinta_a_la_revisada; test_no_activa_si_la_version_activa_cambio_durante_la_revision; test_registro_editado_a_mano_no_se_carga; test_b32_usa_el_prompt_activo_con_system_instruction
- **Estado:** Integrado. El checkpointer es InMemorySaver: una solicitud pendiente se pierde al cerrar el proceso.
- **Nota:** Antes de unificar los flujos este riesgo era 'dos prompts activos a la vez' (4x3, ALTO); se cerró adoptando la idea del repo del ayudante

## R5 (MEDIO, 6). Inyección de prompt vía texto de Wikipedia descargado al corpus con descargar_wikipedia.py

- **Justificación:** Wikipedia es editable por cualquiera y los artículos entran al corpus sin revisión línea por línea.
- **Mitigación:** Detector sobre texto normalizado + etiquetas <contexto> + systemInstruction aparte (Bloque 31), hoy en los tres flujos por el prompt activo (Bloque 32); el filtro de chunks solo corre en procesar_consulta
- **Punto de supervisión:** Revisar los eventos documento_descartado_por_inyeccion de logs_rag.jsonl después de cada descarga nueva
- **Evidencia:** test_detector_no_se_esquiva_con_tildes_ni_invisibles_ni_etiquetas; test_escanear_coleccion_del_corpus_del_repo_sin_falsos_positivos
- **Estado:** Filtro de chunks pendiente en responder() y registro_rag.

## R9 (MEDIO, 6). El enrutador elige mal la intención con mensajes distintos a las 4 consultas ya probadas

- **Justificación:** La muestra demuestra consistencia, no cobertura de las 5 intenciones.
- **Mitigación:** dataset_tuning.py con 15 ejemplos validados como conjunto de regresión (Bloque 29) y decisión de no personalizar (Bloque 30)
- **Punto de supervisión:** no corresponde (MEDIO)
- **Evidencia:** test_dataset_propio_valido; test_dataset_roto_detecta_cada_error
- **Estado:** Falta correr el enrutador sobre los 15 ejemplos.

## R4 (MEDIO, 4). Reserva de cupo en observatorio ejecutada sin una aprobación humana real

- **Justificación:** Es la única acción del sistema con efecto fuera de él; el agente podría invocarla sin que el turista lo pida.
- **Mitigación:** Validación Pydantic antes de pedir aprobación, esperar_aprobacion con timeout = no aprobada y registrar_aprobacion (Bloques 22 y 23)
- **Punto de supervisión:** esperar_aprobacion en reservar_cupo_observatorio (Bloque 23)
- **Evidencia:** test_reserva_pide_aprobacion_y_la_registra; test_aprobacion_por_timeout_devuelve_false; test_reserva_invalida_no_pide_aprobacion
- **Estado:** Integrado y probado en 2 corridas reales (2026-11-09). Aprobación por consola.
- **Nota:** Es la acción de mayor riesgo del sistema. Conectada a un sistema real de reservas sube a impacto 5 (2x5 = 10, ALTO); la aprobación humana ya está
