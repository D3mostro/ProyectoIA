# Gate 2: autoevaluación (27-11-2026)

Puntajes propios de 0 a 100 con los pesos de la lección 20449, estimados después del ensayo real del 27-11 ([`gate2_ensayo.md`](gate2_ensayo.md)). Son una estimación honesta, no una nota: los puntos débiles van en la columna de evidencia.

## Criterios no compensables

| Criterio | Estado | Evidencia |
|---|---|---|
| `pipeline_corre_de_punta_a_punta` | True | Corrida real de `servicio_rag.py` con código de salida 0 sobre `astroturismo_real` (3948 chunks); `test_1_servicio_corre_de_punta_a_punta` y `test_1b_clon_limpio_usa_la_coleccion_del_repo` |
| `respuestas_citan_fuente` | True | La respuesta real termina con `Fuentes: [1] …042_Observatorio_Migrantes.txt; …`; una respuesta sin cita queda en estado `sin_cita` con aviso (`test_2b_sin_cita_no_queda_ok`) |
| `logs_sin_contenido_crudo` | True | `logs_rag.jsonl` guarda largo y huella; `registros_latencia_costo.jsonl` guarda texto solo de preguntas fijas de evaluación; `trazas_agente.jsonl` guarda largo y huella del mensaje del turista (`test_3_*`, `test_3b_*`); `preflight.py` OK |
| `supervision_humana_funciona` | True | `grafo_supervision.py`: `interrupt()` con el diff del prompt y `Command(resume=True/False)`; solo `True` activa (`test_4_*`, `test_4b_*`, `python src/grafo_supervision.py --demo`). La reserva de cupo, la acción con efecto externo, tiene su propia aprobación humana con timeout (`esperar_aprobacion`, Bloque 23) |

## Dimensiones

| Dimensión | Peso | Puntaje | Evidencia y debilidades |
|---|---:|---:|---|
| pipeline_funcional | 0,25 | 90 | Índice persistido en `vectorstore/` con reindexado incremental (Bloque 25) y limpieza de chunks sin gastar embeddings; servicio único; un clon limpio usa `corpus/` |
| recuperacion_citacion | 0,15 | 75 | Citas validadas y fuentes con nombre de archivo. Débil: las métricas del Bloque 27 (k=3: recall 0,97 y precisión 0,40) se midieron sobre `corpus/` (19 chunks sobre la app), no sobre el corpus turístico; en la muestra real, 5 de 20 respuestas y 2 de 6 del perfil familia dicen que el contexto no alcanza (R8) |
| costo_latencia | 0,10 | 85 | `registros_latencia_costo.jsonl` con corridas reales por etapa, tokens y costo (Bloque 28) |
| personalizacion | 0,15 | 85 | `docs/decision_personalizacion.md`: decisión propia (RAG, sin PEFT) con puntaje y sensibilidad (Bloque 30) |
| observabilidad | 0,15 | 90 | Un `trace_id` por consulta en todos los eventos, consumo real y costo (Bloque 31); la traza se reconstruye solo desde `logs_rag.jsonl` |
| seguridad_gobernanza | 0,20 | 80 | Filtro de inyección sobre los `Document` recuperados y sobre la pregunta (0 falsos positivos en 4003 chunks reales), prompts versionados con huella, grafo de aprobación y matriz de 9 riesgos con los 2 ALTO conectados a supervisión. Débil: la supervisión de R2 es un procedimiento (correr `preflight.py` antes del push) |

```python
pesos = {"pipeline_funcional": 0.25, "recuperacion_citacion": 0.15, "costo_latencia": 0.10,
         "personalizacion": 0.15, "observabilidad": 0.15, "seguridad_gobernanza": 0.20}
mi_puntaje = {"pipeline_funcional": 90, "recuperacion_citacion": 75, "costo_latencia": 85,
              "personalizacion": 85, "observabilidad": 90, "seguridad_gobernanza": 80}
# promedio ponderado = 84.5; no compensables: los 4 en True
```

**Promedio ponderado: 84,5** (umbral 80), con los cuatro no compensables en True.

## Qué falta antes de dar por cerrada la entrega

- Ensayo de instalación limpia: clonar el repo en una carpeta nueva, crear un `.venv` nuevo, `pip install -r requirements.txt`, `python src/verificar_entorno.py`, `python -m pytest -q` y `python src/servicio_rag.py --simulado`.
- `git status` limpio y `git log -1` con el commit de entrega.
