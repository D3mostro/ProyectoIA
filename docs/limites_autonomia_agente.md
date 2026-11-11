# Límites de autonomía: agente asesor de astroturismo

Agente de `src/agente_con_herramientas.py` (`construir_agente_con_memoria`): `AgentExecutor` con `gemini-3.5-flash-lite`, tres herramientas, memoria por `session_id` (en memoria del proceso) y `max_iterations=5`.

## Herramientas y nivel de riesgo

- `consultar_fase_lunar`: bajo (solo calcula con efemérides locales de Skyfield, no modifica nada ni sale a internet salvo la primera descarga de `de421.bsp`).
- `pronosticar_nubes`: bajo (solo lee el pronóstico de Open-Meteo; si se equivoca de lugar o fecha, el peor resultado es una recomendación de noche equivocada, no un daño real).
- `reservar_cupo_observatorio`: alto (una reserva real comprometería cupos y dinero del turista, y una visita cancelada o duplicada no siempre se puede deshacer). Hoy es **simulada**: no está conectada a ningún sistema de reservas y solo devuelve un texto.

No hay herramientas de riesgo medio.

## Herramientas que requieren aprobación humana

- `reservar_cupo_observatorio`: primero valida los datos con `ArgumentosReserva` (ver "Segunda corrección"); si son inválidos, no pide aprobación. Si son válidos, llama a `esperar_aprobacion` antes de reservar y registra la decisión con `registrar_aprobacion` (herramienta, argumentos, aprobado, momento), se apruebe o no. Si nadie responde a tiempo (`SEGUNDOS_PARA_APROBAR`: 300 s por defecto, 60 s en la demo del `__main__`), la reserva queda **no aprobada**. La espera usa un hilo y una cola en vez del `select()` del curso, porque `select()` no funciona sobre la entrada estándar en Windows.
- `consultar_fase_lunar` y `pronosticar_nubes` no piden aprobación: son de solo lectura, y pedir confirmación en cada consulta haría que quien aprueba termine aprobando por costumbre.

Límites conocidos de esta implementación:

- La aprobación se pide por consola (`input()`), así que solo sirve en una terminal interactiva. Sin entrada estándar (`EOFError`), la reserva queda no aprobada.
- Después de un timeout, el hilo de `input()` sigue esperando y se queda con la próxima línea que se escriba en la consola.
- `registro_aprobaciones` vive en memoria y se pierde al cerrar el programa; para uso real habría que guardarlo en una base de datos, igual que los historiales.

## Revisión de trazas

Cada consulta del `__main__` se guarda con `registrar_traza` como una línea en `trazas_agente.jsonl` (raíz del repo): `timestamp`, `session_id`, `entrada`, `acciones` (herramienta, argumentos, observación) y `respuesta_final` en texto.

- Frecuencia: una muestra de 10 trazas por semana mientras el agente esté en prueba, y **todas** las trazas donde aparezca `reservar_cupo_observatorio`.
- Qué se busca:
  - `reservar_cupo_observatorio` en una traza sin una entrada en `registro_aprobaciones` para los mismos argumentos: la supervisión no está funcionando.
  - `reservar_cupo_observatorio` invocada sin que la entrada pida reservar de forma explícita (el prompt de sistema lo prohíbe).
  - La misma herramienta repetida seguida con argumentos parecidos, por ejemplo `pronosticar_nubes` con un lugar que no está en `LUGARES`: el agente reintenta sin avanzar y gasta cuota de Gemini.
  - Una respuesta final con un porcentaje de luna o de nubes que no aparece en ninguna observación: el agente lo inventó.
  - En sesiones con memoria, un segundo turno que responde sobre otra fecha u otro lugar que el del turno anterior.

Comandos para revisar el archivo en PowerShell (equivalentes a `wc -l` y `grep -c` del curso):

```powershell
(Get-Content trazas_agente.jsonl).Count
(Select-String -Path trazas_agente.jsonl -Pattern '"herramienta": "reservar_cupo_observatorio"').Count
Select-String -Path src\agente_con_herramientas.py -Pattern "esperar_aprobacion\("
```

## Qué hacer ante un comportamiento inesperado

1. Si el agente reserva sin que se pida, o sin aprobación registrada: quitar `reservar_cupo_observatorio` de `HERRAMIENTAS` de inmediato y dejar el agente solo con las dos herramientas de lectura.
2. Buscar en `trazas_agente.jsonl` todas las líneas de esa sesión (`session_id`) para ver qué entrada y qué historial llevaron a la decisión.
3. Si eligió mal la herramienta: ajustar el docstring de la herramienta (cuándo usarla y cuándo no) y la regla del prompt de sistema en `construir_prompt_agente`.
4. Si se equivocó en los argumentos (fecha en otro formato, lugar desconocido): ajustar las descripciones de `ArgumentosPronostico` o agregar un esquema Pydantic a la herramienta afectada.
5. Agregar el caso como prueba con un modelo falso en `src/test_agente_con_herramientas.py` y correr `python -m pytest -v` antes de volver a activar la herramienta.
6. Si el problema es de cuota (respuestas cortadas o error 429 en `error.log`): bajar `max_iterations` o el número de consultas por corrida antes de tocar el prompt.

## Estado de la evidencia

- Pruebas con modelo falso: aprobación por timeout, aprobación y rechazo registrados, reserva invocada por el agente, traza escrita en un archivo temporal y memoria en dos sesiones de dos turnos (`src/test_agente_con_herramientas.py`).
- Dos corridas reales contra Gemini (2026-11-09, `python src/agente_con_herramientas.py`): 8 líneas en `trazas_agente.jsonl`, cada corrida con dos sesiones de dos turnos.
  - `turista_1`: usó `consultar_fase_lunar` (30 % iluminada el 2026-12-14) y, en el segundo turno, `pronosticar_nubes` con la fecha del turno anterior: la memoria funciona.
  - `turista_2`: usó `reservar_cupo_observatorio` en las dos corridas. En la primera nadie respondió y la reserva quedó **no aprobada por timeout** (5 minutos entre la consulta y la respuesta, el límite de 300 s). En la segunda se respondió "s" y quedó **aprobada**, con su entrada en `registro_aprobaciones`. En el segundo turno recordó la fecha de la reserva sin usar herramientas.
  - Sin herramientas repetidas seguidas.
- Hallazgos de la revisión de esas trazas (patrón "respuesta que no coincide con ninguna observación"):
  - Primera corrida: Open-Meteo respondió 400 para el 14 de diciembre (fecha fuera del pronóstico) y el agente igual afirmó que "históricamente" el Valle del Elqui suele estar despejado. En la segunda corrida, con la misma observación, respondió bien que no hay pronóstico con tanta anticipación: el comportamiento no es estable.
  - Segunda corrida: tras la reserva aprobada, el agente dijo "He registrado con éxito tu reserva" sin aclarar que es simulada, y agregó consejos (llegar 15 minutos antes) que no vienen de ninguna herramienta.
  - Acción pendiente: agregar al prompt que, si una herramienta no entrega datos, lo diga sin completar con conocimiento general.

## Corrección aplicada (después de las dos corridas)

Las 8 líneas actuales de `trazas_agente.jsonl` son la evidencia **antes de la corrección** y no se borraron. Cambios en `src/agente_con_herramientas.py`:

1. **Regla en el prompt de sistema** (`REGLA_OBSERVACIONES`): si una herramienta devuelve un error o no trae datos, el agente lo dice tal cual y no lo completa con conocimiento general ni con datos históricos; cuando usa herramientas, afirma solo lo que digan sus observaciones; y avisa siempre que las reservas son simuladas.
2. **Texto de la reserva**: `reservar_cupo_observatorio` empieza siempre con "RESERVA SIMULADA (no se reservó nada real): ", tanto si se aprueba como si se rechaza.
3. **Aviso de aprobación visible**: `esperar_aprobacion` imprime un encabezado "=== APROBACIÓN REQUERIDA ===" separado del verbose, con el tiempo límite. En el `__main__` el límite baja a 60 s.

Pruebas nuevas: el encabezado y el límite global, el prefijo de la reserva simulada y la regla en el prompt. Suite: 94 passed.

- [x] **Corrida 3, después de la corrección** (2026-11-09, líneas 9 a 12 de `trazas_agente.jsonl`, 12 en total):
  - Ante el 400 de Open-Meteo respondió "No se pudo obtener el pronóstico de nubosidad para Vicuña en esa fecha debido a un error en el servicio", sin agregar clima histórico.
  - La reserva quedó **no aprobada** 13 s después de la consulta, dentro del límite de 60 s: el supervisor respondió "n" ante el encabezado "=== APROBACIÓN REQUERIDA ===" y `registro_aprobaciones` quedó con `aprobado: False`. La respuesta aclara que "las reservas son simuladas y no se ha realizado ninguna gestión real".
  - La memoria sigue funcionando (fecha del turno anterior en ambas sesiones) y no hay herramientas repetidas seguidas.
- [x] **Corrida 4, después de la corrección** (2026-11-09, líneas 13 a 16, 16 en total): cubre el caso **aprobado** que faltaba tras la corrección.
  - El supervisor respondió "s"; la herramienta devolvió "RESERVA SIMULADA (no se reservó nada real): aprobada, 2 cupos en Mamalluca para el 2026-12-14" y la respuesta final dijo "¡Reserva simulada exitosa! [...] Recuerda que esta es una simulación y no una reserva real". `registro_aprobaciones` quedó con `aprobado: True`.
  - Ante el 400 de Open-Meteo volvió a decir solo que no se pudo obtener el pronóstico.
  - Hallazgo menor: en la respuesta lunar agregó "Esto dejará el cielo bastante oscuro para disfrutar de [...] las Gemínidas". Es una inferencia razonable a partir del 30 %, pero no sale de la observación; se deja anotado para la revisión de trazas, sin cambiar el prompt.
- Límite: son dos corridas tras la corrección; la regla del prompt reduce el riesgo, pero no garantiza que Gemini la cumpla siempre. Se sigue revisando en la muestra semanal.

## Segunda corrección (2026-11-10, revisión del código)

Cambios en `src/agente_con_herramientas.py`, sin corrida real nueva: las 16 líneas de `trazas_agente.jsonl` siguen siendo las de las corridas 1 a 4.

1. **La reserva valida antes de pedir aprobación** (`args_schema=ArgumentosReserva`): `personas` entre 1 y 20, `fecha` en formato AAAA-MM-DD real, y `observatorio` dentro de `OBSERVATORIOS` (Mamalluca, Cancana, Alfa Aldea, Cruz del Sur, Collowara, Del Pangue; sin distinguir mayúsculas ni tildes, con o sin "Observatorio"). Si algo falla, `handle_validation_error` devuelve "Datos de reserva no válidos, no se pidió aprobación: ..." como observación, así que el agente no se cae y el supervisor no ve una reserva imposible. El registro guarda el nombre canónico (por ejemplo "Observatorio Mamalluca"; en las trazas antiguas aparece "Mamalluca").
2. **`pronosticar_nubes` revisa el rango antes de llamar a Open-Meteo**: si la fecha es pasada o está a 16 días o más de hoy, devuelve "fuera del rango de pronóstico (hasta 16 días desde hoy)" sin hacer la petición. Probado contra la API el 2026-11-10: acepta hasta hoy + 16 días, pero ese último día llega sin datos de noche, por eso el corte es hoy + 15. Esto evita el 400 que en la corrida 1 terminó en un clima inventado.
3. **`registrar_traza` no rompe la respuesta**: un `OSError` al escribir se avisa por stderr y la consulta sigue; usa `resultado.get("output", "")`.
4. **`momento` en UTC**: `registrar_aprobacion` usa `datetime.now(timezone.utc)`, igual que el `timestamp` de la traza, así que se pueden cruzar sin convertir la hora de Chile. En las corridas 1 a 4 el `momento` era hora local.

Pruebas nuevas (modelo falso, sin Gemini): 6 reservas inválidas que no piden aprobación ni quedan registradas, normalización de nombre y personas, agente que no se cae con datos inválidos, fechas fuera de rango sin llamar a la API, último día del rango, fecha mal formada, `OSError` en la traza, traza sin `output` y `momento` con zona UTC. Suite: 110 passed.

Límites que siguen: la reserva acepta fechas pasadas; la lista de observatorios está escrita a mano; `registro_aprobaciones` y los historiales siguen solo en memoria; nada impide que el agente pida dos reservas en el mismo turno (cada una pide su propia aprobación).
