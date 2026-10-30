# Planificador de astroturismo en Chile

Miniaplicación en Python que consume la API de Google Gemini para responder consultas de astroturismo en Chile (qué se puede ver en un lugar y fecha, y cuándo conviene observar). Proyecto de la **Especialización en Desarrollo con IA Generativa (Accenture / Skillnest)**, Módulos 1 y 2.

> **Estado actual (camino al Gate 2):** Gate 1 entregado (conexión autenticada a Gemini, banco de prompts con prueba de consistencia y comparación de modelos documentada). En el Módulo 2 se agregaron un cliente modular con reintentos y backoff exponencial, registro de costo por operación, pruebas unitarias con `pytest`, un pipeline de limpieza de texto y métricas de precisión, cobertura y F1 sobre las salidas del modelo. Todavía **no** hay interfaz, herramientas ni RAG (ver [Hoja de ruta](#hoja-de-ruta)).

---

## Requisitos

- Python 3.10 o superior (desarrollado con 3.13).
- Una llave de API de Gemini (gratis en [Google AI Studio](https://aistudio.google.com/apikey)).
- Git.

## Instalación

Los comandos son para **Windows (PowerShell)**; entre paréntesis, el equivalente en Linux/macOS.

```powershell
# 1. Clonar el repositorio
git clone https://github.com/D3mostro/ProyectoIA.git
cd ProyectoIA

# 2. Crear y activar el entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1          # (Linux/macOS: source .venv/bin/activate)

# 3. Instalar las dependencias
pip install -r requirements.txt

# 4. Crear el archivo de credenciales
copy .env.example .env              # (Linux/macOS: cp .env.example .env)
```

Luego abre `.env` y reemplaza el valor de ejemplo por tu llave real:

```
GEMINI_API_KEY=tu_llave_real_aqui
```

El archivo `.env` está en `.gitignore` y **nunca** se sube al repositorio.

## Cómo ejecutar

Desde la raíz del repositorio, con el entorno virtual activo:

| Qué hace | Comando |
|---|---|
| Prueba de conexión: una consulta de astroturismo con *system instruction* (muestra latencia y costo) | `python src/consumir_gemini.py` |
| Comparación de los tres modelos candidatos (latencia y respuestas) | `python src/comparar_modelos.py` |
| Prueba de consistencia del banco de prompts (3 intentos por prompt; guarda `prompts/registro_resultados.json`) | `python prompts/test_consistencia.py` |
| Ensayo de rate limit: 10 consultas seguidas con reintento y costo acumulado (Bloque 13) | `python src/cliente.py` |
| Pipeline de texto: coincidencia de palabras antes y después de limpiar (guarda `src/registro_pipeline_texto.json`) | `python src/pipeline_texto.py` |
| Métricas de precisión, cobertura y F1 sobre 20 respuestas reales (guarda `src/resultados_metricas.json`; si se corta, al volver a correrlo retoma lo que falta) | `python src/metricas.py` |
| Revisión manual de 10 casos con la rúbrica del Bloque 17 (guarda `src/revision_manual.json`) | `python src/revision_manual.py` |

Salida esperada de `consumir_gemini.py`: la respuesta del modelo, la latencia en segundos y el costo estimado en USD. Si la llave falta o es incorrecta, el programa lo informa con un mensaje claro.

### Pruebas unitarias

Las pruebas usan mocks (no llaman a la API ni gastan cuota) y viven en `src/`, junto al código que prueban, como indica el curso. Se corren desde la raíz del repositorio:

```powershell
python -m pytest -v
```

| Archivo | Qué prueba |
|---|---|
| `test_cliente.py` | Clasificación de errores HTTP, reintento con backoff, registro de costo y `consultar_modelo` |
| `test_pipeline_texto.py` | Limpieza, tokenización, bolsa de palabras y coincidencia |
| `test_metricas.py` | Precisión, cobertura, F1, casos vacíos, resumen, peores casos y los tres límites de la métrica (sinónimo, dato equivocado, negación) |

`pytest.ini` limita la búsqueda a `src/` (`testpaths = src`). Así `pytest` no recolecta `prompts/test_consistencia.py`, que no es una prueba unitaria (llama a la API real). También funciona `cd src` y luego `python -m pytest -v`.

## Estructura del repositorio

```
.
├── src/
│   ├── consumir_gemini.py     # Cliente de Gemini: arma, envía y valida peticiones
│   ├── comparar_modelos.py    # Mide latencia de los modelos candidatos
│   ├── cliente.py             # Cliente modular: reintento con backoff y registro de costo
│   ├── pipeline_texto.py      # Limpieza, tokenización y bolsa de palabras
│   ├── metricas.py            # Precisión, cobertura y F1 sobre las salidas del modelo
│   ├── casos_evaluacion.py    # 20 consultas con referencia escrita a mano
│   ├── revision_manual.py     # Rúbrica aplicada a mano a 10 casos (Bloque 17)
│   ├── test_cliente.py        # Pruebas unitarias (pytest)
│   ├── test_pipeline_texto.py
│   ├── test_metricas.py
│   ├── registro_pipeline_texto.json   # (se genera al ejecutar pipeline_texto.py)
│   ├── resultados_metricas.json       # (se genera al ejecutar metricas.py)
│   └── revision_manual.json           # (se genera al ejecutar revision_manual.py)
├── prompts/
│   ├── banco.py               # Banco de prompts: system instruction + esquema JSON
│   ├── test_consistencia.py   # Ejecuta cada prompt varias veces y cuenta respuestas válidas
│   └── registro_resultados.json   # (se genera al ejecutar la prueba)
├── docs/
│   ├── decision-arquitectura.md   # Elección del modelo y su justificación
│   ├── limites_metrica.md         # Qué no puede medir el F1 (Bloque 17)
│   ├── diagrama_arquitectura.png  # Diagrama del flujo de la aplicación
│   └── Evidencia_conexion_Autenticada_Gemini.png
├── tests/                     # Viene de la plantilla del curso; hoy vacía (las pruebas están en src/)
├── pytest.ini                 # Indica a pytest que las pruebas están en src/
├── requirements.txt           # Dependencias fijadas
├── .env.example               # Plantilla de variables de entorno (sin secretos)
└── .gitignore
```

## Cómo funciona

El proyecto llama a la **API REST de Gemini con `requests`** (no usa el SDK `google-genai`, aunque esté listado en `requirements.txt` por el entorno del curso). Flujo:

```
consultar_modelo(pregunta, system_instruction)          ← cliente.py
   → armar_peticion → con_reintento(enviar_peticion) → extraer_texto → respuesta
                                │
                                └→ registrar_costo → calcular_costo (usageMetadata) → costo en USD
```

Funciones principales de `src/consumir_gemini.py`:

| Función | Responsabilidad |
|---|---|
| `armar_peticion(pregunta, modelo, system_instruction, response_schema)` | Construye la URL y el body. Lanza `ValueError` si la pregunta está vacía. Agrega `systemInstruction` y `generationConfig` (JSON + esquema) cuando se entregan. |
| `enviar_peticion(url, body)` | Hace el POST. Lanza `TimeoutError` ante timeout, falla de conexión, `429` o error `5xx` (fallas momentáneas, se pueden reintentar); `ValueError` ante llave inválida o sin permiso (`401`, `403` o `400 API_KEY_INVALID`); `RuntimeError` ante otro código distinto de 200 o respuesta que no es JSON. |
| `extraer_texto(datos)` | Obtiene el texto de la respuesta del modelo. Lanza `RuntimeError` si no tiene el formato esperado. |
| `parsear_respuesta_json(texto, claves_requeridas)` | Quita las vallas \`\`\`json, hace `json.loads`, exige que sea un objeto y que tenga las claves esperadas. Lanza `ValueError` si no cumple. La usa `prompts/test_consistencia.py`. |
| `calcular_costo(datos, precio_entrada, precio_salida)` | Estima el costo en USD con `usageMetadata` (tokens de entrada, salida y de razonamiento). |
| `obtener_dato_de_prueba(...)` | Función de demostración: envía, mide latencia, imprime costo. |

Funciones de `src/cliente.py` (Bloques 12 y 13):

| Función | Responsabilidad |
|---|---|
| `con_reintento(funcion, intentos=3, espera_base=2, espera_maxima=30)` | Reintenta solo ante `TimeoutError`, con backoff exponencial (2, 4, 8 s…, con techo de 30 s) y un jitter de hasta 10 %. Los demás errores pasan sin reintentar. |
| `registrar_costo(respuesta)` | Calcula el costo de la operación con los precios de `gemini-3.5-flash-lite` y lo suma a `costo_total_sesion`. |
| `consultar_modelo(pregunta, system_instruction="")` | Punto de entrada del cliente. Si se agotan los reintentos lanza `RuntimeError`; registra el costo solo tras una respuesta exitosa. |

**Criterio de errores:** `ValueError` = el contenido o la credencial están mal (no tiene sentido reintentar); `TimeoutError` = falla momentánea del proveedor (se reintenta); `RuntimeError` = falló la comunicación de forma definitiva. Así cada capa sabe qué hacer en cada caso.

### Pipeline de texto, métricas y sus límites (Bloques 15 a 17)

- `src/pipeline_texto.py`: `limpiar_texto` (minúsculas, sin puntuación, conserva tildes, ñ y números; `TypeError` si no recibe texto), `tokenizar`, `procesar_texto` (los dos pasos juntos), `vectorizar` (bolsa de palabras con `Counter`) y `calcular_coincidencia`.
- `src/metricas.py`: `calcular_precision_cobertura_f1` (cuenta repeticiones con la intersección de `Counter`; devuelve 0.0 si algún texto está vacío), `calcular_metricas`, `resumir_metricas` (promedios) `peores_casos` (ordena por F1 ascendente) y `muestra_para_revisar` (los 5 F1 más bajos y los 5 más altos, sin repetir).

Último ensayo guardado en `src/resultados_metricas.json` (30-10-2026, 20 consultas de astroturismo de `src/casos_evaluacion.py`): precisión 0.56, cobertura 0.57, F1 promedio 0.55. La revisión manual de 10 casos (`src/revision_manual.py`) encontró un falso positivo: una fecha de eclipse equivocada con F1 0.73. Los límites de la métrica y cuándo revisar a mano están en [`docs/limites_metrica.md`](docs/limites_metrica.md).

### Banco de prompts (Bloque 10)

`prompts/banco.py` define tres prompts para la miniaplicación. Cada uno trae una *system instruction* estable, un ejemplo de usuario, las claves esperadas y un `responseSchema`:

| Prompt | Entrada | Salida JSON |
|---|---|---|
| `clasificar_consulta` | Consulta del turista | `categoria` (observacion / clima / logistica / otro), `confianza` (0–1) |
| `extraer_datos_consulta` | Consulta con lugar y fecha | `lugar`, `fecha`, `interes` |
| `resumir_texto` | Texto informativo | `resumen`, `palabras_clave` |

`prompts/test_consistencia.py` ejecuta cada prompt 3 veces y registra cuántas respuestas son JSON válido con las claves esperadas. Un prompt con menos de 2/3 debe ajustarse y volver a medirse.

**Resultados de la última medición** (22-10-2026, `gemini-3.5-flash-lite`, 3 intentos por prompt, sin errores):

| Prompt | Válidas |
|---|---|
| `clasificar_consulta` | 3/3 |
| `extraer_datos_consulta` | 3/3 |
| `resumir_texto` | 3/3 |

El detalle queda en `prompts/registro_resultados.json`. Con 3 intentos la medida es una primera evidencia, no una garantía: conviene repetirla al cambiar un prompt o el modelo.

La medición se repitió el **24-10-2026** en una instalación limpia (ver [Registro de ensayo de instalación limpia](#registro-de-ensayo-de-instalación-limpia)) y dio el mismo resultado: 3/3 en los tres prompts, 9/9 en total.

## Decisión de modelo

Se eligió **`gemini-3.5-flash-lite`** frente a `gemini-3.1-flash-lite` y `gemini-3.8-flash` (medición del 20-10-2026):

| Modelo | Latencia promedio | Calidad (0–6) | Observación |
|---|---|---|---|
| `gemini-3.1-flash-lite` | 7,01 s | 4/6 | Error 503 en una corrida; retiro anunciado el 07-05-2027 |
| **`gemini-3.5-flash-lite`** | **1,51 s** | 3/6 | Sin errores; el más rápido y estable |
| `gemini-3.8-flash` | 15,65 s | no evaluable | Error 429 en el nivel gratuito |

Prima la latencia (demo en vivo). La limitación conocida es la calidad (inventa datos en algunas preguntas), que se compensará en el Gate 2 con herramientas y RAG. Detalle completo, tarifas y fuentes en [`docs/decision-arquitectura.md`](docs/decision-arquitectura.md).

## Seguridad

- La llave vive solo en `.env` (excluido por `.gitignore`); el código la lee con `python-dotenv`.
- `.env.example` contiene únicamente un valor de ejemplo.
- Nunca se imprime ni se registra la llave. Si se expone por error, hay que revocarla en Google AI Studio y crear otra.
- El cliente maneja `timeout`, errores de conexión y códigos HTTP no exitosos con mensajes que no incluyen la URL (la URL contiene la llave).

## Solución de problemas

| Síntoma | Causa probable | Qué hacer |
|---|---|---|
| `Activate.ps1 no se puede cargar porque la ejecución de scripts está deshabilitada` | Política de ejecución de PowerShell | Ejecutar `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y volver a activar el `.venv` |
| `ModuleNotFoundError: No module named 'dotenv'` | Entorno sin dependencias | Activar `.venv` y ejecutar `pip install -r requirements.txt` |
| Mensaje de llave no encontrada | Falta `.env` o la variable | Copiar `.env.example` a `.env` y poner `GEMINI_API_KEY` |
| `Status Code: 400` | Petición mal formada (p. ej. `system_instruction` en vez de `systemInstruction`) | Revisar los nombres de campo en camelCase |
| `Credencial inválida o sin permiso` | Llave inválida o sin permisos (`401`, `403` o `400 API_KEY_INVALID`) | Generar una nueva llave en AI Studio |
| `Límite excedido` | `429`: límite del nivel gratuito | Esperar; `consultar_modelo` (`cliente.py`) reintenta con backoff |
| `El proveedor del servicio no pudo procesar la peticion` | `5xx`: servicio saturado | `consultar_modelo` reintenta solo; si se agotan los intentos, probar más tarde |
| `La respuesta no es un JSON válido` | El servicio devolvió algo que no es JSON | Reintentar en unos segundos |
| `La respuesta del modelo no es JSON válido` | El modelo devolvió texto libre en vez del JSON pedido | Revisar la system instruction y el `responseSchema` |
| `ModuleNotFoundError: No module named 'cliente'` al correr `pytest` | Se ejecutó desde una carpeta que no es la raíz del repositorio ni `src/` | Volver a la raíz y correr `python -m pytest -v` |

## Pruebas de reproducción del entorno

El Gate 1 tiene cuatro condiciones no compensables. Tres de ellas se demuestran reproduciendo el proyecto desde cero: que el código corra contra la API real, que el entorno se pueda reconstruir solo con `requirements.txt`, y que las credenciales nunca hayan salido de `.env`. Esta sección registra esas pruebas. La cuarta condición, el repositorio accesible, se cumple porque el repositorio es público.

### 1. Código ejecutable contra la API real

Ejecutado el 24-10-2026 sobre el clon limpio, con una llave real en `.env` y sin tocar el código:

| Comando | Resultado |
|---|---|
| `python src/consumir_gemini.py` | Sin `Traceback`. Respondió las dos pruebas con latencias de 1,09 s y 0,99 s y costos de 0,000038 y 0,000069 USD. La salida estructurada se parseó como `dict` con las claves esperadas: `{'categoria': 'observacion', 'confianza': 0.95}`. Recorre `armar_peticion` → `enviar_peticion` → `extraer_texto` → `parsear_respuesta_json` de punta a punta. |
| `python prompts/test_consistencia.py` | 9/9 respuestas válidas (3/3 en cada prompt), sin errores. Regeneró `prompts/registro_resultados.json`. |
| `python src/comparar_modelos.py` | Midió `gemini-3.1-flash-lite` (5,90 s) y `gemini-3.5-flash-lite` (1,03 s). `gemini-3.8-flash` devolvió `503`; el reintento con backoff (1, 2, 4 y 8 s) actuó como está documentado y el script terminó sin caerse. |

### 2. Entorno reproducible

- `requirements.txt` se generó con `pip freeze`. Para comprobar que está completo, se instaló en un entorno virtual nuevo y se comparó `pip freeze` de ese entorno con el archivo: son **idénticos** (26 paquetes con versión fijada).
- El ensayo de instalación limpia se hizo en una carpeta nueva, sin el proyecto ni dependencias instaladas de antes, siguiendo solo este README.

### Registro de ensayo de instalación limpia

- **Fecha:** 24-10-2026 (Linux, Python 3.13.17, carpeta nueva clonada desde GitHub, siguiendo exactamente [Instalación (desde cero)](#instalación-desde-cero) y [Cómo ejecutar](#cómo-ejecutar)).
- **Encontrado:** la instalación funcionó tal como está escrita. En un ensayo previo sin llave válida, un error de llave solo mostraba `Status Code: 400`, aunque el README promete un mensaje claro.
- **Corregido:** `src/consumir_gemini.py` ahora indica claramente si la llave falta o no es válida.
- **Resultado tras la corrección:** exitoso. Las 26 dependencias se instalaron sin conflictos y los tres comandos corrieron con llamadas reales a la API (ver la tabla del punto 1).

| Fecha | Entorno | Quién | Resultado |
|---|---|---|---|
| 24-10-2026 | Windows, PowerShell, carpeta nueva | Germán Spuler, clonando desde GitHub | Exitoso, sin incidencias |

### 3. Credenciales protegidas

Verificado el 24-10-2026 sobre el clon del repositorio:

| Comprobación | Comando | Resultado |
|---|---|---|
| `.env` está ignorado | revisar `.gitignore` | Contiene la línea `.env` |
| Existe la plantilla sin secretos | revisar `.env.example` | Solo trae `GEMINI_API_KEY=tu_llave_aqui_como_ejemplo` |
| `.env` no queda para subir | `git status` | Con `.env` creado, no aparece |
| `.env` nunca se subió | `git log --all --oneline -- .env` | Sin resultados |
| No hay llaves en el historial | búsqueda de `AIza` en `git log -p --all` | 0 coincidencias |

### Observaciones

- El `503` de `gemini-3.8-flash` confirma que ese modelo es inestable en el nivel gratuito, igual que en la medición del 20-10-2026 (ahí fue `429`). La tabla de [Decisión de modelo](#decisión-de-modelo) ya lo refleja.
- Las latencias de este ensayo son menores que las del 20-10-2026 (1,03 s frente a 1,51 s para el modelo elegido). Dependen de la red y de la carga del servicio; lo que sostiene la decisión es que el orden entre los modelos se mantiene.

## Hoja de ruta

| Etapa | Contenido | Estado |
|---|---|---|
| Gate 1 | Conexión autenticada, JSON validado, banco de prompts, decisión de modelo, README | Entregado (22-10-2026) |
| Módulo 2 | Cliente modular con reintentos y backoff, costo por operación, pruebas unitarias, pipeline de texto y métricas | En curso |
| Gate 2 | Herramientas (fase lunar, visibilidad, pronóstico) y RAG sobre documentos de astroturismo | Planificado |
| Gate 3 | Aplicación completa | Planificado |

## Reglas del programa

- Ningún secreto va en el código ni en archivos versionados.
- Cada proyecto tiene su propio entorno virtual.
- Todo el trabajo se entrega en un repositorio de GitHub.
