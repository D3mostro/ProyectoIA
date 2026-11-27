"""Baja al corpus el directorio de oferentes de astroturismo de la Fundación Cielos de Chile.

El Portal de Astroturismo (https://astroturismo.cieloschile.cl/) publica una capa pública de
ArcGIS con 96 oferentes de todo Chile (revisado el 26-11-2026): categoría, región, comuna,
servicios, temporada, idiomas, accesibilidad, mascotas y cómo agendar. El corpus actual solo
tiene 8 observatorios; esto cubre preguntas como "¿hay astroturismo en la Araucanía?" o
"¿qué tour acepta mascotas?".

Uso (desde la raíz del repo):
    python src/descargar_oferentes.py            # baja los que no existen todavía
    python src/descargar_oferentes.py --seco     # solo muestra qué bajaría y cuántos chunks aprox.
    python src/descargar_oferentes.py --forzar   # reemplaza los .txt (para refrescar los datos)

- No llama a Gemini: no gasta cuota. Usa la API REST pública de ArcGIS (sin login).
- Escribe un .txt por oferente en originales/oferentes_cielos_chile/, con el mismo encabezado
  que el resto del corpus.
- NO guarda celulares ni correos (el RAG los citaría y quedarían en los .jsonl versionados,
  donde preflight.py los bloquea). Además borra cualquier correo, RUT o celular que aparezca en
  los textos libres, con los mismos patrones de revision_corpus.py.
- Los datos son de junio de 2026 y no traen horarios: sirven para saber qué existe y dónde,
  no para decir a qué hora abre (riesgo ALTO de la matriz del B32).
- La capa no declara licencia: uso educativo, el corpus no se sube a GitHub.
"""
import argparse
import re
import sys
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from revision_corpus import PATRONES_PII

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DESTINO = RAIZ.parent.parent / "Corpus RAG astroturismo" / "originales" / "oferentes_cielos_chile"
CAPA = ("https://services1.arcgis.com/Y48mXVkXeuEquJOz/arcgis/rest/services/"
        "Oferentes_actual/FeatureServer/0")
PORTAL = "https://astroturismo.cieloschile.cl/"
LICENCIA = "No declarada por la fuente (capa pública de ArcGIS; uso educativo, el corpus no se sube a GitHub)"
CABECERAS = {"User-Agent": "ProyectoIA-curso/1.0 (uso educativo)"}
CARACTERES_POR_CHUNK = 250  # chunk_size=300 menos chunk_overlap=50 de indexar_corpus.py

# (campo de la capa, etiqueta en el .txt). Celular, correo, coordenadas e ids quedan fuera a propósito.
CAMPOS = [
    ("Categoría_principal", "Categoría"),
    ("Región", "Región"),
    ("Comuna", "Comuna"),
    ("Dirección", "Dirección"),
    ("Servicios_ofrecidos", "Servicios ofrecidos"),
    ("Actividades_complementarias", "Actividades complementarias"),
    ("Temporada_de_funcionamiento", "Temporada de funcionamiento"),
    ("Horario_diurno_o_nocturno", "Horario"),
    ("Donde_realizan_sus_actividades", "Dónde se realizan las actividades"),
    ("Aire_libre_o_espacios_cerrados", "Aire libre o espacios cerrados"),
    ("Cuentan_con_telescopios", "Telescopios"),
    ("Tipo_de_telescopios", "Tipo de telescopios"),
    ("Tipo_de_equipamiento_ofrecido", "Equipamiento"),
    ("Idiomas", "Idiomas"),
    ("Certificaciones_turísticas", "Certificaciones turísticas"),
    ("Alojamiento", "Alojamiento"),
    ("Permiten_mascotas", "Permiten mascotas"),
    ("Infraestructura_apropiada_para_personas_con_movilidad_reducida",
     "Infraestructura para personas con movilidad reducida"),
    ("Opciones_de_transporte", "Cómo llegar"),
    ("Reestricciones", "Restricciones"),
    ("Procedimiento_para_agendar", "Cómo reservar"),
    ("Página_web", "Sitio web"),
    ("Instagram", "Instagram"),
]
VACIOS = {"", "n/a", "na", "-", "no aplica", "null", "none"}
URL = re.compile(r"https?://[^\s,]+")


def limpiar_valor(valor):
    """Texto sin espacios de sobra, sin datos personales; None si el campo viene vacío."""
    if valor is None:
        return None
    texto = re.sub(r"\s+", " ", str(valor)).strip()
    if texto.lower() in VACIOS:
        return None
    for patron in PATRONES_PII.values():
        texto = patron.sub("[dato de contacto omitido]", texto)
    return texto


def limpiar_web(valor):
    """Primera URL, sin parámetros de rastreo (fbclid, utm_...). Algunas vienen pegadas dos veces."""
    texto = limpiar_valor(valor)
    if not texto:
        return None
    encontrada = URL.search(texto)
    if not encontrada:
        return texto
    url = encontrada.group(0)
    repetida = url.find("http", len("https://"))
    if repetida > 0:
        url = url[:repetida]
    base, _, consulta = url.partition("?")
    parametros = [p for p in consulta.split("&") if p and not re.match(r"(fbclid|utm_\w+|gclid)=", p)]
    return base + ("?" + "&".join(parametros) if parametros else "")


def bajar_oferentes(sesion=requests, tam_pagina=1000):
    """Devuelve (lista de atributos, fecha de última edición de la capa o None). Pagina si hace falta."""
    meta = sesion.get(CAPA, params={"f": "json"}, headers=CABECERAS, timeout=30)
    meta.raise_for_status()
    meta = meta.json()
    if "error" in meta:
        raise RuntimeError(f"ArcGIS respondió con error: {meta['error']}")
    editado = (meta.get("editingInfo") or {}).get("lastEditDate")
    actualizado = (datetime.fromtimestamp(editado / 1000, tz=timezone.utc).date().isoformat()
                   if editado else None)

    campos = ",".join(["ObjectId", "Nombre"] + [c for c, _ in CAMPOS])
    registros, desde = [], 0
    while True:
        params = {"where": "1=1", "outFields": campos, "returnGeometry": "false", "orderByFields": "ObjectId",
                  "resultOffset": desde, "resultRecordCount": tam_pagina, "f": "json"}
        respuesta = sesion.get(CAPA + "/query", params=params, headers=CABECERAS, timeout=30)
        respuesta.raise_for_status()
        datos = respuesta.json()
        if "error" in datos:
            raise RuntimeError(f"ArcGIS respondió con error: {datos['error']}")
        pagina = [f["attributes"] for f in datos.get("features", [])]
        registros += pagina
        if not datos.get("exceededTransferLimit") or not pagina:
            return registros, actualizado
        desde += len(pagina)


def nombre_archivo(registro):
    """Nombre estable y legible: id + nombre sin tildes ni símbolos."""
    nombre = unicodedata.normalize("NFKD", limpiar_valor(registro.get("Nombre")) or "sin_nombre")
    nombre = re.sub(r"[^A-Za-z0-9]+", "_", nombre.encode("ascii", "ignore").decode()).strip("_")[:60]
    return f"{int(registro['ObjectId']):03d}_{nombre or 'sin_nombre'}.txt"


def armar_documento(registro, actualizado=None, hoy=None):
    nombre = limpiar_valor(registro.get("Nombre")) or "Oferente sin nombre"
    lineas = [f"Título: {nombre} (oferente de astroturismo)",
              f"Fuente: {PORTAL} (capa Oferentes_actual, Fundación Cielos de Chile)",
              f"Licencia: {LICENCIA}",
              f"Descargado: {(hoy or date.today()).isoformat()}"]
    if actualizado:
        lineas.append(f"Actualizado en la fuente: {actualizado}")
    lineas += ["", f"{nombre} es un oferente de astroturismo registrado en el Portal de Astroturismo "
                   "de la Fundación Cielos de Chile."]
    for campo, etiqueta in CAMPOS:
        valor = limpiar_web(registro.get(campo)) if campo == "Página_web" else limpiar_valor(registro.get(campo))
        if valor:
            lineas.append(f"{etiqueta}: {valor}")
    lineas.append("Los horarios y cupos no están en esta ficha: hay que confirmarlos con el oferente.")
    return "\n".join(lineas) + "\n"


def guardar(registro, carpeta=CARPETA_DESTINO, actualizado=None, forzar=False, hoy=None):
    """Escribe el .txt. Devuelve (ruta, caracteres) o (None, 0) si ya existía y no se pidió --forzar."""
    ruta = Path(carpeta) / nombre_archivo(registro)
    if ruta.exists() and not forzar:
        return None, 0
    contenido = armar_documento(registro, actualizado, hoy)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(contenido, encoding="utf-8")
    return ruta, len(contenido)


def main(argv=None, sesion=requests):
    parser = argparse.ArgumentParser(description="Baja el directorio de oferentes de astroturismo al corpus.")
    parser.add_argument("--destino", type=Path, default=CARPETA_DESTINO)
    parser.add_argument("--forzar", action="store_true", help="reemplaza los .txt existentes")
    parser.add_argument("--seco", action="store_true", help="no escribe nada")
    args = parser.parse_args(argv)

    if not args.destino.parent.exists():
        raise SystemExit(f"No existe {args.destino.parent}: revisa la ruta del corpus (--destino).")
    try:
        registros, actualizado = bajar_oferentes(sesion)
    except (requests.RequestException, RuntimeError) as error:
        raise SystemExit(f"No se pudo leer la capa de ArcGIS: {error}")
    print(f"{len(registros)} oferentes en la capa (actualizada en la fuente: {actualizado or 'sin fecha'}).")

    nuevos, existentes, caracteres = 0, 0, 0
    for registro in registros:
        if args.seco:
            caracteres += len(armar_documento(registro, actualizado))
            continue
        ruta, largo = guardar(registro, args.destino, actualizado, args.forzar)
        if ruta is None:
            existentes += 1
        else:
            nuevos += 1
            caracteres += largo
    chunks = round(caracteres / CARACTERES_POR_CHUNK)
    if args.seco:
        print(f"[seco] escribiría {len(registros)} .txt, ~{chunks} chunks. No se escribió nada.")
        return
    print(f"guardados: {nuevos}  ya existían: {existentes} (usa --forzar para refrescarlos)  ~{chunks} chunks nuevos")
    if nuevos:
        print('\nAhora reindexa (solo calcula embeddings de lo nuevo):\n'
              '  python src/indexar_corpus.py --corpus "../../Corpus RAG astroturismo" '
              '--patron "*.txt" --coleccion astroturismo_real')


if __name__ == "__main__":
    main(sys.argv[1:])
