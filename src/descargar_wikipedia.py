"""Bloque 30: completa el corpus real con artículos de Wikipedia en español.

En la muestra de 20 respuestas del Bloque 30, el modelo dijo "el contexto no alcanza" en 8.
Revisando los 46 .txt, la información de 5 de ellas no está en el corpus (qué es un año luz,
el eclipse de 2020, el primer santuario de cielo oscuro, si hace falta telescopio para empezar
y el mejor momento del mes para observar). Este script baja esos artículos con el mismo
encabezado que los 38 que ya existen (Título, Fuente, Licencia, Descargado).

Uso (desde la raíz del repo):
    python src/descargar_wikipedia.py                  # baja la lista TITULOS_B30
    python src/descargar_wikipedia.py "Año luz" ...    # baja otros títulos
    python src/descargar_wikipedia.py --seco           # solo muestra qué bajaría

- No llama a Gemini: no gasta cuota. Usa la API pública de Wikipedia (MediaWiki).
- Si un título no existe, no inventa nada: muestra hasta 3 títulos parecidos para elegir.
- No sobrescribe un archivo que ya existe (usa --forzar para volver a bajarlo).
- Los textos de Wikipedia son CC BY-SA 4.0: quedan en la carpeta del corpus, que no se sube
  a GitHub (regla del vault).
"""
import argparse
import sys
from datetime import date
from pathlib import Path
from urllib.parse import quote

import requests

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DESTINO = RAIZ.parent.parent / "Corpus RAG astroturismo" / "originales" / "wikipedia_es"
API = "https://es.wikipedia.org/w/api.php"
CABECERAS = {"User-Agent": "ProyectoIA-curso/1.0 (uso educativo)"}
LICENCIA = "CC BY-SA 4.0 (Wikipedia en español)"
TITULOS_B30 = [
    "Año luz",
    "Eclipse solar del 14 de diciembre de 2020",
    "Astronomía amateur",
    "Reserva de cielo oscuro",
]


class TituloNoEncontrado(LookupError):
    def __init__(self, titulo, sugerencias):
        super().__init__(f"'{titulo}' no existe en es.wikipedia. Parecidos: {sugerencias or 'ninguno'}")
        self.sugerencias = sugerencias


def bajar_articulo(titulo, sesion=requests):
    """Devuelve (título_real, texto_plano). Sigue redirecciones. Lanza TituloNoEncontrado."""
    params = {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1,
              "titles": titulo, "format": "json", "formatversion": 2}
    respuesta = sesion.get(API, params=params, headers=CABECERAS, timeout=30)
    respuesta.raise_for_status()
    paginas = respuesta.json().get("query", {}).get("pages", [])
    pagina = paginas[0] if paginas else {"missing": True}
    if pagina.get("missing") or not (pagina.get("extract") or "").strip():
        raise TituloNoEncontrado(titulo, buscar_parecidos(titulo, sesion))
    return pagina["title"], pagina["extract"].strip()


def buscar_parecidos(titulo, sesion=requests, limite=3):
    params = {"action": "query", "list": "search", "srsearch": titulo, "srlimit": limite,
              "format": "json", "formatversion": 2}
    respuesta = sesion.get(API, params=params, headers=CABECERAS, timeout=30)
    respuesta.raise_for_status()
    return [r["title"] for r in respuesta.json().get("query", {}).get("search", [])]


def nombre_archivo(titulo):
    return titulo.replace(" ", "_").replace("/", "-") + ".txt"


def armar_documento(titulo, texto, hoy=None):
    """Mismo encabezado que los .txt existentes del corpus."""
    fuente = "https://es.wikipedia.org/wiki/" + quote(titulo.replace(" ", "_"))
    return (f"Título: {titulo}\nFuente: {fuente}\nLicencia: {LICENCIA}\n"
            f"Descargado: {(hoy or date.today()).isoformat()}\n\n{texto}\n")


def guardar(titulo, texto, carpeta=CARPETA_DESTINO, forzar=False, hoy=None):
    """Escribe el .txt. Devuelve la ruta, o None si ya existía y no se pidió --forzar."""
    ruta = Path(carpeta) / nombre_archivo(titulo)
    if ruta.exists() and not forzar:
        return None
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(armar_documento(titulo, texto, hoy), encoding="utf-8")
    return ruta


def main(argv=None):
    parser = argparse.ArgumentParser(description="Baja artículos de es.wikipedia al corpus real.")
    parser.add_argument("titulos", nargs="*", default=TITULOS_B30)
    parser.add_argument("--destino", type=Path, default=CARPETA_DESTINO)
    parser.add_argument("--forzar", action="store_true", help="sobrescribe archivos existentes")
    parser.add_argument("--seco", action="store_true", help="no escribe nada")
    args = parser.parse_args(argv)

    if not args.destino.parent.exists():
        raise SystemExit(f"No existe {args.destino.parent}: revisa la ruta del corpus (--destino).")
    nuevos = 0
    for titulo in args.titulos:
        try:
            real, texto = bajar_articulo(titulo)
        except TituloNoEncontrado as error:
            print(f"NO ENCONTRADO  {error}")
            continue
        except requests.RequestException as error:
            print(f"ERROR DE RED   {titulo}: {error}")
            continue
        if args.seco:
            print(f"bajaría       {real} ({len(texto)} caracteres) -> {nombre_archivo(real)}")
            continue
        ruta = guardar(real, texto, args.destino, args.forzar)
        if ruta is None:
            print(f"ya existe     {nombre_archivo(real)} (usa --forzar para reemplazarlo)")
        else:
            nuevos += 1
            print(f"guardado      {ruta.name} ({len(texto)} caracteres)")
    if nuevos:
        print('\nAhora reindexa (solo calcula embeddings de lo nuevo):\n'
              '  python src/indexar_corpus.py --corpus "../../Corpus RAG astroturismo" '
              '--patron "*.txt" --coleccion astroturismo_real')


if __name__ == "__main__":
    main(sys.argv[1:])
