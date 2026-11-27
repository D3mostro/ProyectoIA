"""Convierte a .txt limpios los PDF/HTML oficiales que ya están en el corpus pero no se indexan.

indexar_corpus.py solo lee *.txt, así que los PDF y HTML descargados el 10-oct en
"Corpus RAG astroturismo/originales" nunca entraron al RAG. Este script convierte solo una
lista curada (en español y útil para el turista) y deja fuera lo que ensucia la recuperación:
la estrategia de turismo 2035 (genérica), los Diario Oficial de trámites, las páginas de
licencias, las páginas en inglés y el PDF de AURA 2015 (mitad en inglés).

Uso (desde la raíz del repo):
    python src/convertir_originales.py            # convierte la lista CURADOS
    python src/convertir_originales.py --seco     # solo muestra qué haría y cuántos chunks aprox.
    python src/convertir_originales.py --forzar   # reemplaza .txt ya convertidos

- No llama a Gemini ni a internet: no gasta cuota.
- Escribe en originales/convertidos_es/ con el mismo encabezado que los demás .txt.
- "Descargado" sale de la fecha de modificación del original: si copias la carpeta sin
  conservar fechas, esa fecha cambia.
- No sobrescribe un .txt existente salvo con --forzar.
- Necesita pypdf para los PDF (pip install pypdf==7.1.0).
"""
import argparse
import logging
import re
import sys
from collections import Counter
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_ORIGINALES = RAIZ.parent.parent / "Corpus RAG astroturismo" / "originales"
SUBCARPETA_DESTINO = "convertidos_es"
CARACTERES_POR_CHUNK = 250   # chunk_size=300 menos chunk_overlap=50 de indexar_corpus.py
MINIMO_CARACTERES = 500      # menos que esto suele ser una página que carga con JavaScript

CC_BY_ESO = "CC BY 4.0 (ESO, según licencias_y_terminos/eso_copyright.html)"
SIN_VERIFICAR = "No verificada (uso educativo; el corpus no se sube a GitHub)"
PUBLICO_ESTADO = "Documento público del Estado de Chile (licencia no verificada)"

# ruta relativa a originales/ -> (título, fuente si el archivo no la trae, licencia)
CURADOS = {
    "eso_visitas.html": ("Visitas a los Observatorios de ESO en Chile",
                         "https://www.eso.org/public/chile/about-eso/visitors/", CC_BY_ESO),
    "eso_flyer_visitas_paranal_lasilla.pdf": ("Folleto ESO: visitas a Paranal y La Silla", None, CC_BY_ESO),
    "alma_visitas.html": ("Visitas al Observatorio ALMA", None, SIN_VERIFICAR),
    "chiletravel_4_destinos_2024.html": ("Chile Travel: 4 destinos de cielos despejados", None, SIN_VERIFICAR),
    "chiletravel_4_observatorios_2025.html": ("Chile Travel: 4 observatorios para conocer el cielo nocturno",
                                              None, SIN_VERIFICAR),
    "mma_luminica_norma.html": ("Ministerio del Medio Ambiente: norma lumínica", None, PUBLICO_ESTADO),
    "mma_luminica_preguntas_frecuentes.html": ("Ministerio del Medio Ambiente: preguntas frecuentes norma lumínica",
                                               None, PUBLICO_ESTADO),
    "ds2_2023_areas_astronomicas_diario_oficial_2023-06-27.pdf": (
        "Decreto Supremo 2 de 2023: áreas de valor científico para la observación astronómica", None, PUBLICO_ESTADO),
    "starlight_listado_certificaciones_2025.pdf": ("Listado de certificaciones Starlight", None, SIN_VERIFICAR),
    "sernatur_coquimbo_astroturismo_2019.pdf": ("Sernatur: astroturismo en la Región de Coquimbo (2019)",
                                                None, PUBLICO_ESTADO),
    "hoja_ruta_astroturismo_resumen_2016.pdf": ("Hoja de Ruta Astroturismo Chile 2015-2025 (resumen)",
                                                None, PUBLICO_ESTADO),
    "astronomia_andina/cultura_cielo_nocturno_atacama.html": (
        "Ministerio de las Culturas: Cielo Nocturno, los cielos del desierto de Atacama", None, PUBLICO_ESTADO),
    "astronomia_andina/mineduc_programa_lickanantay.pdf": ("Mineduc: programa de lengua y cultura lickanantay",
                                                           None, PUBLICO_ESTADO),
    "region_metropolitana/turismoastronomico_cl.html": ("Turismo Astronómico en la Región Metropolitana",
                                                        "https://www.turismoastronomico.cl/", SIN_VERIFICAR),
    "region_metropolitana/uc_ruta_astroturistica_santiago_2019.html": (
        "UC: primera ruta astroturística de Santiago (2019)", None, SIN_VERIFICAR),
}

ETIQUETAS_RUIDO = {"script", "style", "noscript", "svg", "nav", "header", "footer", "form",
                   "aside", "select", "button", "iframe", "template"}
ETIQUETAS_BLOQUE = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr",
                    "section", "article", "main", "table", "ul", "ol", "dd", "dt"}


class _ExtractorHTML(HTMLParser):
    """Texto visible del HTML sin menús, pies de página ni scripts. Prefiere <main> si existe."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ruido = 0
        self.en_main = 0
        self.todo, self.main = [], []
        self.canonica = None

    def handle_starttag(self, etiqueta, atributos):
        atributos = dict(atributos)
        if etiqueta == "link" and atributos.get("rel") == "canonical":
            self.canonica = atributos.get("href")
        if etiqueta == "meta" and atributos.get("property") == "og:url" and not self.canonica:
            self.canonica = atributos.get("content")
        if etiqueta in ETIQUETAS_RUIDO:
            self.ruido += 1
        elif etiqueta == "main":
            self.en_main += 1
        if etiqueta in ETIQUETAS_BLOQUE:
            self._agregar("\n")

    def handle_startendtag(self, etiqueta, atributos):
        self.handle_starttag(etiqueta, atributos)
        if etiqueta in ETIQUETAS_RUIDO and self.ruido:
            self.ruido -= 1

    def handle_endtag(self, etiqueta):
        if etiqueta in ETIQUETAS_RUIDO and self.ruido:
            self.ruido -= 1
        elif etiqueta == "main" and self.en_main:
            self.en_main -= 1
        if etiqueta in ETIQUETAS_BLOQUE:
            self._agregar("\n")

    def handle_data(self, datos):
        if not self.ruido:
            self._agregar(datos)

    def _agregar(self, texto):
        self.todo.append(texto)
        if self.en_main:
            self.main.append(texto)


def texto_de_html(html):
    """Devuelve (texto, url_canónica o None)."""
    extractor = _ExtractorHTML()
    extractor.feed(html)
    principal = "".join(extractor.main)
    texto = principal if len(principal.strip()) >= MINIMO_CARACTERES else "".join(extractor.todo)
    return limpiar(texto), extractor.canonica


def texto_de_pdf(ruta):
    from pypdf import PdfReader  # import tardío: los tests de HTML no necesitan pypdf
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # PDFs del Estado traen xref rotos: avisos sin efecto
    paginas = [pagina.extract_text() or "" for pagina in PdfReader(str(ruta)).pages]
    return limpiar("\n".join(paginas))


def limpiar(texto, repeticiones_max=3, largo_cabecera=80):
    """Normaliza espacios y quita cabeceras/pies repetidos (líneas cortas que aparecen 3+ veces)."""
    lineas = [re.sub(r"\s+", " ", linea).strip() for linea in texto.splitlines()]
    lineas = [linea for linea in lineas if linea and not re.fullmatch(r"(página\s*)?\d{1,3}(\s*de\s*\d+)?", linea,
                                                                    flags=re.IGNORECASE)]
    conteo = Counter(linea for linea in lineas if len(linea) <= largo_cabecera)
    repetidas = {linea for linea, n in conteo.items() if n >= repeticiones_max}
    limpias, anterior = [], None
    for linea in lineas:
        if linea in repetidas or linea == anterior:
            continue
        limpias.append(linea)
        anterior = linea
    return "\n".join(limpias)


def armar_documento(titulo, fuente, licencia, descargado, texto, hoy=None):
    """Mismo encabezado que los .txt de Wikipedia, más la fecha de conversión."""
    return (f"Título: {titulo}\nFuente: {fuente}\nLicencia: {licencia}\n"
            f"Descargado: {descargado}\nConvertido: {(hoy or date.today()).isoformat()}\n\n{texto}\n")


def convertir(relativa, carpeta=CARPETA_ORIGINALES, curados=None):
    """Devuelve (contenido_txt, caracteres_de_texto). Lanza FileNotFoundError si falta el original."""
    titulo, fuente, licencia = (CURADOS if curados is None else curados)[relativa]
    ruta = Path(carpeta) / relativa
    if ruta.suffix.lower() == ".pdf":
        texto, canonica = texto_de_pdf(ruta), None
    else:
        texto, canonica = texto_de_html(ruta.read_text(encoding="utf-8", errors="ignore"))
    fuente = fuente or canonica or f"archivo local originales/{relativa}"
    descargado = date.fromtimestamp(ruta.stat().st_mtime).isoformat()
    return armar_documento(titulo, fuente, licencia, descargado, texto), len(texto)


def ruta_destino(relativa, carpeta=CARPETA_ORIGINALES):
    return Path(carpeta) / SUBCARPETA_DESTINO / (Path(relativa).stem + ".txt")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convierte PDF/HTML curados del corpus a .txt.")
    parser.add_argument("--originales", type=Path, default=CARPETA_ORIGINALES)
    parser.add_argument("--forzar", action="store_true", help="reemplaza .txt ya convertidos")
    parser.add_argument("--seco", action="store_true", help="no escribe nada")
    args = parser.parse_args(argv)

    if not args.originales.exists():
        raise SystemExit(f"No existe {args.originales}: revisa la ruta (--originales).")
    total_chunks, nuevos = 0, 0
    for relativa in CURADOS:
        destino = ruta_destino(relativa, args.originales)
        if destino.exists() and not args.forzar and not args.seco:
            print(f"ya existe     {destino.name} (usa --forzar para reemplazarlo)")
            continue
        try:
            contenido, caracteres = convertir(relativa, args.originales)
        except FileNotFoundError:
            print(f"NO ENCONTRADO {relativa}")
            continue
        chunks = round(caracteres / CARACTERES_POR_CHUNK)
        aviso = "  AVISO: casi vacío, revisar a mano" if caracteres < MINIMO_CARACTERES else ""
        if args.seco:
            print(f"convertiría   {relativa}: {caracteres} caracteres, ~{chunks} chunks{aviso}")
            total_chunks += chunks
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(contenido, encoding="utf-8")
        nuevos += 1
        total_chunks += chunks
        print(f"guardado      {destino.name}: {caracteres} caracteres, ~{chunks} chunks{aviso}")
    print(f"\nTotal aproximado: ~{total_chunks} chunks (= embeddings de Gemini al indexar).")
    if nuevos:
        print('Revisa 2 o 3 .txt a mano y luego reindexa (solo calcula embeddings de lo nuevo):\n'
              '  python src/indexar_corpus.py --corpus "../../Corpus RAG astroturismo" '
              '--patron "*.txt" --coleccion astroturismo_real')


if __name__ == "__main__":
    main(sys.argv[1:])
