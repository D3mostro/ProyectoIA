"""Limpieza del corpus turístico: quita chunks que no ayudan a responder, sin gastar embeddings.

Se aplica DESPUÉS de partir cada documento y de numerar sus chunks, así los chunks que quedan
conservan su id y su texto: `indexar()` los salta (ya están guardados) y solo borra de Chroma
los ids descartados. Limpiar no llama a Gemini. Si se cambian estas reglas y un chunk vuelve a
entrar, ese sí se embebe de nuevo (1 embedding por chunk recuperado).

Motivos de descarte (medidos el 27-11-2026 sobre los 174 .txt del corpus real):
- solo_encabezado: el chunk tiene solo las líneas Título/Fuente/Licencia/Descargado.
- ruido: solo títulos de sección, o casi sin letras (fórmulas mal convertidas, calendarios).
- otro_idioma: chunks en inglés (aviso de cookies de ESO, versión en inglés del folleto ESO y
  de Mamalluca, menú de Space): la app responde en español y ese contenido ya está en español.
- cola_wikipedia: Referencias, Notas, Enlaces externos, Véase también, Galería...
- fuera_de_tema: secciones de Wikipedia que no sirven a un turista de cielos en Chile
  (fórmulas, ciencia ficción, Europa, Argentina...).
- localidad_fuera_de_tema: en artículos de ciudades y valles, las secciones de historia,
  administración, economía, deportes, medios... salvo que el chunk hable de astronomía.

La sección de un chunk es la de su primer carácter: un chunk que empieza en "Lugares de interés"
y termina en "Referencias" se conserva entero (no se recorta: recortar cambiaría el texto y
obligaría a embeberlo de nuevo).

No se borra "Historia" en general: en Astronomía_en_Chile o Constelación sí es contenido útil.
"""
import re
from pathlib import Path

# "Sitio" es el encabezado de los .txt de observatorios_turisticos (scrapeados el 10-oct).
CAMPOS_ENCABEZADO = ("Título", "Fuente", "Licencia", "Descargado", "Convertido", "Actualizado en la fuente", "Sitio")
PATRON_ENCABEZADO = re.compile(r"^(?:%s):" % "|".join(map(re.escape, CAMPOS_ENCABEZADO)))
PATRON_SECCION = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$", re.MULTILINE)  # "== Historia ==" (Wikipedia)
# Línea que es solo un título: "== Morfología ==" o "=== https://sitio.cl/contacto/" (páginas scrapeadas).
PATRON_LINEA_TITULO = re.compile(r"^(={2,})\s*(.+?)\s*(\1)?\s*$")
# "Campo: valor" de las fichas de oferentes: corto pero es un dato, nunca se trata como ruido.
PATRON_CAMPO = re.compile(r"^[A-ZÁÉÍÓÚÑ][^:\n]{2,60}:\s+\S", re.MULTILINE)

SECCIONES_COLA = {
    "referencias", "notas", "notas y referencias", "bibliografía", "enlaces externos",
    "véase también", "galería", "fuentes", "lecturas adicionales", "otras lecturas",
}
SECCIONES_FUERA_DE_TEMA = {
    "en ciencia ficción", "en la cultura popular", "vista del cielo desde un hipotético planeta",
    "formulación matemática", "cálculos", "ley de pogson", "recubrimiento óptico",
    "contaminación lumínica en europa", "eclipse en argentina", "eclipse en uruguay",
}
# Artículos de lugares: interesa qué hay para ver y cómo llegar, no la municipalidad ni el fútbol.
ARTICULOS_LOCALIDAD = {"Vicuña_(Chile)", "Combarbalá", "San_Pedro_de_Atacama", "Pisco_Elqui", "Valle_del_Elqui"}
SECCIONES_LOCALIDAD = {
    "historia", "historia del valle de elqui", "administración", "economía", "demografía",
    "deportes", "medios de comunicación", "servicios", "relaciones internacionales",
    "toponimia", "medio ambiente",
}
# Un chunk de una sección de localidad descartada se conserva si habla de esto.
PATRON_ASTRONOMIA = re.compile(
    r"observator|astron|astrotur|estrell|cielo|telescop|constelac|eclips|planeta|galaxi|nebulos|"
    r"v[ií]a l[aá]ctea|starlight|oscur",
    re.IGNORECASE,
)
PALABRAS_EN = {"the", "and", "of", "to", "is", "in", "with", "for", "are", "this", "that", "from",
               "we", "our", "you", "your", "by", "on", "or", "be"}
PALABRAS_ES = {"el", "la", "de", "que", "y", "en", "los", "las", "del", "por", "con", "una", "para",
               "se", "un", "es", "al", "su"}
PATRON_DATO_PRACTICO = re.compile(r"Direcci[oó]n|Tel[eé]fono|WhatsApp|Ubicaci[oó]n")
MINIMO_LETRAS = 20
PROPORCION_MINIMA_LETRAS = 0.5


def _es_solo_encabezado(texto):
    lineas = [linea.strip() for linea in texto.splitlines() if linea.strip()]
    return bool(lineas) and all(PATRON_ENCABEZADO.match(linea) for linea in lineas)


def _es_ruido(texto):
    lineas = [linea.strip() for linea in texto.splitlines() if linea.strip()]
    if lineas and all(PATRON_LINEA_TITULO.match(linea) for linea in lineas):
        return True  # solo títulos de sección, sin contenido
    if PATRON_CAMPO.search(texto):
        return False
    compacto = re.sub(r"\s+", "", texto)
    letras = sum(c.isalpha() for c in compacto)
    return letras < MINIMO_LETRAS or letras < PROPORCION_MINIMA_LETRAS * len(compacto)


def _es_otro_idioma(texto):
    """Inglés: al menos 4 palabras funcionales inglesas y más del doble que las españolas."""
    if PATRON_DATO_PRACTICO.search(texto):
        return False  # p. ej. el chunk de Elqui Domos con su única dirección, entre frases en inglés
    palabras = re.findall(r"[a-záéíóúñ]+", texto.lower())
    en = sum(p in PALABRAS_EN for p in palabras)
    return en >= 4 and en > 2 * sum(p in PALABRAS_ES for p in palabras)


def _secciones(texto_documento):
    """[(posición, nivel, nombre en minúsculas)] de los títulos "== X ==" del documento."""
    return [(m.start(), len(m.group(1)), m.group(2).strip().lower())
            for m in PATRON_SECCION.finditer(texto_documento)]


def _cadena_de_secciones(secciones, posicion):
    """Títulos activos en `posicion`, del más general al más específico."""
    activas = {}
    for inicio, nivel, nombre in secciones:
        if inicio > posicion:
            break
        activas = {n: s for n, s in activas.items() if n < nivel}
        activas[nivel] = nombre
    return [activas[n] for n in sorted(activas)]


def _posiciones(texto_documento, textos):
    """Posición de cada chunk en el documento (None si no se encuentra: entonces no se descarta)."""
    posiciones, desde = [], 0
    for texto in textos:
        i = texto_documento.find(texto, desde)
        if i < 0:
            i = texto_documento.find(texto)
        posiciones.append(i if i >= 0 else None)
        if i >= 0:
            desde = i + 1
    return posiciones


def motivos_de_descarte(texto_documento, textos_chunks, fuente):
    """Lista paralela a `textos_chunks`: el motivo para descartar cada chunk, o None si se queda."""
    articulo = Path(fuente).stem
    secciones = _secciones(texto_documento)
    motivos = []
    for texto, posicion in zip(textos_chunks, _posiciones(texto_documento, textos_chunks)):
        motivo = None
        if _es_solo_encabezado(texto):
            motivo = "solo_encabezado"
        elif _es_ruido(texto):
            motivo = "ruido"
        elif _es_otro_idioma(texto):
            motivo = "otro_idioma"
        elif secciones and posicion is not None:
            cadena = _cadena_de_secciones(secciones, posicion)
            if any(s in SECCIONES_COLA for s in cadena):
                motivo = "cola_wikipedia"
            elif any(s in SECCIONES_FUERA_DE_TEMA for s in cadena):
                motivo = "fuera_de_tema"
            elif (articulo in ARTICULOS_LOCALIDAD and cadena and cadena[0] in SECCIONES_LOCALIDAD
                  and not PATRON_ASTRONOMIA.search(texto)):
                motivo = "localidad_fuera_de_tema"
        motivos.append(motivo)
    return motivos


def filtrar_chunks(partes, ids, texto_documento, fuente, descartes=None):
    """Quita los chunks que no aportan. Los que quedan conservan su id (y su texto).

    Si se pasa `descartes` (lista), le agrega (id, motivo, texto) de cada chunk quitado.
    """
    motivos = motivos_de_descarte(texto_documento, [p.page_content for p in partes], fuente)
    quedan, ids_quedan = [], []
    for parte, id_, motivo in zip(partes, ids, motivos):
        if motivo is None:
            quedan.append(parte)
            ids_quedan.append(id_)
        elif descartes is not None:
            descartes.append((id_, motivo, parte.page_content))
    return quedan, ids_quedan


def resumir_descartes(descartes, total_chunks):
    """Texto corto con cuántos chunks se quitan por motivo y los archivos más afectados."""
    if not descartes:
        return "Limpieza: no se descarta ningún chunk."
    por_motivo, por_archivo = {}, {}
    for id_, motivo, _ in descartes:
        por_motivo[motivo] = por_motivo.get(motivo, 0) + 1
        archivo = id_.rsplit("-", 1)[0]
        por_archivo[archivo] = por_archivo.get(archivo, 0) + 1
    lineas = [f"Limpieza: se descartan {len(descartes)} de {total_chunks} chunks "
              f"({100 * len(descartes) / total_chunks:.0f} %), sin gastar embeddings:"]
    lineas += [f"  {motivo}: {n}" for motivo, n in sorted(por_motivo.items(), key=lambda x: -x[1])]
    mas = sorted(por_archivo.items(), key=lambda x: -x[1])[:5]
    lineas.append("  archivos con más descartes: " + ", ".join(f"{Path(a).name} ({n})" for a, n in mas))
    return "\n".join(lineas)
