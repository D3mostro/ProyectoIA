from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

texto = (
    "La luna del 14 de diciembre estará iluminada en un 30 %, buena noche para ver las Gemínidas.\n\n"
    "Open-Meteo solo pronostica nubes hasta 16 días hacia adelante; más allá no hay datos.\n\n"
    "Toda reserva en un observatorio necesita la aprobación de un supervisor humano."
)

for nombre, splitter in [
    ("FIJO", CharacterTextSplitter(separator="", chunk_size=110, chunk_overlap=0)),
    ("RECURSIVO", RecursiveCharacterTextSplitter(chunk_size=110, chunk_overlap=20)),
]:
    print(f"**** CHUNK {nombre} ****")
    for i, chunk in enumerate(splitter.split_text(texto)):
        print(f"--- chunk {i} ({len(chunk)} caracteres) ---")
        print(chunk)
