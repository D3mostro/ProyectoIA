BANCO_DE_PROMPTS = [
    {
        "nombre": "clasificar_consulta",
        "system_instruction": (
            "Eres un asistente de astroturismo en Chile. Clasifica la consulta del usuario. "
            "Responde únicamente con un objeto JSON válido, sin texto adicional y sin bloques de markdown. "
            "El JSON debe tener exactamente estas claves: 'categoria' (string: 'observacion' si pregunta "
            "qué se puede ver en el cielo, 'clima' si pregunta por nubes o condiciones del tiempo, "
            "'logistica' si pregunta por reservas, horarios, accesos o precios, u 'otro') y "
            "'confianza' (número entre 0 y 1)."
        ),
        "ejemplo_usuario": "¿Puedo ver las Perseidas desde el Valle del Elqui?",
        "claves_esperadas": ["categoria", "confianza"],
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "categoria": {"type": "STRING"},
                "confianza": {"type": "NUMBER"},
            },
            "required": ["categoria", "confianza"],
        },
    },
    {
        "nombre": "extraer_datos_consulta",
        "system_instruction": (
            "Eres un asistente de astroturismo en Chile. Extrae los datos de la consulta del usuario. "
            "Responde únicamente con un objeto JSON válido, sin texto adicional y sin bloques de markdown. "
            "El JSON debe tener exactamente estas claves: 'lugar' (string: lugar de Chile mencionado), "
            "'fecha' (string: fecha o rango de fechas tal como las escribió el usuario) e "
            "'interes' (string: qué quiere saber o hacer el usuario, en pocas palabras). "
            "Si un dato no aparece en la consulta, usa el string 'no indicado'."
        ),
        "ejemplo_usuario": "Voy a Vicuña la noche del 13 al 14 de diciembre de 2026. ¿Qué puedo ver?",
        "claves_esperadas": ["lugar", "fecha", "interes"],
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "lugar": {"type": "STRING"},
                "fecha": {"type": "STRING"},
                "interes": {"type": "STRING"},
            },
            "required": ["lugar", "fecha", "interes"],
        },
    },
    {
        "nombre": "resumir_texto",
        "system_instruction": (
            "Eres un asistente de astroturismo en Chile. Resume el texto que te entregue el usuario. "
            "Responde únicamente con un objeto JSON válido, sin texto adicional y sin bloques de markdown. "
            "El JSON debe tener exactamente estas claves: 'resumen' (string: una sola oración) y "
            "'palabras_clave' (lista de 3 a 5 strings)."
        ),
        "ejemplo_usuario": (
            "La escala de Bortle clasifica la oscuridad del cielo nocturno en nueve niveles, desde 1 "
            "(cielo excepcionalmente oscuro) hasta 9 (cielo de centro de ciudad). Sirve para saber cuántas "
            "estrellas y objetos débiles se pueden ver desde un lugar. El Valle del Elqui suele considerarse "
            "un destino de cielos muy oscuros, aunque la luna llena y las nubes reducen la visibilidad."
        ),
        "claves_esperadas": ["resumen", "palabras_clave"],
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "resumen": {"type": "STRING"},
                "palabras_clave": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["resumen", "palabras_clave"],
        },
    },
]
