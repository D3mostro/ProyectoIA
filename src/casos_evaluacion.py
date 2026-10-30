from pipeline_texto import CASOS

# 20 consultas de astroturismo con referencia escrita a mano .
# Las 3 primeras son las del Bloque 15, varias de las nuevas traen fechas, cifras o negaciones a propósito, porque son los casos donde la métrica falla.
CASOS_EVALUACION = CASOS + [
    {
        "consulta": "¿Cuándo es el máximo de la lluvia de meteoros Gemínidas?",
        "referencia": "La lluvia de las Gemínidas alcanza su máximo cada año alrededor del 13 y 14 de diciembre.",
    },
    {
        "consulta": "¿A qué altura está el observatorio ALMA?",
        "referencia": "ALMA está en el llano de Chajnantor, en la región de Antofagasta, a unos 5.000 metros de altura.",
    },
    {
        "consulta": "¿Dónde está el Observatorio Paranal?",
        "referencia": "El Observatorio Paranal está en el cerro Paranal, en la región de Antofagasta, a 2.635 metros de altura, y alberga el Very Large Telescope.",
    },
    {
        "consulta": "¿Por qué es famoso el Valle del Elqui?",
        "referencia": "El Valle del Elqui, en la región de Coquimbo, es famoso por sus cielos limpios, sus observatorios turísticos y el pisco.",
    },
    {
        "consulta": "¿Cuál fue el primer santuario internacional de cielo oscuro del mundo?",
        "referencia": "El Santuario Gabriela Mistral, en el Valle del Elqui, fue declarado en 2015 el primer Santuario Internacional de Cielo Oscuro del mundo.",
    },
    {
        "consulta": "¿Qué hará el Observatorio Vera C. Rubin?",
        "referencia": "El Observatorio Vera C. Rubin, en el cerro Pachón de la región de Coquimbo, hará un sondeo del cielo austral durante diez años.",
    },
    {
        "consulta": "¿Qué son las Nubes de Magallanes?",
        "referencia": "Las Nubes de Magallanes son dos galaxias enanas satélites de la Vía Láctea que se ven a simple vista desde el hemisferio sur.",
    },
    {
        "consulta": "¿Cuál es el mejor momento del mes para observar estrellas?",
        "referencia": "El mejor momento es cerca de la luna nueva, porque la luz de la Luna no opaca las estrellas débiles.",
    },
    {
        "consulta": "¿Qué es un eclipse total de Sol?",
        "referencia": "Un eclipse total de Sol ocurre cuando la Luna se interpone entre la Tierra y el Sol y lo cubre por completo.",
    },
    {
        "consulta": "¿Cuándo fue el último eclipse total de Sol visible en Chile?",
        "referencia": "El último eclipse total de Sol visible en Chile fue el 14 de diciembre de 2020, en la región de La Araucanía.",
    },
    {
        "consulta": "¿Para qué sirve la Cruz del Sur?",
        "referencia": "La Cruz del Sur es una constelación del cielo austral que sirve para ubicar el polo sur celeste.",
    },
    {
        "consulta": "¿Cuál es el planeta más brillante del cielo nocturno?",
        "referencia": "Venus es el planeta más brillante del cielo nocturno y se ve cerca del horizonte al amanecer o al atardecer.",
    },
    {
        "consulta": "¿Qué es un año luz?",
        "referencia": "Un año luz es la distancia que recorre la luz en un año, unos 9,46 billones de kilómetros.",
    },
    {
        "consulta": "¿Se puede ver la Vía Láctea desde una ciudad grande?",
        "referencia": "No, en una ciudad grande la contaminación lumínica impide ver la Vía Láctea.",
    },
    {
        "consulta": "¿Por qué se ven pocas estrellas en Santiago?",
        "referencia": "En Santiago se ven pocas estrellas por la contaminación lumínica de la ciudad y el smog.",
    },
    {
        "consulta": "¿Qué es el Telescopio Extremadamente Grande (ELT)?",
        "referencia": "El Telescopio Extremadamente Grande de ESO se construye en el cerro Armazones, en la región de Antofagasta, y tendrá un espejo principal de 39 metros.",
    },
    {
        "consulta": "¿Necesito un telescopio para empezar a observar el cielo?",
        "referencia": "No se necesita un telescopio para empezar; bastan unos binoculares, una linterna roja y un mapa del cielo.",
    },
]
