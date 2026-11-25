"""Bloque 32 (Core), Paso 2: punto de supervisión humana para activar una versión de prompt.

Grafo de LangGraph: proponer -> supervisar (interrupt) -> aplicar. Nada se activa hasta que
una persona responda con Command(resume=True). Con Command(resume=False) se mantiene la
versión anterior. La activación aprobada queda escrita en `registro_prompts.json`.

Diferencias con el código de la página y de la profesora (fallos verificados):
- La página y la profesora guardan `aprobado = decision` sin revisar el tipo. Un
  Command(resume="no") es un texto no vacío, o sea verdadero, y despliega. Aquí solo el booleano
  True aprueba; cualquier otro valor cuenta como rechazo y queda anotado en el estado.
- El nodo "proponer" de la página no hace nada. Aquí revisa que la huella exista en el registro
  (el supervisor no debe ver una versión imposible) y arma el diff contra la versión activa.
  La lección pide revisar el diff antes de aprobar, pero su interrupt solo muestra la huella.
- "aplicar" de la página solo imprime. Aquí llama a RegistroDePrompts.activar, con aprobador.
- `resolver` (idea de la lección y de la profesora) revisa get_state antes de reanudar, para
  no aprobar algo que nadie revisó ni aprobar dos veces la misma solicitud. Además exige bool.
- Proponer la huella que ya está activa no pide aprobación: no hay nada que decidir.
- Ideas del repo del ayudante: `resolver` exige la huella que la persona revisó (no se aprueba
  una versión distinta a la del diff) y "aplicar" vuelve a leer el registro y se niega a
  activar si la versión activa cambió mientras se esperaba la aprobación.

Límite: usa InMemorySaver, así que la solicitud pendiente se pierde si se cierra el proceso.
Para que la aprobación sobreviva un reinicio hace falta `langgraph-checkpoint-sqlite` (no
está en requirements.txt) y pasar `checkpointer=SqliteSaver(...)` a `construir_grafo`.

Uso:
    python src/grafo_supervision.py --demo     # aprobación y rechazo con un registro temporal
    python src/grafo_supervision.py --huella <h> --aprobar s|n   # sobre registro_prompts.json
No usa la API de Gemini.
"""
import argparse
import tempfile
import uuid
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from registro_prompts import (
    ARCHIVO_REGISTRO,
    NOMBRE_PROMPT,
    RegistroDePrompts,
    diff_prompts,
    registrar_actuales,
)


class EstadoDespliegue(TypedDict, total=False):
    nombre: str
    huella_nueva: str
    huella_activa: str
    diff: list
    aprobado: bool
    decision_recibida: str
    resultado: str


def construir_grafo(registro, aprobador="German", checkpointer=None):
    """Devuelve el grafo compilado. `registro` es un RegistroDePrompts."""

    def proponer(estado: EstadoDespliegue):
        nombre, huella = estado["nombre"], estado["huella_nueva"]
        nueva = registro.buscar(huella)
        if nueva is None or nueva["nombre"] != nombre:
            raise ValueError(f"La huella {huella} no está registrada para {nombre}.")
        activa = registro.activa(nombre)
        return {
            "huella_activa": activa["huella"] if activa else "",
            "diff": diff_prompts(activa["texto"], nueva["texto"]) if activa else [],
        }

    def supervisar(estado: EstadoDespliegue):
        if estado["huella_activa"] == estado["huella_nueva"]:
            return {"aprobado": False, "decision_recibida": "sin_cambios"}
        decision = interrupt({
            "motivo": "aprobar nueva version de prompt en produccion",
            "nombre": estado["nombre"],
            "huella_propuesta": estado["huella_nueva"],
            "huella_activa": estado["huella_activa"] or None,
            "diff": estado["diff"],
        })
        # Solo True aprueba: "no", "false" o 1 no despliegan nada.
        return {"aprobado": decision is True, "decision_recibida": repr(decision)}

    def aplicar(estado: EstadoDespliegue):
        if estado.get("decision_recibida") == "sin_cambios":
            return {"resultado": f"la version {estado['huella_nueva']} ya estaba activa"}
        if estado["aprobado"]:
            registro.recargar()  # otro proceso pudo activar algo mientras se esperaba
            activa = registro.activa(estado["nombre"])
            if (activa["huella"] if activa else "") != estado["huella_activa"]:
                raise ValueError("La version activa cambio durante la revision "
                                 f"({estado['huella_activa'] or 'ninguna'} -> {activa['huella']}). "
                                 "Inicia otra solicitud.")
            registro.activar(estado["nombre"], estado["huella_nueva"], aprobador,
                             motivo="aprobado en grafo_supervision")
            return {"resultado": f"version {estado['huella_nueva']} activada"}
        return {"resultado": f"version {estado['huella_nueva']} rechazada, se mantiene "
                             f"{estado['huella_activa'] or 'ninguna'}"}

    grafo = StateGraph(EstadoDespliegue)
    grafo.add_node("proponer", proponer)
    grafo.add_node("supervisar", supervisar)
    grafo.add_node("aplicar", aplicar)
    grafo.add_edge(START, "proponer")
    grafo.add_edge("proponer", "supervisar")
    grafo.add_edge("supervisar", "aplicar")
    grafo.add_edge("aplicar", END)
    return grafo.compile(checkpointer=checkpointer or InMemorySaver())


def nuevo_thread_id(prefijo="despliegue"):
    return f"{prefijo}-{uuid.uuid4().hex[:8]}"


def proponer_despliegue(app, huella, nombre=NOMBRE_PROMPT):
    """Corre el grafo hasta el interrupt. Devuelve (config, resultado)."""
    config = {"configurable": {"thread_id": nuevo_thread_id()}}
    resultado = app.invoke({"nombre": nombre, "huella_nueva": huella}, config=config)
    return config, resultado


def pendiente(app, config):
    """El valor del interrupt pendiente, o None si ese thread no espera nada."""
    snapshot = app.get_state(config)
    return snapshot.interrupts[0].value if snapshot.next and snapshot.interrupts else None


def resolver(app, config, aprobar, huella_revisada):
    """Reanuda con la decisión. `huella_revisada` es la que la persona vio en el diff."""
    if not isinstance(aprobar, bool):
        raise TypeError("aprobar debe ser True o False.")
    valor = pendiente(app, config)
    if valor is None:
        raise ValueError("Este thread_id no tiene ninguna aprobación pendiente.")
    if valor["huella_propuesta"] != huella_revisada:
        raise ValueError(f"La huella pendiente ({valor['huella_propuesta']}) no es la revisada "
                         f"({huella_revisada}).")
    return app.invoke(Command(resume=aprobar), config=config)


def demo():
    with tempfile.TemporaryDirectory() as carpeta:
        registro = RegistroDePrompts(Path(carpeta) / "registro_demo.json")
        v1, v2 = registrar_actuales(registro)
        registro.activar(NOMBRE_PROMPT, v1["huella"], "German", "estado inicial (Bloque 26)")
        app = construir_grafo(registro)

        print("1) rechazo de", v2["huella"])
        config, resultado = proponer_despliegue(app, v2["huella"])
        print("   interrumpido?", "__interrupt__" in resultado)
        valor = pendiente(app, config)
        print("   pendiente:", valor["motivo"], valor["huella_activa"], "->", valor["huella_propuesta"])
        print("   diff:", len(valor["diff"]), "lineas")
        print("  ", resolver(app, config, False, v2["huella"])["resultado"])
        print("   activa:", registro.activa(NOMBRE_PROMPT)["huella"])

        print("2) aprobación de", v2["huella"], "(otro thread_id)")
        config2, _ = proponer_despliegue(app, v2["huella"])
        print("  ", resolver(app, config2, True, v2["huella"])["resultado"])
        print("   activa:", registro.activa(NOMBRE_PROMPT)["huella"])

        print("3) aprobar otra vez el mismo thread_id")
        try:
            resolver(app, config2, True, v2["huella"])
        except ValueError as error:
            print("   ValueError:", error)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Punto de supervisión para activar un prompt (B32).")
    parser.add_argument("--demo", action="store_true", help="aprobación y rechazo con registro temporal")
    parser.add_argument("--huella", help="huella registrada que se propone activar")
    parser.add_argument("--aprobar", choices=["s", "n"], help="respuesta del supervisor")
    parser.add_argument("--archivo", default=ARCHIVO_REGISTRO, help="JSON del registro")
    args = parser.parse_args(argv)

    if args.demo or not args.huella:
        return demo()

    registro = RegistroDePrompts(args.archivo)
    app = construir_grafo(registro)
    config, resultado = proponer_despliegue(app, args.huella)
    valor = pendiente(app, config)
    if valor is None:
        print(resultado.get("resultado"))
        return 0
    print("Pendiente:", valor["motivo"])
    print("Activa:", valor["huella_activa"], "| propuesta:", valor["huella_propuesta"])
    for linea in valor["diff"]:
        print(linea)
    respuesta = args.aprobar or input("¿Aprobar? (s/n): ").strip().lower()
    print(resolver(app, config, respuesta == "s", args.huella)["resultado"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
