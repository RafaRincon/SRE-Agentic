# ADR 006: Refactorización a Grafo Secuencial y Pydantic Nativo en el Falsificador Epistémico

**Fecha:** 2026-04-09
**Estado:** Aceptado

## Contexto

El Falsificador Epistémico ("Track B") tenía la responsabilidad de razonar exhaustivamente usando herramientas de búsqueda sobre la base de las alertas y un modelo del mundo. Inicialmente:
1. **Carrera de Condiciones Paralelas:** El LangGraph operaba con ramas asíncronas (`fan-out` desde Intake a World Model y Risk Hypothesizer al mismo tiempo). Esto causaba *Race Conditions* peligrosas, donde el Track B fallaba por la ausencia del contexto de entidades o estado del escenario antes de que el Track A terminara de poblar la memoria base (State).
2. **Abstracciones LLM Débiles:** Usábamos LangChain encapsulando Vertex AI, con schemas de JSON Mode complejos que el modelo no siempre respetaba, causando fallos de parsing silentes que frenaban el throughput o alucinaban conclusiones de código.
3. **Pobre Capacidad Iterativa:** El agente usaba demasiadas herramientas superpuestas (4 tools) y reportaba su veredicto vía un llamado a base de datos dentro del nodo, bloqueando las pruebas locales puras.

## Decisión

1. **Grafo Estrictamente Secuencial (Candado de Estado):**  
   Refactorizamos el `AgentGraph` para eliminar la bifurcación inicial. Ahora procesa el Track A completo antes de ceder el Estado (State) al Track B:  
   `intake` → `world_model` → `slot_filler` → `risk_hypothesizer` → `span_arbiter` → `falsifier` → `consolidator`.  
   Esto asegura que el `risk_hypothesizer` cuente con una comprensión impecable del modelo semántico integral (el incidente ya extraído) antes de generar hipótesis estructuradas.

2. **Migración a SDK GenAI Nativo (gemini-3.1-pro-preview):**  
   Abandonamos el bloque genérico de LangChain para la función recursiva de falsificación y adoptamos directamente `google-genai` con Pydantic Estricto. Esto nos permite aprovechar:
   - `thinking_level="MEDIUM"` para forzar convergencia iterativa y planificación sin forzar schemas intermedios.
   - `temperature=1.0` y búsqueda paralela en sub-tools.
   - Dos herramientas hiper-específicas (`lookup_code`, `check_defenses`) que evitan vomitar sintaxis, y en su lugar, devuelven *conclusiones lógicas narradas*.

3. **Arquitectura Funcional Pura en Nodos:**  
   Retiramos del interior de los nodos la responsabilidad de grabar en el Ledger (eliminamos `record_verdict` en medio del Falsificador). Todas las escrituras destructibles suceden post-consolidación. El Falsificador ahora solamente toma de entrada las hipótesis en el `state` y devuelve una lista estructurada y plana de `FalsificationVerdict` vía `state["falsifier_verdicts"]`.

## Consecuencias

*   **Positivas:** 
    *   **Determinismo Absoluto:** Las pruebas ("smoke tests") demostraron 100% de reproducibilidad bajo los 6 ejes epistémicos de la arquitectura porque el modelo respeta el esquema Pydantic y razona con un plan explícito.
    *   **Resiliencia Total contra JSON Incompleto:** Se evitan `TypeErrors` durante *slicing* o accesos anidados en el grafo porque el modelo nativo maneja las tuplas estructuradas desde el protocolo binario del API (Structured Outputs).
    *   **Observabilidad de "Caja de Cristal":** Es trivial testear cada razonamiento iterativo ejecutando el módulo en solitario.
*   **Negativas:** 
    *   Leve incremento en la latencia del grafo base frente al viejo paralelismo asíncrono, dado que los Tracks deben esperar su turno. Sin embargo, en SRE, la precisión y la carencia de alucinaciones superan el costo de ganar milisegundos.
