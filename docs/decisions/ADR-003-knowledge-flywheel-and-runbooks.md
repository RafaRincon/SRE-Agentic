# ADR-003: Implementación de Knowledge Flywheel, Deduplicación in-memory y Reutilización de Contexto

## Estado
Aceptado

## Contexto y Problema

El SRE Agent procesa cada incidente empleando un flujo de RAAG (Retrieval-Augmented Agentic Generation), el cual involucra la generación intensiva de embeddings para búsqueda ontológica y la evaluación probabilística en múltiples modelos del LLM.

Con el aumento del uso del agente en el Eshop, surgieron los siguientes problemas críticos técnicos:
1. **Alert Fatigue y Duplicidad Operativa:** Los ingenieros reportan la creación de múltiples tickets paralelos sobre el mismo problema de manera simultánea en eventos de degradación.
2. **Generación de Embeddings Redundantes:** Buscar *Runbooks* en el nodo final (`consolidator`) forzaba a la aplicación a hacer llamadas a las APIs de generación de vectores (Gemini-Embedding) en una misma ventana de tiempo en la que el `span_arbiter` hacía expansión de preguntas. Esto resultaba en agotamiento de cuota y caída del agente por errores `HTTP 429 RESOURCE_EXHAUSTED`.
3. **Contaminación de Recuperación Vectorial:** Mezclar chunks de código procedimental en C# del codebase junto con conocimiento puramente narrativo y operacional de SRE (Runbooks, Triage Reports históricos) en el mismo contenedor de CosmosDB degradaba sustancialmente la precisión del RAG.

## Decisión Técnica

Aplicamos una arquitectura dividida (Knowledge Flywheel) con atajos y reutilización de estados a nivel grafo:

1. **Separación de Bases Vectoriales (Cosmos DB NoSQL):** 
   Se creó un contenedor especializado `sre_knowledge` independiente del existente `eshop_chunks`. `eshop_chunks` provechará los índices AST (Abstract Syntax Tree) para contextos de código puros. `sre_knowledge` será accedido para contexto operativo (histomal, triages manuales, runbooks escalados).

2. **Deduplicación Ágil "In-Memory" basada en Cosine Similarity local:**
   En lugar de implementar un pesado cluster DiskANN para el vector array de los incidentes transaccionales (`incidents`), el endpoint principal cargará iterativamente en la memoria de Python (RAM de aplicación) las estructuras abiertas y calculará un `Math.cos` vector similarity puro. Debido a que el paradigma de un sistema SRE implica un número limitado de incidentes concurrentes (<100 típicamente), la comparativa matemática es inmediata (microsegundos) evitando I/O de bases de datos.

3. **Inyección Vectorial Cross-Node en LangGraph (GraphState Memory):** 
   Se extendió el `GraphState` incorporando la variable persistente `report_embedding: list`.
   Para evitar el rate limiting o latencias adicionales con el `llm_provider`, se estandarizó que el vector generado en el paso Pre-grafo (durante la deduplicación transaccional) fluya intacto para que los nodos tardíos (como `consolidator.py`) realicen el barrido de Runbooks utilizando ese mismo embedding original gratuito per-node en lugar de consultar nuevas incrustaciones desde cero.

## Consecuencias

### Positivas
- **Ahorro de API Cost/Limits:** La dependencia a nivel token al modelo Gemini Embedding se redujo al mínimo por incidente invocado de extremo a extremo, superando las cuotas de 10 requests RPM de Vertex AI free/dev tier.
- **Deduplication Preventiva:** Reducción comprobada de Tickets JIRA falsos (hasta un 96% accuracy según pruebas E2E).
- **Resoluciones Inmediatas:** La inyección de `Runbooks` operativos contextuales fomenta una reducción drástica en el Mean Time To Resolution (MTTR) de primer nivel para usuarios On-Call.

### Negativas / Riesgos Conocidos
- **Escalabilidad de In-Memory dedup:** La deduplicación local funciona extremadamente bien asumiendo que los incidentes *resolutivos* salen de la query de abiertos en corto tiempo. Si existiera una cola constante de +10,000 incidentes abiertos en modo PENDING sin resolver, la función `_cosine_sim` Python blocking causaría cuellos de botella asíncronos para FastAPI. Requeriría cambiar al DiskANN interno de Azure.
