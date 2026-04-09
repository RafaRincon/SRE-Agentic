# ADR-002: Selección de Base de Datos Vectorial (Cosmos DB DiskANN vs Chroma DB)

| Field      | Value                                          |
|------------|-------------------------------------------------|
| **Status** | Accepted                                        |
| **Date**   | 2026-04-08                                      |
| **Author** | SRE Agent Team                                  |
| **Scope**  | `app/providers/db_provider.py`, Infrastructure  |

---

## Contexto

Para nuestro agente de Site Reliability Engineering (SRE) que se encarga del triage de incidentes (`Neuro-Symbolic SRE Agent`), la consulta veloz y precisa del *codebase* (vía RAG) es el núcleo de la arquitectura para evitar alucinaciones.

En la etapa de arquitectura, evaluamos si utilizar una base de datos puramente vectorial enfocada en RAG ágil como **Chroma DB**, o aprovechar las capacidades nativas de **Azure Cosmos DB NoSQL con índices DiskANN**. 

La naturaleza del trabajo de SRE exige correlacionar información de múltiples fuentes (código fuente, logs de errores, métricas, tickets de incidentes y documentación) en tiempo real, lo que hace que la arquitectura y escalabilidad de la base de datos sean fundamentales.

## Decisión

**Resolución:** Se decide establecer **Azure Cosmos DB con DiskANN** como el único almacén de datos (Single Source of Truth) para todo el pipeline, descartando alternativas como Chroma DB.

### Justificación Técnica de la Decisión

#### 1. Particionamiento por Metadatos (Sharded DiskANN)
En el triage de SRE, rara vez se busca en todo el repositorio simultáneamente. Lo óptimo es buscar errores o fragmentos aislando por microservicio, repositorio, equipo o rango de fechas (ej. un despliegue reciente). 
Cosmos DB utiliza *Sharded DiskANN*, optimizando la búsqueda vectorial particionando los índices por clave (en nuestro caso, `service_name`). Esto garantiza que, al pre-filtrar, Cosmos DB busca exclusivamente en la partición relevante dictando respuestas resolutivas en baja latencia. Chroma DB administra métadatos con menor eficiencia a gran escala, lo que degrada el rendimiento al inflarse la base del código.

#### 2. Búsqueda Híbrida Nativa (Vectores + BM25)
La búsqueda semántica pura por vectores ocasionalmente falla buscando nombres exactos de variables, IDs de correlación u hashes de commits. Cosmos DB brinda **búsqueda híbrida nativa** (Similitud Semántica + FullTextScore BM25) vía *Reciprocal Rank Fusion* (RRF). Chroma DB es primariamente de naturaleza semántica pura; perder un nombre de función por falta de keyword matching exacto es algo que el framework del SRE Agent no se puede permitir.

#### 3. Arquitectura No-ETL para Datos Operativos
El agente necesita retener y analizar un historial continuo, no solamente consultar código sin estado persistente. Con Cosmos DB manejamos:
- **Vector Store (eshop_chunks):** Ast-based Embeddings.
- **Transaccional (incidents):** Payload operativo JSON y estado en un mismo clúster. 
- **Ledger Inmutable (incident_ledger):** Trazabilidad de cada alucinación corregida.
Chroma DB nos forzaría a segregar los vectores, gestionando una réplica adicional para la metadata y el histórico, introduciendo latencia de red y un patrón anti-ETL muy complejo.

#### 4. Eficiencia de Memoria a Gran Escala
Al acoplar el código histórico, tickets indexados y a futuro, logs (telemetría), el tamaño del vector store crecerá radicalmente. Chroma DB aplica el algoritmo HNSW el cual requiere alojar todo su índice en RAM (Costoso infraestructuralmente a escala). *DiskANN*, al contrario, transacciona directamente limitando en gran disco SSD, promoviendo a la RAM exclusivamente una compresión vectorizada (Cuantizada). Permite indexar volúmenes masivos de fragmentos de resolución con latencias sub-20ms a un coste muy inferior.

## Consecuencias y Próximos Pasos

- **Consolidación Logística:** Se consolida el uso de `azure-cosmos` SDK para LangGraph (`langgraph-checkpoint-cosmosdb`) y el retriever RAG.
- **Preparación Evolutiva:** Al contar con Cosmos DB pre-establecido en el hub transaccional, integrar logs de telemetría continuos (Data Ingestion) bajo esta misma capa será trivial. 
- **Cero Infraestructura Extra:** Mantenemos un setup ágil sin servicios "sidecar" manejados localmente que obstaculicen despliegues empresariales para del hackathon.
