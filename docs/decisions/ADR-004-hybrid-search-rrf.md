# ADR 004: Hybrid Search using Reciprocal Rank Fusion (RRF) en Azure Cosmos DB

## Contexto

El Agente SRE fue diseñado originalmente utilizando una búsqueda puramente vectorial (basada en el índice DiskANN de Azure Cosmos DB NoSQL). Si bien el "Vector Search" es excelente para comprensión semántica (ej. buscar "código que maneje flujos de autenticación" sin saber qué clases están involucradas), presenta limitaciones críticas en escenarios operativos de SRE.

En el triage de incidentes de producción, los programadores a menudo ingresan identificadores exactos y literales dentro del log de la incidencia o en sus consultas:
- Códigos de error específicos (ej. `ERR_SYS_004`).
- Nombres de clases o métodos exactos que lanzaron una excepción (ej. `OrderPaymentFailedIntegrationEvent`).
- Nombres de servicios (ej. `CatalogApi`).
- Track IDs o Correlation IDs.

Los modelos de Embeddings (como Gemini-Embedding) a menudo "suavizan" el significado de estos tokens sintéticos literales, lo que resulta en que el Vector Search puro a veces no logre posicionar el archivo exacto como la mejor coincidencia.

## Decisión Técnica

Hemos decidido evolucionar la estrategia de retención desde una "Búsqueda Puramente Semántica" a una **"Búsqueda Híbrida" (Hybrid Search)** implementando **Reciprocal Rank Fusion (RRF)**, una capacidad soportada nativamente en la versión Preview de Azure Cosmos DB NoSQL.

### Detalles de la Implementación:

1. **Infraestructura de Cosmos DB:** Se han implementado un `FullTextPolicy` y un `FullTextIndex` sobre el campo `/chunk_text` en ambos contenedores donde hacemos recuperación profunda: `eshop_chunks` (código AST) y `sre_knowledge` (Runbooks / Historicos). Esto se suma a los índices `vectorIndexes` usando DiskANN ya existentes.
2. **Consultas SQL Híbridas (No-SQL API):** El `db_provider.py` se ha actualizado para combinar ambos scores en una sola consulta. Reemplazamos `ORDER BY VectorDistance(...)` por `ORDER BY RANK RRF(VectorDistance(...), FullTextScore(...))`.
3. **Mecanismo Fallback Seguro:** Debido a que el entorno de despliegue puede no tener habilitados los índices FullText (o debido a errores en la consulta), se ha implementado un mecanismo try/except. Si la búsqueda RRF falla, el sistema realiza automáticamente un _fallback_ a la búsqueda DiskANN Vector-Only, garantizando que el Agente no falle.
4. **Pasaje del Query Texto Original:** Los nodos del LangGraph (`span_arbiter`, `risk_hypothesizer`, `consolidator`) se han actualizado para pasar tanto el Vector (para el DiskANN) como el query string original (para el motor de BM25 de FullTextScore). 

## Consecuencias

### Positivas
- **Aumento radical de Precisión:** Las pruebas de integración demostraron que en situaciones con "Exact Keywords", la búsqueda Híbrida se comporta igual o sustancialmente mejor que Vector-only. No hubieron casos donde perdiera.
- **Span Arbitration Mejorado:** El `span_arbiter` aísla los literales que el LLM del `risk_hypothesizer` asegura que tienen la culpa. BM25 actúa como un filtro determinista vital, haciendo que las confirmaciones del árbitro sean mucho más rápidas y exactas.
- **Cero latencias agregadas en capa de servicio:** Todo el cálculo RRF ocurre _In-Database_ en el clúster de Azure Cosmos DB. Disminuimos la carga de la aplicación al no tener que cruzar dos listas manualmente en memoria.

### Riesgos o Trade-offs
- **Dependencias Preview:** El soporte BM25/RRF en Cosmos DB NoSQL se encuentra a la fecha en versión preliminar. 
- **Costo de Storage:** Incorporar el FullText index sobre cada chunk aumenta ligeramente el peso del contenedor, aunque irrelevante a los costos de RU de cómputo para este PoC.

## Verificación

Se documentaron pruebas unitarias, smoke tests integrados y pruebas E2E en `tests/test_hybrid_e2e.py` que comprueban y legitiman cómo la arquitectura de recuperación se favorece cuando conviven queries semánticos vs queries literales.
