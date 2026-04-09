# Application Scaling Strategy & Architecture Decisions (SCALING.md)

## 1. Current Architecture & Compute

The application is built around a stateless unified backend:
- **FastAPI Workers:** Serves all HTTP ingress traffic asynchronously.
- **Azure Cosmos DB NoSQL (Serverless):** Single Source of Truth for transactional data, immutable ledgers, and Vector Search (DiskANN).
- **Google Gemini API:** Provides the generative compute (Flash for fast parsing, Pro for epistemic falsification).

**Scaling Strategy:** 
The current hackathon setup ships as a single-container Docker Compose deployment. The backend is intentionally stateless, so it can be scaled horizontally in a fuller deployment by adding FastAPI replicas behind a load balancer while keeping orchestration state in Cosmos DB checkpointers.

## 2. Database Scaling (Cosmos DB over Alternatives)

### Why Azure Cosmos over Chroma DB / pgvector?
1. **Sharded DiskANN (Partitioning by Metadata):** 
   In SRE triage, searches are scoped heavily by service. Cosmos DB leverages Sharded DiskANN using the partition key (`/service_name`). This guarantees the AI only searches vectorially inside the failed domain, eliminating the latency penalty that global vector DBs encounter at massive scales.
2. **No-ETL Architecture:**
   SRE incidents, historical Ledgers, and AST-parsed code chunks all live natively in a single cluster. Using isolated vector databases like Chroma DB would require fragile two-way sync pipelines.
3. **Billion-Scale DiskANN Extensibility:**
   Unlike HNSW algorithms that must hold the entire index in RAM, DiskANN accommodates quantization on enterprise SSDs, keeping RAM free while still providing sub-20ms latencies on tens of millions of records.

## 3. Scale Optimization: The Knowledge Flywheel & Deduplication

API throughput bottlenecks almost always originate from LLM/Embedding limits (RPM/TPM).
- **Early Deduplication (Alert Storm Defense):** The pipeline computes one embedding up front and then performs duplicate detection before the more expensive triage stages. In a 500-error Alert Storm, duplicates can short-circuit the rest of the graph and avoid additional LLM generations.
- **Vector Injection / Flywheel:** The incident's embedding is saved into graph memory (`state["report_embedding"]`). Later, when the `consolidator` node needs to fetch Runbooks from the `sre_knowledge` database, it reuses the same vector and avoids an extra embedding call in the common path.

## 4. Indexing Pipeline (AST Code Chunking)

When the codebase grows to mono-repo size (e.g., 100M+ lines of code), standard regex/character-based chunking breaks down.
- **Solution:** We use `tree-sitter` (AST). Each chunk is a semantically complete C# method or class. 
- **Scale Impact:** While indexing is computationally heavier upfront, retrieval is astronomically more precise. To avoid inflating the index, we enforce a strict minimum threshold (discarding 1-line properties `< 100 chars`) and strip noise files (`.csproj`, `GlobalUsings`). 

## 5. Technical Decisions Summary

| Decision | Rationale |
|---|---|
| **Cosmos DB DiskANN** | Unified transactional + vector engine with strong retrieval ergonomics for this architecture. |
| **Hybrid Search (RRF)** | Merges Vector Semantic similarity with BM25 FullText. Vital when SREs search for exact GUIDs or Error Codes (`ERR_SYS_004`). |
| **AST Chunking (`tree-sitter`)** | Eliminates cut-in-half code chunks. Provides exact `class_name` metadata for deterministic LLM citations. |
| **Sequential Graph Locking** | By running the pipeline linearly instead of parallel fan-out, the later stages always receive the full Track A context. |
| **Pydantic Native (Gemini SDK)** | Migrating away from LangChain's fragile JSON extractors. `gemini-3.1-pro` guarantees byte-perfect structs natively on the wire. |
| **MRL 768 dims** | 4x storage savings with <2% accuracy loss over 3072 dims, perfectly matching our DiskANN scaling logic. |
