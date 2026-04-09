# SCALING.md

## Current Architecture

The application is a single FastAPI service connected to:
- **Azure Cosmos DB NoSQL** (serverless) for data persistence and vector search
- **Google Gemini API** for LLM and embedding generation

## How It Scales

### Compute (FastAPI Workers)

**Current**: Single uvicorn worker.

**Scaling strategy**: Horizontal scaling via Docker replicas:

```yaml
services:
  sre-agent:
    deploy:
      replicas: 4
```

Each replica is stateless — state lives in Cosmos DB. Add a load balancer (nginx, Traefik, or cloud LB) in front.

**Advanced**: Kubernetes Deployment with HPA based on request latency or queue depth.

### Database (Cosmos DB)

**Current**: Serverless provisioning with automatic scaling.

**Scaling factors**:
- **Throughput**: Cosmos DB auto-scales RU/s based on demand (serverless mode). Switch to provisioned throughput with autoscale for predictable high-volume workloads.
- **Partitioning**: `eshop_chunks` uses `/service_name` as partition key — queries scoped to a single service (e.g., "Ordering.API") are partition-local. Cross-partition queries are used for global search but can be optimized with composite indexes.
- **Vector Search (DiskANN)**: DiskANN is designed for billion-scale vector search. Current 949 chunks → supports 100M+ without architecture changes.
- **Read replicas**: Enable multi-region reads for geo-distributed on-call teams.

### LLM (Gemini API)

**Current**: Sequential calls within each track, parallel between Track A and Track B.

**Scaling bottleneck**: Gemini API rate limits (RPM/TPM).

**Mitigation strategies**:
1. **Parallel tracks**: Track A and Track B already run concurrently via LangGraph fan-out
2. **Batch embeddings**: Indexer embeds 20 texts per API call
3. **Caching**: Repeated queries (e.g., same error pattern) could cache world model projections
4. **Rate limit handling**: Exponential backoff on 429 responses (currently handled by node error boundaries)
5. **Model tiering**: Use `gemini-3.1-flash-lite-preview` for slot filling (cheaper/faster), keep `gemini-3-flash-preview` for hypothesis generation

### Indexing Pipeline

**Current**: Full sequential indexing on-demand via `POST /index`.

**Scaling options**:
- **Incremental indexing**: Track git commit SHAs, only re-index changed files
- **Background worker**: Move indexing to a Celery/RQ worker to avoid blocking the API
- **Multi-repo**: Extend `SERVICE_MAP` to support multiple microservice repos
- **Webhook-triggered**: Index on push events via GitHub webhooks

## Assumptions

1. **Incident volume**: Designed for ~100 incidents/day per service. At 1000+/day, implement queue-based processing (Redis + worker pattern).
2. **eShop codebase size**: Current approach handles repos up to ~10K files. Larger monorepos would need chunking optimization (AST-based splitting instead of line-based).
3. **Latency target**: End-to-end triage in <30 seconds. Current: ~15s (dominated by LLM calls). With model caching: <5s.
4. **Team size**: Mocked notifications assume <20 on-call teams. Real PagerDuty/OpsGenie integration would handle arbitrary team sizes.

## Technical Decisions

| Decision | Rationale |
|---|---|
| Cosmos DB over PostgreSQL | Unified vector search (DiskANN) + document store + ledger in one service. No pgvector setup required. |
| MRL 768 dims (not 3072) | 4x storage savings with <2% accuracy loss per Gemini embedding docs. Matches DiskANN index already provisioned. |
| LangGraph over raw async | Built-in parallel fan-out/fan-in, state reducers, and checkpoint compatibility. |
| Deterministic FSM over LLM routing | LLM decides "what" (content); FSM decides "when" (flow). Eliminates state corruption. |
| Fuzzy string matching over exact | Real code citations may have whitespace/formatting differences. `SequenceMatcher` handles this with configurable threshold. |
