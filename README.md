# 🧠 Neuro-Symbolic SRE Agent

**An AI-powered SRE Incident Intake & Triage Agent that combines probabilistic LLM reasoning (Gemini Flash/Pro) with deterministic code verification (Cosmos DB DiskANN) to reduce hallucinated diagnoses.**

---

## 🛑 The Problem: Alert Fatigue & AI Hallucination
On-call SRE teams waste critical minutes during incidents manually reading error reports, cross-referencing code, and routing issues while drowning in "Alert Storms". Existing generative AI tools can output plausible-sounding root causes rapidly — but they regularly hallucinate file paths, fabricate function names, and cite C# code that doesn't actually exist in the production repository, leading engineers on wild goose chases.

## 🔬 Our Solution: Dual-Track Neuro-Symbolic Verification
This agent pioneers a **Dual-Track architecture** inspired by neuroscience's dual-process theory:

- **Track A (Neuronal / Fast):** Google Gemini 3 Flash processes the raw incident report (text + images), projects a compact operational "World Model," and extracts structured entities into strict Pydantic schemas.
- **Track B (Adversarial / Slow):** After Track A has built the full incident context, a separate deep-thinking LLM (`gemini-3.1-pro-preview` with `thinking_mode`) acts as a Popperian Falsifier. It generates root cause hypotheses with **mandatory code citations** (exact spans). A fully **symbolic** Python `Span Arbiter` then verifies every citation against the real codebase indexed in Azure Cosmos DB utilizing Hybrid Search (Reciprocal Rank Fusion - RRF).

**Core Safety Property:** Hypotheses that fail strict citation verification are marked as `HALLUCINATION` and excluded from the conservative severity path. The system pairs LLM generation with symbolic verification so ungrounded citations do not drive the main triage outcome. *The LLM proposes; the Code verifies.*

---

## 🏛️ Comprehensive Architecture

```text
                     ┌──────────────────────────────────────┐
                     │            POST /incident            │
                     │         (Text + Screenshots)         │
                     └──────────────────┬───────────────────┘
                                        │
                         [ EARLY DEDUPLICATION GATE ]
                         (Cosine Similarity Short-Circuit)
                                        │
                     ┌──────────────────▼───────────────────┐
                     │              INTAKE                  │
                     │    (FSM: RECEIVED → TRIAGING)        │
                     └─────────┬──────────────────┬─────────┘
                               │                  │
                ┌──────────────────▼───────────────────┐
                │              WORLD MODEL             │
                │   Compact operational triage view    │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │              SLOT FILLER             │
                │   Structured entities from report    │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │           RISK HYPOTHESIZER          │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │            SPAN ARBITER              │
                │              (SYMBOLIC)              │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │              FALSIFIER               │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │             CONSOLIDATOR             │
                │    Merge + Algebraic Severity +      │
                │      Runbook Vector Retrieval        │
                └──────────────────┬───────────────────┘
                                   │
                ┌──────────────────▼───────────────────┐
                │         CREATE INCIDENT RECORD       │
                │           → ROUTE OWNERSHIP          │
                └──────────────────────────────────────┘
```

Persisted incidents keep `raw_report` as the source of truth, plus compact derived views for `entities` and `world_model`.

## 🛠️ Enterprise Tech Stack

| Component | Technology | Rationale |
|---|---|---|
| **Primary LLMs** | Google Gemini 3 Flash & 3.1 Pro | Flash for fast entity extraction; Pro with `thinking_level` for deep epistemic falsification. |
| **Embeddings** | Gemini Embedding 2 (MRL 768d) | 4x storage reduction utilizing Matryoshka models natively supported by DiskANN. |
| **Orchestration** | LangGraph (StateGraph) | Strict sequential execution with Cosmos DB checkpointers for immutable resume-ability. |
| **Database & Vector** | Azure Cosmos DB NoSQL | Centralized Sharded DiskANN. Supports Hybrid Search (RRF) combining vector math with BM25 text rank natively in-database. |
| **Code Parser** | Tree-Sitter (AST C#) | Abstract Syntax Tree chunking cuts code semantically at method boundaries instead of random characters. |
| **Observability** | Langfuse & Console | Glass Box transparency mapping out exact token usage, latency, and tool selection. |

---

## 🔑 Key Features & Architectural Advancements

### 1. Alert Storm Protection (Knowledge Flywheel)
The system is protected against massive Alert Storms. When an incident arrives, it calculates an in-memory cosine similarity against open resolving tickets. If a duplication threshold logic is breached, it safely short-circuits the run, updates an occurrence counter on the primary ticket, and saves hundreds of dollars in LLM API throughput (RPM/TPM limits).

### 2. Hybrid RRF Search against AST Chunks
Code retrieval matters. The agent chunks the entire eShop codebase using `tree-sitter` (AST), breaking chunks strictly at class/method levels. When querying Azure Cosmos DB, it uses a Hybrid Reciprocal Rank Fusion (RRF) search, combining Dense Vector (intent matching) with BM25 (exact keyword matching for Error Codes like `ERR_500`). 

### 3. Native Pydantic "Schema Escalation" Defense
By leveraging Google GenAI SDK natively, we lock the agent into `Structured Outputs`. Attackers trying Prompt Injection to break JSON boundaries (DAN attacks) are blocked at the protocol layer, making Remote Code Executions or Hallucination Inceptions literally impossible.

### 4. Immutable Audit Ledger
The agent records its own thoughts. Every FSM transition, every hypothesized finding, and every `Span Arbiter` verdict (VERIFIED or HALLUCINATED) is written strictly append-only into the `/incident_ledger` in Cosmos DB.

---

## 🚀 Quick Start (Development)

The entire enterprise architecture shrinks down to a single compose file.

### 1. Setup Your Keys
```bash
git clone <repo-url>
cd <repo-dir>
cp .env.example .env
```
Fill out `GEMINI_API_KEY`, `COSMOS_ENDPOINT`, `COSMOS_KEY`, and `APP_ADMIN_API_KEY` inside `.env`.

### 2. Bootstrap the Environment
```bash
docker compose up --build
```
This lifts the FastAPI worker, initializes the Cosmos DB Collections (Vectors, Ledger, Incidents), and binds ports.

### 3. Interacting (See QUICKGUIDE.md for more)
Use the API immediately to inject the codebase and test the triage flow:
```bash
# Force AST logic to chunk and embed CosmosDB
curl -X POST http://localhost:8000/index \
  -H "X-Admin-Api-Key: $APP_ADMIN_API_KEY"

# Throw a complex multimodal incident
curl -X POST http://localhost:8000/incident \
  -F "report=HTTP 500 on checkout endpoint /api/orders. NullReferenceException inside OrdersController causing failure."
```

## 🌐 API Overview

| Method | Path | Operational Action |
|---|---|---|
| `GET` | `/health` | Check Fast API Liveness |
| `POST` | `/index` | Chunk external repo, generate Vectors, save to Cosmos. |
| `POST` | `/incident` | Main intake endpoint. Accepts multipart form data with `report` plus optional uploaded `image`. |
| `GET` | `/incident/{id}` | Read mathematical Severity and FSM status. |
| `GET` | `/incident/{id}/ledger` | Read the immutable audit trail of the investigation flow (admin API key required). |
| `POST` | `/incident/{id}/resolve` | Close incident, record resolution notes, and persist reporter follow-up metadata (admin API key required). |

---

## 📑 Detailed Documentation Available
For additional architecture and operational detail, please review:
- **[AGENTS_USE.md]**: Deep dive into LangGraph, Falsification Tracks, and Agent Constraints.
- **[SCALING.md]**: Details on Docker Workers, Rate Limiting defense, Cosmos DB Sharding, and DiskANN optimizations.
- **[QUICKGUIDE.md]**: A step-by-step test lab tutorial.

## 📄 License
MIT — see [LICENSE](LICENSE)
