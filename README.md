# 🧠 Neuro-Symbolic SRE Agent

**An AI-powered SRE Incident Intake & Triage Agent that combines LLM reasoning with deterministic code verification to eliminate hallucinated diagnoses.**

## The Problem

On-call SRE teams waste critical minutes during incidents manually reading error reports, cross-referencing code, and routing issues. Existing AI tools can generate plausible-sounding root causes — but they hallucinate file paths, fabricate function names, and cite code that doesn't exist.

## Our Approach: Dual-Track Neuro-Symbolic Verification

This agent uses a **dual-track architecture** inspired by neuroscience's dual-process theory:

- **Track A (Neuronal):** Gemini 3 Flash processes the incident report (text + images), projects a cognitive "World Model," and extracts structured entities.
- **Track B (Adversarial):** A separate LLM generates root cause hypotheses with **mandatory code citations** (exact spans). A fully **symbolic** Span Arbiter then verifies every citation against the real eShop codebase indexed in Cosmos DB with DiskANN vector search.

**The key insight:** Hypotheses that fail citation verification are marked as `HALLUCINATION` and discarded. Only verified root causes reach the triage summary. The LLM proposes; the code verifies.

## Architecture

```
                     ┌──────────────────────────┐
                     │      POST /incident      │
                     │    (text + image)         │
                     └────────┬─────────────────┘
                              │
                     ┌────────▼─────────┐
                     │     INTAKE       │
                     │  (FSM: RECEIVED  │
                     │   → TRIAGING)    │
                     └────┬────────┬────┘
                          │        │
            ┌─────────────▼──┐  ┌──▼──────────────┐
            │   TRACK A      │  │    TRACK B       │
            │  (Neuronal)    │  │  (Adversarial)   │
            │                │  │                  │
            │ World Model    │  │ Risk Hypothesizer│
            │ → Slot Filler  │  │ → Span Arbiter   │
            │                │  │   (SYMBOLIC)     │
            └──────┬─────────┘  └──────┬───────────┘
                   │                   │
                   └───────┬───────────┘
                           │
                  ┌────────▼──────────┐
                  │   CONSOLIDATOR    │
                  │ (merge + severity │
                  │  via FSM rules)   │
                  └────────┬──────────┘
                           │
                  ┌────────▼──────────┐
                  │  CREATE TICKET    │
                  │  → NOTIFY TEAM    │
                  └───────────────────┘
```

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 3 Flash (`gemini-3-flash-preview`) |
| Embeddings | Gemini Embedding 2 (768d via MRL) |
| Orchestration | LangGraph (StateGraph, parallel fan-out) |
| Database | Azure Cosmos DB NoSQL (DiskANN vector index) |
| Backend | FastAPI (Python 3.12) |
| Observability | Langfuse |
| Containerization | Docker Compose |

## Key Features

- **Multimodal Input**: Accepts text reports + screenshot images
- **Anti-Hallucination**: Span Arbiter performs deterministic string matching against indexed codebase
- **Deterministic FSM**: State transitions are algebraic (never LLM-decided)
- **Algebraic Severity**: CRITICAL/HIGH/MEDIUM/LOW computed from verified hypothesis count + blast radius
- **Immutable Audit Trail**: Every decision logged to Cosmos DB ledger
- **eShop Code Index**: 607 files → 949 chunks with DiskANN vector search

## Quick Start

```bash
git clone <repo-url>
cd hackaton
cp .env.example .env
# Fill in your API keys
docker compose up --build
```

Then:
```bash
# Index the eShop codebase
curl -X POST http://localhost:8000/index

# Submit an incident
curl -X POST http://localhost:8000/incident \
  -F "report=HTTP 500 on checkout endpoint /api/orders. NullReferenceException in OrdersController."

# Check status
curl http://localhost:8000/incident/{incident_id}
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/incident` | Submit incident (multipart form) |
| `GET` | `/incident/{id}` | Get incident status |
| `POST` | `/incident/{id}/resolve` | Resolve and notify reporter |
| `GET` | `/incidents` | List all incidents |
| `POST` | `/index` | Trigger eShop code indexing |
| `GET` | `/index/status` | Check index status |
| `GET` | `/incident/{id}/ledger` | View audit trail |

## License

MIT — see [LICENSE](LICENSE)
