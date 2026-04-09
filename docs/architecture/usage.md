# AGENTS_USE.md

## Agent Overview

The **Neuro-Symbolic SRE Agent** is an automated incident intake and triage system that processes incident reports, generates verified root cause hypotheses, creates tickets, and notifies teams — all within seconds.

## Use Cases

### Primary: Incident Triage
1. **Reporter** submits a text description (+ optional screenshot) via `POST /incident`
2. **Agent** runs dual-track analysis:
   - Track A: Understands the incident semantically (service, severity, entities)
   - Track B: Generates code-grounded hypotheses and verifies them against real codebase
3. **Agent** creates a ticket, assigns to the right team, and sends notifications
4. **On-call engineer** reviews the verified triage summary and acts

### Secondary: Incident Resolution
- `POST /incident/{id}/resolve` transitions the FSM and notifies the reporter

### Maintenance: Code Indexing
- `POST /index` clones the eShop repo, chunks code files, generates embeddings, and indexes into Cosmos DB

## Implementation Details

### Dual-Track Pipeline (LangGraph)

The agent uses a **StateGraph** with parallel fan-out:

```
intake ──┬── world_model → slot_filler ──────┬── consolidator → ticket → notify
         └── risk_hypothesizer → span_arbiter ┘
```

**Track A (Neuronal):**
- `world_model`: Projects incident across service dimensions (affected service, blast radius, severity, category)
- `slot_filler`: Extracts structured entities (error codes, endpoints, stack traces)

**Track B (Adversarial):**
- `risk_hypothesizer`: Generates 2-4 root cause hypotheses with **mandatory exact code citations**
- `span_arbiter`: Verifies each citation against indexed codebase using vector search + deterministic string matching

### Anti-Hallucination Mechanism

The Span Arbiter is a **100% symbolic** verification layer:

1. Takes each hypothesis's `exact_span` (a verbatim code quote)
2. Generates an embedding and performs DiskANN vector search to find candidate chunks
3. Applies `difflib.SequenceMatcher` fuzzy matching against candidates
4. Produces a verdict: `VERIFIED`, `PARTIAL_MATCH`, or `HALLUCINATION`

Hypotheses marked as `HALLUCINATION` are discarded and explicitly mentioned in the triage summary as "failed evidence verification."

### Deterministic FSM

State transitions follow algebraic rules — no LLM decides the state:

```
RECEIVED → TRIAGING → TRIAGED → TICKET_CREATED → TEAM_NOTIFIED → RESOLVED → REPORTER_NOTIFIED
```

Each transition has a boolean condition (e.g., "TRIAGED requires world_model != null AND entities != null AND triage_summary != empty AND final_severity != UNKNOWN").

### Severity Computation

Severity is computed deterministically from verified findings:
- **CRITICAL**: ≥2 verified root causes OR blast radius ≥ 3 services
- **HIGH**: 1 verified root cause AND blast radius ≥ 1
- **MEDIUM**: Hypotheses exist but none fully verified
- **LOW**: No hypotheses generated

## Observability

### Langfuse Integration
- All LLM calls are traced via Langfuse
- Each trace includes: prompt, response, latency, token count
- Linked to incident IDs for correlation

### Structured Logging
- All nodes log to stdout with structured format
- Each log includes: timestamp, level, module, incident_id

### Audit Ledger
- Every decision is persisted to the `incident_ledger` container in Cosmos DB
- Entries are immutable (append-only via `create_item`)
- Types: `STATE_TRANSITION`, `HYPOTHESIS_GENERATED`, `SPAN_VERDICT`

## Safety Measures

### Input Validation
- Reports must be ≥10 characters
- Image uploads validated for presence of actual data
- Pydantic schemas enforce strict types on all state fields

### LLM Output Constraints
- All LLM calls use **structured JSON output** with Pydantic schemas
- System prompts explicitly instruct against fabrication
- Span Arbiter acts as a hard verification gate — no unverified hypothesis reaches the output

### Failure Resilience
- If any LLM call fails, the pipeline continues with degraded output
- Each node catches exceptions and appends to `state.errors`
- FSM prevents invalid state transitions even under partial failure

### Data Privacy
- Image data (`image_data_b64`) is stripped before Cosmos DB persistence
- Reporter emails are only used in notification mocks
- All Cosmos DB communications use HTTPS with key-based auth
