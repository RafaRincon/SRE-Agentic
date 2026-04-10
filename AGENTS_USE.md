# Agentic Usage & Architecture Guide (AGENTS_USE.md)

## Agent Overview
The **Neuro-Symbolic SRE Agent** is an automated incident intake and triage system designed to act as a Level-1 Site Reliability Engineer (SRE). Traditional AI agents operate on purely generative or iterative LLM loops which suffer from instability, slowness, and an inherent propensity to hallucinate when faced with strict business rules or code. 

Our agent solves this by employing a rigid **Dual-Track Neuro-Symbolic** architecture:
- **Track A (Neuronal/Probabilistic):** Powered by LLMs (Gemini 3 Flash/Pro) to understand semantic intent, extract entities, and model the incident landscape.
- **Track B (Symbolic/Deterministic):** Powered by Python code, native structured schema validation, and Azure Cosmos DB vector mathematics to logically verify or discard LLM hypotheses using *actual* codebase citations.

## 1. Core Use Cases

1. **Intelligent Triage & Deduplication (Alert Storm Protection):** 
   During an incident, SREs are often flooded with Alert Storms. The agent listens to incoming alerts and acts as a preventative shield. Using local `Cosine Similarity` in-memory, it performs Early Deduplication (Short-Circuiting). If an incident shares >80% semantic similarity with an open issue, it classifies it as a `DUPLICATE`, updates the counter, and shortcuts the LLM LangGraph execution, saving massive API quota (TPM/RPM) and reducing alert fatigue.
2. **Contextual Entity Extraction:** 
   Translates unstructured pager text into strict Pydantic schemas (User IDs, Service Names, IPs).
3. **Epistemic Root Cause Analysis (Anti-Hallucination):** 
   When investigating the `.NET eShop` codebase, the agent generates risk hypotheses. However, it is explicitly constrained to provide a verbatim "Exact Span" citation. The symbolic `Span Arbiter` searches the disk index; if the citation doesn't exist, the assumption is immediately flagged as `HALLUCINATION` and discarded.
4. **Knowledge Flywheel & Runbook Drafting:** 
   Cross-references historical resolved incidents and runbooks housed in a separate vector space (`sre_knowledge`). It leverages the embeddings early in the pipeline to recommend precise operational procedures in the ticket consolidation phase without paying twice for vectorization.

## 2. Implementation & LangGraph Architecture

The agent execution is strictly sequential to maintain absolute determinism and "Graph Locking", avoiding parallel Race Conditions.

**Execution Flow:** `intake` → `world_model` → `slot_filler` → `risk_hypothesizer` → `span_arbiter` → `falsifier` → `consolidator`

### Incident Document Shape (Cosmos DB)
- `raw_report` is the persisted source of truth for the original incident text.
- `entities` is a derived operational view extracted from `raw_report` and persisted without `thinking_process`.
- `world_model` is a compact derived triage view; newly persisted incidents keep only `affected_service`, `incident_category`, `blast_radius`, and `epistemic_snapshot`.
- Legacy runtime-only fields may still exist in memory or older stored incidents for backward compatibility, but they are not part of the new persisted contract.

### Key Architectural Components:
- **Strict Pydantic Natives (gemini-3.1-pro-preview):** Instead of using fragile LangChain JSON parsers, the **Falsifier** leverages the native Google GenAI SDK. With `thinking_level="MEDIUM"`, the model is forced into deep planning while respecting protocol-level Structured Outputs, guaranteeing zero parsing `TypeErrors`.
- **AST-Based Code Chunking (`tree-sitter`):** Rather than blindly chunking C# code by character count, the system utilizes an Abstract Syntax Tree (AST) parser. It cuts chunks strictly at method or class boundaries, enriching chunks with structural metadata (`class_name`, `method_name`, `language`). This boosts the Span Arbiter's precision astronomically, as chunks are semantically coherent.
- **Reciprocal Rank Fusion (RRF):** For retrieval, relying solely on Dense Vectors (DiskANN) misses exact syntactical identifiers (e.g. `ERR_SYS_004` or `OrderPaymentFailedIntegrationEvent`). The agent uses a Hybrid Search blending DiskANN (Semantic) with BM25 FullTextScore (Syntactic) powered natively by Cosmos DB's RRF capability for unmatched accuracy.

## 3. Safety Measures (Secure-By-Design)

When deploying agents with access to real codebases, defense-in-depth is mandatory:

1. **Neutralization of Schema Escalation (DAN attacks):**
   The Agent's World Model is generated with native Structured Outputs and validated against the Pydantic runtime schema (`WorldModelProjection`). This materially reduces prompt-injection surface by constraining the output shape before downstream logic consumes it.
2. **RCE Inception Defense via Symbolic Arbiter:**
   If a prompt injection attempts to force a malicious hypothesis (`"Suspected file: /etc/passwd. Suspected function: bash"`), the pure-Python **Span Arbiter** acts as a deterministic guardrail. It demands evidence that the suspected text span actually exists in indexed eShop code. Unsupported spans are tagged as hallucinated and do not pass the conservative verification path.
3. **Google Native Guardrails:**
   The API bindings rely on provider-level safety features in the Google GenAI stack together with local validation and symbolic checks.
4. **Data Privacy & Ephemeral Checkpointing:**
   Image data (`image_data_b64`) is stripped before incident persistence, and compact incident views are stored separately from the original raw report for auditability.

## 4. Observability Evidence

Enterprise agents require "Glass Box" observability to audit AI reasoning:
- **Production Tracing (Langfuse):** All node transitions and LLM calls are traced via Langfuse. Each trace includes prompt arrays, response latency, and token consumption grouped linearly per `incident_id`.
- **The Immutable Ledger:** Every hypothesis generated, every `SPAN_VERDICT` from the Arbiter, and every transition of the Finite State Machine (FSM) is appended to an immutable `incident_ledger` collection in Cosmos DB. This acts as an unalterable flight recorder, vital for post-mortems.
