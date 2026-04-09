"""
SRE Agent — Risk Hypothesizer Node (Track B)

Implements a Hybrid RAG pattern with QUERY EXPANSION:

1. LLM-powered Query Expansion: Generates multiple search perspectives
   from the incident (error-centric, service-centric, code-pattern-centric)
2. Parallel Vector Retrieval: Each expanded query searches DiskANN independently
3. Deduplication + Ranking: Merge results, remove duplicates, rank by relevance
4. Grounded Hypothesis Generation: LLM sees REAL code and cites from it

The Span Arbiter (next node) double-checks all citations.
"""

import logging
from pydantic import BaseModel, Field
from app.agents.state import (
    EpistemicStatus,
    RiskHypothesis,
    make_epistemic_claim,
    merge_epistemic_snapshots,
    snapshot_is_empty,
)
from app.providers import llm_provider, db_provider
from app.ledger.audit import record_hypothesis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ExpandedQueries(BaseModel):
    """Multi-perspective query expansion for SRE incidents."""
    error_query: str = Field(
        description="Search query focused on the error type, exception, and stack trace"
    )
    service_query: str = Field(
        description="Search query focused on the affected service's handlers, controllers, and entry points"
    )
    pattern_query: str = Field(
        description="Search query focused on code patterns that commonly cause this type of failure (null checks, dependency injection, serialization)"
    )
    dependency_query: str = Field(
        description="Search query focused on inter-service dependencies, event bus, and integration points"
    )


class HypothesesOutput(BaseModel):
    """Wrapper for structured output with multiple hypotheses."""
    hypotheses: list[RiskHypothesis] = Field(
        default_factory=list,
        description="List of risk hypotheses with mandatory code citations"
    )


def _build_hypothesis_snapshot(hypothesis: RiskHypothesis):
    observed = []
    inferred = []
    unknown = []

    if hypothesis.exact_span:
        observed.append(
            make_epistemic_claim(
                label=f"exact_span={hypothesis.exact_span[:120]}",
                status=EpistemicStatus.OBSERVED,
                evidence=hypothesis.exact_span,
                source="risk_hypothesizer",
            )
        )

    if hypothesis.suspected_file:
        observed.append(
            make_epistemic_claim(
                label=f"suspected_file={hypothesis.suspected_file}",
                status=EpistemicStatus.OBSERVED,
                evidence=hypothesis.suspected_file,
                source="risk_hypothesizer",
            )
        )

    inferred.append(
        make_epistemic_claim(
            label=hypothesis.description,
            status=EpistemicStatus.INFERRED,
            evidence="Causal interpretation grounded in the retrieved code span.",
            source="risk_hypothesizer",
        )
    )

    unknown.append(
        make_epistemic_claim(
            label="upstream_validation_or_wiring",
            status=EpistemicStatus.UNKNOWN,
            evidence=(
                "The retrieved code does not prove whether upstream validation, "
                "dependency wiring, or environment-specific config blocks this path."
            ),
            source="risk_hypothesizer",
        )
    )

    return merge_epistemic_snapshots(
        {"observed": observed, "inferred": inferred, "unknown": unknown}
    )




# ---------------------------------------------------------------------------
# Query Expansion
# ---------------------------------------------------------------------------


async def expand_queries(raw_report: str, world_model: dict, entities: dict) -> list[str]:
    """
    Generate multiple search queries from different perspectives.
    Uses LLM to produce semantically diverse queries for better recall.
    Falls back to manual extraction if LLM fails.
    """
    context = f"""Incident: {raw_report[:500]}
Service: {world_model.get('affected_service') or 'unknown'}
Error: {entities.get('error_code') or ''} {entities.get('error_message') or ''}
Endpoint: {entities.get('endpoint_affected') or ''}
Stack: {(entities.get('stack_trace') or '')[:200]}"""

    from app.agents.prompts import RISK_EXPANSION_SYSTEM, build_risk_expansion_prompt
    
    try:
        expanded = await llm_provider.generate_structured(
            prompt=build_risk_expansion_prompt(context),
            response_schema=ExpandedQueries,
            system_instruction=RISK_EXPANSION_SYSTEM,
        )

        queries = [
            expanded.error_query,
            expanded.service_query,
            expanded.pattern_query,
            expanded.dependency_query,
        ]

        logger.info(
            f"[query_expansion] Generated {len(queries)} expanded queries: "
            f"{[q[:50] + '...' for q in queries]}"
        )
        return queries

    except Exception as e:
        logger.warning(f"[query_expansion] LLM expansion failed, using manual: {e}")
        return _manual_expand(raw_report, world_model, entities)


def _manual_expand(raw_report: str, world_model: dict, entities: dict) -> list[str]:
    """Fallback: manually extract multiple query perspectives."""
    queries = []

    # Error-centric
    error_parts = []
    if entities.get("error_message"):
        error_parts.append(entities["error_message"])
    if entities.get("error_code"):
        error_parts.append(str(entities["error_code"]))
    if error_parts:
        queries.append(" ".join(error_parts))

    # Service-centric
    if world_model.get("affected_service"):
        service = world_model["affected_service"]
        queries.append(f"{service} controller handler endpoint")

    # Endpoint-centric
    if entities.get("endpoint_affected"):
        queries.append(f"{entities['endpoint_affected']} route handler action")

    # Stack trace
    if entities.get("stack_trace"):
        queries.append(entities["stack_trace"][:200])

    # Fallback: raw report
    if not queries:
        queries.append(raw_report[:500])

    return queries


# ---------------------------------------------------------------------------
# Multi-Query Retrieval
# ---------------------------------------------------------------------------


async def retrieve_with_expansion(
    queries: list[str],
    service_filter: str | None = None,
    top_k_per_query: int = 5,
) -> list[dict]:
    """
    Execute vector search for each expanded query and merge results.
    Deduplicates by chunk ID and ranks by best similarity score.
    """
    seen_ids = set()
    all_chunks = []

    for query in queries:
        try:
            query_embedding = await llm_provider.generate_embedding(
                query,
                task_type="RETRIEVAL_QUERY",
            )

            chunks = db_provider.vector_search(
                query_vector=query_embedding,
                query_text=query,
                top_k=top_k_per_query,
                service_filter=service_filter,
            )

            for chunk in chunks:
                chunk_id = chunk.get("id", "")
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_chunks.append(chunk)

        except Exception as e:
            logger.warning(f"[retrieval] Query failed: {e}")
            continue

    # Sort by similarity score (higher = better), take top 15
    all_chunks.sort(
        key=lambda c: c.get("similarity_score", 0),
        reverse=True,
    )

    deduplicated = all_chunks[:15]

    logger.info(
        f"[retrieval] {len(queries)} queries → "
        f"{len(all_chunks)} unique chunks → "
        f"top {len(deduplicated)} selected"
    )

    return deduplicated


# ---------------------------------------------------------------------------
# Main Node
# ---------------------------------------------------------------------------


async def risk_hypothesizer_node(state: dict) -> dict:
    """
    Generate risk hypotheses using Hybrid RAG with query expansion:
    1. Expand incident context into 4 diverse search queries
    2. Retrieve code chunks from each query (parallel, deduplicated)
    3. Generate hypotheses grounded in real retrieved code
    """
    raw_report = state.get("raw_report", "")
    world_model = state.get("world_model", {})
    entities = state.get("entities", {})

    # --- Step 1: Query Expansion (LLM-powered) ---
    expanded_queries = await expand_queries(raw_report, world_model, entities)

    # --- Step 2: Multi-Query Retrieval (deduplicated) ---
    service_filter = world_model.get("affected_service")
    retrieved_chunks = await retrieve_with_expansion(
        queries=expanded_queries,
        service_filter=service_filter,
        top_k_per_query=5,
    )

    # --- Step 2b: Retrieve historical incidents (FLYWHEEL) ---
    historical_chunks = []
    try:
        # Use the primary search query to find similar past incidents
        primary_query = expanded_queries[0] if expanded_queries else raw_report[:500]
        history_embedding = await llm_provider.generate_embedding(
            primary_query,
            task_type="RETRIEVAL_QUERY",
        )
        historical_chunks = db_provider.knowledge_search(
            query_vector=history_embedding,
            query_text=primary_query,
            top_k=5,
            service_filter=service_filter,
        )
        if historical_chunks:
            # Apply temporal decay to prioritize recent incidents
            from app.indexer.knowledge_indexer import apply_temporal_decay
            historical_chunks = apply_temporal_decay(historical_chunks)

        logger.info(
            f"[risk_hypothesizer] Retrieved {len(historical_chunks)} "
            f"historical knowledge chunks"
        )
    except Exception as e:
        logger.warning(f"[risk_hypothesizer] Historical retrieval failed: {e}")

    # --- Step 3: Format retrieved code for the LLM ---
    if retrieved_chunks:
        code_context = "\n\n".join([
            f"--- FILE: {c.get('file_path', 'unknown')} "
            f"(lines {c.get('start_line', '?')}-{c.get('end_line', '?')}) ---\n"
            f"{c.get('chunk_text', '')}"
            for c in retrieved_chunks
        ])
    else:
        code_context = (
            "(No code chunks retrieved — codebase may not be indexed yet. "
            "Generate hypotheses based on knowledge but with LOW confidence.)"
        )

    # --- Step 3b: Format historical context ---
    if historical_chunks:
        history_context = "\n\n".join([
            f"--- PAST INCIDENT [{c.get('source_id', '?')}] "
            f"[Severity: {c.get('metadata', {}).get('severity', '?')}] "
            f"[Role: {c.get('metadata', {}).get('chunk_role', '?')}] ---\n"
            f"{c.get('chunk_text', '')[:500]}"
            for c in historical_chunks
        ])
        recurrence_count = len([
            c for c in historical_chunks
            if c.get("similarity_score", 0) > 0.7
        ])
    else:
        history_context = "(No historical incidents found — this may be the first occurrence.)"
        recurrence_count = 0

    # --- Step 4: Generate hypotheses grounded in real code + history ---
    from app.agents.prompts import RISK_HYPOTHESIS_SYSTEM, build_risk_hypothesis_prompt

    prompt = build_risk_hypothesis_prompt(
        raw_report=raw_report,
        world_model=world_model,
        entities=entities,
        code_context=code_context,
        history_context=history_context,
        retrieved_chunks_len=len(retrieved_chunks),
        expanded_queries_len=len(expanded_queries),
        historical_chunks_len=len(historical_chunks),
        recurrence_count=recurrence_count
    )

    try:
        result = await llm_provider.generate_structured(
            prompt=prompt,
            response_schema=HypothesesOutput,
            system_instruction=RISK_HYPOTHESIS_SYSTEM,
        )

        # Filter out hypotheses without spans
        valid_hypotheses = [
            h for h in result.hypotheses
            if h.exact_span and len(h.exact_span.strip()) > 5
        ]

        for hypothesis in valid_hypotheses:
            if snapshot_is_empty(hypothesis.epistemic_snapshot):
                hypothesis.epistemic_snapshot = _build_hypothesis_snapshot(hypothesis)

        logger.info(
            f"[risk_hypothesizer] Generated {len(result.hypotheses)} hypotheses, "
            f"{len(valid_hypotheses)} with valid spans "
            f"(from {len(retrieved_chunks)} chunks via {len(expanded_queries)}-query expansion)"
        )

        # Build historical context for consolidator
        historical_context = {
            "similar_past_incidents": [
                {
                    "source_id": c.get("source_id"),
                    "severity": c.get("metadata", {}).get("severity"),
                    "resolution_notes": c.get("metadata", {}).get("resolution_notes", ""),
                    "mttr_minutes": c.get("metadata", {}).get("mttr_minutes"),
                    "chunk_role": c.get("metadata", {}).get("chunk_role", ""),
                    "similarity": c.get("similarity_score", 0),
                }
                for c in historical_chunks
                if c.get("metadata", {}).get("chunk_role") in ("resolution", "root_cause")
            ],
            "recurrence_count": recurrence_count,
        }

        # Write each hypothesis to the audit ledger (feeds flywheel REASONING_TRACE)
        incident_id = state.get("incident_id", "unknown")
        for h in valid_hypotheses:
            try:
                record_hypothesis(
                    incident_id=incident_id,
                    hypothesis=h.model_dump(),
                    node_name="risk_hypothesizer",
                )
            except Exception as e:
                logger.warning(f"[risk_hypothesizer] Ledger write failed: {e}")

        return {
            "hypotheses": [h.model_dump() for h in valid_hypotheses],
            "historical_context": historical_context,
        }

    except Exception as e:
        logger.error(f"[risk_hypothesizer] Error: {e}")
        return {
            "errors": state.get("errors", []) + [f"Risk hypothesizer failed: {str(e)}"],
        }
