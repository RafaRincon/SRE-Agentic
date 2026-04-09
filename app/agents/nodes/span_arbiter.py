"""
SRE Agent — Span Arbiter Node (Track B, Symbolic)

Verifies each hypothesis's exact_span against the indexed eShop codebase.
Uses vector search to find candidate code chunks, then applies
deterministic string matching to verify the citation.

If the span is NOT found → the hypothesis is marked as HALLUCINATION.
If found → the hypothesis advances to the Epistemic Falsifier.
"""

import logging
from app.agents.state import SpanVerdict
from app.symbolic.span_matcher import fuzzy_match_span
from app.providers import llm_provider, db_provider
from app.ledger.audit import record_verdict, record_state_transition

logger = logging.getLogger(__name__)


async def span_arbiter_node(state: dict) -> dict:
    """
    For each hypothesis, verify its exact_span against the real codebase.
    100% symbolic after the initial vector search retrieval.
    """
    hypotheses = state.get("hypotheses", [])
    if not hypotheses:
        logger.info("[span_arbiter] No hypotheses to verify")
        return {"span_verdicts": []}

    verdicts = []

    for hyp in hypotheses:
        hypothesis_id = hyp.get("hypothesis_id", "unknown")
        exact_span = hyp.get("exact_span", "")
        suspected_file = hyp.get("suspected_file", "")

        logger.info(f"[span_arbiter] Verifying hypothesis {hypothesis_id}: '{exact_span[:50]}...'")

        try:
            # Step 1: Use vector search to find candidate chunks near the span
            query_embedding = await llm_provider.generate_embedding(
                exact_span,
                task_type="RETRIEVAL_QUERY",
            )

            candidates = db_provider.vector_search(
                query_vector=query_embedding,
                query_text=exact_span,
                top_k=5,
            )

            if suspected_file:
                # Pre-filter candidates by the file the LLM claims an issue is in
                file_candidates = [c for c in candidates if suspected_file in c.get("file_path", "")]
                if file_candidates:
                    candidates = file_candidates

            # Step 2: Deterministic string matching against candidates
            best_match = False
            best_score = 0.0
            matched_file = None
            matched_line = None

            for candidate in candidates:
                chunk_text = candidate.get("chunk_text", "")
                file_path = candidate.get("file_path", "")

                matched, score = fuzzy_match_span(exact_span, chunk_text, threshold=0.6)

                if score > best_score:
                    best_score = score
                    best_match = matched
                    matched_file = file_path
                    matched_line = candidate.get("start_line")

            # Step 3: Produce verdict
            if best_match and best_score >= 0.6:
                verdict = "VERIFIED"
            elif best_score >= 0.4:
                verdict = "PARTIAL_MATCH"
            else:
                verdict = "HALLUCINATION"

            span_verdict = SpanVerdict(
                hypothesis_id=hypothesis_id,
                span_found=best_match,
                matched_file=matched_file,
                matched_line=matched_line,
                similarity_score=best_score,
                verdict=verdict,
            )

            logger.info(
                f"[span_arbiter] Hypothesis {hypothesis_id}: "
                f"{verdict} (score={best_score:.2f}, file={matched_file})"
            )

        except Exception as e:
            logger.error(f"[span_arbiter] Error verifying {hypothesis_id}: {e}")
            span_verdict = SpanVerdict(
                hypothesis_id=hypothesis_id,
                span_found=False,
                similarity_score=0.0,
                verdict="ERROR",
            )

        verdicts.append(span_verdict.model_dump())

        # Write verdict to audit ledger (feeds the flywheel REASONING_TRACE)
        try:
            record_verdict(
                incident_id=state.get("incident_id", "unknown"),
                verdict=span_verdict.model_dump(),
                verdict_type="SPAN_VERDICT",
                node_name="span_arbiter",
            )
        except Exception as e:
            logger.warning(f"[span_arbiter] Ledger write failed: {e}")

    return {"span_verdicts": verdicts}
