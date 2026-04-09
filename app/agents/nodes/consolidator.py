"""
SRE Agent — Consolidator Node

Merges Track A (World Model + Entities) and Track B (verified hypotheses)
into a final triage summary. Computes severity deterministically via FSM.
Transitions state to TRIAGED.
"""

import copy
import logging
from app.agents.state import (
    IncidentStatus,
    Severity,
    ensure_epistemic_snapshot,
    merge_epistemic_snapshots,
)
from app.symbolic.fsm import compute_severity
from app.agents.state import IncidentState
from app.providers import db_provider, llm_provider
from app.ledger.audit import record_state_transition, record_entry

logger = logging.getLogger(__name__)


def _snapshot_lines(snapshot) -> list[str]:
    normalized = ensure_epistemic_snapshot(snapshot)
    lines = []

    for claim in normalized.observed:
        lines.append(f"Observed: {claim.label} | evidence={claim.evidence}")
    for claim in normalized.inferred:
        lines.append(f"Inferred: {claim.label} | evidence={claim.evidence}")
    for claim in normalized.unknown:
        lines.append(f"Unknown: {claim.label} | evidence={claim.evidence}")

    return lines


def _format_epistemic_context(context: dict) -> str:
    lines = []
    for bucket in ("observed", "inferred", "unknown"):
        claims = context.get(bucket, [])
        lines.append(f"{bucket.capitalize()}:")
        if claims:
            lines.extend(
                [
                    f"- {claim.get('label', '')} | evidence={claim.get('evidence', '')}"
                    for claim in claims[:6]
                ]
            )
        else:
            lines.append("- None")
    return "\n".join(lines)


def _build_final_epistemic_context(
    world_model: dict,
    entities: dict,
    verified_payloads: list[dict],
    discarded_payloads: list[dict],
) -> dict:
    world_snapshot = ensure_epistemic_snapshot(world_model.get("epistemic_snapshot"))
    entity_snapshot = ensure_epistemic_snapshot(entities.get("epistemic_snapshot"))

    verified_snapshots = [
        merge_epistemic_snapshots(
            payload.get("hypothesis", {}).get("epistemic_snapshot"),
            payload.get("span_verdict", {}).get("epistemic_snapshot"),
            payload.get("falsifier_verdict", {}).get("epistemic_snapshot"),
        )
        for payload in verified_payloads
    ]
    merged = merge_epistemic_snapshots(
        world_snapshot,
        entity_snapshot,
        *verified_snapshots,
    )

    return {
        "observed": [claim.model_dump() for claim in merged.observed],
        "inferred": [claim.model_dump() for claim in merged.inferred],
        "unknown": [claim.model_dump() for claim in merged.unknown],
        "world_model": world_snapshot.model_dump(),
        "entities": entity_snapshot.model_dump(),
        "verified_hypotheses": [
            {
                "hypothesis_id": payload.get("hypothesis", {}).get("hypothesis_id", ""),
                "description": payload.get("hypothesis", {}).get("description", ""),
                "hypothesis_snapshot": ensure_epistemic_snapshot(
                    payload.get("hypothesis", {}).get("epistemic_snapshot")
                ).model_dump(),
                "span_verdict": payload.get("span_verdict", {}),
                "falsifier_verdict": payload.get("falsifier_verdict", {}),
            }
            for payload in verified_payloads
        ],
        "discarded_hypotheses": [
            {
                "hypothesis_id": payload.get("hypothesis", {}).get("hypothesis_id", ""),
                "description": payload.get("hypothesis", {}).get("description", ""),
                "hypothesis_snapshot": ensure_epistemic_snapshot(
                    payload.get("hypothesis", {}).get("epistemic_snapshot")
                ).model_dump(),
                "span_verdict": payload.get("span_verdict", {}),
                "falsifier_verdict": payload.get("falsifier_verdict", {}),
            }
            for payload in discarded_payloads
        ],
    }


def _format_historical_context(context: dict) -> str:
    """Format historical context from the flywheel for the triage prompt."""
    if not context:
        return "- No historical data available (knowledge base may be empty)."

    past_incidents = context.get("similar_past_incidents", [])
    recurrence = context.get("recurrence_count", 0)

    if not past_incidents:
        return "- No similar past incidents found. This appears to be a first occurrence."

    lines = [f"Found {len(past_incidents)} similar past incidents"
             f" ({recurrence} highly similar):"]

    for p in past_incidents[:3]:  # Top 3 precedents
        mttr = p.get("mttr_minutes")
        mttr_str = f", MTTR: {mttr}min" if mttr else ""
        lines.append(
            f"- {p.get('source_id', '?')}: "
            f"Severity={p.get('severity', '?')}{mttr_str}"
        )
        if p.get("resolution_notes"):
            lines.append(f"  Resolution: {p['resolution_notes'][:150]}")

    if recurrence >= 3:
        lines.append(f"⚠️ RECURRING ISSUE: This pattern has appeared {recurrence}+ times.")

    return "\n".join(lines)


async def consolidator_node(state: dict) -> dict:
    """
    Merge findings from both tracks and produce the final triage summary.
    """
    world_model = state.get("world_model") or {}
    entities = state.get("entities") or {}
    hypotheses = state.get("hypotheses", [])
    span_verdicts = state.get("span_verdicts", [])

    span_by_id = {verdict.get("hypothesis_id"): verdict for verdict in span_verdicts}
    falsifier_verdicts = state.get("falsifier_verdicts", [])
    falsifier_by_id = {
        verdict.get("hypothesis_id"): verdict for verdict in falsifier_verdicts
    }

    verified_payloads = []
    discarded_payloads = []

    for hypothesis in hypotheses:
        hyp_id = hypothesis.get("hypothesis_id")
        hyp_desc = hypothesis.get("description", "")[:80]
        span_verdict = span_by_id.get(hyp_id, {})
        falsifier_verdict = falsifier_by_id.get(hyp_id, {})

        span_passed = span_verdict.get("verdict") in ("VERIFIED", "PARTIAL_MATCH")
        was_falsified = falsifier_verdict.get("verdict") == "FALSIFIED"

        payload = {
            "hypothesis": hypothesis,
            "span_verdict": span_verdict,
            "falsifier_verdict": falsifier_verdict,
        }

        if span_passed and not was_falsified:
            verified_payloads.append(payload)
        else:
            discarded_payloads.append(payload)
            if not span_passed:
                logger.debug(f"[consolidator] REJECTED (span): {hyp_desc}")
            elif was_falsified:
                logger.info(f"[consolidator] FALSIFIED (Popper): {hyp_desc}")

    verified_hypotheses = [payload["hypothesis"] for payload in verified_payloads]
    verified_root_causes = [
        f"{hypothesis.get('suspected_file', 'unknown')}: {hypothesis.get('description', '')}"
        for hypothesis in verified_hypotheses
    ]

    falsifier_corroborated = {
        hyp_id for hyp_id, verdict in falsifier_by_id.items()
        if verdict.get("verdict") == "CORROBORATED"
    }
    falsifier_falsified = {
        hyp_id for hyp_id, verdict in falsifier_by_id.items()
        if verdict.get("verdict") == "FALSIFIED"
    }
    span_verified = {
        hyp_id for hyp_id, verdict in span_by_id.items()
        if verdict.get("verdict") in ("VERIFIED", "PARTIAL_MATCH")
    }

    logger.info(
        f"[consolidator] Epistemic filter: "
        f"{len(hypotheses)} total → {len(span_verified)} span-verified → "
        f"{len(verified_hypotheses)} passed falsification "
        f"({len(falsifier_falsified)} falsified, "
        f"{len(falsifier_corroborated)} corroborated)"
    )

    # --- Compute severity deterministically ---
    # Build a minimal state for severity computation
    temp_state = IncidentState(
        hypotheses=hypotheses,
        verified_root_causes=verified_root_causes,
    )
    if world_model:
        from app.agents.state import WorldModelProjection
        temp_state.world_model = WorldModelProjection(**world_model)

    final_severity = compute_severity(temp_state)

    # --- Generate triage summary using LLM ---
    discarded = [payload["hypothesis"] for payload in discarded_payloads]
    epistemic_context = _build_final_epistemic_context(
        world_model,
        entities,
        verified_payloads,
        discarded_payloads,
    )

    # --- Search for matching runbooks ---
    suggested_runbooks = []
    try:
        service = world_model.get("affected_service", "")

        # Reuse the report_embedding from the dedup phase if available (no extra API call)
        runbook_embedding = state.get("report_embedding")

        if runbook_embedding is None:
            # Fallback: generate embedding from verified causes (shorter, less likely to rate limit)
            error_msg = entities.get("error_message", "")
            category = world_model.get("incident_category", "")
            search_text = (
                f"{service} {category} {error_msg}"
                if error_msg
                else f"{service} {state.get('raw_report', '')[:150]}"
            )
            runbook_query = f"RUNBOOK steps escalation: {search_text}"[:300]
            if len(runbook_query.strip()) > 20:
                runbook_embedding = await llm_provider.generate_embedding(
                    runbook_query, task_type="RETRIEVAL_QUERY"
                )

        if runbook_embedding:
            # Build a text query for BM25 from available context
            runbook_text_query = locals().get('runbook_query') or f"{service} runbook escalation"
            runbook_results = db_provider.knowledge_search(
                query_vector=runbook_embedding,
                query_text=runbook_text_query,
                top_k=10,
                service_filter=service if service else None,
            )
            suggested_runbooks = [
                r for r in runbook_results
                if r.get("doc_type") == "RUNBOOK"
            ]
            logger.info(
                f"[consolidator] Found {len(suggested_runbooks)} matching runbooks "
                f"(from {len(runbook_results)} total results)"
            )
    except Exception as e:
        logger.warning(f"[consolidator] Runbook search failed: {e}")

    # Format runbooks for prompt
    if suggested_runbooks:
        runbook_section = "\n\n".join([
            f"📋 {r.get('chunk_text', '')[:500]}"
            for r in suggested_runbooks[:2]  # Top 2 runbooks
        ])
    else:
        runbook_section = "- No matching runbooks found."

    from app.agents.prompts import CONSOLIDATOR_SYSTEM, build_consolidator_prompt

    summary_prompt = build_consolidator_prompt(
        world_model=world_model,
        entities=entities,
        final_severity=final_severity.value,
        verified_root_causes=verified_root_causes,
        discarded=discarded,
        historical_context_formatted=_format_historical_context(state.get('historical_context', {})),
        runbook_section=runbook_section,
        epistemic_context_formatted=_format_epistemic_context(epistemic_context),
    )

    try:
        triage_summary = await llm_provider.generate_text(
            prompt=summary_prompt,
            system_instruction=CONSOLIDATOR_SYSTEM,
        )
    except Exception as e:
        logger.error(f"[consolidator] Summary generation failed: {e}")
        triage_summary = (
            f"AUTOMATED TRIAGE: {final_severity.value} severity incident in "
            f"{world_model.get('affected_service', 'unknown')}. "
            f"{len(verified_root_causes)} verified root causes, "
            f"{len(discarded)} hypotheses discarded (failed evidence check)."
        )

    logger.info(
        f"[consolidator] Final severity: {final_severity.value}, "
        f"verified: {len(verified_root_causes)}, discarded: {len(discarded)}"
    )

    # Write FSM transition + triage result to audit ledger (feeds flywheel)
    incident_id = state.get("incident_id", "unknown")
    try:
        record_state_transition(
            incident_id=incident_id,
            from_state="TRIAGING",
            to_state="TRIAGED",
            node_name="consolidator",
        )
        record_entry(
            incident_id=incident_id,
            event_type="TRIAGE_COMPLETE",
            node_name="consolidator",
            data={
                "final_severity": final_severity.value,
                "verified_causes_count": len(verified_root_causes),
                "discarded_count": len(discarded),
                "recurrence_count": state.get("historical_context", {}).get(
                    "recurrence_count", 0
                ),
                "epistemic_context": copy.deepcopy(epistemic_context),
            },
        )
    except Exception as e:
        logger.warning(f"[consolidator] Ledger write failed: {e}")

    return {
        "triage_summary": triage_summary,
        "final_severity": final_severity.value,
        "verified_root_causes": verified_root_causes,
        "epistemic_context": copy.deepcopy(epistemic_context),
        "suggested_runbooks": [
            {
                "runbook_id": r.get("metadata", {}).get("runbook_id", ""),
                "title": r.get("chunk_text", "").split("\n")[0],
                "escalation_path": r.get("metadata", {}).get("escalation_path", ""),
                "estimated_resolution_time": r.get("metadata", {}).get(
                    "estimated_resolution_time", ""
                ),
            }
            for r in suggested_runbooks
        ],
        "status": IncidentStatus.TRIAGED,
    }
