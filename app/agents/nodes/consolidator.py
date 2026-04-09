"""
SRE Agent — Consolidator Node

Merges Track A (World Model + Entities) and Track B (verified hypotheses)
into a final triage summary. Computes severity deterministically via FSM.
Transitions state to TRIAGED.
"""

import logging
from app.agents.state import IncidentStatus, Severity
from app.symbolic.fsm import compute_severity
from app.agents.state import IncidentState
from app.providers import db_provider, llm_provider
from app.ledger.audit import record_state_transition, record_entry

logger = logging.getLogger(__name__)


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
    world_model = state.get("world_model", {})
    entities = state.get("entities", {})
    hypotheses = state.get("hypotheses", [])
    span_verdicts = state.get("span_verdicts", [])

    # --- Determine verified root causes (survived span arbitration) ---
    verified_hypotheses = []
    for hyp in hypotheses:
        hyp_id = hyp.get("hypothesis_id")
        verdict = next(
            (v for v in span_verdicts if v.get("hypothesis_id") == hyp_id),
            None,
        )
        if verdict and verdict.get("verdict") in ("VERIFIED", "PARTIAL_MATCH"):
            verified_hypotheses.append(hyp)

    verified_root_causes = [
        f"{h.get('suspected_file', 'unknown')}: {h.get('description', '')}"
        for h in verified_hypotheses
    ]

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
    discarded = [
        h for h in hypotheses
        if h.get("hypothesis_id") not in [v.get("hypothesis_id") for v in verified_hypotheses]
    ]

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

    summary_prompt = f"""Generate a concise technical triage summary for this incident.

INCIDENT OVERVIEW:
- Service: {world_model.get('affected_service', 'unknown')}
- Category: {world_model.get('incident_category', 'unknown')}
- Severity: {final_severity.value}
- Blast Radius: {world_model.get('blast_radius', [])}

EXTRACTED ENTITIES:
- Error: {entities.get('error_code', 'N/A')} — {entities.get('error_message', 'N/A')}
- Endpoint: {entities.get('endpoint_affected', 'N/A')}

VERIFIED ROOT CAUSES ({len(verified_hypotheses)} hypotheses survived verification):
{chr(10).join(f'- {rc}' for rc in verified_root_causes) or '- No verified root causes'}

DISCARDED HYPOTHESES ({len(discarded)} failed span verification — likely hallucinated):
{chr(10).join(f'- {h.get("description", "")} [REASON: citation not found in codebase]' for h in discarded) or '- None'}

HISTORICAL PRECEDENTS:
{_format_historical_context(state.get('historical_context', {}))}

Write a 3-5 sentence summary for the on-call engineering team. Include specific file names and functions only for VERIFIED causes. Mention that {len(discarded)} hypotheses were discarded due to failed evidence verification. If precedents exist, mention them and suggest proven resolutions. If this is a RECURRING issue, flag it explicitly.

SUGGESTED RUNBOOKS:
{runbook_section}

If a matching runbook exists, include the runbook ID and key action steps in your summary."""

    try:
        triage_summary = await llm_provider.generate_text(
            prompt=summary_prompt,
            system_instruction="You are an SRE writing a concise technical incident triage summary. Be direct, factual, and actionable.",
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
            },
        )
    except Exception as e:
        logger.warning(f"[consolidator] Ledger write failed: {e}")

    return {
        "triage_summary": triage_summary,
        "final_severity": final_severity.value,
        "verified_root_causes": verified_root_causes,
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
