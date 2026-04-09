"""
SRE Agent — Slot Filler Node (Track A)

Extracts structured entities from the raw incident report.
The LLM extracts, then symbolic validators verify types and formats.
"""

import logging
from app.agents.state import (
    EpistemicStatus,
    ExtractedEntity,
    make_epistemic_claim,
    merge_epistemic_snapshots,
    snapshot_is_empty,
)
from app.providers import llm_provider

logger = logging.getLogger(__name__)


def _contains_report_text(raw_report: str, value: str) -> bool:
    return bool(value and value.lower() in raw_report.lower())


def _build_entities_snapshot(raw_report: str, entities: ExtractedEntity):
    observed = []
    inferred = []
    unknown = []

    scalar_fields = {
        "error_code": entities.error_code,
        "error_message": entities.error_message,
        "stack_trace": entities.stack_trace,
        "endpoint_affected": entities.endpoint_affected,
        "reporter_name": entities.reporter_name,
        "reporter_email": entities.reporter_email,
        "timestamp_reported": entities.timestamp_reported,
    }

    for field_name, value in scalar_fields.items():
        if value:
            status = (
                EpistemicStatus.OBSERVED
                if _contains_report_text(raw_report, str(value))
                else EpistemicStatus.INFERRED
            )
            (observed if status == EpistemicStatus.OBSERVED else inferred).append(
                make_epistemic_claim(
                    label=f"{field_name}={value}",
                    status=status,
                    evidence=str(value),
                    source="slot_filler",
                )
            )
        else:
            unknown.append(
                make_epistemic_claim(
                    label=field_name,
                    status=EpistemicStatus.UNKNOWN,
                    evidence=f"{field_name} not present in the report.",
                    source="slot_filler",
                )
            )

    if entities.file_references:
        for ref in entities.file_references:
            status = (
                EpistemicStatus.OBSERVED
                if _contains_report_text(raw_report, ref)
                else EpistemicStatus.INFERRED
            )
            (observed if status == EpistemicStatus.OBSERVED else inferred).append(
                make_epistemic_claim(
                    label=f"file_reference={ref}",
                    status=status,
                    evidence=ref,
                    source="slot_filler",
                )
            )
    else:
        unknown.append(
            make_epistemic_claim(
                label="file_references",
                status=EpistemicStatus.UNKNOWN,
                evidence="No file references mentioned in the report.",
                source="slot_filler",
            )
        )

    return merge_epistemic_snapshots(
        {"observed": observed, "inferred": inferred, "unknown": unknown}
    )


async def slot_filler_node(state: dict) -> dict:
    """Extract structured entities from the incident report."""
    from app.providers.llm_provider import _langfuse, _LANGFUSE_ENABLED, _noop_ctx

    raw_report = state.get("raw_report", "")
    incident_id = state.get("incident_id", "unknown")

    from app.agents.prompts import SLOT_FILLER_SYSTEM, build_slot_filler_prompt
    prompt = build_slot_filler_prompt(raw_report)

    node_ctx = (
        _langfuse.start_as_current_observation(
            as_type="span",
            name="node:slot_filler",
            input={"incident_id": incident_id},
            metadata={"node": "slot_filler", "incident_id": incident_id},
        )
        if _LANGFUSE_ENABLED
        else _noop_ctx()
    )
    try:
        with node_ctx as node_obs:
            entities = await llm_provider.generate_structured(
                prompt=prompt,
                response_schema=ExtractedEntity,
                system_instruction=SLOT_FILLER_SYSTEM,
            )

            logger.info(
                f"[slot_filler] Extracted: error_code={entities.error_code}, "
                f"endpoint={entities.endpoint_affected}, "
                f"files={entities.file_references}"
            )

            if snapshot_is_empty(entities.epistemic_snapshot):
                entities.epistemic_snapshot = _build_entities_snapshot(raw_report, entities)

            if _LANGFUSE_ENABLED and node_obs:
                node_obs.update(
                    output={
                        "error_code": entities.error_code,
                        "error_message": (entities.error_message or "")[:200],
                        "endpoint_affected": entities.endpoint_affected,
                        "file_references": entities.file_references,
                        "thinking_process": entities.thinking_process[:500] if entities.thinking_process else "",
                    }
                )
            return {"entities": entities.model_dump()}

    except Exception as e:
        logger.error(f"[slot_filler] Error: {e}")
        return {
            "errors": state.get("errors", []) + [f"Slot filler failed: {str(e)}"],
        }
