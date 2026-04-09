"""
SRE Agent — World Model Node (Track A)

Uses the LLM to create a cognitive projection of the incident.
This is NOT a diagnostic — it's a structured "imagination" of the
incident's state across multiple business dimensions.

Output: WorldModelProjection (service, severity, blast radius, category).
"""

import base64
import logging
from app.agents.state import (
    EpistemicStatus,
    WorldModelProjection,
    make_epistemic_claim,
    merge_epistemic_snapshots,
    snapshot_is_empty,
)
from app.providers import llm_provider

logger = logging.getLogger(__name__)


def _contains_report_text(raw_report: str, value: str) -> bool:
    return bool(value and value.lower() in raw_report.lower())


def _build_world_model_snapshot(raw_report: str, projection: WorldModelProjection):
    observed = []
    inferred = []
    unknown = []

    if projection.affected_service:
        status = (
            projection.affected_service_confidence
            if projection.affected_service_confidence != EpistemicStatus.UNKNOWN
            else (
                EpistemicStatus.OBSERVED
                if _contains_report_text(raw_report, projection.affected_service)
                else EpistemicStatus.INFERRED
            )
        )
        (observed if status == EpistemicStatus.OBSERVED else inferred).append(
            make_epistemic_claim(
                label=f"affected_service={projection.affected_service}",
                status=status,
                evidence=projection.affected_service,
                source="world_model",
            )
        )
    else:
        unknown.append(
            make_epistemic_claim(
                label="affected_service",
                status=EpistemicStatus.UNKNOWN,
                evidence="Service not identified in the report.",
                source="world_model",
            )
        )

    if projection.incident_category:
        status = (
            EpistemicStatus.OBSERVED
            if _contains_report_text(raw_report, projection.incident_category)
            else EpistemicStatus.INFERRED
        )
        (observed if status == EpistemicStatus.OBSERVED else inferred).append(
            make_epistemic_claim(
                label=f"incident_category={projection.incident_category}",
                status=status,
                evidence=projection.incident_category,
                source="world_model",
            )
        )
    else:
        unknown.append(
            make_epistemic_claim(
                label="incident_category",
                status=EpistemicStatus.UNKNOWN,
                evidence="Category was not explicit in the report.",
                source="world_model",
            )
        )

    if projection.estimated_severity.value != "UNKNOWN":
        inferred.append(
            make_epistemic_claim(
                label=f"estimated_severity={projection.estimated_severity.value}",
                status=EpistemicStatus.INFERRED,
                evidence=projection.severity_rationale or "Projected from triage context.",
                source="world_model",
            )
        )
    else:
        unknown.append(
            make_epistemic_claim(
                label="estimated_severity",
                status=EpistemicStatus.UNKNOWN,
                evidence="Severity could not be projected confidently.",
                source="world_model",
            )
        )

    if projection.temporal_context:
        status = (
            EpistemicStatus.OBSERVED
            if _contains_report_text(raw_report, projection.temporal_context)
            else EpistemicStatus.INFERRED
        )
        (observed if status == EpistemicStatus.OBSERVED else inferred).append(
            make_epistemic_claim(
                label=f"temporal_context={projection.temporal_context}",
                status=status,
                evidence=projection.temporal_context,
                source="world_model",
            )
        )

    if projection.blast_radius:
        for service in projection.blast_radius:
            status = (
                EpistemicStatus.OBSERVED
                if _contains_report_text(raw_report, service)
                else EpistemicStatus.INFERRED
            )
            (observed if status == EpistemicStatus.OBSERVED else inferred).append(
                make_epistemic_claim(
                    label=f"blast_radius={service}",
                    status=status,
                    evidence=service,
                    source="world_model",
                )
            )
    else:
        unknown.append(
            make_epistemic_claim(
                label="blast_radius",
                status=EpistemicStatus.UNKNOWN,
                evidence="No impacted downstream services were explicit.",
                source="world_model",
            )
        )

    return merge_epistemic_snapshots(
        {"observed": observed, "inferred": inferred, "unknown": unknown}
    )




async def world_model_node(state: dict) -> dict:
    """
    Project a cognitive model of the incident.
    Uses multimodal input if an image is attached.
    """
    raw_report = state.get("raw_report", "")
    has_image = state.get("has_image", False)
    image_data_b64 = state.get("image_data_b64", "")
    image_mime_type = state.get("image_mime_type", "image/png")

    from app.agents.prompts import WORLD_MODEL_SYSTEM, build_world_model_prompt
    prompt = build_world_model_prompt(raw_report)

    try:
        if has_image and image_data_b64:
            image_bytes = base64.b64decode(image_data_b64)
            projection = await llm_provider.generate_multimodal(
                text_prompt=prompt,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                system_instruction=WORLD_MODEL_SYSTEM,
                response_schema=WorldModelProjection,
            )
        else:
            projection = await llm_provider.generate_structured(
                prompt=prompt,
                response_schema=WorldModelProjection,
                system_instruction=WORLD_MODEL_SYSTEM,
            )

        logger.info(
            f"[world_model] Projected: service={projection.affected_service}, "
            f"severity={projection.estimated_severity}, "
            f"blast_radius={projection.blast_radius}"
        )

        if snapshot_is_empty(projection.epistemic_snapshot):
            projection.epistemic_snapshot = _build_world_model_snapshot(
                raw_report,
                projection,
            )

        return {"world_model": projection.model_dump()}

    except Exception as e:
        logger.error(f"[world_model] Error: {e}")
        return {
            "errors": state.get("errors", []) + [f"World Model failed: {str(e)}"],
        }
