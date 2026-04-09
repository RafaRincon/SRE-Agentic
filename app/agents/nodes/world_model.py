"""
SRE Agent — World Model Node (Track A)

Uses the LLM to create a cognitive projection of the incident.
This is NOT a diagnostic — it's a structured "imagination" of the
incident's state across multiple business dimensions.

Output: WorldModelProjection (service, severity, blast radius, category).
"""

import base64
import logging
from app.agents.state import WorldModelProjection
from app.providers import llm_provider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert SRE (Site Reliability Engineer) performing initial triage on an incident report for the eShop e-commerce application (.NET microservices architecture).

The eShop application consists of these key services:
- Catalog.API: Product catalog management
- Ordering.API: Order processing and management
- Basket.API: Shopping cart functionality
- Payment.API: Payment processing
- Identity.API: User authentication
- WebApp: Frontend web application
- OrderProcessor: Background order processing worker
- WebhookClient: Webhook delivery service

Your task is to project a "World Model" — a structured assessment of the incident's impact across multiple dimensions. You are NOT diagnosing yet, you are IMAGINING the state of the system based on the report.

For each field, classify your confidence:
- OBSERVED: explicitly stated in the report
- INFERRED: you deduced it from context clues
- UNKNOWN: not enough information

Be precise. Do not hallucinate services or components that aren't in eShop."""


async def world_model_node(state: dict) -> dict:
    """
    Project a cognitive model of the incident.
    Uses multimodal input if an image is attached.
    """
    raw_report = state.get("raw_report", "")
    has_image = state.get("has_image", False)
    image_data_b64 = state.get("image_data_b64", "")
    image_mime_type = state.get("image_mime_type", "image/png")

    prompt = f"""Analyze this incident report and produce a World Model projection:

--- INCIDENT REPORT ---
{raw_report}
--- END REPORT ---

Project the incident state across all dimensions."""

    try:
        if has_image and image_data_b64:
            image_bytes = base64.b64decode(image_data_b64)
            projection = await llm_provider.generate_multimodal(
                text_prompt=prompt,
                image_bytes=image_bytes,
                image_mime_type=image_mime_type,
                system_instruction=SYSTEM_PROMPT,
                response_schema=WorldModelProjection,
            )
        else:
            projection = await llm_provider.generate_structured(
                prompt=prompt,
                response_schema=WorldModelProjection,
                system_instruction=SYSTEM_PROMPT,
            )

        logger.info(
            f"[world_model] Projected: service={projection.affected_service}, "
            f"severity={projection.estimated_severity}, "
            f"blast_radius={projection.blast_radius}"
        )

        return {"world_model": projection.model_dump()}

    except Exception as e:
        logger.error(f"[world_model] Error: {e}")
        return {
            "errors": state.get("errors", []) + [f"World Model failed: {str(e)}"],
        }
