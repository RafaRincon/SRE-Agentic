"""
SRE Agent — Slot Filler Node (Track A)

Extracts structured entities from the raw incident report.
The LLM extracts, then symbolic validators verify types and formats.
"""

import logging
from app.agents.state import ExtractedEntity
from app.providers import llm_provider

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an entity extraction system for SRE incident reports.

Extract the following structured fields from the report:
- error_code: HTTP status code or application error code (e.g., "500", "TIMEOUT", "NullReferenceException")
- error_message: The error message text
- stack_trace: Any stack trace or exception details
- file_references: List of file paths mentioned (e.g., "OrdersController.cs", "PaymentService.cs")
- endpoint_affected: API endpoint or URL that failed (e.g., "/api/orders", "/checkout")
- reporter_name: Name of the person reporting
- reporter_email: Email of the reporter
- timestamp_reported: When the issue was first observed

If a field is not present in the report, leave it as null or empty.
Do NOT fabricate information. Only extract what is explicitly stated."""


async def slot_filler_node(state: dict) -> dict:
    """Extract structured entities from the incident report."""
    raw_report = state.get("raw_report", "")

    prompt = f"""Extract all structured entities from this incident report:

--- INCIDENT REPORT ---
{raw_report}
--- END REPORT ---

Return the extracted entities as structured JSON."""

    try:
        entities = await llm_provider.generate_structured(
            prompt=prompt,
            response_schema=ExtractedEntity,
            system_instruction=SYSTEM_PROMPT,
        )

        logger.info(
            f"[slot_filler] Extracted: error_code={entities.error_code}, "
            f"endpoint={entities.endpoint_affected}, "
            f"files={entities.file_references}"
        )

        return {"entities": entities.model_dump()}

    except Exception as e:
        logger.error(f"[slot_filler] Error: {e}")
        return {
            "errors": state.get("errors", []) + [f"Slot filler failed: {str(e)}"],
        }
