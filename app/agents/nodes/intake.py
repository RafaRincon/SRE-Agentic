"""
SRE Agent — Intake Node

Receives the raw incident report (text + optional image),
validates basic inputs, and transitions FSM to TRIAGING.
"""

import base64
import logging
from app.agents.state import IncidentState, IncidentStatus

logger = logging.getLogger(__name__)


async def intake_node(state: dict) -> dict:
    """
    Entry point for the incident pipeline.
    Validates the report exists and advances FSM: RECEIVED → TRIAGING.
    """
    logger.info(f"[intake] Processing incident {state.get('incident_id', 'unknown')}")

    raw_report = state.get("raw_report", "")
    if not raw_report or len(raw_report.strip()) < 10:
        return {
            "errors": state.get("errors", []) + ["Report too short or empty. Minimum 10 characters."],
        }

    # Validate image if present
    has_image = state.get("has_image", False)
    if has_image:
        image_data = state.get("image_data_b64", "")
        if not image_data:
            has_image = False
            logger.warning("[intake] has_image=True but no image data provided")

    # Transition FSM: RECEIVED → TRIAGING
    return {
        "status": IncidentStatus.TRIAGING,
        "has_image": has_image,
    }
