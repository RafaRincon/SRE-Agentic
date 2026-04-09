from __future__ import annotations

"""
SRE Agent — Ledger Module

Provides utility functions to record immutable audit entries
at each stage of the pipeline. This creates a full trace of
every decision, hypothesis, and verdict — critical for:
- Post-incident review
- Regulatory compliance
- Debugging agent behavior
"""

import copy
import uuid
import logging
from datetime import datetime, timezone
from typing import Any

from app.providers import db_provider

logger = logging.getLogger(__name__)


def record_entry(
    incident_id: str,
    event_type: str,
    data: dict[str, Any],
    node_name: str = "",
) -> dict:
    """
    Append an immutable entry to the audit ledger.

    Args:
        incident_id: The incident this entry belongs to.
        event_type: TYPE of event (e.g., 'STATE_TRANSITION', 'HYPOTHESIS_GENERATED', 'SPAN_VERDICT').
        data: Arbitrary data payload for the entry.
        node_name: Which graph node produced this entry.

    Returns:
        The created ledger document.
    """
    entry = {
        "id": uuid.uuid4().hex,
        "incident_id": incident_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "node_name": node_name,
        "data": copy.deepcopy(data),
    }

    try:
        result = db_provider.append_ledger_entry(entry)
        logger.debug(f"[ledger] Recorded {event_type} for {incident_id}")
        return result
    except Exception as e:
        logger.warning(f"[ledger] Failed to record entry: {e}")
        return entry


def record_state_transition(
    incident_id: str,
    from_state: str,
    to_state: str,
    node_name: str = "",
) -> dict:
    """Record an FSM state transition."""
    return record_entry(
        incident_id=incident_id,
        event_type="STATE_TRANSITION",
        node_name=node_name,
        data={
            "from_state": from_state,
            "to_state": to_state,
        },
    )


def record_hypothesis(
    incident_id: str,
    hypothesis: dict,
    node_name: str = "risk_hypothesizer",
) -> dict:
    """Record a generated hypothesis."""
    return record_entry(
        incident_id=incident_id,
        event_type="HYPOTHESIS_GENERATED",
        node_name=node_name,
        data=hypothesis,
    )


def record_verdict(
    incident_id: str,
    verdict: dict,
    verdict_type: str = "SPAN_VERDICT",
    node_name: str = "span_arbiter",
) -> dict:
    """Record a verification verdict."""
    return record_entry(
        incident_id=incident_id,
        event_type=verdict_type,
        node_name=node_name,
        data=verdict,
    )
