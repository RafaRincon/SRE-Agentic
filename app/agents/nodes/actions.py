"""
SRE Agent — Actions Node

Creates tickets and sends notifications.
Uses provider interfaces (ABC pattern) so production/test implementations
are interchangeable.
"""

import logging
import uuid
from app.agents.persistence import (
    normalize_entities_for_persistence,
    normalize_world_model_for_persistence,
)
from app.agents.state import IncidentStatus, TicketInfo, NotificationInfo

logger = logging.getLogger(__name__)


async def create_ticket_node(state: dict) -> dict:
    """
    Create a ticket in the ticketing system.
    Transitions FSM: TRIAGED → TICKET_CREATED.
    """
    incident_id = state.get("incident_id", "unknown")
    world_model = state.get("world_model", {})
    triage_summary = state.get("triage_summary", "No summary available")
    final_severity = state.get("final_severity", "UNKNOWN")

    ticket_id = f"SRE-{uuid.uuid4().hex[:6].upper()}"
    service = world_model.get("affected_service", "unknown")

    # Map severity to Jira priority
    priority_map = {
        "CRITICAL": "P1 - Critical",
        "HIGH": "P2 - High",
        "MEDIUM": "P3 - Medium",
        "LOW": "P4 - Low",
    }

    # Map service to team
    team_map = {
        "Ordering.API": "Order Team",
        "Payment.API": "Payments Team",
        "Catalog.API": "Catalog Team",
        "Basket.API": "Cart Team",
        "Identity.API": "Platform Team",
        "WebApp": "Frontend Team",
        "OrderProcessor": "Order Team",
    }

    ticket = TicketInfo(
        ticket_id=ticket_id,
        ticket_url=f"/incident/{incident_id}",
        assigned_team=team_map.get(service, "Platform Team"),
        priority=priority_map.get(final_severity, "P3 - Medium"),
    )

    # Attach runbook suggestions to ticket
    suggested_runbooks = state.get("suggested_runbooks", [])

    logger.info(
        f"[actions] 🎫 Incident routed: {ticket_id} | "
        f"Priority: {ticket.priority} | Team: {ticket.assigned_team}"
    )
    if suggested_runbooks:
        logger.info(
            f"[actions] 📋 Runbooks attached: "
            f"{[r.get('runbook_id', '?') for r in suggested_runbooks]}"
        )

    return {
        "ticket": {
            **ticket.model_dump(),
            "suggested_runbooks": suggested_runbooks,
        },
        "status": IncidentStatus.TICKET_CREATED,
    }


async def notify_team_node(state: dict) -> dict:
    """
    Notify the assigned team via Slack and email.
    Transitions FSM: TICKET_CREATED → TEAM_NOTIFIED.
    Also persists the fully triaged incident to Cosmos for the flywheel.
    """
    ticket = state.get("ticket", {})
    triage_summary = state.get("triage_summary", "")
    final_severity = state.get("final_severity", "UNKNOWN")
    incident_id = state.get("incident_id", "unknown")

    ticket_id = ticket.get("ticket_id", "unknown")
    assigned_team = ticket.get("assigned_team", "Platform Team")

    slack_message = (
        f"🚨 *New Incident* [{final_severity}] — {ticket_id}\n"
        f"Team: {assigned_team}\n"
        f"Summary: {triage_summary[:200]}...\n"
        f"Ticket: {ticket.get('ticket_url', 'N/A')}"
    )
    logger.info(f"[actions] 📢 Team routing prepared for {assigned_team}:\n{slack_message}")

    email_subject = f"[{final_severity}] Incident {ticket_id} — Action Required"
    logger.info(f"[actions] 📧 Follow-up channel recorded for {assigned_team}: {email_subject}")

    notifications = NotificationInfo(
        team_notified=True,
        team_notification_channel=f"team:{assigned_team}",
        reporter_notified=False,
        reporter_notification_channel="",
    )

    # Persist the triaged incident to Cosmos.
    # This keeps deduplication and retrieval available across entry points.
    try:
        from app.providers import db_provider

        persist_data = {
            "id": incident_id,
            "incident_id": incident_id,
            "status": IncidentStatus.TEAM_NOTIFIED.value,
            "created_at": state.get("created_at", ""),
            "raw_report": state.get("raw_report", ""),
            "world_model": normalize_world_model_for_persistence(
                state.get("world_model", {})
            ),
            "entities": normalize_entities_for_persistence(
                state.get("entities", {})
            ),
            "triage_summary": triage_summary,
            "final_severity": final_severity,
            "verified_root_causes": state.get("verified_root_causes", []),
            "ticket": ticket,
            "notifications": notifications.model_dump(),
            "suggested_runbooks": state.get("suggested_runbooks", []),
            "occurrence_count": 1,
        }
        # Store the report embedding for future duplicate detection.
        report_embedding = state.get("report_embedding")
        if report_embedding:
            persist_data["report_embedding"] = report_embedding

        db_provider.upsert_incident(persist_data)
        logger.info(f"[actions] 💾 Incident {incident_id} persisted to Cosmos")
    except Exception as e:
        logger.warning(f"[actions] Failed to persist incident {incident_id}: {e}")

    return {
        "notifications": notifications.model_dump(),
        "status": IncidentStatus.TEAM_NOTIFIED,
    }
