"""
SRE Agent — Actions Node

Creates tickets and sends notifications.
Uses provider interfaces (ABC pattern) so real/mock implementations
are interchangeable.
"""

import logging
import uuid
from datetime import datetime, timezone
from app.agents.state import IncidentStatus, TicketInfo, NotificationInfo

logger = logging.getLogger(__name__)


async def create_ticket_node(state: dict) -> dict:
    """
    Create a ticket in the ticketing system (mock).
    Transitions FSM: TRIAGED → TICKET_CREATED.
    """
    incident_id = state.get("incident_id", "unknown")
    world_model = state.get("world_model", {})
    triage_summary = state.get("triage_summary", "No summary available")
    final_severity = state.get("final_severity", "UNKNOWN")

    # --- Mock Jira ticket creation ---
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
        ticket_url=f"https://jira.example.com/browse/{ticket_id}",
        assigned_team=team_map.get(service, "Platform Team"),
        priority=priority_map.get(final_severity, "P3 - Medium"),
    )

    # Attach runbook suggestions to ticket
    suggested_runbooks = state.get("suggested_runbooks", [])

    logger.info(
        f"[actions] 🎫 Ticket created: {ticket_id} | "
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
    Notify the assigned team via Slack and Email (mock).
    Transitions FSM: TICKET_CREATED → TEAM_NOTIFIED.
    """
    ticket = state.get("ticket", {})
    triage_summary = state.get("triage_summary", "")
    final_severity = state.get("final_severity", "UNKNOWN")
    incident_id = state.get("incident_id", "unknown")

    ticket_id = ticket.get("ticket_id", "unknown")
    assigned_team = ticket.get("assigned_team", "Platform Team")

    # --- Mock Slack notification ---
    slack_message = (
        f"🚨 *New Incident* [{final_severity}] — {ticket_id}\n"
        f"Team: {assigned_team}\n"
        f"Summary: {triage_summary[:200]}...\n"
        f"Ticket: {ticket.get('ticket_url', 'N/A')}"
    )
    logger.info(f"[actions] 📢 Slack notification sent to #{assigned_team}:\n{slack_message}")

    # --- Mock Email notification ---
    email_subject = f"[{final_severity}] Incident {ticket_id} — Action Required"
    logger.info(f"[actions] 📧 Email sent to {assigned_team}@eshop.com: {email_subject}")

    notifications = NotificationInfo(
        team_notified=True,
        team_notification_channel=f"slack:#{assigned_team}, email:{assigned_team}@eshop.com",
        reporter_notified=False,
        reporter_notification_channel="",
    )

    return {
        "notifications": notifications.model_dump(),
        "status": IncidentStatus.TEAM_NOTIFIED,
    }
