"""
SRE Agent — Deterministic Finite State Machine

Controls incident lifecycle transitions with ALGEBRAIC conditions.
The LLM NEVER decides the state — only pure boolean logic does.

Anti-pattern: NEVER let an LLM decide "what state should this be in?"
"""

from app.agents.state import IncidentState, IncidentStatus, Severity
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transition rules (purely algebraic)
# ---------------------------------------------------------------------------

TRANSITIONS = {
    IncidentStatus.RECEIVED: {
        "next": IncidentStatus.TRIAGING,
        "condition": lambda s: bool(s.raw_report),  # Must have a report
    },
    IncidentStatus.TRIAGING: {
        "next": IncidentStatus.TRIAGED,
        "condition": lambda s: (
            s.world_model is not None
            and s.entities is not None
            and s.triage_summary != ""
            and s.final_severity != Severity.UNKNOWN
        ),
    },
    IncidentStatus.TRIAGED: {
        "next": IncidentStatus.TICKET_CREATED,
        "condition": lambda s: (
            s.ticket is not None
            and s.ticket.ticket_id != ""
        ),
    },
    IncidentStatus.TICKET_CREATED: {
        "next": IncidentStatus.TEAM_NOTIFIED,
        "condition": lambda s: (
            s.notifications is not None
            and s.notifications.team_notified
        ),
    },
    IncidentStatus.TEAM_NOTIFIED: {
        "next": IncidentStatus.RESOLVED,
        "condition": lambda s: True,  # Requires external trigger (resolve endpoint)
    },
    IncidentStatus.RESOLVED: {
        "next": IncidentStatus.REPORTER_NOTIFIED,
        "condition": lambda s: (
            s.notifications is not None
            and s.notifications.reporter_notified
        ),
    },
}


def can_transition(state: IncidentState) -> bool:
    """Check if the current state can transition to the next."""
    current = state.status
    if current not in TRANSITIONS:
        return False
    rule = TRANSITIONS[current]
    return rule["condition"](state)


def try_transition(state: IncidentState) -> IncidentState:
    """
    Attempt to advance the FSM by one step.
    Returns the state with an updated status if the transition is valid.
    """
    current = state.status
    if current not in TRANSITIONS:
        logger.info(f"FSM: terminal state {current.value}, no transition possible")
        return state

    rule = TRANSITIONS[current]
    if rule["condition"](state):
        new_status = rule["next"]
        logger.info(f"FSM: {current.value} → {new_status.value}")
        state.status = new_status
        return state
    else:
        logger.info(f"FSM: conditions not met for {current.value} → {rule['next'].value}")
        return state


def force_transition(state: IncidentState, target: IncidentStatus) -> IncidentState:
    """
    Force a transition to a specific state (used for external triggers like resolution).
    Validates that the transition is legal (target must be the next state).
    """
    current = state.status
    if current in TRANSITIONS and TRANSITIONS[current]["next"] == target:
        logger.info(f"FSM: forced transition {current.value} → {target.value}")
        state.status = target
        return state
    else:
        raise ValueError(
            f"Illegal transition: {current.value} → {target.value}. "
            f"Expected next state: {TRANSITIONS.get(current, {}).get('next', 'N/A')}"
        )


def compute_severity(state: IncidentState) -> Severity:
    """
    Compute severity deterministically based on verified findings.

    Rules (algebraic, not LLM):
    - CRITICAL: ≥2 verified root causes OR blast_radius ≥ 3 services
    - HIGH: 1 verified root cause AND blast_radius ≥ 1
    - MEDIUM: hypotheses exist but none fully verified
    - LOW: no hypotheses generated (likely informational)
    """
    num_verified = len(state.verified_root_causes)
    blast_radius = len(state.world_model.blast_radius) if state.world_model else 0

    if num_verified >= 2 or blast_radius >= 3:
        return Severity.CRITICAL
    elif num_verified >= 1 and blast_radius >= 1:
        return Severity.HIGH
    elif len(state.hypotheses) > 0:
        return Severity.MEDIUM
    else:
        return Severity.LOW
