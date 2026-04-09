import pytest

from app.agents.state import (
    IncidentState,
    IncidentStatus,
    RiskHypothesis,
    Severity,
    WorldModelProjection,
)
from app.symbolic.fsm import can_transition, compute_severity, force_transition, try_transition
from app.symbolic.span_matcher import exact_match_span, fuzzy_match_span


def test_fuzzy_match_span_exact_substring():
    match, score = fuzzy_match_span(
        "NullReferenceException",
        "A NullReferenceException occurred while processing the order.",
    )

    assert match is True
    assert score == 1.0


def test_fuzzy_match_span_case_insensitive_match():
    match, score = fuzzy_match_span(
        "NullReferenceException",
        "a nullreferenceexception occurred while processing the order.",
    )

    assert match is True
    assert score == 0.95


def test_exact_match_span_rejects_missing_or_empty_values():
    assert exact_match_span("", "anything") == (False, 0.0)
    assert exact_match_span("NullReferenceException", "Everything works perfectly") == (
        False,
        0.0,
    )


def test_can_transition_received_requires_report():
    ready = IncidentState(raw_report="Checkout fails with HTTP 500")
    blocked = IncidentState(raw_report="")

    assert can_transition(ready) is True
    assert can_transition(blocked) is False


def test_try_transition_advances_when_conditions_are_met():
    state = IncidentState(raw_report="Checkout fails with HTTP 500")

    updated = try_transition(state)

    assert updated.status == IncidentStatus.TRIAGING


def test_try_transition_keeps_state_when_conditions_are_missing():
    state = IncidentState(
        status=IncidentStatus.TRIAGING,
        triage_summary="",
        final_severity=Severity.UNKNOWN,
    )

    updated = try_transition(state)

    assert updated.status == IncidentStatus.TRIAGING


def test_force_transition_validates_next_state():
    state = IncidentState(status=IncidentStatus.TEAM_NOTIFIED)

    updated = force_transition(state, IncidentStatus.RESOLVED)

    assert updated.status == IncidentStatus.RESOLVED

    with pytest.raises(ValueError):
        force_transition(updated, IncidentStatus.TRIAGED)


@pytest.mark.parametrize(
        ("verified_root_causes", "blast_radius", "hypotheses", "expected"),
        [
            (["a", "b"], ["WebApp"], [], Severity.CRITICAL),
            (["a"], ["WebApp"], [], Severity.HIGH),
            (
                [],
                [],
                [
                    RiskHypothesis(
                        hypothesis_id="h1",
                        description="Missing null guard",
                        suspected_file="OrdersController.cs",
                        exact_span="throw new NullReferenceException();",
                    )
                ],
                Severity.MEDIUM,
            ),
            ([], [], [], Severity.LOW),
        ],
    )
def test_compute_severity_rules(
    verified_root_causes,
    blast_radius,
    hypotheses,
    expected,
):
    state = IncidentState(
        hypotheses=hypotheses,
        verified_root_causes=verified_root_causes,
    )
    state.world_model = WorldModelProjection(
        affected_service="Ordering.API",
        blast_radius=blast_radius,
        estimated_severity=Severity.UNKNOWN,
    )

    assert compute_severity(state) == expected
