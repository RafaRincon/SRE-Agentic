from app.ledger.audit import record_entry, record_state_transition, record_verdict


def test_record_entry_matches_ledger_contract(monkeypatch):
    stored = []

    monkeypatch.setattr(
        "app.ledger.audit.db_provider.append_ledger_entry",
        lambda entry: stored.append(entry) or entry,
    )

    result = record_entry(
        incident_id="inc-1",
        event_type="TEST_EVENT",
        data={"foo": "bar"},
        node_name="unit-test",
    )

    assert result["incident_id"] == "inc-1"
    assert result["event_type"] == "TEST_EVENT"
    assert result["node_name"] == "unit-test"
    assert result["data"] == {"foo": "bar"}
    assert "id" in result
    assert "timestamp" in result
    assert stored


def test_record_entry_copies_payload_before_persisting(monkeypatch):
    persisted = []
    payload = {"nested": {"status": "VERIFIED"}}

    monkeypatch.setattr(
        "app.ledger.audit.db_provider.append_ledger_entry",
        lambda entry: persisted.append(entry) or entry,
    )

    result = record_entry(
        incident_id="inc-1",
        event_type="SPAN_VERDICT",
        data=payload,
    )
    payload["nested"]["status"] = "MUTATED"

    assert result["data"]["nested"]["status"] == "VERIFIED"
    assert persisted[0]["data"]["nested"]["status"] == "VERIFIED"


def test_state_transition_and_verdict_use_expected_event_types(monkeypatch):
    monkeypatch.setattr(
        "app.ledger.audit.db_provider.append_ledger_entry",
        lambda entry: entry,
    )

    transition = record_state_transition(
        incident_id="inc-1",
        from_state="TRIAGING",
        to_state="TRIAGED",
        node_name="consolidator",
    )
    verdict = record_verdict(
        incident_id="inc-1",
        verdict={"hypothesis_id": "h1", "verdict": "VERIFIED"},
    )

    assert transition["event_type"] == "STATE_TRANSITION"
    assert transition["data"]["from_state"] == "TRIAGING"
    assert transition["data"]["to_state"] == "TRIAGED"
    assert verdict["event_type"] == "SPAN_VERDICT"
    assert verdict["data"]["verdict"] == "VERIFIED"
