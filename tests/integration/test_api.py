import sys
import types

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.agents.state import IncidentStatus


class FakeGraph:
    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error
        self.calls = []

    async def ainvoke(self, state, config=None):
        self.calls.append((state, config))
        if self.error:
            raise self.error
        return self.result

    def get_state(self, config=None):
        return None


def test_health_check():
    graph = FakeGraph()

    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "sre-agent"
    assert "timestamp" in payload


def test_root_serves_frontend():
    with TestClient(main_module.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "SRE Agent" in response.text


def test_incident_submission_success(monkeypatch):
    graph = FakeGraph(
        result={
            "status": IncidentStatus.TRIAGED.value,
            "triage_summary": "Null guard missing in OrdersController.",
            "final_severity": "HIGH",
            "verified_root_causes": ["OrdersController.cs: Missing null guard"],
            "suggested_runbooks": [
                {
                    "runbook_id": "RB-42",
                    "title": "Checkout 500 Runbook",
                    "escalation_path": "Order Team -> Platform",
                    "estimated_resolution_time": "15m",
                }
            ],
            "ticket": {
                "ticket_id": "SRE-123456",
                "ticket_url": "https://jira.example.com/browse/SRE-123456",
                "assigned_team": "Order Team",
            },
            "errors": [],
        }
    )
    persisted = []

    monkeypatch.setattr(main_module, "get_graph", lambda: graph)
    monkeypatch.setattr(
        main_module.db_provider,
        "upsert_incident",
        lambda incident: persisted.append(incident) or incident,
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/incident",
            data={
                "report": "HTTP 500 on checkout after deployment",
                "reporter_name": "Hector",
                "reporter_email": "hector@example.com",
            },
            files={"image": ("error.png", b"fake-image", "image/png")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["incident_id"]
    assert payload["status"] == IncidentStatus.TRIAGED.value
    assert payload["ticket_id"] == "SRE-123456"
    assert payload["suggested_runbooks"][0]["runbook_id"] == "RB-42"

    submitted_state, submitted_config = graph.calls[0]
    assert submitted_state["has_image"] is True
    assert submitted_state["image_mime_type"] == "image/png"
    assert submitted_state["image_data_b64"]
    assert submitted_state["raw_report"].startswith("Reporter: Hector (hector@example.com)")
    assert submitted_config["configurable"]["thread_id"] == payload["incident_id"]

    assert "image_data_b64" not in persisted[0]
    assert persisted[0]["incident_id"] == payload["incident_id"]


def test_incident_submission_returns_500_when_pipeline_fails(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "get_graph",
        lambda: FakeGraph(error=RuntimeError("pipeline offline")),
    )

    with TestClient(main_module.app) as client:
        response = client.post("/incident", data={"report": "HTTP 500 on checkout"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Triage pipeline failed: pipeline offline"


def test_get_incident_success(monkeypatch):
    monkeypatch.setattr(
        main_module.db_provider,
        "get_incident",
        lambda incident_id: {"id": incident_id, "status": IncidentStatus.TRIAGED.value},
    )

    with TestClient(main_module.app) as client:
        response = client.get("/incident/inc-1")

    assert response.status_code == 200
    assert response.json()["id"] == "inc-1"


def test_get_incident_not_found(monkeypatch):
    monkeypatch.setattr(main_module.db_provider, "get_incident", lambda incident_id: None)

    with TestClient(main_module.app) as client:
        response = client.get("/incident/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Incident not found"


@pytest.mark.asyncio
async def test_maybe_await_handles_sync_and_async_values():
    async def async_value():
        return "async"

    assert await main_module._maybe_await(async_value()) == "async"
    assert await main_module._maybe_await("sync") == "sync"


def test_resolve_incident_success(monkeypatch):
    stored = []
    
    async def fake_index_resolved_incident(incident, resolution_notes):
        return {
            "chunks_indexed": 2,
            "source_incident": incident["incident_id"],
            "resolution_notes": resolution_notes,
        }

    monkeypatch.setattr(
        main_module.db_provider,
        "get_incident",
        lambda incident_id: {
            "incident_id": incident_id,
            "status": IncidentStatus.TEAM_NOTIFIED.value,
            "notifications": {},
        },
    )
    monkeypatch.setattr(
        main_module.db_provider,
        "upsert_incident",
        lambda incident: stored.append(incident) or incident,
    )
    monkeypatch.setitem(
        sys.modules,
        "app.indexer.knowledge_indexer",
        types.SimpleNamespace(index_resolved_incident=fake_index_resolved_incident),
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/incident/inc-1/resolve",
            data={"resolution_notes": "Added null guard"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == IncidentStatus.REPORTER_NOTIFIED.value
    assert payload["knowledge_indexed"] is True
    assert stored[0]["resolution_notes"] == "Added null guard"
    assert stored[0]["notifications"]["reporter_notified"] is True


def test_resolve_incident_rejects_wrong_state(monkeypatch):
    monkeypatch.setattr(
        main_module.db_provider,
        "get_incident",
        lambda incident_id: {"incident_id": incident_id, "status": IncidentStatus.TRIAGING.value},
    )

    with TestClient(main_module.app) as client:
        response = client.post("/incident/inc-1/resolve")

    assert response.status_code == 400
    assert "Cannot resolve incident" in response.json()["detail"]


def test_list_incidents(monkeypatch):
    monkeypatch.setattr(
        main_module.db_provider,
        "list_incidents",
        lambda: [{"incident_id": "inc-1"}, {"incident_id": "inc-2"}],
    )

    with TestClient(main_module.app) as client:
        response = client.get("/incidents")

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_index_endpoints(monkeypatch):
    async def fake_index_repo(force=False):
        return {"indexed": True, "force": force}

    monkeypatch.setitem(
        sys.modules,
        "app.indexer.repo_indexer",
        types.SimpleNamespace(index_repo=fake_index_repo),
    )
    monkeypatch.setattr(main_module.db_provider, "count_chunks", lambda: 7)

    with TestClient(main_module.app) as client:
        trigger = client.post("/index?force=true")
        status = client.get("/index/status")

    assert trigger.status_code == 200
    assert trigger.json()["force"] is True
    assert status.json() == {"status": "indexed", "chunk_count": 7}


def test_index_endpoint_accepts_sync_indexer(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "app.indexer.repo_indexer",
        types.SimpleNamespace(index_repo=lambda force=False: {"indexed": True, "force": force}),
    )

    with TestClient(main_module.app) as client:
        trigger = client.post("/index")

    assert trigger.status_code == 200
    assert trigger.json()["indexed"] is True


def test_ledger_and_knowledge_endpoints(monkeypatch):
    async def fake_seed_historical_incidents():
        return {"seeded": 5}

    monkeypatch.setattr(
        main_module.db_provider,
        "get_ledger_entries",
        lambda incident_id: [{"event_type": "STATE_TRANSITION"}],
    )
    monkeypatch.setattr(main_module.db_provider, "count_chunks", lambda: 3)
    monkeypatch.setattr(main_module.db_provider, "count_knowledge_chunks", lambda: 4)
    monkeypatch.setitem(
        sys.modules,
        "app.indexer.seed_incidents",
        types.SimpleNamespace(seed_historical_incidents=fake_seed_historical_incidents),
    )

    with TestClient(main_module.app) as client:
        ledger = client.get("/incident/inc-1/ledger")
        knowledge = client.get("/knowledge/status")
        seed = client.post("/knowledge/seed")

    assert ledger.status_code == 200
    assert ledger.json()["entries"] == [{"event_type": "STATE_TRANSITION"}]
    assert knowledge.json()["flywheel_active"] is True
    assert knowledge.json()["total_indexed"] == 7
    assert seed.json() == {"seeded": 5}


def test_seed_endpoint_accepts_sync_seeder(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "app.indexer.seed_incidents",
        types.SimpleNamespace(seed_historical_incidents=lambda: {"seeded": 2}),
    )

    with TestClient(main_module.app) as client:
        response = client.post("/knowledge/seed")

    assert response.status_code == 200
    assert response.json() == {"seeded": 2}
