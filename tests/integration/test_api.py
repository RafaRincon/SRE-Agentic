import importlib
import sys
import types
from types import SimpleNamespace

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
        self.calls.append({"state": state, "config": config})
        if self.error:
            raise self.error
        return self.result


def build_settings(**overrides):
    data = {
        "app_disable_admin_endpoints": False,
        "app_admin_api_key": "test-admin-key",
        "app_max_upload_bytes": 5 * 1024 * 1024,
        "app_require_index_ready": False,
        "log_level": "INFO",
        "app_env": "development",
        "cosmos_endpoint": "https://cosmos.test",
        "cosmos_database": "sre",
        "gemini_model": "gemini-test",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


@pytest.fixture(autouse=True)
def stub_runtime(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: build_settings(),
    )
    monkeypatch.setattr(
        main_module,
        "_startup_or_raise",
        lambda: {"ready": True, "components": {}},
    )


@pytest.fixture
def admin_headers():
    return {"X-Admin-Api-Key": "test-admin-key"}


def test_liveness_aliases():
    with TestClient(main_module.app) as client:
        live = client.get("/livez")
        health = client.get("/health")

    assert live.status_code == 200
    assert health.status_code == 200
    assert live.json()["status"] == "healthy"
    assert health.json()["service"] == "sre-agent"


def test_readyz_ready(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "_collect_readiness_status",
        lambda require_index_ready=None: {
            "service": "sre-agent",
            "status": "ready",
            "ready": True,
            "components": {"database": {"ready": True}},
        },
    )

    with TestClient(main_module.app) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_readyz_degraded(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "_collect_readiness_status",
        lambda require_index_ready=None: {
            "service": "sre-agent",
            "status": "degraded",
            "ready": False,
            "components": {"code_index": {"ready": False, "chunk_count": 0}},
        },
    )

    with TestClient(main_module.app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["components"]["code_index"]["ready"] is False


def test_incident_submission_success(monkeypatch):
    graph = FakeGraph(
        result={
            "status": IncidentStatus.TRIAGED.value,
            "triage_summary": "Null guard missing in OrdersController.",
            "final_severity": "HIGH",
            "verified_root_causes": ["OrdersController.cs: Missing null guard"],
            "epistemic_context": {
                "observed": [{"label": "error_code=500", "evidence": "500"}],
                "inferred": [{"label": "Missing null guard", "evidence": "hypothesis"}],
                "unknown": [{"label": "upstream_validation_or_wiring", "evidence": "not indexed"}],
            },
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
    monkeypatch.setattr(main_module, "get_graph", lambda: graph)
    async def fake_generate_embedding(*args, **kwargs):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(main_module.llm_provider, "generate_embedding", fake_generate_embedding)
    monkeypatch.setattr(
        main_module.db_provider,
        "find_duplicate_incident",
        lambda **kwargs: None,
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
    assert "epistemic_context" not in payload
    assert payload["suggested_runbooks"][0]["runbook_id"] == "RB-42"

    submitted_state = graph.calls[0]["state"]
    assert submitted_state["has_image"] is True
    assert submitted_state["image_mime_type"] == "image/png"
    assert submitted_state["image_data_b64"]
    assert submitted_state["raw_report"].startswith("Reporter: Hector (hector@example.com)")
    assert graph.calls[0]["config"]["configurable"]["thread_id"] == payload["incident_id"]


def test_incident_submission_rejects_unsupported_image():
    with TestClient(main_module.app) as client:
        response = client.post(
            "/incident",
            data={"report": "HTTP 500 on checkout"},
            files={"image": ("error.txt", b"oops", "text/plain")},
        )

    assert response.status_code == 415
    assert "Unsupported image type" in response.json()["detail"]


def test_incident_submission_rejects_oversized_image(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: build_settings(app_max_upload_bytes=4),
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/incident",
            data={"report": "HTTP 500 on checkout"},
            files={"image": ("error.png", b"12345", "image/png")},
        )

    assert response.status_code == 413
    assert "Image exceeds" in response.json()["detail"]


def test_incident_submission_returns_500_when_pipeline_fails(monkeypatch):
    async def fake_generate_embedding(*args, **kwargs):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        main_module,
        "get_graph",
        lambda: FakeGraph(error=RuntimeError("pipeline offline")),
    )
    monkeypatch.setattr(main_module.llm_provider, "generate_embedding", fake_generate_embedding)
    monkeypatch.setattr(
        main_module.db_provider,
        "find_duplicate_incident",
        lambda **kwargs: None,
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


@pytest.mark.parametrize(
    "incident",
    [
        {
            "id": "inc-legacy",
            "status": IncidentStatus.TRIAGED.value,
            "world_model": {
                "affected_service": "Ordering.API",
                "incident_category": "RuntimeException",
                "blast_radius": ["WebApp"],
                "estimated_severity": "HIGH",
                "severity_rationale": "legacy-only",
            },
            "entities": {
                "error_code": "500",
                "thinking_process": "legacy-only",
            },
        },
        {
            "id": "inc-compact",
            "status": IncidentStatus.TRIAGED.value,
            "world_model": {
                "affected_service": "Ordering.API",
                "incident_category": "RuntimeException",
                "blast_radius": ["WebApp"],
            },
            "entities": {
                "error_code": "500",
            },
        },
    ],
)
def test_get_incident_supports_legacy_and_compact_shapes(monkeypatch, incident):
    monkeypatch.setattr(
        main_module.db_provider,
        "get_incident",
        lambda incident_id: incident,
    )
    monkeypatch.setattr(
        main_module,
        "get_graph",
        lambda: types.SimpleNamespace(get_state=lambda config: None),
    )

    with TestClient(main_module.app) as client:
        response = client.get(f"/incident/{incident['id']}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["world_model"]["affected_service"] == "Ordering.API"
    assert payload["entities"]["error_code"] == "500"


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


def test_resolve_incident_success(monkeypatch, admin_headers):
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
            headers=admin_headers,
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == IncidentStatus.REPORTER_NOTIFIED.value
    assert payload["knowledge_indexed"] is True
    assert stored[0]["resolution_notes"] == "Added null guard"
    assert stored[0]["notifications"]["reporter_notified"] is True


def test_resolve_incident_requires_admin_key():
    with TestClient(main_module.app) as client:
        response = client.post("/incident/inc-1/resolve")

    assert response.status_code == 401


def test_resolve_incident_rejects_wrong_state(monkeypatch, admin_headers):
    monkeypatch.setattr(
        main_module.db_provider,
        "get_incident",
        lambda incident_id: {"incident_id": incident_id, "status": IncidentStatus.TRIAGING.value},
    )

    with TestClient(main_module.app) as client:
        response = client.post("/incident/inc-1/resolve", headers=admin_headers)

    assert response.status_code == 400
    assert "Cannot resolve incident" in response.json()["detail"]


def test_list_incidents(monkeypatch, admin_headers):
    monkeypatch.setattr(
        main_module.db_provider,
        "list_incidents",
        lambda: [{"incident_id": "inc-1"}, {"incident_id": "inc-2"}],
    )

    with TestClient(main_module.app) as client:
        response = client.get("/incidents", headers=admin_headers)

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_index_endpoints(monkeypatch, admin_headers):
    async def fake_index_repo(force=False):
        return {"indexed": True, "force": force}

    monkeypatch.setitem(
        sys.modules,
        "app.indexer.repo_indexer",
        types.SimpleNamespace(index_repo=fake_index_repo),
    )
    monkeypatch.setattr(main_module.db_provider, "count_chunks", lambda: 7)

    with TestClient(main_module.app) as client:
        trigger = client.post("/index?force=true", headers=admin_headers)
        status = client.get("/index/status")

    assert trigger.status_code == 200
    assert trigger.json()["force"] is True
    assert status.json() == {"status": "indexed", "chunk_count": 7}


def test_index_endpoint_accepts_sync_indexer(monkeypatch, admin_headers):
    monkeypatch.setitem(
        sys.modules,
        "app.indexer.repo_indexer",
        types.SimpleNamespace(index_repo=lambda force=False: {"indexed": True, "force": force}),
    )

    with TestClient(main_module.app) as client:
        trigger = client.post("/index", headers=admin_headers)

    assert trigger.status_code == 200
    assert trigger.json()["indexed"] is True


def test_ledger_and_knowledge_endpoints(monkeypatch, admin_headers):
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
        ledger = client.get("/incident/inc-1/ledger", headers=admin_headers)
        knowledge = client.get("/knowledge/status")
        seed = client.post("/knowledge/seed", headers=admin_headers)

    assert ledger.status_code == 200
    assert ledger.json()["entries"] == [{"event_type": "STATE_TRANSITION"}]
    assert knowledge.json()["flywheel_active"] is True
    assert knowledge.json()["total_indexed"] == 7
    assert seed.json() == {"seeded": 5}


def test_seed_endpoint_accepts_sync_seeder(monkeypatch, admin_headers):
    monkeypatch.setitem(
        sys.modules,
        "app.indexer.seed_incidents",
        types.SimpleNamespace(seed_historical_incidents=lambda: {"seeded": 2}),
    )

    with TestClient(main_module.app) as client:
        response = client.post("/knowledge/seed", headers=admin_headers)

    assert response.status_code == 200
    assert response.json() == {"seeded": 2}


def test_startup_fails_fast_on_missing_settings(monkeypatch):
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("missing settings")),
    )

    with pytest.raises(RuntimeError, match="missing settings"):
        with TestClient(main_module.app):
            pass


def test_admin_endpoints_can_be_disabled(monkeypatch, admin_headers):
    monkeypatch.setattr(
        main_module,
        "get_settings",
        lambda: build_settings(app_disable_admin_endpoints=True),
    )

    with TestClient(main_module.app) as client:
        response = client.get("/incidents", headers=admin_headers)

    assert response.status_code == 404


def test_cors_restricts_to_configured_origin(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("APP_CORS_ORIGINS", "https://allowed.example.com")
    monkeypatch.setenv("APP_ENABLE_DOCS", "false")

    reloaded = importlib.reload(main_module)
    monkeypatch.setattr(
        reloaded,
        "_startup_or_raise",
        lambda: {"ready": True, "components": {}},
    )
    monkeypatch.setattr(
        reloaded,
        "get_settings",
        lambda: build_settings(app_env="production", app_cors_origins=["https://allowed.example.com"]),
    )

    with TestClient(reloaded.app) as client:
        allowed = client.options(
            "/health",
            headers={
                "Origin": "https://allowed.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        blocked = client.options(
            "/health",
            headers={
                "Origin": "https://blocked.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )

    assert allowed.headers["access-control-allow-origin"] == "https://allowed.example.com"
    assert "access-control-allow-origin" not in blocked.headers

    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("APP_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("APP_ENABLE_DOCS", raising=False)
    importlib.reload(main_module)
