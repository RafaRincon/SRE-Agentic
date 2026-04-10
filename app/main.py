from __future__ import annotations

"""
SRE Agent — FastAPI Application

Main entry point for the SRE Incident Intake & Triage Agent.
Provides REST endpoints for:
- POST /incident  — Submit a new incident report
- GET  /incident/{id} — Get incident status
- POST /incident/{id}/resolve — Mark incident as resolved
- GET  /livez, /readyz, /health — Health and readiness checks
"""

import base64
import inspect
import logging
import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.agents.graph import get_graph, get_graph_status
from app.agents.state import IncidentStatus
from app.config import get_settings
from app.providers import db_provider, llm_provider

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_cors_origins() -> list[str]:
    raw = os.getenv("APP_CORS_ORIGINS", "").strip()
    if not raw:
        return ["*"] if os.getenv("APP_ENV", "development").strip().lower() != "production" else []
    if raw.startswith("[") and raw.endswith("]"):
        import json

        return [str(item).strip() for item in json.loads(raw) if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _docs_enabled_from_env() -> bool:
    if "APP_ENABLE_DOCS" in os.environ:
        return _env_flag("APP_ENABLE_DOCS")
    return os.getenv("APP_ENV", "development").strip().lower() != "production"


def _mask_endpoint(endpoint: str) -> str:
    if "://" not in endpoint:
        return endpoint
    scheme, remainder = endpoint.split("://", 1)
    host = remainder.split("/", 1)[0]
    return f"{scheme}://{host}"


def _configure_runtime_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(level)
    for noisy_logger in ("azure.cosmos", "azure.core.pipeline", "httpx", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def _collect_readiness_status(*, require_index_ready: bool | None = None) -> dict:
    settings = get_settings()
    check_index = settings.app_require_index_ready if require_index_ready is None else require_index_ready

    status_payload = {
        "service": "sre-agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ready": False,
        "status": "degraded",
        "components": {},
    }

    db_status = db_provider.get_runtime_health(require_index_ready=check_index)
    status_payload["components"].update(db_status["components"])
    ready = db_status["ready"]

    try:
        graph_status = get_graph_status()
        checkpointer_ready = bool(graph_status.get("checkpointer_enabled"))
        status_payload["components"]["checkpointer"] = {
            "ready": checkpointer_ready,
            "compiled": bool(graph_status.get("compiled")),
            "error": graph_status.get("checkpointer_error", ""),
        }
        ready = ready and checkpointer_ready
    except Exception as exc:
        ready = False
        status_payload["components"]["checkpointer"] = {
            "ready": False,
            "error": str(exc),
        }

    status_payload["ready"] = ready
    status_payload["status"] = "ready" if ready else "degraded"
    return status_payload


def _startup_or_raise() -> dict:
    readiness = _collect_readiness_status()
    if not readiness["ready"]:
        raise RuntimeError(
            "Startup readiness check failed: "
            f"{readiness['components']}"
        )
    return readiness


def require_admin_access(
    x_admin_api_key: Annotated[str | None, Header(alias="X-Admin-Api-Key")] = None,
) -> None:
    settings = get_settings()
    if settings.app_disable_admin_endpoints:
        raise HTTPException(status_code=404, detail="Endpoint disabled")

    expected_key = settings.app_admin_api_key
    if not expected_key:
        raise HTTPException(status_code=503, detail="Admin endpoints are not configured")

    if x_admin_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid admin API key")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    settings = get_settings()
    _configure_runtime_logging(settings.log_level)

    logger.info("=" * 60)
    logger.info("SRE Agent starting up")
    logger.info("   Cosmos DB: %s", _mask_endpoint(settings.cosmos_endpoint))
    logger.info("   Database:  %s", settings.cosmos_database)
    logger.info("   LLM:       %s", settings.gemini_model)
    logger.info("   Env:       %s", settings.app_env)
    logger.info("=" * 60)

    readiness = _startup_or_raise()
    app.state.startup_readiness = readiness
    logger.info("Runtime dependencies ready")

    yield

    logger.info("SRE Agent shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


app = FastAPI(
    title="SRE Incident Intake & Triage Agent",
    description=(
        "Neuro-Symbolic SRE Agent that ingests incident reports, "
        "performs dual-track triage (neuronal + adversarial), "
        "and routes issues with verified evidence."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled_from_env() else None,
    redoc_url="/redoc" if _docs_enabled_from_env() else None,
    openapi_url="/openapi.json" if _docs_enabled_from_env() else None,
)

_cors_origins = _env_cors_origins()
STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Serve the single-page frontend without changing backend API routes."""
    return FileResponse(INDEX_FILE)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IncidentSubmission(BaseModel):
    """Request body for submitting an incident (JSON mode)."""

    report: str
    reporter_name: str = ""
    reporter_email: str = ""


class IncidentResponse(BaseModel):
    """Response after processing an incident."""

    incident_id: str
    status: str
    triage_summary: str = ""
    final_severity: str = ""
    verified_root_causes: list[str] = Field(default_factory=list)
    suggested_runbooks: list[dict] = Field(default_factory=list)
    ticket_id: str = ""
    ticket_url: str = ""
    assigned_team: str = ""
    duplicate_of: str = ""
    errors: list[str] = Field(default_factory=list)


async def _maybe_await(value):
    """Await values only when the callee returned an awaitable."""
    if inspect.isawaitable(value):
        return await value
    return value


def _validate_image_upload(image: UploadFile, size_bytes: int) -> None:
    settings = get_settings()
    mime_type = image.content_type or "application/octet-stream"
    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image type '{mime_type}'. Allowed types: {sorted(ALLOWED_IMAGE_MIME_TYPES)}",
        )
    if size_bytes > settings.app_max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {settings.app_max_upload_bytes} bytes",
        )
def _persistable_state(state: dict) -> dict:
    return {
        "id": state["incident_id"],
        "incident_id": state["incident_id"],
        **{key: value for key, value in state.items() if key != "image_data_b64"},
    }


# ---------------------------------------------------------------------------
# Health / Readiness
# ---------------------------------------------------------------------------


@app.get("/livez")
async def livez():
    """Liveness probe."""
    return {
        "status": "healthy",
        "service": "sre-agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
async def health():
    """Backward-compatible alias for liveness."""
    return await livez()


@app.get("/readyz")
async def readyz():
    """Readiness probe for dependencies and index availability."""
    readiness = _collect_readiness_status()
    if readiness["ready"]:
        return readiness
    return JSONResponse(status_code=503, content=readiness)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/incident", response_model=IncidentResponse)
async def submit_incident(
    incident_id: str = Form(""),
    report: str = Form(...),
    reporter_name: str = Form(""),
    reporter_email: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    """
    Submit an incident report for triage.

    Dedup and persistence are handled inside the graph (n_dedup, n_notify_team)
    so the logic is consistent regardless of invocation method.
    """
    incident_id = (incident_id or "").strip() or uuid.uuid4().hex
    logger.info(f"📥 New incident submitted: {incident_id}")

    initial_state = {
        "incident_id": incident_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_report": report,
        "has_image": False,
        "image_mime_type": "",
        "image_data_b64": "",
        "status": IncidentStatus.RECEIVED.value,
        "hypotheses": [],
        "span_verdicts": [],
        "falsifier_verdicts": [],
        "verified_root_causes": [],
        "epistemic_context": {},
        "suggested_runbooks": [],
        "errors": [],
    }

    if image:
        image_bytes = await image.read()
        _validate_image_upload(image, len(image_bytes))
        initial_state["has_image"] = True
        initial_state["image_mime_type"] = image.content_type or "image/png"
        initial_state["image_data_b64"] = base64.b64encode(image_bytes).decode()
        logger.info("Image attached: %s (%s)", image.filename, image.content_type)

    if reporter_name or reporter_email:
        initial_state["raw_report"] = (
            f"Reporter: {reporter_name} ({reporter_email})\n\n"
            + initial_state["raw_report"]
        )

    graph = get_graph()
    initial_state["ticket"] = {}
    initial_state["notifications"] = {}
    initial_state["historical_context"] = {}

    try:
        db_provider.upsert_incident(_persistable_state(initial_state))
    except Exception as e:
        logger.error(f"❌ Failed to create initial incident document for {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register incident: {str(e)}",
        )

    # --- DEDUPLICATION CHECK ---
    # Before running the expensive pipeline, check if a similar open incident
    # already exists. Uses cosine similarity in-memory (viable for <100 open incidents).
    report_embedding = None
    try:
        report_embedding = await llm_provider.generate_embedding(
            report[:500], task_type="RETRIEVAL_QUERY"
        )
        duplicate_result = db_provider.find_duplicate_incident(
            query_vector=report_embedding,
            similarity_threshold=0.80,
            exclude_incident_id=incident_id,
        )
        if duplicate_result:
            parent_id = duplicate_result["incident_id"]
            similarity = duplicate_result["similarity"]
            logger.info(
                f"🔁 Duplicate detected: {incident_id} → {parent_id} "
                f"(similarity={similarity:.2f})"
            )

            # Link this report to the parent incident
            db_provider.upsert_incident({
                "id": incident_id,
                "incident_id": incident_id,
                "status": "DUPLICATE",
                "duplicate_of": parent_id,
                "raw_report": report,
                "created_at": initial_state["created_at"],
                "similarity_score": similarity,
            })

            # Increment occurrence count on parent
            parent = db_provider.get_incident(parent_id)
            if parent:
                occurrence_count = parent.get("occurrence_count", 1) + 1
                parent["occurrence_count"] = occurrence_count
                db_provider.upsert_incident(parent)
                logger.info(
                    f"📈 Parent incident {parent_id} now has "
                    f"{occurrence_count} occurrences"
                )

            return IncidentResponse(
                incident_id=incident_id,
                status="DUPLICATE",
                duplicate_of=parent_id,
                triage_summary=(
                    f"This incident is a duplicate of {parent_id} "
                    f"(similarity: {similarity:.0%}). "
                    f"See existing ticket for triage details."
                ),
                final_severity=duplicate_result.get("severity", ""),
                ticket_id=duplicate_result.get("ticket_id", ""),
                ticket_url=duplicate_result.get("ticket_url", ""),
                assigned_team=duplicate_result.get("assigned_team", ""),
            )

    except Exception as e:
        logger.warning(f"⚠️ Dedup check failed (proceeding with full triage): {e}")

    # Run the LangGraph pipeline
    graph = get_graph()

    # Inject the report embedding (already generated for dedup) so the consolidator
    # can reuse it for runbook search with zero extra API calls
    if report_embedding:
        initial_state["report_embedding"] = report_embedding

    initial_state["status"] = IncidentStatus.TRIAGING.value
    try:
        db_provider.upsert_incident(_persistable_state(initial_state))
    except Exception as e:
        logger.warning(f"⚠️ Failed to persist triaging state for {incident_id}: {e}")

    # Thread config: each incident_id is its own checkpointer thread.
    # This means the full node-by-node state is persisted in Cosmos
    # agent_checkpoints and can be resumed or inspected at any time.
    thread_config = {"configurable": {"thread_id": incident_id}}

    try:
        result = await graph.ainvoke(initial_state, config=thread_config)
    except Exception as exc:
        logger.error("Pipeline failed for %s: %s", incident_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Triage pipeline failed: {str(exc)}",
        ) from exc

    ticket = result.get("ticket", {})
    response = IncidentResponse(
        incident_id=incident_id,
        status=result.get("status", "UNKNOWN"),
        triage_summary=result.get("triage_summary", ""),
        final_severity=result.get("final_severity", ""),
        verified_root_causes=result.get("verified_root_causes", []),
        suggested_runbooks=result.get("suggested_runbooks", ticket.get("suggested_runbooks", [])),
        ticket_id=ticket.get("ticket_id", ""),
        ticket_url=ticket.get("ticket_url", ""),
        assigned_team=ticket.get("assigned_team", ""),
        duplicate_of=result.get("duplicate_of", ""),
        errors=result.get("errors", []),
    )

    logger.info(
        "Incident %s triaged: severity=%s, ticket=%s",
        incident_id,
        response.final_severity,
        response.ticket_id,
    )
    return response


@app.get("/incident/{incident_id}")
async def get_incident(incident_id: str):
    """
    Get the current state of an incident.

    Returns the Cosmos DB incident document enriched with the last LangGraph
    checkpoint state (if the graph ran with the CosmosDB checkpointer).
    """
    incident = db_provider.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    incident = {
        key: value
        for key, value in incident.items()
        if key != "epistemic_context"
    }
    if isinstance(incident.get("ticket"), dict):
        incident["ticket"] = {
            key: value
            for key, value in incident["ticket"].items()
            if key != "epistemic_context"
        }

    try:
        graph = get_graph()
        thread_config = {"configurable": {"thread_id": incident_id}}
        snapshot = graph.get_state(thread_config)
        if snapshot and snapshot.values:
            snapshot_values = snapshot.values
            # Attach lightweight checkpoint metadata (exclude large fields)
            incident["_checkpoint"] = {
                "next_nodes": list(snapshot.next) if snapshot.next else [],
                "last_node": (
                    list(snapshot.metadata.get("writes", {}).keys())[-1]
                    if snapshot.metadata and snapshot.metadata.get("writes")
                    else None
                ),
                "status": snapshot_values.get("status", ""),
                "final_severity": snapshot_values.get("final_severity", ""),
            }
    except Exception:
        pass

    return incident


@app.post("/incident/{incident_id}/resolve", dependencies=[Depends(require_admin_access)])
async def resolve_incident(
    incident_id: str,
    resolution_notes: str = Form(""),
):
    """
    Mark an incident as resolved and record the reporter follow-up.
    """
    incident = db_provider.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    current_status = incident.get("status", "")
    if current_status != IncidentStatus.TEAM_NOTIFIED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resolve incident in state '{current_status}'. Expected 'TEAM_NOTIFIED'.",
        )

    incident["status"] = IncidentStatus.RESOLVED.value
    incident["resolution_notes"] = resolution_notes
    logger.info("Incident %s marked as RESOLVED", incident_id)

    notifications = incident.get("notifications", {})
    notifications["reporter_notified"] = True
    reporter_email = (
        incident.get("entities", {}).get("reporter_email")
        or incident.get("reporter_email")
        or ""
    )
    notifications["reporter_notification_channel"] = (
        f"reporter:{reporter_email}" if reporter_email else "reporter:unavailable"
    )
    incident["notifications"] = notifications
    incident["status"] = IncidentStatus.REPORTER_NOTIFIED.value

    logger.info("Reporter follow-up recorded for incident %s", incident_id)
    db_provider.upsert_incident(incident)

    knowledge_stats = None
    try:
        from app.indexer.knowledge_indexer import index_resolved_incident

        knowledge_stats = await _maybe_await(
            index_resolved_incident(incident, resolution_notes=resolution_notes)
        )
        logger.info(
            "Flywheel indexed %s knowledge chunks for incident %s",
            knowledge_stats.get("chunks_indexed", 0),
            incident_id,
        )
    except Exception as exc:
        logger.warning("Flywheel indexing failed for %s: %s", incident_id, exc)

    return {
        "incident_id": incident_id,
        "status": incident["status"],
        "message": "Incident resolved and reporter follow-up recorded.",
        "knowledge_indexed": knowledge_stats is not None,
        "knowledge_stats": knowledge_stats,
    }


@app.get("/incidents", dependencies=[Depends(require_admin_access)])
async def list_incidents():
    """List all incidents."""
    return db_provider.list_incidents()


# ---------------------------------------------------------------------------
# Indexer Endpoints
# ---------------------------------------------------------------------------


@app.post("/index", dependencies=[Depends(require_admin_access)])
async def index_eshop(force: bool = False):
    """
    Trigger indexing of the eShop repository.
    """
    from app.indexer.repo_indexer import index_repo

    logger.info("Index triggered (force=%s)", force)
    stats = await _maybe_await(index_repo(force=force))
    return stats


@app.get("/index/status")
async def index_status():
    """Check the current indexing status."""
    chunk_count = db_provider.count_chunks()
    return {
        "status": "indexed" if chunk_count > 0 else "not_indexed",
        "chunk_count": chunk_count,
    }


# ---------------------------------------------------------------------------
# Ledger / Audit Trail
# ---------------------------------------------------------------------------


@app.get("/incident/{incident_id}/ledger", dependencies=[Depends(require_admin_access)])
async def get_incident_ledger(incident_id: str):
    """Get full audit trail for an incident."""
    entries = db_provider.get_ledger_entries(incident_id)
    return {"incident_id": incident_id, "entries": entries}


# ---------------------------------------------------------------------------
# Knowledge Flywheel
# ---------------------------------------------------------------------------


@app.get("/knowledge/status")
async def knowledge_status():
    """Check the current status of the knowledge flywheel."""
    knowledge_count = db_provider.count_knowledge_chunks()
    code_count = db_provider.count_chunks()
    return {
        "code_chunks": code_count,
        "knowledge_chunks": knowledge_count,
        "flywheel_active": knowledge_count > 0,
        "total_indexed": code_count + knowledge_count,
    }


@app.post("/knowledge/seed", dependencies=[Depends(require_admin_access)])
async def seed_knowledge():
    """
    Seed the knowledge base with synthetic historical incidents.
    """
    from app.indexer.seed_incidents import seed_historical_incidents

    logger.info("Seeding knowledge base with synthetic incidents")
    stats = await _maybe_await(seed_historical_incidents())
    return stats
