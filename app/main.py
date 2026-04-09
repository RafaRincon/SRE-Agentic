from __future__ import annotations

"""
SRE Agent — FastAPI Application

Main entry point for the SRE Incident Intake & Triage Agent.
Provides REST endpoints for:
- POST /incident  — Submit a new incident report
- GET  /incident/{id} — Get incident status
- POST /incident/{id}/resolve — Mark incident as resolved
- GET  /health — Health check
"""

import base64
import inspect
import logging
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from app.config import get_settings
from app.agents.graph import get_graph
from app.agents.state import IncidentStatus
from app.providers import db_provider, llm_provider

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    settings = get_settings()
    logger.info("=" * 60)
    logger.info("🚀 SRE Agent starting up")
    logger.info(f"   Cosmos DB: {settings.cosmos_endpoint}")
    logger.info(f"   Database:  {settings.cosmos_database}")
    logger.info(f"   LLM:       {settings.gemini_model}")
    logger.info(f"   Env:       {settings.app_env}")
    logger.info("=" * 60)

    # Pre-compile the graph
    _ = get_graph()
    logger.info("✅ LangGraph pipeline ready")

    yield

    logger.info("👋 SRE Agent shutting down")


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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "sre-agent",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/incident", response_model=IncidentResponse)
async def submit_incident(
    report: str = Form(...),
    reporter_name: str = Form(""),
    reporter_email: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    """
    Submit an incident report for triage.

    Accepts:
    - report: Text description of the incident (required)
    - reporter_name: Name of the reporter (optional)
    - reporter_email: Email of the reporter (optional)
    - image: Screenshot or log image (optional, multimodal)
    """
    incident_id = uuid.uuid4().hex
    logger.info(f"📥 New incident submitted: {incident_id}")

    # Build initial state
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
        "suggested_runbooks": [],
        "errors": [],
    }

    # Handle image upload
    if image:
        image_bytes = await image.read()
        initial_state["has_image"] = True
        initial_state["image_mime_type"] = image.content_type or "image/png"
        initial_state["image_data_b64"] = base64.b64encode(image_bytes).decode()
        logger.info(f"🖼️  Image attached: {image.filename} ({image.content_type})")

    # Inject reporter info into report
    if reporter_name or reporter_email:
        initial_state["raw_report"] = (
            f"Reporter: {reporter_name} ({reporter_email})\n\n"
            + initial_state["raw_report"]
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

    # Thread config: each incident_id is its own checkpointer thread.
    # This means the full node-by-node state is persisted in Cosmos
    # agent_checkpoints and can be resumed or inspected at any time.
    thread_config = {"configurable": {"thread_id": incident_id}}

    try:
        result = await graph.ainvoke(initial_state, config=thread_config)
    except Exception as e:
        logger.error(f"❌ Pipeline failed for {incident_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Triage pipeline failed: {str(e)}",
        )

    # Persist the final state
    try:
        persist_data = {
            "id": incident_id,
            "incident_id": incident_id,
            **{k: v for k, v in result.items() if k != "image_data_b64"},
        }
        # Store the report embedding for future dedup checks
        if report_embedding:
            persist_data["report_embedding"] = report_embedding
        db_provider.upsert_incident(persist_data)
    except Exception as e:
        logger.warning(f"⚠️  Failed to persist incident {incident_id}: {e}")

    # Build response
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
        f"✅ Incident {incident_id} triaged: "
        f"severity={response.final_severity}, ticket={response.ticket_id}"
    )

    return response


@app.get("/incident/{incident_id}")
async def get_incident(incident_id: str):
    """
    Get the current state of an incident.

    Returns the Cosmos DB incident document enriched with the last LangGraph
    checkpoint state (if the graph ran with the CosmosDB checkpointer).
    This shows granular node-level progress without re-running the pipeline.
    """
    incident = db_provider.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    # Enrich with last checkpoint state (non-blocking — skips if unavailable)
    try:
        graph = get_graph()
        thread_config = {"configurable": {"thread_id": incident_id}}
        snapshot = graph.get_state(thread_config)
        if snapshot and snapshot.values:
            # Attach lightweight checkpoint metadata (exclude large fields)
            incident["_checkpoint"] = {
                "next_nodes": list(snapshot.next) if snapshot.next else [],
                "last_node": (
                    list(snapshot.metadata.get("writes", {}).keys())[-1]
                    if snapshot.metadata and snapshot.metadata.get("writes")
                    else None
                ),
                "status": snapshot.values.get("status", ""),
                "final_severity": snapshot.values.get("final_severity", ""),
            }
    except Exception:
        pass  # Checkpoint enrichment is best-effort

    return incident


@app.post("/incident/{incident_id}/resolve")
async def resolve_incident(
    incident_id: str,
    resolution_notes: str = Form(""),
):
    """
    Mark an incident as resolved and notify the reporter.
    Transitions FSM: TEAM_NOTIFIED → RESOLVED → REPORTER_NOTIFIED.

    Also triggers the KNOWLEDGE FLYWHEEL: the resolved incident is
    automatically chunked, embedded, and indexed into sre_knowledge
    for future retrieval. The agent learns from every resolution.
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

    # Transition to RESOLVED
    incident["status"] = IncidentStatus.RESOLVED.value
    incident["resolution_notes"] = resolution_notes
    logger.info(f"✅ Incident {incident_id} marked as RESOLVED")

    # Notify reporter (mock)
    notifications = incident.get("notifications", {})
    notifications["reporter_notified"] = True
    notifications["reporter_notification_channel"] = "email:reporter@example.com"
    incident["notifications"] = notifications
    incident["status"] = IncidentStatus.REPORTER_NOTIFIED.value

    logger.info(f"📧 Reporter notified for incident {incident_id}")

    # Persist
    db_provider.upsert_incident(incident)

    # 🔄 FLYWHEEL: Auto-index the resolution for future retrieval
    knowledge_stats = None
    try:
        from app.indexer.knowledge_indexer import index_resolved_incident
        knowledge_stats = await _maybe_await(
            index_resolved_incident(incident, resolution_notes=resolution_notes)
        )
        logger.info(
            f"🧠 Flywheel indexed {knowledge_stats.get('chunks_indexed', 0)} "
            f"knowledge chunks for incident {incident_id}"
        )
    except Exception as e:
        logger.warning(f"⚠️ Flywheel indexing failed for {incident_id}: {e}")

    return {
        "incident_id": incident_id,
        "status": incident["status"],
        "message": "Incident resolved and reporter notified.",
        "knowledge_indexed": knowledge_stats is not None,
        "knowledge_stats": knowledge_stats,
    }


@app.get("/incidents")
async def list_incidents():
    """List all incidents."""
    return db_provider.list_incidents()


# ---------------------------------------------------------------------------
# Indexer Endpoints
# ---------------------------------------------------------------------------


@app.post("/index")
async def index_eshop(force: bool = False):
    """
    Trigger indexing of the eShop repository.
    Clones the repo, chunks C# files, generates embeddings, and indexes into Cosmos DB.
    """
    from app.indexer.repo_indexer import index_repo

    logger.info(f"📦 Index triggered (force={force})")
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


@app.get("/incident/{incident_id}/ledger")
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


@app.post("/knowledge/seed")
async def seed_knowledge():
    """
    Seed the knowledge base with synthetic historical incidents.
    Useful for demos and testing the flywheel retrieval.
    """
    from app.indexer.seed_incidents import seed_historical_incidents

    logger.info("🌱 Seeding knowledge base with synthetic incidents...")
    stats = await _maybe_await(seed_historical_incidents())
    return stats
