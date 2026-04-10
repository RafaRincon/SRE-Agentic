from __future__ import annotations

"""
SRE Agent — LangGraph Definition

Assembles the incident-processing pipeline:

Track A: intake → world_model → slot_filler
Track B: intake → risk_hypothesizer → span_arbiter → falsifier
Merge:               consolidator → create_ticket → notify_team

The falsifier evaluates each hypothesis against indexed evidence and
removes hypotheses that do not hold up under review.

Conditional edges based on FSM state flags, never LLM decisions.

Checkpointer:
    CosmosDBSaver (langgraph_checkpoint_cosmosdb) persists every node
    transition under thread_id=incident_id in the agent_checkpoints
    container.  This enables resume-on-failure and state inspection
    via GET /incident/{id} without re-running the pipeline.
"""

import logging
from typing import Annotated, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.agents.nodes.intake import intake_node
from app.agents.nodes.dedup import dedup_node
from app.agents.nodes.world_model import world_model_node
from app.agents.nodes.slot_filler import slot_filler_node
from app.agents.nodes.risk_hypothesizer import risk_hypothesizer_node
from app.agents.nodes.span_arbiter import span_arbiter_node
from app.agents.nodes.falsifier import falsifier_node
from app.agents.nodes.consolidator import consolidator_node
from app.agents.nodes.actions import create_ticket_node, notify_team_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------

def merge_lists(a: list, b: list) -> list:
    """Reducer: merge two lists by concatenation (supports parallel fan-in)."""
    return a + b


def last_value(a: Any, b: Any) -> Any:
    """Reducer: last-writer-wins for scalar keys (FSM status, severity, etc.)."""
    return b


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    State schema for the LangGraph pipeline.
    Uses reducers for list fields to support parallel fan-in.
    Scalar keys that are written by multiple nodes use last_value (last-writer-wins).
    """
    # Identity
    incident_id: str
    created_at: str

    # Input
    raw_report: str
    has_image: bool
    image_mime_type: str
    image_data_b64: str
    image_extracted_context: str  # Text transcript of image content — populated by world_model, read by downstream nodes
    report_embedding: list  # Pre-computed from dedup phase — reused by consolidator

    # FSM — multiple nodes write status (intake → consolidator → actions)
    status: Annotated[str, last_value]

    # Track A
    world_model: dict
    entities: dict

    # Track B
    hypotheses: Annotated[list, merge_lists]
    span_verdicts: Annotated[list, merge_lists]
    falsifier_verdicts: Annotated[list, merge_lists]

    # Consolidated
    triage_summary: str
    final_severity: str
    verified_root_causes: Annotated[list, merge_lists]
    epistemic_context: dict

    # Historical correlation (flywheel)
    historical_context: dict
    recurrence_count: int

    # Runbook suggestions
    suggested_runbooks: Annotated[list, merge_lists]

    # Actions
    ticket: dict
    notifications: dict

    # Dedup gate
    is_duplicate: bool
    duplicate_of: str
    duplicate_similarity: float
    duplicate_ticket_id: str
    duplicate_severity: str

    # Errors
    errors: Annotated[list, merge_lists]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _dedup_router(state: dict) -> str:
    """Conditional edge: skip full triage if duplicate found."""
    if state.get("is_duplicate"):
        return "__end__"
    return "n_world_model"


def build_graph(checkpointer=None):
    """
    Construct and compile the LangGraph pipeline.

    Args:
        checkpointer: Optional LangGraph checkpointer.  When provided,
                      every node transition is persisted under:
                        - thread_id  = incident_id
                        - checkpoint = node name
    """
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("n_intake", intake_node)
    builder.add_node("n_dedup", dedup_node)
    builder.add_node("n_world_model", world_model_node)
    builder.add_node("n_slot_filler", slot_filler_node)
    builder.add_node("n_risk_hypothesizer", risk_hypothesizer_node)
    builder.add_node("n_span_arbiter", span_arbiter_node)
    builder.add_node("n_falsifier", falsifier_node)
    builder.add_node("n_consolidator", consolidator_node)
    builder.add_node("n_create_ticket", create_ticket_node)
    builder.add_node("n_notify_team", notify_team_node)

    # Entry
    builder.set_entry_point("n_intake")

    # Dedup gate (right after intake, before any expensive LLM work)
    builder.add_edge("n_intake", "n_dedup")
    builder.add_conditional_edges("n_dedup", _dedup_router)

    # Sequential execution to guarantee context is fully built and available to Track B
    builder.add_edge("n_world_model", "n_slot_filler")

    # Track B starts running with the context built by Track A
    builder.add_edge("n_slot_filler", "n_risk_hypothesizer")

    # Track B internal chain
    builder.add_edge("n_risk_hypothesizer", "n_span_arbiter")
    builder.add_edge("n_span_arbiter", "n_falsifier")

    # Merge into logical finalizer
    builder.add_edge("n_falsifier", "n_consolidator")

    # Post-consolidation
    builder.add_edge("n_consolidator", "n_create_ticket")
    builder.add_edge("n_create_ticket", "n_notify_team")
    builder.add_edge("n_notify_team", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Checkpointer factory
# ---------------------------------------------------------------------------

def _build_with_checkpointer():
    """
    Build the graph backed by CosmosDBSaver.
    Falls back to a stateless graph if the checkpointer is unavailable
    (for example, due to missing credentials or a network issue).
    """
    try:
        import os
        from langgraph_checkpoint_cosmosdb import CosmosDBSaver
        from app.config import get_settings
        settings = get_settings()

        # CosmosDBSaver v0.2.5 reads endpoint/key from environment variables.
        os.environ.setdefault("COSMOSDB_ENDPOINT", settings.cosmos_endpoint)
        os.environ.setdefault("COSMOSDB_KEY", settings.cosmos_key)

        checkpointer = CosmosDBSaver(
            database_name=settings.cosmos_database,
            container_name=settings.cosmos_container_checkpoints,
        )

        # CosmosDBSaver builds its own CosmosClient with the default 65s read timeout.
        # Checkpoint payloads can reach ~780 KB per incident, which is enough to make
        # the final write hit that timeout. Replace the internal container with one
        # created from a higher-timeout client.
        from azure.cosmos import CosmosClient as _CosmosClient
        _hi_client = _CosmosClient(
            settings.cosmos_endpoint,
            settings.cosmos_key,
            connection_timeout=30,    # TCP connect (was default ~10s, explicit is safer)
            read_timeout=180,         # 3 minutes — enough for any payload size
        )
        checkpointer.container = (
            _hi_client
            .get_database_client(settings.cosmos_database)
            .get_container_client(settings.cosmos_container_checkpoints)
        )
        logger.info("[graph] ✅ Checkpointer container patched with 180s read timeout")

        graph = build_graph(checkpointer=checkpointer)
        _graph_status["checkpointer_enabled"] = True
        _graph_status["checkpointer_error"] = ""
        logger.info(
            "[graph] ✅ LangGraph compiled with CosmosDB checkpointer "
            f"(container: {settings.cosmos_container_checkpoints})"
        )
        return graph

    except Exception as e:
        logger.warning(
            f"[graph] ⚠️  CosmosDB checkpointer unavailable ({e}), "
            "falling back to stateless graph"
        )
        graph = build_graph(checkpointer=None)
        _graph_status["checkpointer_enabled"] = False
        _graph_status["checkpointer_error"] = str(e)
        logger.info("[graph] LangGraph pipeline compiled (stateless mode)")
        return graph


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_graph = None
_graph_status = {
    "compiled": False,
    "checkpointer_enabled": False,
    "checkpointer_error": "",
}


def get_graph():
    """
    Return the compiled graph singleton.

    Thread model: one thread_id per incident_id.
    LangGraph checkpointer config is passed on every ainvoke call via:
        config={"configurable": {"thread_id": incident_id}}
    """
    global _graph
    if _graph is None:
        _graph = _build_with_checkpointer()
        _graph_status["compiled"] = True
    return _graph


def get_graph_status() -> dict[str, object]:
    """Return the current graph/checkpointer readiness snapshot."""
    get_graph()
    return dict(_graph_status)


def get_stateless_graph():
    """
    Return a compiled graph with NO checkpointer.
    Useful for local runs where persistence is not required.
    """
    return build_graph(checkpointer=None)
