from __future__ import annotations

"""
SRE Agent — LangGraph Definition

Assembles the complete Dual-Track Neuro-Symbolic pipeline:

Track A (Neuronal):  intake → world_model → slot_filler
Track B (Adversarial): intake → risk_hypothesizer → span_arbiter → falsifier
Merge:               consolidator → create_ticket → notify_team

The Falsifier implements Popperian epistemology: each hypothesis gets
its own agent (via create_agent) with RAG tools that actively tries to
falsify it. Only corroborated hypotheses survive to the consolidator.

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


# ---------------------------------------------------------------------------
# Graph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict, total=False):
    """
    State schema for the LangGraph pipeline.
    Uses reducers for list fields to support parallel fan-in.
    """
    # Identity
    incident_id: str
    created_at: str

    # Input
    raw_report: str
    has_image: bool
    image_mime_type: str
    image_data_b64: str
    report_embedding: list  # Pre-computed from dedup phase — reused by consolidator

    # FSM
    status: str

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

    # Errors
    errors: Annotated[list, merge_lists]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

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

    # Sequential execution to guarantee context is fully built and available to Track B
    builder.add_edge("n_intake", "n_world_model")
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
    (e.g. missing credentials, network issue) so the agent never hard-fails
    at startup due to an infrastructure problem.
    """
    try:
        import os
        from langgraph_checkpoint_cosmosdb import CosmosDBSaver
        from app.config import get_settings
        settings = get_settings()

        # CosmosDBSaver v0.2.5 reads endpoint/key from env vars
        os.environ.setdefault("COSMOSDB_ENDPOINT", settings.cosmos_endpoint)
        os.environ.setdefault("COSMOSDB_KEY", settings.cosmos_key)

        checkpointer = CosmosDBSaver(
            database_name=settings.cosmos_database,
            container_name=settings.cosmos_container_checkpoints,
        )
        graph = build_graph(checkpointer=checkpointer)
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
        logger.info("[graph] LangGraph pipeline compiled (stateless mode)")
        return graph


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_graph = None


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
    return _graph
