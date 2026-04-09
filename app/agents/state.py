from __future__ import annotations

"""
SRE Agent — Incident State Schema

Central Pydantic models that flow through the LangGraph state.
This is the "contract" between all nodes in the graph.

Design principle: The LLM populates fields, but the FSM controls transitions.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
import uuid
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class IncidentStatus(str, Enum):
    """FSM states — transitions are controlled algebraically, never by an LLM."""
    RECEIVED = "RECEIVED"
    TRIAGING = "TRIAGING"
    TRIAGED = "TRIAGED"
    TICKET_CREATED = "TICKET_CREATED"
    TEAM_NOTIFIED = "TEAM_NOTIFIED"
    RESOLVED = "RESOLVED"
    REPORTER_NOTIFIED = "REPORTER_NOTIFIED"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class EpistemicStatus(str, Enum):
    """Classification of knowledge about a fact."""
    OBSERVED = "OBSERVED"      # Explicitly stated in the report
    INFERRED = "INFERRED"      # Deduced by the LLM from context
    UNKNOWN = "UNKNOWN"        # Missing, needs investigation


# ---------------------------------------------------------------------------
# Sub-models (populated by Track A nodes)
# ---------------------------------------------------------------------------


class WorldModelProjection(BaseModel):
    """Output of the World Model node — cognitive projection of the incident."""
    affected_service: str = Field(description="Primary service affected (e.g., 'OrderingService', 'PaymentService')")
    affected_service_confidence: EpistemicStatus = EpistemicStatus.UNKNOWN
    blast_radius: list[str] = Field(default_factory=list, description="Other services potentially impacted")
    estimated_severity: Severity = Severity.UNKNOWN
    severity_rationale: str = Field(default="", description="Why this severity was assigned")
    incident_category: str = Field(default="", description="e.g., 'RuntimeException', 'Timeout', 'DataCorruption'")
    temporal_context: str = Field(default="", description="When did this start? Is it ongoing?")


class ExtractedEntity(BaseModel):
    """Output of the Slot Filler — structured entities from the report."""
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    file_references: list[str] = Field(default_factory=list)
    endpoint_affected: Optional[str] = None
    reporter_name: str = ""
    reporter_email: str = ""
    timestamp_reported: str = ""


# ---------------------------------------------------------------------------
# Sub-models (populated by Track B nodes)
# ---------------------------------------------------------------------------


class RiskHypothesis(BaseModel):
    """A single hypothesis from the Risk Hypothesizer, with mandatory citation."""
    hypothesis_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str = Field(description="What the LLM thinks might be wrong")
    suspected_file: str = Field(description="File path in eShop repo (e.g., 'src/Ordering.API/Controllers/OrdersController.cs')")
    suspected_function: str = Field(default="", description="Function or method name")
    exact_span: str = Field(description="EXACT quote from the code/log that supports this hypothesis")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SpanVerdict(BaseModel):
    """Result of the Span Arbiter — did the citation check pass?"""
    hypothesis_id: str
    span_found: bool = Field(description="True if the exact_span was found in the indexed codebase")
    matched_file: Optional[str] = None
    matched_line: Optional[int] = None
    similarity_score: float = 0.0
    verdict: str = Field(default="UNVERIFIED", description="VERIFIED | HALLUCINATION | PARTIAL_MATCH")


class FalsifierVerdict(BaseModel):
    """Result of the Epistemic Falsifier."""
    hypothesis_id: str
    axiom_tested: str = Field(description="Which epistemic axiom was tested (e.g., 'FILE_EXISTS', 'FUNCTION_EXISTS', 'ERROR_CODE_VALID')")
    passed: bool
    evidence: str = Field(default="", description="Evidence supporting the verdict")
    verdict: str = Field(default="UNFALSIFIED", description="SURVIVED | FALSIFIED | INCONCLUSIVE")


# ---------------------------------------------------------------------------
# Ticket and Notification models
# ---------------------------------------------------------------------------


class TicketInfo(BaseModel):
    """Info about the created ticket."""
    ticket_id: str = ""
    ticket_url: str = ""
    assigned_team: str = ""
    priority: str = ""


class NotificationInfo(BaseModel):
    """Info about notifications sent."""
    team_notified: bool = False
    team_notification_channel: str = ""
    reporter_notified: bool = False
    reporter_notification_channel: str = ""


# ---------------------------------------------------------------------------
# Main LangGraph State (this flows through the entire graph)
# ---------------------------------------------------------------------------


class IncidentState(BaseModel):
    """
    The central state object that flows through the LangGraph pipeline.
    Every node reads from and writes to this state.
    """
    # --- Identity ---
    incident_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # --- Input ---
    raw_report: str = Field(default="", description="Original text from the reporter")
    has_image: bool = False
    image_mime_type: str = ""
    image_data_b64: str = Field(default="", description="Base64-encoded image data")

    # --- FSM ---
    status: IncidentStatus = IncidentStatus.RECEIVED

    # --- Track A: World Model + Slot Filler ---
    world_model: Optional[WorldModelProjection] = None
    entities: Optional[ExtractedEntity] = None

    # --- Track B: Risk Hypothesizer + Arbiter + Falsifier ---
    hypotheses: list[RiskHypothesis] = Field(default_factory=list)
    span_verdicts: list[SpanVerdict] = Field(default_factory=list)
    falsifier_verdicts: list[FalsifierVerdict] = Field(default_factory=list)

    # --- Consolidated output ---
    triage_summary: str = Field(default="", description="Final technical summary after consolidation")
    final_severity: Severity = Severity.UNKNOWN
    verified_root_causes: list[str] = Field(default_factory=list, description="Hypotheses that survived both arbiter and falsifier")

    # --- Actions ---
    ticket: Optional[TicketInfo] = None
    notifications: Optional[NotificationInfo] = None

    # --- Errors ---
    errors: list[str] = Field(default_factory=list, description="Any errors encountered during processing")
