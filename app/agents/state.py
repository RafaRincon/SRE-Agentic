from __future__ import annotations

"""
SRE Agent — Incident State Schema

Central Pydantic models that flow through the LangGraph state.
These models define the data exchanged between graph nodes.

The LLM may populate fields, but state transitions remain under FSM control.
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


class EpistemicClaim(BaseModel):
    """A single epistemic claim with explicit evidence provenance."""
    label: str = Field(description="Short statement or field name being classified")
    status: EpistemicStatus
    evidence: str = Field(default="", description="Direct evidence or rationale for the classification")
    source: str = Field(default="", description="Where this claim came from (report, code, falsifier, etc.)")


class EpistemicSnapshot(BaseModel):
    """Immutable snapshot of what is observed, inferred, and unknown at a given stage."""
    observed: list[EpistemicClaim] = Field(default_factory=list)
    inferred: list[EpistemicClaim] = Field(default_factory=list)
    unknown: list[EpistemicClaim] = Field(default_factory=list)


def make_epistemic_claim(
    label: str,
    status: EpistemicStatus,
    *,
    evidence: str = "",
    source: str = "",
) -> EpistemicClaim:
    return EpistemicClaim(
        label=label,
        status=status,
        evidence=evidence,
        source=source,
    )


def empty_epistemic_snapshot() -> EpistemicSnapshot:
    return EpistemicSnapshot()


def ensure_epistemic_snapshot(snapshot: EpistemicSnapshot | dict | None) -> EpistemicSnapshot:
    if snapshot is None:
        return EpistemicSnapshot()
    if isinstance(snapshot, EpistemicSnapshot):
        return snapshot
    return EpistemicSnapshot.model_validate(snapshot)


def snapshot_is_empty(snapshot: EpistemicSnapshot | dict | None) -> bool:
    normalized = ensure_epistemic_snapshot(snapshot)
    return not (normalized.observed or normalized.inferred or normalized.unknown)


def merge_epistemic_snapshots(
    *snapshots: EpistemicSnapshot | dict | None,
) -> EpistemicSnapshot:
    """Merge snapshots without mutating any upstream stage output."""
    merged = EpistemicSnapshot()
    seen: set[tuple[str, str, str, str]] = set()

    for snapshot in snapshots:
        normalized = ensure_epistemic_snapshot(snapshot)
        for bucket_name in ("observed", "inferred", "unknown"):
            bucket = getattr(normalized, bucket_name)
            target = getattr(merged, bucket_name)
            for claim in bucket:
                key = (
                    claim.label,
                    claim.status.value,
                    claim.evidence,
                    claim.source,
                )
                if key in seen:
                    continue
                seen.add(key)
                target.append(claim.model_copy(deep=True))

    return merged


# ---------------------------------------------------------------------------
# Sub-models (populated by Track A nodes)
# ---------------------------------------------------------------------------


class WorldModelProjection(BaseModel):
    """Output of the World Model node — compact operational triage context."""
    # Reasoning is retained for traceability and debugging.
    thinking_process: str = Field(
        default="",
        description="Step-by-step reasoning chain. This is kept for observability/debugging and is not part of the persisted incident document."
    )
    affected_service: str = Field(description="Primary service affected (e.g., 'OrderingService', 'PaymentService')")
    affected_service_confidence: EpistemicStatus = EpistemicStatus.UNKNOWN
    blast_radius: list[str] = Field(default_factory=list, description="Other services potentially impacted")
    estimated_severity: Severity = Severity.UNKNOWN
    severity_rationale: str = Field(default="", description="Legacy runtime-only rationale retained for backward compatibility; not persisted.")
    incident_category: str = Field(default="", description="e.g., 'RuntimeException', 'Timeout', 'DataCorruption'")
    temporal_context: str = Field(default="", description="Legacy runtime-only context retained for backward compatibility; not persisted.")
    image_extracted_context: str = Field(
        default="",
        description=(
            "If an image was provided, transcribe ALL diagnostic text visible in it: "
            "exact error messages, stack traces, metric values, log lines, dashboard readings. "
            "This field carries image evidence downstream so other nodes never need the raw blob. "
            "Empty string if no image was provided."
        )
    )
    epistemic_snapshot: EpistemicSnapshot = Field(default_factory=EpistemicSnapshot)


class ExtractedEntity(BaseModel):
    """Output of the Slot Filler — structured entities from the report."""
    # Reasoning is retained for traceability and debugging.
    thinking_process: str = Field(
        default="",
        description="Step-by-step reasoning: which parts of the report were used to extract each entity, and why any field was left null. This is not persisted in the incident document."
    )
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    file_references: list[str] = Field(default_factory=list)
    endpoint_affected: Optional[str] = None
    reporter_name: str = ""
    reporter_email: str = ""
    timestamp_reported: str = ""
    epistemic_snapshot: EpistemicSnapshot = Field(default_factory=EpistemicSnapshot)


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
    epistemic_snapshot: EpistemicSnapshot = Field(default_factory=EpistemicSnapshot)


class SpanVerdict(BaseModel):
    """Result of the Span Arbiter — did the citation check pass?"""
    hypothesis_id: str
    span_found: bool = Field(description="True if the exact_span was found in the indexed codebase")
    matched_file: Optional[str] = None
    matched_line: Optional[int] = None
    similarity_score: float = 0.0
    verdict: str = Field(default="UNVERIFIED", description="VERIFIED | HALLUCINATION | PARTIAL_MATCH")
    hypothesis_epistemic_snapshot: EpistemicSnapshot = Field(default_factory=EpistemicSnapshot)
    epistemic_snapshot: EpistemicSnapshot = Field(default_factory=EpistemicSnapshot)


class FalsifierVerdict(BaseModel):
    """Result of the Epistemic Falsifier."""
    hypothesis_id: str
    axiom_tested: str = Field(description="Which epistemic axiom was tested (e.g., 'FILE_EXISTS', 'FUNCTION_EXISTS', 'ERROR_CODE_VALID')")
    passed: bool
    evidence: str = Field(default="", description="Evidence supporting the verdict")
    verdict: str = Field(default="UNFALSIFIED", description="SURVIVED | FALSIFIED | INCONCLUSIVE")
    reasoning: str = ""
    counter_evidence: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    hypothesis_epistemic_snapshot: EpistemicSnapshot = Field(default_factory=EpistemicSnapshot)
    epistemic_snapshot: EpistemicSnapshot = Field(default_factory=EpistemicSnapshot)


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
    verified_root_causes: list[str] = Field(default_factory=list, description="Hypotheses retained in the consolidated output, including their advisory Span/Falsifier verdict tags.")
    epistemic_context: dict = Field(default_factory=dict, description="Final immutable IOU context exposed downstream")

    # --- Actions ---
    ticket: Optional[TicketInfo] = None
    notifications: Optional[NotificationInfo] = None

    # --- Errors ---
    errors: list[str] = Field(default_factory=list, description="Any errors encountered during processing")
