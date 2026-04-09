from __future__ import annotations

"""
SRE Agent — Knowledge Flywheel Indexer

Automatically converts resolved incidents into retrievable knowledge.
This is the core of the virtuous cycle: every resolution makes
the agent smarter for future incidents.

The flywheel reads BOTH the incident document AND the full ledger
trail to build rich knowledge chunks that capture:
- SYMPTOM: What the incident looked like (for matching future symptoms)
- ROOT_CAUSE: What was verified as the cause (for grounding future hypotheses)
- RESOLUTION: How it was fixed (for suggesting future resolutions)
- REASONING_TRACE: The agent's own decision path (from the ledger)
"""

import hashlib
import logging
import math
from datetime import datetime, timezone

from app.providers import llm_provider, db_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------


def _generate_chunk_id(incident_id: str, chunk_role: str) -> str:
    """Generate a deterministic ID for a knowledge chunk."""
    raw = f"knowledge:{incident_id}:{chunk_role}"
    return hashlib.md5(raw.encode()).hexdigest()


def _extract_tags(incident: dict) -> list[str]:
    """Extract searchable tags from incident data."""
    tags = []
    entities = incident.get("entities", {})
    world_model = incident.get("world_model", {})

    if entities.get("error_code"):
        tags.append(entities["error_code"])
    if entities.get("error_message"):
        # Extract the exception type from the message
        msg = entities["error_message"]
        for exc_type in ["NullReferenceException", "TimeoutException",
                         "ArgumentException", "InvalidOperationException",
                         "HttpRequestException", "SocketException"]:
            if exc_type.lower() in msg.lower():
                tags.append(exc_type)

    if entities.get("endpoint_affected"):
        tags.append(entities["endpoint_affected"])
    if world_model.get("incident_category"):
        tags.append(world_model["incident_category"])
    if world_model.get("affected_service"):
        tags.append(world_model["affected_service"])

    # Add blast radius services as tags
    for svc in world_model.get("blast_radius", []):
        tags.append(svc)

    return list(set(tags))


# ---------------------------------------------------------------------------
# Build knowledge chunks from a resolved incident + its ledger
# ---------------------------------------------------------------------------


def _build_resolution_chunks(
    incident: dict,
    ledger_entries: list[dict],
    resolution_notes: str = "",
) -> list[dict]:
    """
    Convert a resolved incident into retrievable knowledge chunks.

    Uses BOTH the incident document AND the ledger audit trail to build
    comprehensive chunks that capture the full lifecycle.

    Generates up to 4 chunks:
    1. SYMPTOM — for matching future incidents by similar symptoms
    2. ROOT_CAUSE — for grounding future hypotheses
    3. RESOLUTION — for suggesting proven fixes
    4. REASONING_TRACE — the agent's decision path (from ledger)
    """
    incident_id = incident.get("incident_id", incident.get("id", "unknown"))
    service = incident.get("world_model", {}).get("affected_service", "unknown")
    created_at = incident.get("created_at", "")
    severity = incident.get("final_severity", "UNKNOWN")
    entities = incident.get("entities", {})
    verified_causes = incident.get("verified_root_causes", [])
    triage_summary = incident.get("triage_summary", "")

    # Compute MTTR from ledger timestamps
    mttr_minutes = _compute_mttr(incident, ledger_entries)

    base_metadata = {
        "severity": severity,
        "resolution_status": "RESOLVED",
        "created_at": created_at,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "mttr_minutes": mttr_minutes,
        "tags": _extract_tags(incident),
    }

    base = {
        "doc_type": "TICKET",
        "service_name": service,
        "source_id": f"incident-{incident_id}",
    }

    chunks = []

    # --- Chunk 1: SYMPTOM ---
    raw_report = incident.get("raw_report", "")
    symptom_text = (
        f"INCIDENT SYMPTOM — {service}\n"
        f"Severity: {severity}\n"
        f"Error: {entities.get('error_code', '')} {entities.get('error_message', '')}\n"
        f"Endpoint: {entities.get('endpoint_affected', '')}\n"
        f"Category: {incident.get('world_model', {}).get('incident_category', '')}\n"
        f"Blast Radius: {', '.join(incident.get('world_model', {}).get('blast_radius', []))}\n"
        f"Report: {raw_report[:600]}"
    )
    chunks.append({
        **base,
        "id": _generate_chunk_id(incident_id, "symptom"),
        "chunk_text": symptom_text,
        "metadata": {**base_metadata, "chunk_role": "symptom"},
    })

    # --- Chunk 2: ROOT_CAUSE (only if we have verified causes) ---
    if verified_causes:
        # Enrich with span verdicts from ledger
        span_verdicts = [
            e["data"] for e in ledger_entries
            if e.get("event_type") == "SPAN_VERDICT"
            and e.get("data", {}).get("verdict") in ("VERIFIED", "PARTIAL_MATCH")
        ]

        cause_text = (
            f"VERIFIED ROOT CAUSE — {service}\n"
            f"Incident: {incident_id}\n"
            f"Causes:\n" + "\n".join(f"  - {c}" for c in verified_causes) + "\n"
            f"Original error: {entities.get('error_code', '')} "
            f"{entities.get('error_message', '')}\n"
            f"Severity: {severity}\n"
        )

        # Add verified span evidence from ledger
        if span_verdicts:
            cause_text += "Evidence (verified spans):\n"
            for sv in span_verdicts[:3]:
                cause_text += (
                    f"  - File: {sv.get('matched_file', '?')}"
                    f" Line: {sv.get('matched_line', '?')}"
                    f" Score: {sv.get('similarity_score', 0):.2f}\n"
                )

        chunks.append({
            **base,
            "id": _generate_chunk_id(incident_id, "root_cause"),
            "chunk_text": cause_text,
            "metadata": {**base_metadata, "chunk_role": "root_cause"},
        })

    # --- Chunk 3: RESOLUTION ---
    if triage_summary or resolution_notes:
        # Count discarded hypotheses from ledger
        total_hypotheses = len([
            e for e in ledger_entries
            if e.get("event_type") == "HYPOTHESIS_GENERATED"
        ])
        discarded = len([
            e for e in ledger_entries
            if e.get("event_type") == "SPAN_VERDICT"
            and e.get("data", {}).get("verdict") == "HALLUCINATION"
        ])

        resolution_text = (
            f"RESOLUTION — {service}\n"
            f"Incident: {incident_id}\n"
            f"Triage Summary: {triage_summary}\n"
        )
        if resolution_notes:
            resolution_text += f"Human Resolution Notes: {resolution_notes}\n"
        resolution_text += (
            f"Verified causes: {', '.join(verified_causes)}\n"
            f"Severity: {severity}\n"
            f"MTTR: {mttr_minutes} minutes\n"
            f"Hypotheses generated: {total_hypotheses}, "
            f"Discarded (hallucinated): {discarded}\n"
        )

        chunks.append({
            **base,
            "id": _generate_chunk_id(incident_id, "resolution"),
            "chunk_text": resolution_text,
            "metadata": {**base_metadata, "chunk_role": "resolution",
                         "resolution_notes": resolution_notes},
        })

    # --- Chunk 4: REASONING_TRACE (from ledger) ---
    if ledger_entries:
        trace_text = _build_reasoning_trace(incident_id, service, ledger_entries)
        if trace_text:
            chunks.append({
                **base,
                "id": _generate_chunk_id(incident_id, "reasoning_trace"),
                "chunk_text": trace_text,
                "metadata": {**base_metadata, "chunk_role": "reasoning_trace"},
            })

    return chunks


def _build_reasoning_trace(
    incident_id: str,
    service: str,
    ledger_entries: list[dict],
) -> str:
    """
    Build a condensed reasoning trace from the ledger entries.
    This captures the agent's decision path for future reference.
    """
    lines = [f"REASONING TRACE — {service} — Incident {incident_id}"]

    for entry in ledger_entries:
        event_type = entry.get("event_type", "")
        node = entry.get("node_name", "")
        data = entry.get("data", {})
        ts = entry.get("timestamp", "")

        if event_type == "STATE_TRANSITION":
            lines.append(
                f"[{ts}] FSM: {data.get('from_state')} → {data.get('to_state')}"
            )
        elif event_type == "HYPOTHESIS_GENERATED":
            lines.append(
                f"[{ts}] HYPOTHESIS ({node}): {data.get('description', '')[:100]}"
                f" | file={data.get('suspected_file', '?')}"
                f" | confidence={data.get('confidence', '?')}"
            )
        elif event_type == "SPAN_VERDICT":
            verdict = data.get("verdict", "?")
            score = data.get("similarity_score", 0)
            lines.append(
                f"[{ts}] VERDICT ({node}): {verdict}"
                f" | score={score:.2f}"
                f" | file={data.get('matched_file', '?')}"
            )

    # Cap at 2000 chars to stay within chunk size
    trace = "\n".join(lines)
    return trace[:2000] if len(trace) > 2000 else trace


def _compute_mttr(incident: dict, ledger_entries: list[dict]) -> int | None:
    """
    Compute Mean Time To Resolution from ledger timestamps.
    Returns minutes between first RECEIVED and RESOLVED transitions.
    """
    received_ts = None
    resolved_ts = None

    for entry in ledger_entries:
        if entry.get("event_type") != "STATE_TRANSITION":
            continue
        data = entry.get("data", {})
        if data.get("to_state") == "RECEIVED" and received_ts is None:
            received_ts = entry.get("timestamp")
        if data.get("to_state") == "RESOLVED":
            resolved_ts = entry.get("timestamp")

    # Fallback to incident created_at
    if not received_ts:
        received_ts = incident.get("created_at")

    if received_ts and resolved_ts:
        try:
            t0 = datetime.fromisoformat(received_ts)
            t1 = datetime.fromisoformat(resolved_ts)
            return max(1, int((t1 - t0).total_seconds() / 60))
        except (ValueError, TypeError):
            pass

    return None


# ---------------------------------------------------------------------------
# Temporal decay for retrieval scoring
# ---------------------------------------------------------------------------


def apply_temporal_decay(
    results: list[dict],
    half_life_days: int = 90,
) -> list[dict]:
    """
    Boost recent documents in retrieval results.

    Uses exponential decay with a configurable half-life.
    Code chunks (no timestamp) receive max weight since they're
    always current with the indexed repo version.
    """
    now = datetime.now(timezone.utc)

    for r in results:
        created_at = r.get("metadata", {}).get("created_at")
        if created_at:
            try:
                age_days = (now - datetime.fromisoformat(created_at)).days
                decay = math.exp(-0.693 * age_days / half_life_days)
                r["temporal_weight"] = round(decay, 4)
            except (ValueError, TypeError):
                r["temporal_weight"] = 1.0
        else:
            r["temporal_weight"] = 1.0  # Code chunks = always relevant

    return sorted(results, key=lambda x: x.get("temporal_weight", 0), reverse=True)


# ---------------------------------------------------------------------------
# Main flywheel entry point
# ---------------------------------------------------------------------------


async def index_resolved_incident(
    incident: dict,
    resolution_notes: str = "",
) -> dict:
    """
    Called automatically when an incident transitions to RESOLVED.

    1. Fetch the full ledger trail for this incident
    2. Build symptom/root_cause/resolution/reasoning_trace chunks
    3. Generate embeddings for all chunks
    4. Upsert to the sre_knowledge container

    Returns indexing stats.
    """
    incident_id = incident.get("incident_id", incident.get("id", "unknown"))

    # Step 1: Get the full ledger trail — this is the gold mine
    ledger_entries = db_provider.get_ledger_entries(incident_id)
    logger.info(
        f"[flywheel] Fetched {len(ledger_entries)} ledger entries "
        f"for incident {incident_id}"
    )

    # Step 2: Build knowledge chunks
    chunks = _build_resolution_chunks(incident, ledger_entries, resolution_notes)

    if not chunks:
        logger.warning(f"[flywheel] No knowledge chunks generated for {incident_id}")
        return {"incident_id": incident_id, "chunks_indexed": 0}

    # Step 3: Generate embeddings
    texts = [c["chunk_text"] for c in chunks]
    try:
        embeddings = await llm_provider.generate_embeddings_batch(
            texts, task_type="RETRIEVAL_DOCUMENT"
        )
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb
    except Exception as e:
        logger.error(f"[flywheel] Embedding generation failed: {e}")
        return {"incident_id": incident_id, "chunks_indexed": 0, "error": str(e)}

    # Step 4: Upsert to sre_knowledge
    indexed = 0
    for chunk in chunks:
        try:
            db_provider.upsert_knowledge_chunk(chunk)
            indexed += 1
        except Exception as e:
            logger.warning(f"[flywheel] Failed to upsert chunk {chunk['id']}: {e}")

    logger.info(
        f"[flywheel] ✅ Indexed {indexed} knowledge chunks "
        f"for incident {incident_id} "
        f"(roles: {[c['metadata']['chunk_role'] for c in chunks]})"
    )

    return {
        "incident_id": incident_id,
        "chunks_indexed": indexed,
        "chunk_roles": [c["metadata"]["chunk_role"] for c in chunks],
        "ledger_entries_used": len(ledger_entries),
    }
