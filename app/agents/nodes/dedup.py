"""
SRE Agent — Dedup Gate Node

Computes the report embedding and checks for duplicate open incidents
in Cosmos DB. If a duplicate is found above the similarity threshold,
the pipeline short-circuits to a DUPLICATE status.

This node lives inside the graph so that dedup works regardless of
how the graph is invoked (API, local runs, tests, and similar entry points).
"""

import logging
from app.agents.state import IncidentStatus
from app.providers import llm_provider, db_provider

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.80


async def dedup_node(state: dict) -> dict:
    """
    Compute the report embedding and check for semantic duplicates.

    Outputs:
        report_embedding: the 768-dim vector (reused by consolidator for runbooks)
        is_duplicate: bool — consumed by the conditional edge
        duplicate_of: parent incident_id if duplicate
    """
    raw_report = state.get("raw_report", "")
    incident_id = state.get("incident_id", "unknown")

    # Step 1: Compute embedding (reused downstream for runbook search)
    report_embedding = None
    try:
        report_embedding = await llm_provider.generate_embedding(
            raw_report[:500],
            task_type="RETRIEVAL_QUERY",
        )
    except Exception as e:
        logger.warning(f"[dedup] Embedding generation failed: {e}")
        return {
            "is_duplicate": False,
            "report_embedding": [],
        }

    # Step 2: Check for duplicates
    try:
        duplicate_result = await db_provider.async_find_duplicate_incident(
            query_vector=report_embedding,
            similarity_threshold=SIMILARITY_THRESHOLD,
            exclude_incident_id=incident_id,
        )
    except Exception as e:
        logger.warning(f"[dedup] Duplicate check failed (continuing): {e}")
        return {
            "is_duplicate": False,
            "report_embedding": report_embedding,
        }

    if duplicate_result:
        parent_id = duplicate_result["incident_id"]
        similarity = duplicate_result["similarity"]
        logger.info(
            f"[dedup] 🔗 Duplicate detected: {incident_id} → {parent_id} "
            f"(similarity={similarity:.2f})"
        )

        # Persist the duplicate record
        try:
            db_provider.upsert_incident({
                "id": incident_id,
                "incident_id": incident_id,
                "status": "DUPLICATE",
                "duplicate_of": parent_id,
                "raw_report": raw_report,
                "created_at": state.get("created_at", ""),
                "similarity_score": similarity,
            })

            # Bump occurrence count on parent
            parent = db_provider.get_incident(parent_id)
            if parent:
                parent["occurrence_count"] = parent.get("occurrence_count", 1) + 1
                db_provider.upsert_incident(parent)
        except Exception as e:
            logger.warning(f"[dedup] Failed to persist duplicate: {e}")

        return {
            "is_duplicate": True,
            "duplicate_of": parent_id,
            "duplicate_similarity": similarity,
            "duplicate_ticket_id": duplicate_result.get("ticket_id", ""),
            "duplicate_severity": duplicate_result.get("severity", ""),
            "report_embedding": report_embedding,
            "triage_summary": (
                f"This incident is a duplicate of {parent_id} "
                f"(similarity: {similarity:.0%}). "
                f"See existing ticket for triage details."
            ),
            "status": IncidentStatus.TEAM_NOTIFIED,
        }

    logger.info(f"[dedup] No duplicate found for {incident_id}")
    return {
        "is_duplicate": False,
        "report_embedding": report_embedding,
    }
