from __future__ import annotations

"""
SRE Agent — Azure Cosmos DB NoSQL Provider

Unified data layer for:
- Hybrid search: RRF(DiskANN vectors + BM25 full-text) on eshop_chunks & sre_knowledge
- Incident state (incidents)
- Audit trail (incident_ledger)
- LangGraph checkpointing (agent_checkpoints)

Uses the native azure-cosmos SDK (NOT MongoDB vCore).
BM25 full-text indexing enabled on /chunk_text for both vector containers.
"""

from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceExistsError
from typing import Any
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton clients
# ---------------------------------------------------------------------------

_cosmos_client: CosmosClient | None = None
_containers: dict[str, Any] = {}


def _get_cosmos_client() -> CosmosClient:
    """Return a cached Cosmos DB client."""
    global _cosmos_client
    if _cosmos_client is None:
        settings = get_settings()
        _cosmos_client = CosmosClient(
            url=settings.cosmos_endpoint,
            credential=settings.cosmos_key,
        )
    return _cosmos_client


def get_database():
    """Get the Cosmos DB database client."""
    settings = get_settings()
    client = _get_cosmos_client()
    return client.get_database_client(settings.cosmos_database)


def get_container(container_name: str):
    """Get a cached container client."""
    if container_name not in _containers:
        db = get_database()
        _containers[container_name] = db.get_container_client(container_name)
    return _containers[container_name]


# ---------------------------------------------------------------------------
# Hybrid Search (eshop_chunks) — RRF: DiskANN + BM25
# ---------------------------------------------------------------------------


def vector_search(
    query_vector: list[float],
    *,
    query_text: str | None = None,
    top_k: int = 5,
    service_filter: str | None = None,
) -> list[dict]:
    """
    Perform hybrid search on eshop_chunks using Reciprocal Rank Fusion (RRF)
    combining DiskANN vector similarity with BM25 full-text scoring.

    Falls back to vector-only search if hybrid query fails.

    Args:
        query_vector: The embedding vector to search with (768 dims).
        query_text: The original query text for BM25 keyword matching.
        top_k: Number of results to return.
        service_filter: Optional partition key filter (service_name).

    Returns:
        List of matching documents with similarity/RRF scores.
    """
    settings = get_settings()
    container = get_container(settings.cosmos_container_chunks)

    # --- Attempt hybrid RRF search (vector + BM25) ---
    if query_text:
        try:
            return _hybrid_search_chunks(container, query_vector, query_text, top_k)
        except Exception as e:
            logger.warning(f"Hybrid search failed, falling back to vector-only: {e}")

    # --- Fallback: vector-only search ---
    return _vector_only_search_chunks(container, query_vector, top_k)


def _hybrid_search_chunks(
    container, query_vector: list[float], query_text: str, top_k: int
) -> list[dict]:
    """RRF hybrid search combining DiskANN + BM25 on eshop_chunks."""
    query = """
    SELECT TOP @top_k
        c.id,
        c.file_path,
        c.service_name,
        c.chunk_text,
        c.start_line,
        c.end_line,
        c.language,
        c.class_name,
        c.method_name,
        c.chunk_type
    FROM c
    ORDER BY RANK RRF(VectorDistance(c.embedding, @queryVector), FullTextScore(c.chunk_text, @queryText))
    """

    parameters = [
        {"name": "@top_k", "value": top_k},
        {"name": "@queryVector", "value": query_vector},
        {"name": "@queryText", "value": query_text},
    ]

    results = list(
        container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        )
    )

    logger.info(f"Hybrid RRF search returned {len(results)} results (query_text='{query_text[:50]}...')")
    return results


def _vector_only_search_chunks(
    container, query_vector: list[float], top_k: int
) -> list[dict]:
    """Fallback: DiskANN vector-only search on eshop_chunks."""
    query = """
    SELECT TOP @top_k
        c.id,
        c.file_path,
        c.service_name,
        c.chunk_text,
        c.start_line,
        c.end_line,
        c.language,
        c.class_name,
        c.method_name,
        c.chunk_type,
        VectorDistance(c.embedding, @queryVector) AS similarity_score
    FROM c
    ORDER BY VectorDistance(c.embedding, @queryVector)
    """

    parameters = [
        {"name": "@top_k", "value": top_k},
        {"name": "@queryVector", "value": query_vector},
    ]

    results = list(
        container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        )
    )

    logger.info(f"Vector-only search returned {len(results)} results")
    return results


# ---------------------------------------------------------------------------
# Incident CRUD (incidents)
# ---------------------------------------------------------------------------


def upsert_incident(incident: dict) -> dict:
    """Create or update an incident document."""
    settings = get_settings()
    container = get_container(settings.cosmos_container_incidents)
    return container.upsert_item(incident)


def get_incident(incident_id: str) -> dict | None:
    """Get an incident by ID."""
    settings = get_settings()
    container = get_container(settings.cosmos_container_incidents)
    try:
        return container.read_item(item=incident_id, partition_key=incident_id)
    except Exception:
        return None


def list_incidents(limit: int = 50) -> list[dict]:
    """List recent incidents."""
    settings = get_settings()
    container = get_container(settings.cosmos_container_incidents)
    query = "SELECT TOP @limit * FROM c ORDER BY c._ts DESC"
    return list(
        container.query_items(
            query=query,
            parameters=[{"name": "@limit", "value": limit}],
            enable_cross_partition_query=True,
        )
    )


def find_duplicate_incident(
    query_vector: list[float],
    similarity_threshold: float = 0.80,
) -> dict | None:
    """
    Find an open incident that is semantically similar to the new report.

    Uses in-memory cosine similarity (no DiskANN needed on incidents container).
    Only considers incidents that are NOT resolved/duplicate.

    Args:
        query_vector: Embedding of the new incident report.
        similarity_threshold: Minimum similarity to consider a duplicate.

    Returns:
        Dict with parent incident info if duplicate found, None otherwise.
    """
    import math

    settings = get_settings()
    container = get_container(settings.cosmos_container_incidents)

    # Get open incidents that have an embedding stored
    query = (
        "SELECT c.id, c.incident_id, c.status, c.raw_report, "
        "c.report_embedding, c.final_severity, c.triage_summary, "
        "c.ticket, c.occurrence_count "
        "FROM c WHERE c.status NOT IN ('RESOLVED', 'REPORTER_NOTIFIED', 'DUPLICATE') "
        "AND IS_DEFINED(c.report_embedding)"
    )

    try:
        open_incidents = list(
            container.query_items(
                query=query,
                enable_cross_partition_query=True,
            )
        )
    except Exception as e:
        logger.warning(f"[dedup] Failed to query open incidents: {e}")
        return None

    if not open_incidents:
        return None

    # Compute cosine similarity in-memory
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    best_match = None
    best_score = 0.0

    for inc in open_incidents:
        inc_embedding = inc.get("report_embedding")
        if not inc_embedding:
            continue

        score = _cosine_sim(query_vector, inc_embedding)
        if score > best_score:
            best_score = score
            best_match = inc

    if best_match and best_score >= similarity_threshold:
        ticket = best_match.get("ticket", {})
        logger.info(
            f"[dedup] Found duplicate: {best_match['incident_id']} "
            f"(similarity={best_score:.2f})"
        )
        return {
            "incident_id": best_match["incident_id"],
            "similarity": best_score,
            "severity": best_match.get("final_severity", ""),
            "ticket_id": ticket.get("ticket_id", ""),
            "ticket_url": ticket.get("ticket_url", ""),
            "assigned_team": ticket.get("assigned_team", ""),
        }

    return None


# ---------------------------------------------------------------------------
# Ledger / Audit Trail (incident_ledger)
# ---------------------------------------------------------------------------


def append_ledger_entry(entry: dict) -> dict:
    """
    Append an IMMUTABLE entry to the audit trail.
    Each entry records a decision, hypothesis, verdict, or state transition.
    """
    settings = get_settings()
    container = get_container(settings.cosmos_container_ledger)
    return container.create_item(entry)


def get_ledger_entries(incident_id: str) -> list[dict]:
    """Get all ledger entries for an incident, ordered by timestamp."""
    settings = get_settings()
    container = get_container(settings.cosmos_container_ledger)
    query = """
    SELECT * FROM c
    WHERE c.incident_id = @incident_id
    ORDER BY c.timestamp ASC
    """
    return list(
        container.query_items(
            query=query,
            parameters=[{"name": "@incident_id", "value": incident_id}],
            partition_key=incident_id,
        )
    )


# ---------------------------------------------------------------------------
# Upsert chunks (for indexer)
# ---------------------------------------------------------------------------


def upsert_chunk(chunk: dict) -> dict:
    """Upsert a code chunk with its embedding into eshop_chunks."""
    settings = get_settings()
    container = get_container(settings.cosmos_container_chunks)
    return container.upsert_item(chunk)


def count_chunks() -> int:
    """Count the number of indexed chunks."""
    settings = get_settings()
    container = get_container(settings.cosmos_container_chunks)
    query = "SELECT VALUE COUNT(1) FROM c"
    results = list(
        container.query_items(
            query=query,
            enable_cross_partition_query=True,
        )
    )
    return results[0] if results else 0


# ---------------------------------------------------------------------------
# Knowledge Flywheel (sre_knowledge)
# ---------------------------------------------------------------------------


def upsert_knowledge_chunk(chunk: dict) -> dict:
    """Upsert a knowledge chunk (resolved incident data) into sre_knowledge."""
    settings = get_settings()
    container = get_container(settings.cosmos_container_knowledge)
    return container.upsert_item(chunk)


def knowledge_search(
    query_vector: list[float],
    *,
    query_text: str | None = None,
    top_k: int = 5,
    service_filter: str | None = None,
) -> list[dict]:
    """
    Perform hybrid RRF search on sre_knowledge (DiskANN + BM25).

    Searches historical incident knowledge (symptoms, root causes,
    resolutions, reasoning traces) for similar patterns.
    Falls back to vector-only if hybrid fails.

    Args:
        query_vector: The embedding vector to search with (768 dims).
        query_text: The original query text for BM25 keyword matching.
        top_k: Number of results to return.
        service_filter: Optional partition key filter (service_name).

    Returns:
        List of matching documents with similarity/RRF scores.
    """
    settings = get_settings()
    container = get_container(settings.cosmos_container_knowledge)

    # --- Attempt hybrid RRF search ---
    if query_text:
        try:
            query = """
            SELECT TOP @top_k
                c.id,
                c.doc_type,
                c.service_name,
                c.chunk_text,
                c.metadata,
                c.source_id
            FROM c
            ORDER BY RANK RRF(VectorDistance(c.embedding, @queryVector), FullTextScore(c.chunk_text, @queryText))
            """

            parameters = [
                {"name": "@top_k", "value": top_k},
                {"name": "@queryVector", "value": query_vector},
                {"name": "@queryText", "value": query_text},
            ]

            results = list(
                container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True,
                )
            )
            logger.info(f"Knowledge hybrid RRF returned {len(results)} results")
            return results
        except Exception as e:
            logger.warning(f"Knowledge hybrid search failed, falling back: {e}")

    # --- Fallback: vector-only ---
    query = """
    SELECT TOP @top_k
        c.id,
        c.doc_type,
        c.service_name,
        c.chunk_text,
        c.metadata,
        c.source_id,
        VectorDistance(c.embedding, @queryVector) AS similarity_score
    FROM c
    ORDER BY VectorDistance(c.embedding, @queryVector)
    """

    parameters = [
        {"name": "@top_k", "value": top_k},
        {"name": "@queryVector", "value": query_vector},
    ]

    results = list(
        container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        )
    )

    logger.info(f"Knowledge vector-only returned {len(results)} results")
    return results


def count_knowledge_chunks() -> int:
    """Count the number of indexed knowledge chunks."""
    settings = get_settings()
    container = get_container(settings.cosmos_container_knowledge)
    query = "SELECT VALUE COUNT(1) FROM c"
    results = list(
        container.query_items(
            query=query,
            enable_cross_partition_query=True,
        )
    )
    return results[0] if results else 0
