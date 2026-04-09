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
from azure.cosmos.exceptions import CosmosHttpResponseError, CosmosResourceNotFoundError
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


def reset_clients() -> None:
    """Clear cached SDK clients. Useful in tests."""
    global _cosmos_client, _containers
    _cosmos_client = None
    _containers = {}


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


def _vector_embedding_policy(dimensions: int) -> dict[str, Any]:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": dimensions,
            }
        ]
    }


def _vector_indexing_policy() -> dict[str, Any]:
    return {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [{"path": '/"_etag"/?'}],
        "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
        "fullTextIndexes": [{"path": "/chunk_text"}],
    }


def _full_text_policy() -> dict[str, Any]:
    return {
        "defaultLanguage": "en-US",
        "fullTextPaths": [
            {"path": "/chunk_text", "language": "en-US"},
        ],
    }


def get_container_definitions() -> dict[str, dict[str, Any]]:
    """Return the Cosmos DB container contract required by the application."""
    settings = get_settings()
    vector_policy = _vector_embedding_policy(settings.gemini_embedding_dimensions)
    vector_indexing = _vector_indexing_policy()
    full_text = _full_text_policy()

    return {
        settings.cosmos_container_chunks: {
            "partition_key": "/service_name",
            "indexing_policy": vector_indexing,
            "vector_embedding_policy": vector_policy,
            "full_text_policy": full_text,
        },
        settings.cosmos_container_knowledge: {
            "partition_key": "/service_name",
            "indexing_policy": vector_indexing,
            "vector_embedding_policy": vector_policy,
            "full_text_policy": full_text,
        },
        settings.cosmos_container_incidents: {
            "partition_key": "/incident_id",
            "indexing_policy": {
                "indexingMode": "consistent",
                "automatic": True,
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": [{"path": '/"_etag"/?'}],
            },
        },
        settings.cosmos_container_ledger: {
            "partition_key": "/incident_id",
            "indexing_policy": {
                "indexingMode": "consistent",
                "automatic": True,
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": [{"path": '/"_etag"/?'}],
            },
        },
        settings.cosmos_container_checkpoints: {
            "partition_key": "/partition_key",
            "indexing_policy": {
                "indexingMode": "consistent",
                "automatic": True,
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": [{"path": '/"_etag"/?'}],
            },
        },
    }


def _verify_container_properties(
    *,
    container_name: str,
    properties: dict[str, Any],
    expected: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []

    actual_partition_paths = properties.get("partitionKey", {}).get("paths", [])
    actual_partition_path = actual_partition_paths[0] if actual_partition_paths else None
    if actual_partition_path != expected["partition_key"]:
        warnings.append(
            f"{container_name}: partition key mismatch "
            f"(expected {expected['partition_key']}, got {actual_partition_path})"
        )

    indexing_policy = properties.get("indexingPolicy", {})
    expected_indexing = expected.get("indexing_policy", {})

    expected_vector_indexes = expected_indexing.get("vectorIndexes", [])
    actual_vector_indexes = indexing_policy.get("vectorIndexes", [])
    if expected_vector_indexes and actual_vector_indexes != expected_vector_indexes:
        warnings.append(f"{container_name}: vectorIndexes differ from expected policy")

    expected_full_text_indexes = expected_indexing.get("fullTextIndexes", [])
    actual_full_text_indexes = indexing_policy.get("fullTextIndexes", [])
    if expected_full_text_indexes and actual_full_text_indexes != expected_full_text_indexes:
        warnings.append(f"{container_name}: fullTextIndexes differ from expected policy")

    expected_full_text_policy = expected.get("full_text_policy")
    actual_full_text_policy = properties.get("fullTextPolicy")
    if expected_full_text_policy and actual_full_text_policy != expected_full_text_policy:
        warnings.append(f"{container_name}: fullTextPolicy differs from expected policy")

    expected_vector_policy = expected.get("vector_embedding_policy")
    actual_vector_policy = properties.get("vectorEmbeddingPolicy")
    if expected_vector_policy and actual_vector_policy != expected_vector_policy:
        warnings.append(f"{container_name}: vectorEmbeddingPolicy differs from expected policy")

    return warnings


def ensure_database_and_containers() -> dict[str, Any]:
    """
    Create the database and required containers if missing.

    Existing container policies are verified and surfaced as warnings to avoid
    destructive replacements during a production bootstrap.
    """
    settings = get_settings()
    client = _get_cosmos_client()
    definitions = get_container_definitions()

    database = client.create_database_if_not_exists(id=settings.cosmos_database)
    created: list[str] = []
    verified: list[str] = []
    warnings: list[str] = []

    for container_name, definition in definitions.items():
        db = client.get_database_client(settings.cosmos_database)
        before_exists = True
        try:
            existing = db.get_container_client(container_name)
            properties = existing.read()
        except CosmosResourceNotFoundError:
            before_exists = False
            properties = None
        except CosmosHttpResponseError as exc:
            before_exists = False
            logger.warning("[bootstrap] Container lookup failed for %s: %s", container_name, exc)
            properties = None

        kwargs = {
            "id": container_name,
            "partition_key": PartitionKey(path=definition["partition_key"]),
            "indexing_policy": definition.get("indexing_policy"),
            "vector_embedding_policy": definition.get("vector_embedding_policy"),
            "full_text_policy": definition.get("full_text_policy"),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        container = database.create_container_if_not_exists(**kwargs)
        _containers[container_name] = container

        if not before_exists:
            created.append(container_name)
            properties = container.read()
        else:
            verified.append(container_name)

        warnings.extend(
            _verify_container_properties(
                container_name=container_name,
                properties=properties or container.read(),
                expected=definition,
            )
        )

    return {
        "database": settings.cosmos_database,
        "created_containers": created,
        "verified_containers": verified,
        "warnings": warnings,
    }


def get_runtime_health(*, require_index_ready: bool = False) -> dict[str, Any]:
    """
    Check database and container readiness for runtime traffic.
    """
    settings = get_settings()
    definitions = get_container_definitions()
    components: dict[str, dict[str, Any]] = {}
    ready = True

    try:
        db_properties = get_database().read()
        components["database"] = {
            "ready": True,
            "id": db_properties.get("id", settings.cosmos_database),
        }
    except Exception as exc:
        ready = False
        components["database"] = {"ready": False, "error": str(exc)}
        return {"ready": ready, "components": components}

    for container_name in definitions:
        try:
            properties = get_container(container_name).read()
            components[container_name] = {
                "ready": True,
                "id": properties.get("id", container_name),
            }
        except Exception as exc:
            ready = False
            components[container_name] = {"ready": False, "error": str(exc)}

    if require_index_ready and ready:
        try:
            chunk_count = count_chunks()
            index_ready = chunk_count > 0
            components["code_index"] = {
                "ready": index_ready,
                "chunk_count": chunk_count,
            }
            if not index_ready:
                ready = False
        except Exception as exc:
            ready = False
            components["code_index"] = {"ready": False, "error": str(exc)}

    return {"ready": ready, "components": components}


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
