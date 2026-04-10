from __future__ import annotations

"""
SRE Agent — Seed Historical Incidents

Generates realistic synthetic incident data for the eShop application
and indexes them into the sre_knowledge container. This provides the
flywheel with initial "memory" so retrieval has historical
correlation from the first local run.

These incidents are designed to cover common eShop failure modes:
- NullReferenceException in OrdersController
- Timeout in Payment processing
- Catalog API returning stale data
- Basket API concurrency issues
- OrderProcessor message queue failures
"""

import logging
from datetime import datetime, timezone, timedelta

from app.indexer.knowledge_indexer import (
    _generate_chunk_id,
    _extract_tags,
)
from app.providers import llm_provider, db_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic incident data
# ---------------------------------------------------------------------------

SEED_INCIDENTS = [
    {
        "incident_id": "seed-001",
        "service_name": "Ordering.API",
        "severity": "HIGH",
        "error_code": "HTTP 500",
        "error_message": "NullReferenceException at OrdersController.CreateOrderAsync",
        "endpoint": "/api/orders",
        "category": "RuntimeException",
        "blast_radius": ["WebApp", "OrderProcessor"],
        "root_causes": [
            "src/Ordering.API/Controllers/OrdersController.cs: Null check missing "
            "for basket items before creating order — basketItems can be null when "
            "session expires mid-checkout"
        ],
        "triage_summary": (
            "HIGH severity incident in Ordering.API. NullReferenceException thrown "
            "in CreateOrderAsync when basket items are null due to expired session. "
            "1 verified root cause in OrdersController.cs. Impact extended to WebApp "
            "and OrderProcessor via failed order events."
        ),
        "resolution_notes": (
            "Added null check for basketItems in CreateOrderAsync. Also added "
            "session validation middleware to return 401 before reaching controller "
            "when session is expired."
        ),
        "mttr_minutes": 45,
        "days_ago": 30,
    },
    {
        "incident_id": "seed-002",
        "service_name": "Ordering.API",
        "severity": "MEDIUM",
        "error_code": "HTTP 500",
        "error_message": "NullReferenceException in OrdersController.GetOrderAsync — order not found",
        "endpoint": "/api/orders/{id}",
        "category": "RuntimeException",
        "blast_radius": ["WebApp"],
        "root_causes": [
            "src/Ordering.API/Controllers/OrdersController.cs: GetOrderAsync returns "
            "null when order ID doesn't exist, but controller doesn't handle null "
            "response from repository, causing NullReferenceException on serialization"
        ],
        "triage_summary": (
            "MEDIUM severity incident in Ordering.API. GetOrderAsync throws "
            "NullReferenceException when querying non-existent order IDs. Root cause "
            "verified in OrdersController.cs — missing null guard on repository response."
        ),
        "resolution_notes": (
            "Added null check after repository.GetOrderAsync() call. Returns 404 "
            "NotFound instead of letting serialization fail. Added unit test for "
            "non-existent order ID scenario."
        ),
        "mttr_minutes": 25,
        "days_ago": 15,
    },
    {
        "incident_id": "seed-003",
        "service_name": "Payment.API",
        "severity": "CRITICAL",
        "error_code": "TIMEOUT",
        "error_message": "Payment gateway timeout after 30s — orders stuck in PENDING",
        "endpoint": "/api/payments/process",
        "category": "Timeout",
        "blast_radius": ["Ordering.API", "OrderProcessor", "WebApp"],
        "root_causes": [
            "src/Payment.API/Services/PaymentService.cs: HttpClient timeout not "
            "configured — defaults to 100s. Payment gateway intermittently slow, "
            "causing thread pool exhaustion when multiple payments queue up"
        ],
        "triage_summary": (
            "CRITICAL severity incident in Payment.API. Payment gateway timeout "
            "causing cascading failures. HttpClient default timeout of 100s leads "
            "to thread pool exhaustion. Orders stuck in PENDING state across "
            "Ordering.API and OrderProcessor."
        ),
        "resolution_notes": (
            "Set HttpClient timeout to 10s with Polly retry policy (3 retries, "
            "exponential backoff). Added circuit breaker pattern. Implemented "
            "payment status reconciliation job for stuck orders."
        ),
        "mttr_minutes": 120,
        "days_ago": 60,
    },
    {
        "incident_id": "seed-004",
        "service_name": "Catalog.API",
        "severity": "LOW",
        "error_code": "HTTP 200 (stale data)",
        "error_message": "Catalog prices showing outdated values after bulk update",
        "endpoint": "/api/catalog/items",
        "category": "DataStaleness",
        "blast_radius": ["WebApp"],
        "root_causes": [
            "src/Catalog.API/Services/CatalogService.cs: In-memory cache not "
            "invalidated after bulk price update via admin endpoint. Cache TTL "
            "set to 1 hour but admin updates are immediate"
        ],
        "triage_summary": (
            "LOW severity incident in Catalog.API. Catalog prices stale after "
            "bulk update because in-memory cache TTL of 1hr is not invalidated "
            "on admin writes. Users see old prices until cache expires."
        ),
        "resolution_notes": (
            "Added cache invalidation on CatalogService.UpdateItemsAsync(). "
            "Admin bulk update endpoint now publishes CacheInvalidation event "
            "via EventBus to all Catalog.API instances."
        ),
        "mttr_minutes": 60,
        "days_ago": 45,
    },
    {
        "incident_id": "seed-005",
        "service_name": "Basket.API",
        "severity": "HIGH",
        "error_code": "HTTP 409",
        "error_message": "Concurrency conflict updating basket — ETag mismatch",
        "endpoint": "/api/basket",
        "category": "ConcurrencyConflict",
        "blast_radius": ["WebApp"],
        "root_causes": [
            "src/Basket.API/Services/BasketService.cs: Redis optimistic concurrency "
            "using WATCH/MULTI not handling race condition when user adds items "
            "from multiple browser tabs simultaneously"
        ],
        "triage_summary": (
            "HIGH severity incident in Basket.API. Concurrent basket updates from "
            "multiple tabs cause ETag mismatch errors. Redis WATCH/MULTI pattern "
            "in BasketService doesn't retry on conflict, just throws 409."
        ),
        "resolution_notes": (
            "Implemented retry-on-conflict with 3 attempts and jittered backoff "
            "in BasketService.UpdateBasketAsync(). Added merge logic for concurrent "
            "add-item operations instead of failing."
        ),
        "mttr_minutes": 90,
        "days_ago": 20,
    },
    {
        "incident_id": "seed-006",
        "service_name": "Ordering.API",
        "severity": "HIGH",
        "error_code": "HTTP 500",
        "error_message": "NullReferenceException in OrdersController — shipping address null",
        "endpoint": "/api/orders",
        "category": "RuntimeException",
        "blast_radius": ["WebApp", "OrderProcessor"],
        "root_causes": [
            "src/Ordering.API/Controllers/OrdersController.cs: CreateOrderAsync "
            "doesn't validate shipping address field before passing to "
            "OrderService. Guest checkout flow omits address validation step"
        ],
        "triage_summary": (
            "HIGH severity incident in Ordering.API. Third NullReferenceException "
            "in OrdersController in 30 days. This time caused by null shipping "
            "address from guest checkout flow. ⚠️ RECURRING PATTERN: OrdersController "
            "has insufficient null guards across multiple code paths."
        ),
        "resolution_notes": (
            "Added FluentValidation validator for CreateOrderCommand. Now validates "
            "all required fields before controller logic executes. Recommended "
            "architectural review of OrdersController null safety patterns."
        ),
        "mttr_minutes": 35,
        "days_ago": 7,
    },
]


# ---------------------------------------------------------------------------
# Seed Runbooks — operational action guides
# ---------------------------------------------------------------------------

SEED_RUNBOOKS = [
    {
        "runbook_id": "RB-001",
        "title": "NullReferenceException in Ordering.API",
        "service_name": "Ordering.API",
        "trigger_pattern": "NullReferenceException AND (OrdersController OR CreateOrderAsync OR GetOrderAsync)",
        "severity_filter": ["HIGH", "CRITICAL"],
        "steps": [
            "1. Check recent deployments: `git log --since='4 hours ago' -- src/Ordering.API/`",
            "2. Validate null guards in OrdersController.cs — check basketItems, shippingAddress, and paymentMethod fields",
            "3. Verify user session state: check if sessions are expiring prematurely (Redis TTL)",
            "4. Restart affected pods: `kubectl rollout restart deployment ordering-api`",
            "5. Monitor error rate for 10 minutes post-restart",
            "6. If recurring (3+ times in 30 days): escalate to architecture review of OrdersController null safety",
        ],
        "estimated_resolution_time": "30-60 minutes",
        "escalation_path": "Order Team → Platform Team (if infra-related)",
    },
    {
        "runbook_id": "RB-002",
        "title": "Payment Gateway Timeout",
        "service_name": "Payment.API",
        "trigger_pattern": "Timeout AND (PaymentService OR payment gateway OR HttpClient)",
        "severity_filter": ["CRITICAL", "HIGH"],
        "steps": [
            "1. Check payment gateway status page: https://status.paymentgateway.example.com",
            "2. Verify HttpClient timeout configuration in PaymentService.cs (should be ≤10s, not default 100s)",
            "3. Enable circuit breaker if not active: set PAYMENT_CIRCUIT_BREAKER=true",
            "4. Run payment reconciliation job for stuck orders: `POST /api/payments/reconcile`",
            "5. Monitor thread pool metrics: `dotnet-counters monitor --process-id <PID>`",
            "6. If gateway is down: activate fallback payment processor",
        ],
        "estimated_resolution_time": "60-120 minutes",
        "escalation_path": "Payments Team → External vendor (if gateway issue)",
    },
    {
        "runbook_id": "RB-003",
        "title": "Catalog Data Staleness / Cache Invalidation",
        "service_name": "Catalog.API",
        "trigger_pattern": "stale data OR cache OR outdated prices OR CatalogService",
        "severity_filter": ["LOW", "MEDIUM"],
        "steps": [
            "1. Flush Redis cache for Catalog: `redis-cli FLUSHDB` on catalog-cache instance",
            "2. Verify EventBus connectivity: check RabbitMQ management UI for CacheInvalidation queue",
            "3. Force cache re-sync: `POST /api/catalog/cache/invalidate`",
            "4. Check if admin bulk update endpoint triggered cache invalidation event",
            "5. Review CatalogService.cs cache TTL settings (should be ≤15min for price-sensitive data)",
            "6. Monitor catalog response freshness for 30 minutes",
        ],
        "estimated_resolution_time": "15-30 minutes",
        "escalation_path": "Catalog Team",
    },
    {
        "runbook_id": "RB-004",
        "title": "Basket Concurrency Conflict (ETag / Optimistic Locking)",
        "service_name": "Basket.API",
        "trigger_pattern": "409 Conflict OR ETag mismatch OR concurrency OR WATCH/MULTI",
        "severity_filter": ["HIGH", "MEDIUM"],
        "steps": [
            "1. Check Redis cluster health: `redis-cli CLUSTER INFO` — look for cluster_state:ok",
            "2. Review BasketService.cs: ensure retry-on-conflict with ≥3 attempts and jittered backoff",
            "3. Check WATCH/MULTI logs for excessive retries: `redis-cli MONITOR | grep WATCH`",
            "4. Scale Basket.API replicas if load is high: `kubectl scale deployment basket-api --replicas=4`",
            "5. Verify client-side retry logic: ensure frontend retries 409s gracefully",
            "6. If persistent under normal load: review Redis key design for hot-key issues",
        ],
        "estimated_resolution_time": "45-90 minutes",
        "escalation_path": "Cart Team → Platform Team (if Redis issue)",
    },
]


# ---------------------------------------------------------------------------
# Seed logic
# ---------------------------------------------------------------------------


def _build_seed_chunks(incident_data: dict) -> list[dict]:
    """Build knowledge chunks from a synthetic incident definition."""
    now = datetime.now(timezone.utc)
    created_at = (now - timedelta(days=incident_data["days_ago"])).isoformat()
    resolved_at = (
        now - timedelta(days=incident_data["days_ago"])
        + timedelta(minutes=incident_data["mttr_minutes"])
    ).isoformat()

    service = incident_data["service_name"]
    incident_id = incident_data["incident_id"]
    severity = incident_data["severity"]

    base_metadata = {
        "severity": severity,
        "resolution_status": "RESOLVED",
        "created_at": created_at,
        "resolved_at": resolved_at,
        "mttr_minutes": incident_data["mttr_minutes"],
        "tags": [
            service,
            incident_data["error_code"],
            incident_data["category"],
        ],
    }

    base = {
        "doc_type": "TICKET",
        "service_name": service,
        "source_id": f"incident-{incident_id}",
    }

    chunks = []

    # SYMPTOM
    symptom_text = (
        f"INCIDENT SYMPTOM — {service}\n"
        f"Severity: {severity}\n"
        f"Error: {incident_data['error_code']} {incident_data['error_message']}\n"
        f"Endpoint: {incident_data['endpoint']}\n"
        f"Category: {incident_data['category']}\n"
        f"Blast Radius: {', '.join(incident_data['blast_radius'])}"
    )
    chunks.append({
        **base,
        "id": _generate_chunk_id(incident_id, "symptom"),
        "chunk_text": symptom_text,
        "metadata": {**base_metadata, "chunk_role": "symptom"},
    })

    # ROOT_CAUSE
    cause_text = (
        f"VERIFIED ROOT CAUSE — {service}\n"
        f"Incident: {incident_id}\n"
        f"Causes:\n" + "\n".join(f"  - {c}" for c in incident_data["root_causes"])
        + f"\nOriginal error: {incident_data['error_code']} "
        f"{incident_data['error_message']}\n"
        f"Severity: {severity}"
    )
    chunks.append({
        **base,
        "id": _generate_chunk_id(incident_id, "root_cause"),
        "chunk_text": cause_text,
        "metadata": {**base_metadata, "chunk_role": "root_cause"},
    })

    # RESOLUTION
    resolution_text = (
        f"RESOLUTION — {service}\n"
        f"Incident: {incident_id}\n"
        f"Triage Summary: {incident_data['triage_summary']}\n"
        f"Human Resolution Notes: {incident_data['resolution_notes']}\n"
        f"Verified causes: {', '.join(incident_data['root_causes'][:1])}\n"
        f"Severity: {severity}\n"
        f"MTTR: {incident_data['mttr_minutes']} minutes"
    )
    chunks.append({
        **base,
        "id": _generate_chunk_id(incident_id, "resolution"),
        "chunk_text": resolution_text,
        "metadata": {
            **base_metadata,
            "chunk_role": "resolution",
            "resolution_notes": incident_data["resolution_notes"],
        },
    })

    return chunks


def _build_runbook_chunks(runbook_data: dict) -> list[dict]:
    """Build a single knowledge chunk from a runbook definition."""
    runbook_id = runbook_data["runbook_id"]
    service = runbook_data["service_name"]

    steps_text = "\n".join(runbook_data["steps"])
    chunk_text = (
        f"RUNBOOK {runbook_id}: {runbook_data['title']}\n"
        f"Service: {service}\n"
        f"Trigger: {runbook_data['trigger_pattern']}\n"
        f"Applicable severities: {', '.join(runbook_data['severity_filter'])}\n"
        f"Estimated resolution time: {runbook_data['estimated_resolution_time']}\n"
        f"Escalation: {runbook_data['escalation_path']}\n\n"
        f"STEPS:\n{steps_text}"
    )

    return [{
        "id": _generate_chunk_id(runbook_id, "runbook"),
        "doc_type": "RUNBOOK",
        "service_name": service,
        "source_id": f"runbook-{runbook_id}",
        "chunk_text": chunk_text,
        "metadata": {
            "chunk_role": "runbook",
            "runbook_id": runbook_id,
            "trigger_pattern": runbook_data["trigger_pattern"],
            "severity_filter": runbook_data["severity_filter"],
            "estimated_resolution_time": runbook_data["estimated_resolution_time"],
            "escalation_path": runbook_data["escalation_path"],
        },
    }]


async def seed_historical_incidents() -> dict:
    """
    Generate and index synthetic historical incidents AND runbooks.
    Returns stats about the seeding process.
    """
    total_chunks = 0
    total_incidents = 0
    total_runbooks = 0

    # --- Seed incidents ---
    for incident_data in SEED_INCIDENTS:
        chunks = _build_seed_chunks(incident_data)

        # Generate embeddings
        texts = [c["chunk_text"] for c in chunks]
        try:
            embeddings = await llm_provider.generate_embeddings_batch(
                texts, task_type="RETRIEVAL_DOCUMENT"
            )
            for chunk, emb in zip(chunks, embeddings):
                chunk["embedding"] = emb
        except Exception as e:
            logger.error(
                f"[seed] Embedding failed for {incident_data['incident_id']}: {e}"
            )
            continue

        # Upsert to sre_knowledge
        for chunk in chunks:
            try:
                db_provider.upsert_knowledge_chunk(chunk)
                total_chunks += 1
            except Exception as e:
                logger.warning(f"[seed] Upsert failed: {e}")

        total_incidents += 1
        logger.info(
            f"[seed] Indexed {incident_data['incident_id']} "
            f"({len(chunks)} chunks, service={incident_data['service_name']})"
        )

    # --- Seed runbooks ---
    for runbook_data in SEED_RUNBOOKS:
        chunks = _build_runbook_chunks(runbook_data)

        texts = [c["chunk_text"] for c in chunks]
        try:
            embeddings = await llm_provider.generate_embeddings_batch(
                texts, task_type="RETRIEVAL_DOCUMENT"
            )
            for chunk, emb in zip(chunks, embeddings):
                chunk["embedding"] = emb
        except Exception as e:
            logger.error(
                f"[seed] Embedding failed for {runbook_data['runbook_id']}: {e}"
            )
            continue

        for chunk in chunks:
            try:
                db_provider.upsert_knowledge_chunk(chunk)
                total_chunks += 1
            except Exception as e:
                logger.warning(f"[seed] Runbook upsert failed: {e}")

        total_runbooks += 1
        logger.info(
            f"[seed] Indexed runbook {runbook_data['runbook_id']} "
            f"(service={runbook_data['service_name']})"
        )

    logger.info(
        f"[seed] ✅ Seeded {total_incidents} incidents + {total_runbooks} runbooks, "
        f"{total_chunks} total knowledge chunks"
    )

    return {
        "status": "completed",
        "incidents_seeded": total_incidents,
        "runbooks_seeded": total_runbooks,
        "chunks_indexed": total_chunks,
        "services_covered": list(set(
            i["service_name"] for i in SEED_INCIDENTS
        )),
    }
