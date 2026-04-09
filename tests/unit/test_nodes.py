import base64
import sys
import types

import pytest

import app.agents.nodes.actions as actions_module
import app.agents.nodes.consolidator as consolidator_module
import app.agents.nodes.intake as intake_module
import app.agents.nodes.risk_hypothesizer as risk_module
import app.agents.nodes.slot_filler as slot_filler_module
import app.agents.nodes.span_arbiter as span_arbiter_module
import app.agents.nodes.world_model as world_model_module
from app.agents.state import (
    ExtractedEntity,
    NotificationInfo,
    RiskHypothesis,
    Severity,
    WorldModelProjection,
)


@pytest.mark.asyncio
async def test_intake_node_rejects_short_reports():
    result = await intake_module.intake_node({"raw_report": "short"})

    assert result["errors"] == ["Report too short or empty. Minimum 10 characters."]


@pytest.mark.asyncio
async def test_intake_node_accepts_valid_report_and_normalizes_missing_image():
    result = await intake_module.intake_node(
        {
            "incident_id": "inc-1",
            "raw_report": "Checkout fails with HTTP 500 after deployment",
            "has_image": True,
            "image_data_b64": "",
        }
    )

    assert result["status"].value == "TRIAGING"
    assert result["has_image"] is False


@pytest.mark.asyncio
async def test_world_model_node_uses_structured_generation_without_image(monkeypatch):
    projection = WorldModelProjection(
        affected_service="Ordering.API",
        blast_radius=["WebApp"],
        estimated_severity=Severity.HIGH,
        incident_category="RuntimeException",
    )

    async def fake_generate_structured(**kwargs):
        assert kwargs["response_schema"] is WorldModelProjection
        return projection

    monkeypatch.setattr(
        world_model_module.llm_provider,
        "generate_structured",
        fake_generate_structured,
    )

    result = await world_model_module.world_model_node({"raw_report": "HTTP 500"})

    assert result["world_model"]["affected_service"] == "Ordering.API"


@pytest.mark.asyncio
async def test_world_model_node_uses_multimodal_with_image(monkeypatch):
    projection = WorldModelProjection(
        affected_service="WebApp",
        blast_radius=["Ordering.API"],
        estimated_severity=Severity.CRITICAL,
    )
    calls = []

    async def fake_generate_multimodal(**kwargs):
        calls.append(kwargs)
        return projection

    monkeypatch.setattr(
        world_model_module.llm_provider,
        "generate_multimodal",
        fake_generate_multimodal,
    )

    result = await world_model_module.world_model_node(
        {
            "raw_report": "Screenshot attached",
            "has_image": True,
            "image_mime_type": "image/png",
            "image_data_b64": base64.b64encode(b"fake-image").decode(),
        }
    )

    assert result["world_model"]["affected_service"] == "WebApp"
    assert calls[0]["image_bytes"] == b"fake-image"


@pytest.mark.asyncio
async def test_world_model_node_returns_error_when_llm_fails(monkeypatch):
    async def fake_generate_structured(**kwargs):
        raise RuntimeError("llm down")

    monkeypatch.setattr(
        world_model_module.llm_provider,
        "generate_structured",
        fake_generate_structured,
    )

    result = await world_model_module.world_model_node(
        {"raw_report": "HTTP 500", "errors": []}
    )

    assert result["errors"] == ["World Model failed: llm down"]


@pytest.mark.asyncio
async def test_slot_filler_node_extracts_entities(monkeypatch):
    entities = ExtractedEntity(
        error_code="500",
        error_message="NullReferenceException",
        endpoint_affected="/api/orders",
    )

    async def fake_generate_structured(**kwargs):
        assert kwargs["response_schema"] is ExtractedEntity
        return entities

    monkeypatch.setattr(
        slot_filler_module.llm_provider,
        "generate_structured",
        fake_generate_structured,
    )

    result = await slot_filler_module.slot_filler_node({"raw_report": "HTTP 500"})

    assert result["entities"]["error_code"] == "500"


@pytest.mark.asyncio
async def test_slot_filler_node_returns_error_on_failure(monkeypatch):
    async def fake_generate_structured(**kwargs):
        raise RuntimeError("schema mismatch")

    monkeypatch.setattr(
        slot_filler_module.llm_provider,
        "generate_structured",
        fake_generate_structured,
    )

    result = await slot_filler_module.slot_filler_node({"raw_report": "HTTP 500"})

    assert result["errors"] == ["Slot filler failed: schema mismatch"]


@pytest.mark.asyncio
async def test_span_arbiter_returns_empty_verdicts_without_hypotheses():
    result = await span_arbiter_module.span_arbiter_node({"hypotheses": []})

    assert result == {"span_verdicts": []}


@pytest.mark.asyncio
async def test_span_arbiter_verifies_matching_span_and_records_ledger(monkeypatch):
    recorded = []

    async def fake_generate_embedding(text, task_type):
        assert task_type == "RETRIEVAL_QUERY"
        return [0.1, 0.2]

    def fake_vector_search(**kwargs):
        return [
            {
                "file_path": "src/Ordering.API/OrdersController.cs",
                "chunk_text": "if (order == null) throw new NullReferenceException();",
                "start_line": 42,
            }
        ]

    monkeypatch.setattr(
        span_arbiter_module.llm_provider,
        "generate_embedding",
        fake_generate_embedding,
    )
    monkeypatch.setattr(
        span_arbiter_module.db_provider,
        "vector_search",
        fake_vector_search,
    )
    monkeypatch.setattr(
        span_arbiter_module,
        "record_verdict",
        lambda **kwargs: recorded.append(kwargs),
    )

    result = await span_arbiter_module.span_arbiter_node(
        {
            "incident_id": "inc-1",
            "hypotheses": [
                {
                    "hypothesis_id": "hyp-1",
                    "exact_span": "throw new NullReferenceException();",
                    "suspected_file": "OrdersController.cs",
                }
            ],
        }
    )

    verdict = result["span_verdicts"][0]
    assert verdict["verdict"] == "VERIFIED"
    assert verdict["matched_line"] == 42
    assert recorded[0]["incident_id"] == "inc-1"


@pytest.mark.asyncio
async def test_span_arbiter_marks_partial_match(monkeypatch):
    async def fake_generate_embedding(text, task_type):
        return [0.1]

    monkeypatch.setattr(
        span_arbiter_module.llm_provider,
        "generate_embedding",
        fake_generate_embedding,
    )
    monkeypatch.setattr(
        span_arbiter_module.db_provider,
        "vector_search",
        lambda **kwargs: [{"file_path": "src/file.cs", "chunk_text": "candidate", "start_line": 1}],
    )
    monkeypatch.setattr(
        span_arbiter_module,
        "fuzzy_match_span",
        lambda span, chunk_text, threshold=0.6: (False, 0.5),
    )
    monkeypatch.setattr(span_arbiter_module, "record_verdict", lambda **kwargs: kwargs)

    result = await span_arbiter_module.span_arbiter_node(
        {"hypotheses": [{"hypothesis_id": "hyp-1", "exact_span": "foo"}]}
    )

    assert result["span_verdicts"][0]["verdict"] == "PARTIAL_MATCH"


@pytest.mark.asyncio
async def test_span_arbiter_marks_error_when_retrieval_fails(monkeypatch):
    async def fake_generate_embedding(text, task_type):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(
        span_arbiter_module.llm_provider,
        "generate_embedding",
        fake_generate_embedding,
    )
    monkeypatch.setattr(span_arbiter_module, "record_verdict", lambda **kwargs: kwargs)

    result = await span_arbiter_module.span_arbiter_node(
        {"hypotheses": [{"hypothesis_id": "hyp-1", "exact_span": "foo"}]}
    )

    assert result["span_verdicts"][0]["verdict"] == "ERROR"


@pytest.mark.asyncio
async def test_risk_hypothesizer_manual_expand_uses_available_context():
    queries = risk_module._manual_expand(
        "Checkout fails",
        {"affected_service": "Ordering.API"},
        {
            "error_code": "500",
            "error_message": "NullReferenceException",
            "endpoint_affected": "/api/orders",
            "stack_trace": "OrdersController.Handle",
        },
    )

    assert "NullReferenceException 500" in queries
    assert "Ordering.API controller handler endpoint" in queries
    assert "/api/orders route handler action" in queries


@pytest.mark.asyncio
async def test_retrieve_with_expansion_deduplicates_and_skips_failed_queries(monkeypatch):
    embeddings = {
        "q1": [1.0],
        "q2": [2.0],
    }

    async def fake_generate_embedding(query, task_type):
        if query == "q3":
            raise RuntimeError("boom")
        return embeddings[query]

    def fake_vector_search(query_vector, top_k, service_filter):
        if query_vector == [1.0]:
            return [
                {"id": "a", "similarity_score": 0.2},
                {"id": "b", "similarity_score": 0.9},
            ]
        return [
            {"id": "b", "similarity_score": 0.7},
            {"id": "c", "similarity_score": 0.8},
        ]

    monkeypatch.setattr(
        risk_module.llm_provider,
        "generate_embedding",
        fake_generate_embedding,
    )
    monkeypatch.setattr(
        risk_module.db_provider,
        "vector_search",
        fake_vector_search,
    )

    result = await risk_module.retrieve_with_expansion(
        ["q1", "q2", "q3"],
        service_filter="Ordering.API",
    )

    assert [chunk["id"] for chunk in result] == ["b", "c", "a"]


@pytest.mark.asyncio
async def test_risk_hypothesizer_node_returns_grounded_hypotheses(monkeypatch):
    async def fake_expand_queries(*args, **kwargs):
        return ["primary query", "secondary query"]

    async def fake_retrieve_with_expansion(*args, **kwargs):
        return [
            {
                "file_path": "src/Ordering.API/OrdersController.cs",
                "start_line": 10,
                "end_line": 18,
                "chunk_text": "if (order == null) throw new NullReferenceException();",
            }
        ]

    async def fake_generate_embedding(*args, **kwargs):
        return [0.5]

    async def fake_generate_structured(**kwargs):
        return risk_module.HypothesesOutput(
            hypotheses=[
                RiskHypothesis(
                    hypothesis_id="h1",
                    description="Missing null guard",
                    suspected_file="src/Ordering.API/OrdersController.cs",
                    suspected_function="CreateOrder",
                    exact_span="throw new NullReferenceException();",
                    confidence=0.9,
                ),
                RiskHypothesis(
                    hypothesis_id="h2",
                    description="Discarded because span is blank",
                    suspected_file="src/Ordering.API/OrdersController.cs",
                    exact_span="   ",
                    confidence=0.1,
                ),
            ]
        )

    recorded = []
    monkeypatch.setattr(risk_module, "expand_queries", fake_expand_queries)
    monkeypatch.setattr(
        risk_module,
        "retrieve_with_expansion",
        fake_retrieve_with_expansion,
    )
    monkeypatch.setattr(
        risk_module.llm_provider,
        "generate_embedding",
        fake_generate_embedding,
    )
    monkeypatch.setattr(
        risk_module.llm_provider,
        "generate_structured",
        fake_generate_structured,
    )
    monkeypatch.setattr(
        risk_module.db_provider,
        "knowledge_search",
        lambda **kwargs: [
            {
                "source_id": "INC-123",
                "similarity_score": 0.85,
                "chunk_text": "Resolved by adding null checks.",
                "metadata": {
                    "severity": "HIGH",
                    "chunk_role": "resolution",
                    "resolution_notes": "Add null guards.",
                    "mttr_minutes": 15,
                },
            }
        ],
    )
    monkeypatch.setattr(
        risk_module,
        "record_hypothesis",
        lambda **kwargs: recorded.append(kwargs),
    )
    monkeypatch.setitem(
        sys.modules,
        "app.indexer.knowledge_indexer",
        types.SimpleNamespace(apply_temporal_decay=lambda chunks: chunks),
    )

    result = await risk_module.risk_hypothesizer_node(
        {
            "incident_id": "inc-1",
            "raw_report": "Checkout fails with NullReferenceException",
            "world_model": {"affected_service": "Ordering.API", "incident_category": "RuntimeException"},
            "entities": {"error_code": "500", "error_message": "NullReferenceException"},
        }
    )

    assert len(result["hypotheses"]) == 1
    assert result["historical_context"]["recurrence_count"] == 1
    assert recorded[0]["incident_id"] == "inc-1"


@pytest.mark.asyncio
async def test_risk_hypothesizer_node_returns_error_when_generation_fails(monkeypatch):
    async def fake_expand_queries(*args, **kwargs):
        return ["primary query"]

    async def fake_retrieve_with_expansion(*args, **kwargs):
        return []

    async def fake_generate_embedding(*args, **kwargs):
        return [0.1]

    async def fake_generate_structured(**kwargs):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(risk_module, "expand_queries", fake_expand_queries)
    monkeypatch.setattr(
        risk_module,
        "retrieve_with_expansion",
        fake_retrieve_with_expansion,
    )
    monkeypatch.setattr(
        risk_module.llm_provider,
        "generate_embedding",
        fake_generate_embedding,
    )
    monkeypatch.setattr(
        risk_module.db_provider,
        "knowledge_search",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        risk_module.llm_provider,
        "generate_structured",
        fake_generate_structured,
    )

    result = await risk_module.risk_hypothesizer_node(
        {"raw_report": "Checkout fails", "world_model": {}, "entities": {}, "errors": []}
    )

    assert result["errors"] == ["Risk hypothesizer failed: generation failed"]


@pytest.mark.asyncio
async def test_consolidator_node_merges_verified_causes_and_records_ledger(monkeypatch):
    ledger_calls = []

    async def fake_generate_text(**kwargs):
        return "Ordering.API failed because a verified null guard was missing."

    monkeypatch.setattr(
        consolidator_module.llm_provider,
        "generate_text",
        fake_generate_text,
    )
    monkeypatch.setattr(
        consolidator_module,
        "record_state_transition",
        lambda **kwargs: ledger_calls.append(("transition", kwargs)),
    )
    monkeypatch.setattr(
        consolidator_module,
        "record_entry",
        lambda **kwargs: ledger_calls.append(("entry", kwargs)),
    )

    result = await consolidator_module.consolidator_node(
        {
            "incident_id": "inc-1",
            "world_model": {
                "affected_service": "Ordering.API",
                "incident_category": "RuntimeException",
                "blast_radius": ["WebApp"],
                "estimated_severity": "UNKNOWN",
            },
            "entities": {
                "error_code": "500",
                "error_message": "NullReferenceException",
                "endpoint_affected": "/api/orders",
            },
            "hypotheses": [
                {
                    "hypothesis_id": "h1",
                    "suspected_file": "OrdersController.cs",
                    "description": "Missing null guard",
                    "exact_span": "throw new NullReferenceException();",
                },
                {
                    "hypothesis_id": "h2",
                    "suspected_file": "PaymentService.cs",
                    "description": "Noise hypothesis",
                    "exact_span": "paymentClient.DoThing();",
                },
            ],
            "span_verdicts": [
                {"hypothesis_id": "h1", "verdict": "VERIFIED"},
                {"hypothesis_id": "h2", "verdict": "HALLUCINATION"},
            ],
            "historical_context": {"similar_past_incidents": [], "recurrence_count": 0},
        }
    )

    assert result["status"].value == "TRIAGED"
    assert result["verified_root_causes"] == ["OrdersController.cs: Missing null guard"]
    assert ledger_calls


@pytest.mark.asyncio
async def test_consolidator_formats_historical_context_variants():
    assert (
        consolidator_module._format_historical_context({})
        == "- No historical data available (knowledge base may be empty)."
    )
    assert (
        consolidator_module._format_historical_context(
            {"similar_past_incidents": [], "recurrence_count": 0}
        )
        == "- No similar past incidents found. This appears to be a first occurrence."
    )

    formatted = consolidator_module._format_historical_context(
        {
            "similar_past_incidents": [
                {
                    "source_id": "INC-1",
                    "severity": "HIGH",
                    "mttr_minutes": 12,
                    "resolution_notes": "Restart worker and clear poisoned messages.",
                },
                {
                    "source_id": "INC-2",
                    "severity": "MEDIUM",
                },
            ],
            "recurrence_count": 3,
        }
    )

    assert "Found 2 similar past incidents (3 highly similar):" in formatted
    assert "Resolution: Restart worker and clear poisoned messages." in formatted
    assert "RECURRING ISSUE" in formatted


@pytest.mark.asyncio
async def test_consolidator_searches_runbooks_and_falls_back_on_summary_failure(monkeypatch):
    async def fake_generate_embedding(text, task_type):
        assert "RUNBOOK steps escalation:" in text
        return [0.8]

    async def fake_generate_text(**kwargs):
        raise RuntimeError("summary offline")

    monkeypatch.setattr(
        consolidator_module.llm_provider,
        "generate_embedding",
        fake_generate_embedding,
    )
    monkeypatch.setattr(
        consolidator_module.db_provider,
        "knowledge_search",
        lambda **kwargs: [
            {
                "doc_type": "RUNBOOK",
                "chunk_text": "RB-42\n1. Roll back deployment\n2. Restart ordering pods",
                "metadata": {
                    "runbook_id": "RB-42",
                    "escalation_path": "Order Team -> Platform",
                    "estimated_resolution_time": "15m",
                },
            },
            {
                "doc_type": "NOTE",
                "chunk_text": "non-runbook note",
                "metadata": {},
            },
        ],
    )
    monkeypatch.setattr(
        consolidator_module.llm_provider,
        "generate_text",
        fake_generate_text,
    )
    monkeypatch.setattr(
        consolidator_module,
        "record_state_transition",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("ledger unavailable")),
    )

    result = await consolidator_module.consolidator_node(
        {
            "incident_id": "inc-2",
            "raw_report": "Ordering.API returns HTTP 500 during checkout after deployment",
            "world_model": {
                "affected_service": "Ordering.API",
                "incident_category": "RuntimeException",
                "blast_radius": ["WebApp", "OrderProcessor", "Basket.API"],
                "estimated_severity": "UNKNOWN",
            },
            "entities": {
                "error_code": "500",
                "error_message": "",
                "endpoint_affected": "/api/orders",
            },
            "hypotheses": [],
            "span_verdicts": [],
            "historical_context": {
                "similar_past_incidents": [
                    {"source_id": "INC-7", "severity": "HIGH", "resolution_notes": "Rollback"}
                ],
                "recurrence_count": 1,
            },
        }
    )

    assert result["final_severity"] == "CRITICAL"
    assert result["triage_summary"].startswith("AUTOMATED TRIAGE: CRITICAL severity incident")
    assert result["suggested_runbooks"] == [
        {
            "runbook_id": "RB-42",
            "title": "RB-42",
            "escalation_path": "Order Team -> Platform",
            "estimated_resolution_time": "15m",
        }
    ]


@pytest.mark.asyncio
async def test_create_ticket_and_notify_team_nodes():
    created = await actions_module.create_ticket_node(
        {
            "incident_id": "inc-1",
            "world_model": {"affected_service": "Ordering.API"},
            "triage_summary": "Critical checkout outage.",
            "final_severity": "CRITICAL",
        }
    )
    notified = await actions_module.notify_team_node(
        {
            "incident_id": "inc-1",
            "ticket": created["ticket"],
            "triage_summary": "Critical checkout outage.",
            "final_severity": "CRITICAL",
        }
    )

    assert created["status"].value == "TICKET_CREATED"
    assert created["ticket"]["assigned_team"] == "Order Team"
    assert notified["status"].value == "TEAM_NOTIFIED"
    assert notified["notifications"]["team_notified"] is True
