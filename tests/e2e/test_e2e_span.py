import os
import sys

import pytest
from pydantic import BaseModel, Field

from app.agents.nodes.risk_hypothesizer import expand_queries
from app.agents.state import RiskHypothesis
from app.providers import db_provider, llm_provider
from app.symbolic.span_matcher import fuzzy_match_span


INCIDENT = {
    "raw_report": (
        "CRITICAL: Checkout completely broken. Users get HTTP 500 when clicking Place Order. "
        "POST /api/v1/orders fails. Stack trace: System.NullReferenceException at "
        "Ordering.API.Application.Commands.CreateOrderCommandHandler.Handle(). "
        "Started 20 minutes after last deployment. 2000 users affected per hour. "
        "Payment service healthy. Cosmos DB connections fine."
    ),
    "world_model": {
        "affected_service": "Ordering.API",
        "incident_category": "NULL_REFERENCE",
        "estimated_severity": "CRITICAL",
        "blast_radius": ["WebApp", "Basket.API", "OrderProcessor"],
    },
    "entities": {
        "error_code": "500",
        "error_message": "NullReferenceException",
        "endpoint_affected": "POST /api/v1/orders",
        "stack_trace": "Ordering.API.Application.Commands.CreateOrderCommandHandler.Handle()",
        "file_references": [],
    },
}


class HypothesesOutput(BaseModel):
    hypotheses: list[RiskHypothesis] = Field(default_factory=list)


async def run_span_pipeline(
    raw_report: str,
    world_model: dict,
    entities: dict,
) -> dict:
    expanded = await expand_queries(raw_report, world_model, entities)

    service_filter = world_model.get("affected_service")
    seen_ids = set()
    retrieved = []

    for query in expanded:
        embedding = await llm_provider.generate_embedding(
            query,
            task_type="RETRIEVAL_QUERY",
        )
        chunks = db_provider.vector_search(
            query_vector=embedding,
            top_k=5,
            service_filter=service_filter,
        )
        for chunk in chunks:
            chunk_id = chunk.get("id")
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                retrieved.append(chunk)

    retrieved.sort(key=lambda chunk: chunk.get("similarity_score", 0), reverse=True)
    top_chunks = retrieved[:15]

    code_context = "\n\n".join(
        f"--- FILE: {chunk.get('file_path', 'unknown')} "
        f"(lines {chunk.get('start_line', '?')}-{chunk.get('end_line', '?')}) ---\n"
        f"{chunk.get('chunk_text', '')}"
        for chunk in top_chunks
    )

    hypotheses_result = await llm_provider.generate_structured(
        prompt=(
            "Based on this incident and the REAL CODE, generate root cause hypotheses.\n\n"
            f"INCIDENT:\n{raw_report}\n\n"
            f"SERVICE: {world_model.get('affected_service')} | "
            f"ERROR: {entities.get('error_message')} | "
            f"ENDPOINT: {entities.get('endpoint_affected')}\n\n"
            f"REAL CODE ({len(top_chunks)} chunks):\n{code_context}\n\n"
            "IMPORTANT: exact_span must be a VERBATIM copy from the code above."
        ),
        response_schema=HypothesesOutput,
        system_instruction=(
            "You are a PARANOID SRE investigator. "
            "You MUST base your hypotheses ONLY on the CODE CHUNKS provided."
        ),
    )

    hypotheses = [
        hypothesis
        for hypothesis in hypotheses_result.hypotheses
        if hypothesis.exact_span and len(hypothesis.exact_span.strip()) > 5
    ]

    verdicts = []
    for hypothesis in hypotheses:
        span_embedding = await llm_provider.generate_embedding(
            hypothesis.exact_span,
            task_type="RETRIEVAL_QUERY",
        )
        candidates = db_provider.vector_search(
            query_vector=span_embedding,
            top_k=5,
            service_filter=None,
        )

        suspected_file = hypothesis.suspected_file or ""
        file_candidates = [
            candidate
            for candidate in candidates
            if suspected_file and suspected_file in candidate.get("file_path", "")
        ]
        if file_candidates:
            candidates = file_candidates

        best_score = 0.0
        best_match = False
        matched_file = None
        matched_line = None

        for candidate in candidates:
            matched, score = fuzzy_match_span(
                hypothesis.exact_span,
                candidate.get("chunk_text", ""),
                threshold=0.6,
            )
            if score > best_score:
                best_score = score
                best_match = matched
                matched_file = candidate.get("file_path")
                matched_line = candidate.get("start_line")

        if best_match and best_score >= 0.6:
            verdict = "VERIFIED"
        elif best_score >= 0.4:
            verdict = "PARTIAL_MATCH"
        else:
            verdict = "HALLUCINATION"

        verdicts.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "verdict": verdict,
                "score": best_score,
                "matched_file": matched_file,
                "matched_line": matched_line,
            }
        )

    return {
        "expanded_queries": expanded,
        "top_chunks": top_chunks,
        "hypotheses": hypotheses,
        "verdicts": verdicts,
    }


@pytest.mark.asyncio
async def test_e2e_span_pipeline_is_grounded_and_deterministic(monkeypatch):
    expanded_queries = [
        "NullReferenceException checkout handler",
        "Ordering.API order creation endpoint",
        "null guard missing in order command",
        "dependency injection order workflow",
    ]
    embedding_ids = {}

    async def fake_expand_queries(raw_report, world_model, entities):
        assert world_model["affected_service"] == "Ordering.API"
        assert entities["error_message"] == "NullReferenceException"
        return expanded_queries

    async def fake_generate_embedding(text, task_type):
        assert task_type == "RETRIEVAL_QUERY"
        if text not in embedding_ids:
            embedding_ids[text] = float(len(embedding_ids) + 1)
        return [embedding_ids[text]]

    query_results = {
        expanded_queries[0]: [
            {
                "id": "chunk-a",
                "file_path": "src/Ordering.API/Application/Commands/CreateOrderCommandHandler.cs",
                "chunk_text": "if (order == null) throw new NullReferenceException();",
                "start_line": 42,
                "end_line": 47,
                "similarity_score": 0.92,
            }
        ],
        expanded_queries[1]: [
            {
                "id": "chunk-b",
                "file_path": "src/Ordering.API/Controllers/OrdersController.cs",
                "chunk_text": "return await mediator.Send(command);",
                "start_line": 18,
                "end_line": 24,
                "similarity_score": 0.88,
            }
        ],
        expanded_queries[2]: [
            {
                "id": "chunk-a",
                "file_path": "src/Ordering.API/Application/Commands/CreateOrderCommandHandler.cs",
                "chunk_text": "if (order == null) throw new NullReferenceException();",
                "start_line": 42,
                "end_line": 47,
                "similarity_score": 0.91,
            }
        ],
        expanded_queries[3]: [
            {
                "id": "chunk-c",
                "file_path": "src/Ordering.API/DependencyInjection.cs",
                "chunk_text": "services.AddScoped<IOrderService, OrderService>();",
                "start_line": 10,
                "end_line": 14,
                "similarity_score": 0.75,
            }
        ],
        "throw new NullReferenceException();": [
            {
                "id": "verify-a",
                "file_path": "src/Ordering.API/Application/Commands/CreateOrderCommandHandler.cs",
                "chunk_text": "if (order == null) throw new NullReferenceException();",
                "start_line": 42,
                "end_line": 47,
                "similarity_score": 0.95,
            }
        ],
        "imaginary line from nowhere": [
            {
                "id": "verify-b",
                "file_path": "src/Ordering.API/Application/Commands/CreateOrderCommandHandler.cs",
                "chunk_text": "if (order == null) throw new NullReferenceException();",
                "start_line": 42,
                "end_line": 47,
                "similarity_score": 0.20,
            }
        ],
    }

    def fake_vector_search(query_vector, top_k, service_filter):
        query_text = next(
            text for text, embedding_id in embedding_ids.items() if embedding_id == query_vector[0]
        )
        return query_results[query_text]

    async def fake_generate_structured(prompt, response_schema, system_instruction):
        assert response_schema is HypothesesOutput
        assert "REAL CODE" in prompt
        return HypothesesOutput(
            hypotheses=[
                RiskHypothesis(
                    hypothesis_id="h-verified",
                    description="Missing null guard causes checkout failure",
                    suspected_file="CreateOrderCommandHandler.cs",
                    suspected_function="Handle",
                    exact_span="throw new NullReferenceException();",
                    confidence=0.91,
                    epistemic_snapshot={
                        "observed": [
                            {
                                "label": "exact_span=throw new NullReferenceException();",
                                "status": "OBSERVED",
                                "evidence": "throw new NullReferenceException();",
                                "source": "risk_hypothesizer",
                            }
                        ],
                        "inferred": [
                            {
                                "label": "Missing null guard causes checkout failure",
                                "status": "INFERRED",
                                "evidence": "Derived from the retrieved code path.",
                                "source": "risk_hypothesizer",
                            }
                        ],
                        "unknown": [
                            {
                                "label": "upstream_validation_or_wiring",
                                "status": "UNKNOWN",
                                "evidence": "Bootstrap and upstream validation were not inspected in this test.",
                                "source": "risk_hypothesizer",
                            }
                        ],
                    },
                ),
                RiskHypothesis(
                    hypothesis_id="h-hallucinated",
                    description="Invented code path with no grounding",
                    suspected_file="CreateOrderCommandHandler.cs",
                    suspected_function="Handle",
                    exact_span="imaginary line from nowhere",
                    confidence=0.25,
                ),
            ]
        )

    monkeypatch.setattr(sys.modules[__name__], "expand_queries", fake_expand_queries)
    monkeypatch.setattr(
        llm_provider,
        "generate_embedding",
        fake_generate_embedding,
    )
    monkeypatch.setattr(
        db_provider,
        "vector_search",
        fake_vector_search,
    )
    monkeypatch.setattr(
        llm_provider,
        "generate_structured",
        fake_generate_structured,
    )

    result = await run_span_pipeline(
        INCIDENT["raw_report"],
        INCIDENT["world_model"],
        INCIDENT["entities"],
    )

    assert result["expanded_queries"] == expanded_queries
    assert [chunk["id"] for chunk in result["top_chunks"]] == ["chunk-a", "chunk-b", "chunk-c"]
    assert [hypothesis.hypothesis_id for hypothesis in result["hypotheses"]] == [
        "h-verified",
        "h-hallucinated",
    ]
    assert result["hypotheses"][0].epistemic_snapshot.unknown[0].label == "upstream_validation_or_wiring"
    assert result["verdicts"][0] == {
        "hypothesis_id": "h-verified",
        "verdict": "VERIFIED",
        "score": 1.0,
        "matched_file": "src/Ordering.API/Application/Commands/CreateOrderCommandHandler.cs",
        "matched_line": 42,
    }
    assert result["verdicts"][1]["hypothesis_id"] == "h-hallucinated"
    assert result["verdicts"][1]["verdict"] == "HALLUCINATION"
    assert result["verdicts"][1]["matched_file"] == (
        "src/Ordering.API/Application/Commands/CreateOrderCommandHandler.cs"
    )
    assert result["verdicts"][1]["matched_line"] == 42
    assert result["verdicts"][1]["score"] < 0.4


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_E2E"),
    reason="live span e2e requires external LLM/Cosmos services",
)
@pytest.mark.asyncio
async def test_e2e_span_live():
    result = await run_span_pipeline(
        INCIDENT["raw_report"],
        INCIDENT["world_model"],
        INCIDENT["entities"],
    )

    assert result["expanded_queries"]
    assert result["hypotheses"]
    assert result["verdicts"]
