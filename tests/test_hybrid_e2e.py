"""
Integration & E2E Test: Hybrid Search vs Vector-Only
Compares results from both strategies on real Cosmos DB data.
Validates that BM25 keyword matching improves retrieval for exact identifiers.
"""
import sys, asyncio
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.providers import llm_provider, db_provider

# Queries designed to show BM25's advantage: exact identifiers
# that semantic embedding alone might miss
TEST_CASES = [
    {
        "name": "Exact method name (BM25 advantage)",
        "query": "GetItemsByBrandId CatalogApi",
        "expect_keyword": "GetItemsByBrandId",
    },
    {
        "name": "Error class name (BM25 advantage)",
        "query": "OrderPaymentFailedIntegrationEvent payment",
        "expect_keyword": "PaymentFailed",
    },
    {
        "name": "Semantic-only (no exact keywords)",
        "query": "code that handles user login authentication flow",
        "expect_keyword": "Identity",
    },
    {
        "name": "Mixed: semantic + exact class",
        "query": "BasketService gRPC timeout handling",
        "expect_keyword": "BasketService",
    },
]


def _result_contains_keyword(results: list[dict], keyword: str) -> tuple[bool, int]:
    """Check if any result contains the keyword. Returns (found, position)."""
    for i, r in enumerate(results):
        text = f"{r.get('chunk_text', '')} {r.get('class_name', '')} {r.get('method_name', '')} {r.get('file_path', '')}"
        if keyword.lower() in text.lower():
            return True, i + 1
    return False, -1


async def run_integration_test():
    print("=" * 70)
    print("INTEGRATION TEST: Hybrid (RRF) vs Vector-Only")
    print("=" * 70)

    wins_hybrid = 0
    wins_vector = 0
    ties = 0

    for tc in TEST_CASES:
        print(f"\n--- {tc['name']} ---")
        print(f"  Query: {tc['query']}")

        embedding = await llm_provider.generate_embedding(
            tc["query"], task_type="RETRIEVAL_QUERY"
        )

        # Run hybrid (RRF)
        hybrid_results = db_provider.vector_search(
            query_vector=embedding,
            query_text=tc["query"],
            top_k=5,
        )

        # Run vector-only
        vector_results = db_provider.vector_search(
            query_vector=embedding,
            top_k=5,
        )

        keyword = tc["expect_keyword"]
        h_found, h_pos = _result_contains_keyword(hybrid_results, keyword)
        v_found, v_pos = _result_contains_keyword(vector_results, keyword)

        # Top-1 comparison
        h_top1 = f"{hybrid_results[0].get('class_name', '—')}.{hybrid_results[0].get('method_name', '—')}" if hybrid_results else "EMPTY"
        v_top1 = f"{vector_results[0].get('class_name', '—')}.{vector_results[0].get('method_name', '—')}" if vector_results else "EMPTY"

        print(f"  Hybrid top-1:  {h_top1}")
        print(f"  Vector top-1:  {v_top1}")
        print(f"  Keyword '{keyword}': hybrid={'#' + str(h_pos) if h_found else '❌'}, vector={'#' + str(v_pos) if v_found else '❌'}")

        if h_found and (not v_found or h_pos < v_pos):
            print(f"  → 🏆 HYBRID wins")
            wins_hybrid += 1
        elif v_found and (not h_found or v_pos < h_pos):
            print(f"  → 🏆 VECTOR wins")
            wins_vector += 1
        else:
            print(f"  → 🤝 TIE")
            ties += 1

    print(f"\n{'=' * 70}")
    print(f"SCOREBOARD: Hybrid={wins_hybrid}  Vector={wins_vector}  Ties={ties}")
    total = len(TEST_CASES)
    hybrid_pct = (wins_hybrid / total) * 100
    print(f"Hybrid win rate: {hybrid_pct:.0f}%")

    # E2E assertion: hybrid should beat or tie vector-only
    assert wins_hybrid >= wins_vector, (
        f"Hybrid search should perform at least as well as vector-only. "
        f"Got hybrid={wins_hybrid}, vector={wins_vector}"
    )
    print(f"\n✅ E2E ASSERTION PASSED: Hybrid search >= vector-only performance")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_integration_test())
