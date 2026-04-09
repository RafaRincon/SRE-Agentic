import os

import pytest

from app.providers import db_provider, llm_provider


pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_E2E"),
    reason="live retrieval test requires external LLM/Cosmos services",
)


QUERIES = [
    "NullReferenceException in OrdersController checkout",
    "gRPC BasketService AddItem timeout error",
    "Catalog.API GetItemsByBrand SQL query performance",
]


@pytest.mark.asyncio
async def test_retrieval_live_queries_return_ranked_chunks():
    for query in QUERIES:
        embedding = await llm_provider.generate_embedding(
            query,
            task_type="RETRIEVAL_QUERY",
        )
        results = db_provider.vector_search(query_vector=embedding, top_k=5)

        assert results, f"No results returned for query: {query}"
        assert all("file_path" in item for item in results)
        assert all("similarity_score" in item for item in results)
