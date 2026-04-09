import os

import pytest

from app.providers import db_provider, llm_provider


pytestmark = pytest.mark.skipif(
    not os.getenv("RUN_LIVE_E2E"),
    reason="live semantic retrieval test requires external LLM/Cosmos services",
)


SEMANTIC_QUERIES = [
    "code that handles order creation and saves to database",
    "authentication token validation and user identity verification",
    "shopping cart persistence and session state management",
]


@pytest.mark.asyncio
async def test_semantic_retrieval_live_queries_return_matches():
    for query in SEMANTIC_QUERIES:
        embedding = await llm_provider.generate_embedding(
            query,
            task_type="RETRIEVAL_QUERY",
        )
        results = db_provider.vector_search(query_vector=embedding, top_k=3)

        assert results, f"No semantic matches returned for query: {query}"
        assert all("chunk_text" in item for item in results)
