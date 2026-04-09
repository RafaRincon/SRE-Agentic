import types

import pytest
from pydantic import BaseModel

import app.providers.db_provider as db_provider_module
import app.providers.llm_provider as llm_provider_module


class ExampleSchema(BaseModel):
    message: str


class FakeSettings:
    gemini_api_key = "test-key"
    gemini_model = "gemini-test"
    gemini_embedding_model = "embedding-test"
    gemini_embedding_dimensions = 3
    cosmos_endpoint = "https://cosmos.local"
    cosmos_key = "cosmos-key"
    cosmos_database = "db"
    cosmos_container_chunks = "chunks"
    cosmos_container_incidents = "incidents"
    cosmos_container_ledger = "ledger"
    cosmos_container_knowledge = "knowledge"


class FakeLLMModels:
    def __init__(self):
        self.generate_calls = []
        self.embed_calls = []

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        response_text = kwargs["contents"]
        if isinstance(response_text, list):
            response_text = '{"message":"structured from multimodal"}'
        elif "structured" in response_text:
            response_text = '{"message":"structured output"}'
        return types.SimpleNamespace(text=response_text if response_text.startswith("{") else "plain output")

    def embed_content(self, **kwargs):
        self.embed_calls.append(kwargs)
        contents = kwargs["contents"]
        if isinstance(contents, list):
            embeddings = [
                types.SimpleNamespace(values=[float(index), float(index + 1), float(index + 2)])
                for index, _ in enumerate(contents, start=1)
            ]
        else:
            embeddings = [types.SimpleNamespace(values=[0.1, 0.2, 0.3])]
        return types.SimpleNamespace(embeddings=embeddings)


class FakeLLMClient:
    def __init__(self):
        self.models = FakeLLMModels()


class FakeContainer:
    def __init__(self, *, query_results=None, query_map=None, read_item_result=None):
        self.query_results = query_results or []
        self.query_map = query_map or {}
        self.read_item_result = read_item_result
        self.query_calls = []
        self.upserted_items = []
        self.created_items = []

    def query_items(self, **kwargs):
        self.query_calls.append(kwargs)
        query = kwargs.get("query", "")
        for needle, result in self.query_map.items():
            if needle in query:
                return result
        return self.query_results

    def upsert_item(self, item):
        self.upserted_items.append(item)
        return item

    def read_item(self, item, partition_key):
        if isinstance(self.read_item_result, Exception):
            raise self.read_item_result
        return self.read_item_result

    def create_item(self, item):
        self.created_items.append(item)
        return item


def test_get_client_caches_singleton(monkeypatch):
    created = []

    def fake_client(api_key):
        created.append(api_key)
        return FakeLLMClient()

    llm_provider_module._client = None
    monkeypatch.setattr(llm_provider_module, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(llm_provider_module.genai, "Client", fake_client)

    first = llm_provider_module._get_client()
    second = llm_provider_module._get_client()

    assert first is second
    assert created == ["test-key"]


@pytest.mark.asyncio
async def test_llm_provider_generation_helpers(monkeypatch):
    client = FakeLLMClient()
    llm_provider_module._client = client
    monkeypatch.setattr(llm_provider_module, "get_settings", lambda: FakeSettings())

    text = await llm_provider_module.generate_text("plain prompt", system_instruction="sys")
    structured = await llm_provider_module.generate_structured(
        "structured prompt",
        ExampleSchema,
    )
    multimodal = await llm_provider_module.generate_multimodal(
        "prompt",
        image_bytes=b"img",
        response_schema=ExampleSchema,
    )

    assert text == "plain output"
    assert structured.message == "structured output"
    assert multimodal.message == "structured from multimodal"
    assert client.models.generate_calls[0]["config"].system_instruction == "sys"


@pytest.mark.asyncio
async def test_llm_provider_embedding_helpers(monkeypatch):
    client = FakeLLMClient()
    llm_provider_module._client = client
    monkeypatch.setattr(llm_provider_module, "get_settings", lambda: FakeSettings())

    embedding = await llm_provider_module.generate_embedding("prompt", task_type="RETRIEVAL_QUERY")
    batch = await llm_provider_module.generate_embeddings_batch(
        ["one", "two"],
        task_type="RETRIEVAL_QUERY",
    )

    assert embedding == [0.1, 0.2, 0.3]
    assert batch == [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
    assert client.models.embed_calls[0]["config"].task_type == "RETRIEVAL_QUERY"


def test_db_provider_client_and_container_caching(monkeypatch):
    created = []

    class FakeDatabase:
        def __init__(self):
            self.container_requests = []

        def get_container_client(self, name):
            self.container_requests.append(name)
            return {"name": name}

    fake_database = FakeDatabase()

    class FakeCosmosClient:
        def __init__(self, url, credential):
            created.append((url, credential))

        def get_database_client(self, name):
            assert name == "db"
            return fake_database

    db_provider_module._cosmos_client = None
    db_provider_module._containers = {}
    monkeypatch.setattr(db_provider_module, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(db_provider_module, "CosmosClient", FakeCosmosClient)

    client = db_provider_module._get_cosmos_client()
    database = db_provider_module.get_database()
    first = db_provider_module.get_container("chunks")
    second = db_provider_module.get_container("chunks")

    assert client is db_provider_module._cosmos_client
    assert database is fake_database
    assert first == second == {"name": "chunks"}
    assert created == [("https://cosmos.local", "cosmos-key")]


def test_db_provider_query_and_mutation_helpers(monkeypatch):
    containers = {
        "chunks": FakeContainer(
            query_results=[{"id": "c1", "similarity_score": 0.9}],
            query_map={"SELECT VALUE COUNT(1)": [2]},
        ),
        "incidents": FakeContainer(
            query_results=[{"id": "inc-1"}],
            read_item_result={"id": "inc-1"},
        ),
        "ledger": FakeContainer(query_results=[{"event_type": "STATE_TRANSITION"}]),
        "knowledge": FakeContainer(
            query_results=[{"id": "k1", "similarity_score": 0.8}],
            query_map={"SELECT VALUE COUNT(1)": [5]},
        ),
    }

    monkeypatch.setattr(db_provider_module, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        db_provider_module,
        "get_container",
        lambda name: containers[name],
    )

    results = db_provider_module.vector_search([0.1], top_k=2)
    incident = db_provider_module.upsert_incident({"id": "inc-1"})
    fetched = db_provider_module.get_incident("inc-1")
    incidents = db_provider_module.list_incidents(limit=10)
    ledger_entry = db_provider_module.append_ledger_entry({"incident_id": "inc-1"})
    ledger = db_provider_module.get_ledger_entries("inc-1")
    chunk = db_provider_module.upsert_chunk({"id": "chunk-1"})
    count = db_provider_module.count_chunks()
    knowledge_chunk = db_provider_module.upsert_knowledge_chunk({"id": "know-1"})
    knowledge_results = db_provider_module.knowledge_search([0.2], top_k=3)
    knowledge_count = db_provider_module.count_knowledge_chunks()

    assert results[0]["id"] == "c1"
    assert incident["id"] == "inc-1"
    assert fetched["id"] == "inc-1"
    assert incidents == [{"id": "inc-1"}]
    assert ledger_entry["incident_id"] == "inc-1"
    assert ledger == [{"event_type": "STATE_TRANSITION"}]
    assert chunk["id"] == "chunk-1"
    assert count == 2
    assert knowledge_chunk["id"] == "know-1"
    assert knowledge_results[0]["id"] == "k1"
    assert knowledge_count == 5


def test_get_incident_returns_none_when_read_fails(monkeypatch):
    containers = {
        "incidents": FakeContainer(read_item_result=RuntimeError("missing")),
    }

    monkeypatch.setattr(db_provider_module, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(
        db_provider_module,
        "get_container",
        lambda name: containers[name],
    )

    assert db_provider_module.get_incident("missing") is None
