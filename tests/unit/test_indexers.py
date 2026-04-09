import builtins
from datetime import datetime, timedelta, timezone

import pytest

import app.indexer.chunker as chunker_module
import app.indexer.knowledge_indexer as knowledge_indexer_module
import app.indexer.repo_indexer as repo_indexer_module
import app.indexer.seed_incidents as seed_incidents_module


class FakeIndexerSettings:
    eshop_cache_dir = ".test-eshop-cache"
    eshop_repo_url = "https://example.com/eshop.git"


class FakeNode:
    def __init__(
        self,
        node_type,
        *,
        text=None,
        children=None,
        start_byte=0,
        end_byte=0,
        start_point=(0, 0),
        end_point=(0, 0),
    ):
        self.type = node_type
        self.text = text
        self.children = children or []
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.start_point = start_point
        self.end_point = end_point


class FakeParser:
    def __init__(self, language, root_node):
        self.language = language
        self.root_node = root_node

    def parse(self, payload):
        return type("FakeTree", (), {"root_node": self.root_node})()


def test_detect_service_name_and_chunk_file_helpers():
    assert (
        chunker_module.detect_service_name(
            "src/Ordering.API/Controllers/OrdersController.cs"
        )
        == "Ordering.API"
    )
    assert chunker_module.detect_service_name("src/Unknown/file.txt") == "unknown"
    assert chunker_module.chunk_file("README.md", "   ") == []


def test_chunker_helper_functions():
    identifier_node = FakeNode(
        "method_declaration",
        children=[FakeNode("identifier", text=b"CreateOrder")],
    )
    nameless_node = FakeNode("method_declaration", children=[FakeNode("keyword", text=b"public")])

    assert chunker_module._generate_chunk_id("src/a.cs", 10) == chunker_module._generate_chunk_id(
        "src/a.cs", 10
    )
    assert chunker_module._get_identifier(identifier_node) == "CreateOrder"
    assert chunker_module._get_identifier(nameless_node) is None
    assert (
        chunker_module._extract_namespace("namespace Ordering.API.Controllers {}")
        == "Ordering.API.Controllers"
    )
    assert chunker_module._extract_namespace("class OrdersController {}") is None
    assert chunker_module._build_context_header(
        "Ordering.API",
        "OrdersController",
        "CreateOrder",
    ) == (
        "// Namespace: Ordering.API\n"
        "// Class: OrdersController\n"
        "// Method: CreateOrder\n"
    )


def test_chunk_file_uses_text_fallback_for_non_csharp():
    chunks = chunker_module.chunk_file(
        "docs/runbook.md",
        "Line one\n\nLine two\n\nLine three\n" * 20,
    )

    assert chunks
    assert all(chunk["language"] == "markdown" for chunk in chunks)
    assert all(chunk["chunk_type"] == "text_segment" for chunk in chunks)


def test_chunk_text_fallback_handles_find_errors():
    class BrokenContent(str):
        def find(self, sub, start=None, end=None):
            raise RuntimeError("broken find")

    chunks = chunker_module._chunk_text_fallback(
        "src/Ordering.API/Controllers/OrdersController.cs",
        BrokenContent("line one\nline two\nline three\n" * 30),
        "Ordering.API",
        "csharp",
    )

    assert chunks
    assert chunks[0]["start_line"] == 1
    assert chunks[0]["end_line"] == 1


def test_chunk_csharp_ast_falls_back_when_tree_sitter_is_unavailable(monkeypatch):
    monkeypatch.setattr(chunker_module, "TREE_SITTER_AVAILABLE", False)
    monkeypatch.setattr(
        chunker_module,
        "_chunk_text_fallback",
        lambda file_path, content, service_name, language: [{"id": "fallback"}],
    )

    chunks = chunker_module._chunk_csharp_ast("src/a.cs", "class A {}", "Ordering.API")

    assert chunks == [{"id": "fallback"}]


def test_chunk_csharp_ast_builds_semantic_chunks_and_subchunks(monkeypatch):
    large_body = (
        "namespace Ordering.API;\n"
        "public class OrdersController {\n"
        + ("x" * 2500)
        + "\n}"
    )
    identifier = FakeNode("identifier", text=b"CreateOrder")
    method_node = FakeNode(
        "method_declaration",
        children=[identifier],
        start_byte=0,
        end_byte=len(large_body),
        start_point=(4, 0),
        end_point=(40, 0),
    )
    class_identifier = FakeNode("identifier", text=b"OrdersController")
    class_node = FakeNode(
        "class_declaration",
        children=[class_identifier, method_node],
    )
    root_node = FakeNode("root", children=[class_node])

    monkeypatch.setattr(chunker_module, "TREE_SITTER_AVAILABLE", True)
    monkeypatch.setattr(chunker_module, "CSHARP_LANG", object())
    monkeypatch.setattr(
        chunker_module,
        "Parser",
        lambda language: FakeParser(language, root_node),
        raising=False,
    )

    class FakeSplitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def split_text(self, text):
            assert text.startswith("// Namespace: Ordering.API")
            return ["part one", "part two"]

    monkeypatch.setattr(chunker_module, "RecursiveCharacterTextSplitter", FakeSplitter)

    chunks = chunker_module._chunk_csharp_ast(
        "src/Ordering.API/Controllers/OrdersController.cs",
        large_body,
        "Ordering.API",
    )

    assert [chunk["method_name"] for chunk in chunks] == ["CreateOrder_part1", "CreateOrder_part2"]
    assert all(chunk["class_name"] == "OrdersController" for chunk in chunks)
    assert all(chunk["chunk_type"] == "method_declaration" for chunk in chunks)


def test_chunk_csharp_ast_skips_tiny_properties_and_falls_back_when_no_chunks(monkeypatch):
    small_body = "namespace Ordering.API;\npublic class OrdersController { public string Name { get; set; } }"
    identifier = FakeNode("identifier", text=b"Name")
    property_node = FakeNode(
        "property_declaration",
        children=[identifier],
        start_byte=0,
        end_byte=20,
        start_point=(1, 0),
        end_point=(1, 10),
    )
    root_node = FakeNode("root", children=[property_node])

    monkeypatch.setattr(chunker_module, "TREE_SITTER_AVAILABLE", True)
    monkeypatch.setattr(chunker_module, "CSHARP_LANG", object())
    monkeypatch.setattr(
        chunker_module,
        "Parser",
        lambda language: FakeParser(language, root_node),
        raising=False,
    )
    monkeypatch.setattr(
        chunker_module,
        "_chunk_text_fallback",
        lambda file_path, content, service_name, language: [{"id": "fallback-no-chunks"}],
    )

    chunks = chunker_module._chunk_csharp_ast(
        "src/Ordering.API/Controllers/OrdersController.cs",
        small_body,
        "Ordering.API",
    )

    assert chunks == [{"id": "fallback-no-chunks"}]


def test_walk_repo_filters_noise_and_build_artifacts(tmp_path):
    repo = tmp_path / "repo"
    src = repo / "src" / "Ordering.API"
    build = repo / "bin"
    tests_dir = repo / "tests"
    src.mkdir(parents=True)
    build.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    (src / "OrdersController.cs").write_text("class A {}\n" * 20, encoding="utf-8")
    (src / "GlobalUsings.cs").write_text("global using System;", encoding="utf-8")
    (build / "Ignored.cs").write_text("class Ignored {}", encoding="utf-8")
    (tests_dir / "Ignored.md").write_text("ignored\n" * 20, encoding="utf-8")
    (src / "tiny.md").write_text("short", encoding="utf-8")

    files = chunker_module.walk_repo(str(repo))

    assert files == [
        (
            "src/Ordering.API/OrdersController.cs",
            (src / "OrdersController.cs").read_text(encoding="utf-8"),
        )
    ]


def test_walk_repo_skips_unsupported_extensions_and_logs_read_errors(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    src = repo / "src" / "Ordering.API"
    src.mkdir(parents=True)

    readable = src / "README.md"
    unreadable = src / "broken.md"
    ignored = src / "notes.txt"

    readable.write_text("useful docs\n" * 20, encoding="utf-8")
    unreadable.write_text("broken docs\n" * 20, encoding="utf-8")
    ignored.write_text("ignore me\n" * 20, encoding="utf-8")

    warnings = []
    original_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if str(path).endswith("broken.md"):
            raise OSError("cannot read")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    monkeypatch.setattr(chunker_module.logger, "warning", lambda message: warnings.append(message))

    files = chunker_module.walk_repo(str(repo))

    assert files == [("src/Ordering.API/README.md", readable.read_text(encoding="utf-8"))]
    assert warnings and "Skipping src/Ordering.API/broken.md" in warnings[0]


def test_extract_tags_and_reasoning_trace():
    incident = {
        "entities": {
            "error_code": "500",
            "error_message": "NullReferenceException in CreateOrder",
            "endpoint_affected": "/api/orders",
        },
        "world_model": {
            "incident_category": "RuntimeException",
            "affected_service": "Ordering.API",
            "blast_radius": ["WebApp", "OrderProcessor"],
        },
    }
    tags = knowledge_indexer_module._extract_tags(incident)

    assert "500" in tags
    assert "NullReferenceException" in tags
    assert "Ordering.API" in tags
    assert "WebApp" in tags

    trace = knowledge_indexer_module._build_reasoning_trace(
        "inc-1",
        "Ordering.API",
        [
            {
                "timestamp": "2026-04-08T12:00:00+00:00",
                "event_type": "STATE_TRANSITION",
                "data": {"from_state": "RECEIVED", "to_state": "TRIAGING"},
            },
            {
                "timestamp": "2026-04-08T12:01:00+00:00",
                "event_type": "HYPOTHESIS_GENERATED",
                "node_name": "risk_hypothesizer",
                "data": {
                    "description": "Missing null guard",
                    "suspected_file": "OrdersController.cs",
                    "confidence": 0.9,
                },
            },
        ],
    )

    assert "REASONING TRACE" in trace
    assert "Missing null guard" in trace


def test_compute_mttr_and_temporal_decay():
    created_at = datetime.now(timezone.utc) - timedelta(minutes=20)
    resolved_at = created_at + timedelta(minutes=12)
    mttr = knowledge_indexer_module._compute_mttr(
        {"created_at": created_at.isoformat()},
        [
            {
                "event_type": "STATE_TRANSITION",
                "timestamp": resolved_at.isoformat(),
                "data": {"to_state": "RESOLVED"},
            }
        ],
    )

    results = knowledge_indexer_module.apply_temporal_decay(
        [
            {"id": "recent", "metadata": {"created_at": datetime.now(timezone.utc).isoformat()}},
            {"id": "old", "metadata": {"created_at": "2020-01-01T00:00:00+00:00"}},
            {"id": "code", "metadata": {}},
        ]
    )

    assert mttr == 12
    assert results[0]["id"] in {"recent", "code"}
    assert all("temporal_weight" in item for item in results)


def test_build_resolution_chunks_creates_expected_roles():
    incident = {
        "incident_id": "inc-1",
        "created_at": "2026-04-08T12:00:00+00:00",
        "final_severity": "HIGH",
        "raw_report": "Checkout fails with HTTP 500",
        "triage_summary": "Missing null guard in OrdersController.",
        "verified_root_causes": ["OrdersController.cs: Missing null guard"],
        "entities": {
            "error_code": "500",
            "error_message": "NullReferenceException",
            "endpoint_affected": "/api/orders",
        },
        "world_model": {
            "affected_service": "Ordering.API",
            "incident_category": "RuntimeException",
            "blast_radius": ["WebApp"],
        },
    }
    ledger_entries = [
        {
            "event_type": "HYPOTHESIS_GENERATED",
            "timestamp": "2026-04-08T12:01:00+00:00",
            "data": {"description": "Missing null guard"},
        },
        {
            "event_type": "SPAN_VERDICT",
            "timestamp": "2026-04-08T12:02:00+00:00",
            "data": {
                "verdict": "VERIFIED",
                "matched_file": "OrdersController.cs",
                "matched_line": 42,
                "similarity_score": 0.91,
            },
        },
    ]

    chunks = knowledge_indexer_module._build_resolution_chunks(
        incident,
        ledger_entries,
        resolution_notes="Added null guard",
    )
    roles = [chunk["metadata"]["chunk_role"] for chunk in chunks]

    assert roles == ["symptom", "root_cause", "resolution", "reasoning_trace"]
    assert any("verified spans" in chunk["chunk_text"] for chunk in chunks)


@pytest.mark.asyncio
async def test_index_resolved_incident_success(monkeypatch):
    upserted = []
    monkeypatch.setattr(
        knowledge_indexer_module.db_provider,
        "get_ledger_entries",
        lambda incident_id: [],
    )
    monkeypatch.setattr(
        knowledge_indexer_module,
        "_build_resolution_chunks",
        lambda incident, ledger_entries, resolution_notes: [
            {"chunk_text": "symptom", "metadata": {"chunk_role": "symptom"}},
            {"chunk_text": "resolution", "metadata": {"chunk_role": "resolution"}},
        ],
    )

    async def fake_generate_embeddings_batch(texts, task_type):
        assert task_type == "RETRIEVAL_DOCUMENT"
        return [[0.1], [0.2]]

    monkeypatch.setattr(
        knowledge_indexer_module.llm_provider,
        "generate_embeddings_batch",
        fake_generate_embeddings_batch,
    )
    monkeypatch.setattr(
        knowledge_indexer_module.db_provider,
        "upsert_knowledge_chunk",
        lambda chunk: upserted.append(chunk),
    )

    stats = await knowledge_indexer_module.index_resolved_incident(
        {"incident_id": "inc-1"},
        resolution_notes="Added null guard",
    )

    assert stats["chunks_indexed"] == 2
    assert stats["chunk_roles"] == ["symptom", "resolution"]
    assert len(upserted) == 2


@pytest.mark.asyncio
async def test_index_resolved_incident_returns_error_when_embedding_fails(monkeypatch):
    monkeypatch.setattr(
        knowledge_indexer_module.db_provider,
        "get_ledger_entries",
        lambda incident_id: [],
    )
    monkeypatch.setattr(
        knowledge_indexer_module,
        "_build_resolution_chunks",
        lambda incident, ledger_entries, resolution_notes: [
            {"chunk_text": "symptom", "metadata": {"chunk_role": "symptom"}}
        ],
    )

    async def fake_generate_embeddings_batch(texts, task_type):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(
        knowledge_indexer_module.llm_provider,
        "generate_embeddings_batch",
        fake_generate_embeddings_batch,
    )

    stats = await knowledge_indexer_module.index_resolved_incident({"incident_id": "inc-1"})

    assert stats["chunks_indexed"] == 0
    assert stats["error"] == "embedding failed"


@pytest.mark.asyncio
async def test_clone_repo_returns_cached_repo(monkeypatch, tmp_path):
    cache_dir = tmp_path / "eshop"
    (cache_dir / ".git").mkdir(parents=True)
    settings = FakeIndexerSettings()
    settings.eshop_cache_dir = str(cache_dir)

    monkeypatch.setattr(repo_indexer_module, "get_settings", lambda: settings)

    path = await repo_indexer_module.clone_repo()

    assert path == str(cache_dir)


@pytest.mark.asyncio
async def test_clone_repo_clones_when_missing(monkeypatch, tmp_path):
    calls = []
    settings = FakeIndexerSettings()
    settings.eshop_cache_dir = str(tmp_path / "eshop")

    monkeypatch.setattr(repo_indexer_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        repo_indexer_module.git.Repo,
        "clone_from",
        lambda url, cache_dir, depth, single_branch, branch: calls.append(
            (url, cache_dir, depth, single_branch, branch)
        ),
    )

    path = await repo_indexer_module.clone_repo()

    assert path == settings.eshop_cache_dir
    assert calls == [
        (settings.eshop_repo_url, settings.eshop_cache_dir, 1, True, "main")
    ]


@pytest.mark.asyncio
async def test_index_repo_skips_when_chunks_exist(monkeypatch):
    monkeypatch.setattr(repo_indexer_module.db_provider, "count_chunks", lambda: 5)

    stats = await repo_indexer_module.index_repo()

    assert stats == {"status": "skipped", "existing_chunks": 5}


@pytest.mark.asyncio
async def test_index_repo_embeds_and_upserts_chunks(monkeypatch):
    async def fake_clone_repo():
        return "/tmp/repo"

    monkeypatch.setattr(repo_indexer_module, "clone_repo", fake_clone_repo)
    monkeypatch.setattr(
        repo_indexer_module,
        "walk_repo",
        lambda repo_path: [("src/a.cs", "content-a"), ("src/b.cs", "content-b")],
    )
    monkeypatch.setattr(
        repo_indexer_module,
        "chunk_file",
        lambda rel_path, content: [
            {"id": f"{rel_path}-1", "chunk_text": f"{content}-1", "embedding": []},
            {"id": f"{rel_path}-2", "chunk_text": f"{content}-2", "embedding": []},
        ],
    )
    monkeypatch.setattr(repo_indexer_module, "BATCH_SIZE", 2)

    async def fake_generate_embeddings_batch(texts, task_type):
        if "content-b-1" in texts[0]:
            raise RuntimeError("batch failed")
        return [[0.1, 0.2, 0.3] for _ in texts]

    upserted = []
    monkeypatch.setattr(
        repo_indexer_module.llm_provider,
        "generate_embeddings_batch",
        fake_generate_embeddings_batch,
    )
    monkeypatch.setattr(
        repo_indexer_module.db_provider,
        "upsert_chunk",
        lambda chunk: upserted.append(chunk),
    )

    stats = await repo_indexer_module.index_repo(force=True)

    assert stats["status"] == "completed"
    assert stats["files_processed"] == 2
    assert stats["chunks_generated"] == 4
    assert stats["chunks_embedded"] == 2
    assert stats["chunks_indexed"] == 2
    assert len(upserted) == 2


def test_build_seed_and_runbook_chunks():
    seed_chunks = seed_incidents_module._build_seed_chunks(seed_incidents_module.SEED_INCIDENTS[0])
    runbook_chunks = seed_incidents_module._build_runbook_chunks(seed_incidents_module.SEED_RUNBOOKS[0])

    assert [chunk["metadata"]["chunk_role"] for chunk in seed_chunks] == [
        "symptom",
        "root_cause",
        "resolution",
    ]
    assert runbook_chunks[0]["doc_type"] == "RUNBOOK"
    assert runbook_chunks[0]["metadata"]["chunk_role"] == "runbook"


@pytest.mark.asyncio
async def test_seed_historical_incidents_indexes_incidents_and_runbooks(monkeypatch):
    upserted = []

    async def fake_generate_embeddings_batch(texts, task_type):
        return [[0.1] for _ in texts]

    monkeypatch.setattr(
        seed_incidents_module.llm_provider,
        "generate_embeddings_batch",
        fake_generate_embeddings_batch,
    )
    monkeypatch.setattr(
        seed_incidents_module.db_provider,
        "upsert_knowledge_chunk",
        lambda chunk: upserted.append(chunk),
    )

    stats = await seed_incidents_module.seed_historical_incidents()

    assert stats["status"] == "completed"
    assert stats["incidents_seeded"] == len(seed_incidents_module.SEED_INCIDENTS)
    assert stats["runbooks_seeded"] == len(seed_incidents_module.SEED_RUNBOOKS)
    assert stats["chunks_indexed"] == len(upserted)
    assert "Ordering.API" in stats["services_covered"]


@pytest.mark.asyncio
async def test_seed_historical_incidents_continues_when_embeddings_fail(monkeypatch):
    calls = {"count": 0}

    async def fake_generate_embeddings_batch(texts, task_type):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("embedding failed")
        return [[0.1] for _ in texts]

    upserted = []
    monkeypatch.setattr(
        seed_incidents_module.llm_provider,
        "generate_embeddings_batch",
        fake_generate_embeddings_batch,
    )
    monkeypatch.setattr(
        seed_incidents_module.db_provider,
        "upsert_knowledge_chunk",
        lambda chunk: upserted.append(chunk),
    )

    stats = await seed_incidents_module.seed_historical_incidents()

    assert stats["incidents_seeded"] == len(seed_incidents_module.SEED_INCIDENTS) - 1
    assert stats["runbooks_seeded"] == len(seed_incidents_module.SEED_RUNBOOKS)
    assert upserted
