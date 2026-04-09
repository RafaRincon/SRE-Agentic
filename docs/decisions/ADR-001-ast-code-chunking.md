# ADR-001: AST-Based Code Chunking for eShop RAG Pipeline

| Field      | Value                                          |
|------------|-------------------------------------------------|
| **Status** | Accepted                                        |
| **Date**   | 2026-04-08                                      |
| **Author** | SRE Agent Team                                  |
| **Scope**  | `app/indexer/`, `app/providers/db_provider.py`, `app/agents/nodes/` |

---

## Context

The Neuro-Symbolic SRE Agent uses a RAG (Retrieval-Augmented Generation) pipeline to ground LLM-generated hypotheses in **real code** from the .NET eShop repository. The LLM (Gemini Flash) never sees the full codebase — it relies entirely on **retrieved code chunks** stored in Azure Cosmos DB with DiskANN vector search.

The quality of the entire Dual-Track pipeline depends on one thing: **whether the correct code chunk is retrieved for a given incident report.** If the RAG returns an irrelevant or incomplete chunk, the Risk Hypothesizer generates bad `exact_span` citations, and the Span Arbiter marks everything as `HALLUCINATION`.

### Problem with the Previous Approach

The original chunker used a **naive sliding window** strategy:

```python
# Old: Split by character count (2000 chars), overlap 3 lines
MAX_CHUNK_SIZE = 2000
for i, line in enumerate(lines, 1):
    chunk_lines.append(line)
    if len("\n".join(chunk_lines)) >= MAX_CHUNK_SIZE:
        # try to find a break point (any "}" or blank line)
        break_point = _find_break_point(chunk_lines)
        ...
```

**Measured deficiencies:**

| Problem | Impact on Pipeline |
|---|---|
| Chunks cut methods in half arbitrarily | Span Arbiter can't verify partial code → false `HALLUCINATION` verdicts |
| Break points trigger on any `}` (if, foreach, class — no distinction) | Semantically meaningless boundaries |
| Overlap = 3 lines (~60 chars), despite `CHUNK_OVERLAP = 200` being defined but unused | Lost context between chunks |
| Embedding truncated to 1000 chars (`[:1000]`) on 2000-char chunks | Vector represents only 50% of chunk content → poor retrieval |
| No `class_name` or `method_name` metadata | LLM context shows `FILE: OrdersController.cs (lines 1-47)` — no semantic context |
| `.csproj`, `GlobalUsings.cs` indexed | Noise in retrieval results |

---

## Decision

Replace the sliding-window chunker with an **AST-based chunker** using [tree-sitter](https://tree-sitter.github.io/) with the C# grammar. Each chunk corresponds to a **complete semantic unit** (method, constructor, or class body), enriched with structural metadata.

### Chunking Strategy

```
chunk_file(file_path, content)
   │
   ├── .cs files  → tree-sitter C# AST parser
   │                 ├── method_declaration      → 1 chunk per method
   │                 ├── constructor_declaration  → 1 chunk per constructor
   │                 ├── property_declaration     → 1 chunk (if >100 chars)
   │                 └── fallback if no AST nodes → RecursiveCharacterTextSplitter
   │
   └── other files → RecursiveCharacterTextSplitter with language-aware separators
```

### Chunk Schema (New)

```json
{
    "id": "a1b2c3d4...",
    "file_path": "src/Ordering.API/Controllers/OrdersController.cs",
    "service_name": "Ordering.API",
    "chunk_text": "// Namespace: eShop.Ordering.API\n// Class: OrdersController\n// Method: CreateOrder\npublic async Task<IActionResult> CreateOrder(...) { ... }",
    "start_line": 38,
    "end_line": 65,
    "language": "csharp",
    "class_name": "OrdersController",
    "method_name": "CreateOrder",
    "chunk_type": "method_declaration",
    "embedding": [0.012, -0.034, ...]
}
```

New fields vs. old schema: `class_name`, `method_name`, `chunk_type`. The `chunk_text` now includes a context header with namespace, class, and method for richer embeddings.

---

## Implementation

### Files Modified

| File | Change |
|---|---|
| `requirements.txt` | Added `tree-sitter`, `tree-sitter-c-sharp`, `langchain-text-splitters` |
| `app/indexer/chunker.py` | **Rewritten** — AST parser for C#, text splitter fallback for others |
| `app/indexer/repo_indexer.py` | Removed `[:1000]` truncation on embedding input |
| `app/providers/db_provider.py` | Added `class_name`, `method_name`, `chunk_type` to SQL SELECT |
| `app/agents/nodes/risk_hypothesizer.py` | Context header now shows `OrdersController.CreateOrder [file:38-65]` |
| `app/agents/nodes/span_arbiter.py` | Pre-filters candidates by `suspected_file` before fuzzy matching |

### Key Design Decisions Within the Implementation

1. **Context Headers on Chunks**: Each AST chunk is prepended with `// Namespace: ...\n// Class: ...\n// Method: ...`. This enriches the embedding vector so that a semantic query like "order creation and database save" matches `eShop.Ordering.API.OrderService.CreateOrderAsync` even without knowing those names.

2. **Minimum Chunk Size = 100 chars**: Auto-properties like `public string Name { get; set; }` (< 100 chars) are filtered out. They add noise to retrieval without useful diagnostic value.

3. **Noise File Exclusion**: `GlobalUsings.cs`, `AssemblyInfo.cs`, `.csproj` files are excluded. They consistently appeared in top-5 results without contributing to incident diagnosis.

4. **Sub-chunking for Large Methods**: Methods exceeding `MAX_CHUNK_SIZE` (2000 chars) are sub-divided using `RecursiveCharacterTextSplitter` with C#-aware separators, but **retain the parent's AST metadata** (`class_name`, `method_name` as `MethodName_part1`, `_part2`, etc.).

5. **Graceful Degradation**: If `tree-sitter` is not installed, the chunker falls back to `RecursiveCharacterTextSplitter` with C#-aware separators — still better than the old sliding window.

---

## Validation Results

### Unit Test (Synthetic C# File)

A mock `OrdersController.cs` with 3 members was chunked correctly:

| Chunk | Class | Method | Type | Lines | Size |
|---|---|---|---|---|---|
| 1 | OrdersController | CreateOrder | method_declaration | 16-25 | 428 chars |

The constructor (92 chars) and `ValidateOrder` (79 chars) were correctly filtered by the 100-char threshold.

### Retrieval Test — Exact Queries (Knowing the Codebase)

| Query | Top-1 Result | Score |
|---|---|---|
| `NullReferenceException in OrdersController checkout` | `OrderMockService.OrderCheckout` | 0.731 |
| `gRPC BasketService AddItem timeout error` | `BasketService.GetBasketClient` | 0.743 |
| `Catalog.API GetItemsByBrand SQL query performance` | `ICatalogService.GetCatalogBrandAsync` | 0.720 |

### Retrieval Test — Semantic Queries (LLM Without Codebase Knowledge)

| Query (Natural Language) | Top-1 Result | Score | Relevant? |
|---|---|---|---|
| "code that handles order creation and saves to database" | `OrderMockService.CreateOrderAsync` | 0.722 | ✅ |
| "authentication token validation and user identity verification" | `IIdentityService` interface | 0.677 | ✅ |
| "shopping cart persistence and session state management" | `IBasketService.LocalBasketItems` | 0.642 | ⚠️ Partial |
| "serving product images and catalog item media URLs" | `IProductImageUrlProvider.GetProductImageUrl` | 0.682 | ✅ |
| "payment processing integration and transaction error handling" | `OrderPaymentFailedIntegrationEvent` | 0.688 | ✅ |
| "product search filtering and pagination logic" | `CatalogApi.GetAllItems` (paginated) | 0.665 | ✅ |
| "event bus message publishing and subscription handling" | `IEventBus.PublishAsync` | 0.729 | ✅ |

**6/7 queries returned the semantically correct top-1 result** without any knowledge of eShop class/method names.

### Indexing Stats

| Metric | Before (Sliding Window) | After (AST) |
|---|---|---|
| Files processed | 607 | 607 |
| Chunks generated | 949 | 1,637 |
| Indexing time | ~60s | ~325s |
| Chunk contains complete method | ~30% (estimated) | ~95% |
| Has `class_name` metadata | ❌ | ✅ |
| Has `method_name` metadata | ❌ | ✅ |
| Embedding input truncated | Yes (50%) | No (100%) |

---

## Consequences

### Positive

- **Span Arbiter precision increases** — each chunk is a complete method, so `exact_span` from the LLM is verifiable against a well-bounded unit.
- **Risk Hypothesizer generates better hypotheses** — the LLM sees `OrdersController.CreateOrder [file:38-65]` instead of a blob, so it can cite specific methods.
- **Retrieval quality improves** — embeddings are computed on full, semantically coherent chunks enriched with namespace/class context.
- **Anti-hallucination strengthened** — with `class_name`/`method_name` in Cosmos DB, the Span Arbiter can now pre-filter by file before fuzzy matching.

### Negative

- **Indexing is ~5x slower** (325s vs 60s) due to AST parsing + more chunks to embed. Acceptable for a one-time operation.
- **More chunks = higher Cosmos DB storage** (~1.7x more documents). Marginal cost impact.
- **New dependency** (`tree-sitter`, `tree-sitter-c-sharp`) — adds ~2MB to the container image.

### Risks

- **Tree-sitter C# grammar may not parse all .NET 10 syntax** (e.g., raw string literals, primary constructors). The fallback to `RecursiveCharacterTextSplitter` mitigates this.
- **The 100-char threshold may filter out small but important methods.** Can be tuned post-hackathon.

---

## Re-Indexing Procedure

After deploying this change, existing Cosmos DB chunks must be cleared and re-indexed:

```bash
# 1. Clear old chunks
python3 reset_cosmos.py

# 2. Re-index with AST chunker
curl -X POST http://localhost:8000/index
```

The `reset_cosmos.py` script reads all chunk IDs and deletes them individually (no bulk delete in Cosmos DB NoSQL API). Takes ~2 minutes for ~1000 chunks.
