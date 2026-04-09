from __future__ import annotations

"""
SRE Agent — Code Chunker

Chunks C# source files from the eShop repository into indexable pieces using AST.
Each chunk includes metadata about file path, service name, line numbers,
language, class name, and method name — critical for the Span Arbiter's verification.
"""

import os
import hashlib
import logging
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

try:
    from tree_sitter import Language as TSLanguage, Parser
    import tree_sitter_c_sharp as tscsharp
    TREE_SITTER_AVAILABLE = True
    CSHARP_LANG = TSLanguage(tscsharp.language())
except ImportError:
    TREE_SITTER_AVAILABLE = False
    CSHARP_LANG = None

logger = logging.getLogger(__name__)

# eShop service directories → service names
SERVICE_MAP = {
    "Catalog.API": "Catalog.API",
    "Ordering.API": "Ordering.API",
    "Ordering.Domain": "Ordering.Domain",
    "Ordering.Infrastructure": "Ordering.Infrastructure",
    "Basket.API": "Basket.API",
    "Payment.API": "Payment.API",
    "Identity.API": "Identity.API",
    "WebApp": "WebApp",
    "OrderProcessor": "OrderProcessor",
    "WebhookClient": "WebhookClient",
    "EventBus": "EventBus",
    "EventBusRabbitMQ": "EventBusRabbitMQ",
    "ServiceDefaults": "ServiceDefaults",
    "eShop.AppHost": "eShop.AppHost",
}

# File extensions to index (removed .csproj — project configs are noise for RAG)
INDEXABLE_EXTENSIONS = {".cs", ".razor", ".json", ".yml", ".yaml", ".md"}

# Files to skip entirely — they pollute retrieval with low-signal content
SKIP_FILENAMES = {
    "GlobalUsings.cs",
    "AssemblyInfo.cs",
    "Usings.cs",
}

MAX_CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Custom C# Separators for fallback/sub-chunking
CSHARP_SEPARATORS = [
    "\npublic class ", "\nprivate class ", "\ninternal class ",
    "\npublic interface ", "\npublic enum ",
    "\npublic ", "\nprivate ", "\nprotected ", "\ninternal ",
    "\nasync Task", "\noverride ",
    "\n\n", "\n", " ", ""
]


def detect_service_name(file_path: str) -> str:
    """Map a file path to its eShop service name using the SERVICE_MAP."""
    parts = Path(file_path).parts
    for part in parts:
        for key, service in SERVICE_MAP.items():
            if key in part:
                return service
    return "unknown"


def _generate_chunk_id(file_path: str, start_line: int) -> str:
    """Generate a deterministic ID for a chunk."""
    raw = f"{file_path}:{start_line}"
    return hashlib.md5(raw.encode()).hexdigest()


def _get_identifier(node) -> str | None:
    """Extract identifier text from a tree-sitter node if available."""
    for child in node.children:
        if child.type == "identifier" or child.type == "name":
            return child.text.decode("utf8") if hasattr(child, "text") and child.text else None
    return None


def _extract_namespace(content: str) -> str | None:
    """Extract the namespace from C# source using simple regex."""
    import re
    match = re.search(r'namespace\s+([\w.]+)', content)
    return match.group(1) if match else None


def _build_context_header(namespace: str | None, class_name: str | None, method_name: str | None) -> str:
    """Build a semantic context header to prepend to chunk text for richer embeddings."""
    parts = []
    if namespace:
        parts.append(f"// Namespace: {namespace}")
    if class_name:
        parts.append(f"// Class: {class_name}")
    if method_name:
        parts.append(f"// Method: {method_name}")
    return "\n".join(parts) + "\n" if parts else ""


def _chunk_csharp_ast(file_path: str, content: str, service_name: str) -> list[dict]:
    """Parse C# files with Tree-sitter and chunk by semantic methods and classes."""
    if not TREE_SITTER_AVAILABLE:
        logger.warning("Tree-sitter not available, falling back to text chunking for C#")
        return _chunk_text_fallback(file_path, content, service_name, "csharp")

    csharp_parser = Parser(CSHARP_LANG)
    tree = csharp_parser.parse(bytes(content, "utf8"))
    namespace = _extract_namespace(content)
    
    chunks = []
    
    TARGET_NODES = {
        "method_declaration",
        "constructor_declaration",
        "property_declaration"
    }

    CLASS_NODES = {
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "struct_declaration",
    }

    sub_splitter = RecursiveCharacterTextSplitter(
        separators=CSHARP_SEPARATORS,
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    def process_node(node, parent_class=None):
        if node.type in TARGET_NODES:
            start_byte = node.start_byte
            end_byte = node.end_byte
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            
            chunk_text = content[start_byte:end_byte]
            method_name = _get_identifier(node)
            
            base_metadata = {
                "file_path": file_path,
                "service_name": service_name,
                "language": "csharp",
                "class_name": parent_class,
                "method_name": method_name,
                "chunk_type": node.type,
                "embedding": [],
            }

            # Prepend semantic header for richer embeddings
            context_header = _build_context_header(namespace, parent_class, method_name)
            enriched_text = context_header + chunk_text

            if len(enriched_text) > MAX_CHUNK_SIZE:
                # Sub-chunk very large methods while preserving metadata
                sub_chunks = sub_splitter.split_text(enriched_text)
                for i, sub in enumerate(sub_chunks):
                    chunks.append({
                        **base_metadata,
                        "id": _generate_chunk_id(file_path, start_line + i),
                        "chunk_text": sub,
                        "method_name": f"{method_name}_part{i+1}" if method_name else f"part{i+1}",
                        "start_line": start_line,
                        "end_line": end_line,
                    })
            elif len(chunk_text) > 100:  # Skip trivial properties (e.g. auto-getters)
                chunks.append({
                    **base_metadata,
                    "id": _generate_chunk_id(file_path, start_line),
                    "chunk_text": enriched_text,
                    "start_line": start_line,
                    "end_line": end_line,
                })
        
        elif node.type in CLASS_NODES:
            # We don't chunk the class itself, but we capture its name for children
            class_name = _get_identifier(node)
            # Add field and small properties of class as one generic chunk? 
            # Skipping, as we just want method boundaries for precision.
            for child in node.children:
                process_node(child, parent_class=class_name)
            return # Processed children 

        # Continue traversing down
        for child in node.children:
            process_node(child, parent_class)

    process_node(tree.root_node)
    
    if not chunks:
        # Fallback if no methods/classes found (e.g. GlobalStatements)
        return _chunk_text_fallback(file_path, content, service_name, "csharp")
        
    return chunks


def _chunk_text_fallback(file_path: str, content: str, service_name: str, language: str) -> list[dict]:
    """Fallback chunking using LangChain's RecursiveCharacterTextSplitter."""
    separators = ["\n\n", "\n", " ", ""]
    if language == "csharp":
        separators = CSHARP_SEPARATORS
        
    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    docs = splitter.create_documents([content])
    chunks = []
    
    current_line = 1
    lines = content.split("\n")
    
    for i, doc in enumerate(docs):
        # Approximate line numbers by finding substring
        # Accurate start/end lines are less deterministic here due to overlapping text
        chunk_text = doc.page_content
        
        # very naive start_line calculation (good enough for fallback)
        try:
            char_idx = content.find(chunk_text[:50])
            start_line = content.count("\n", 0, char_idx) + 1
            end_line = start_line + chunk_text.count("\n")
        except:
            start_line = 1
            end_line = 1
            
        chunk_id = _generate_chunk_id(file_path, start_line)
        
        chunks.append({
            "id": chunk_id,
            "file_path": file_path,
            "service_name": service_name,
            "chunk_text": chunk_text,
            "start_line": start_line,
            "end_line": end_line,
            "language": language,
            "class_name": None,
            "method_name": None,
            "chunk_type": "text_segment",
            "embedding": [],
        })
        
    return chunks


def chunk_file(file_path: str, content: str) -> list[dict]:
    """
    Split a source file into semantic overlapping chunks with metadata.

    Each chunk is a dict ready for Cosmos DB insertion:
    {
        "id": deterministic hash,
        "file_path": relative path in eShop repo,
        "service_name": partition key,
        "chunk_text": the code text,
        "start_line": first line number,
        "end_line": last line number,
        "language": "csharp" | "json" | etc.,
        "class_name": className (if C# AST),
        "method_name": methodName (if C# AST),
        "chunk_type": AST node type or 'text_segment',
        "embedding": [] (populated later by the indexer)
    }
    """
    if not content.strip():
        return []

    ext = Path(file_path).suffix.lower()
    lang_map = {
        ".cs": "csharp",
        ".razor": "razor",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".md": "markdown",
    }
    language = lang_map.get(ext, "text")
    service_name = detect_service_name(file_path)

    if language == "csharp":
        return _chunk_csharp_ast(file_path, content, service_name)
    else:
        return _chunk_text_fallback(file_path, content, service_name, language)


def walk_repo(repo_path: str) -> list[tuple[str, str]]:
    """
    Walk the eShop repo and return (relative_path, content) for indexable files.
    Skips binary files, test files, and build artifacts.
    """
    files = []
    skip_dirs = {
        "bin", "obj", "node_modules", ".git", ".vs",
        "TestResults", "tests", "test",
    }

    for root, dirs, filenames in os.walk(repo_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for filename in filenames:
            # Skip noise files that pollute retrieval
            if filename in SKIP_FILENAMES:
                continue

            ext = Path(filename).suffix.lower()
            if ext not in INDEXABLE_EXTENSIONS:
                continue

            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, repo_path)

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Skip very small or very large files
                if len(content) < 50 or len(content) > 100_000:
                    continue

                files.append((rel_path, content))
            except Exception as e:
                logger.warning(f"Skipping {rel_path}: {e}")

    logger.info(f"Found {len(files)} indexable files in {repo_path}")
    return files

