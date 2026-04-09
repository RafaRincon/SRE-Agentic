from __future__ import annotations

"""
SRE Agent — eShop Repository Indexer

Clones the eShop repository, chunks the code, generates embeddings,
and indexes everything into Cosmos DB with DiskANN vectors.

This runs:
- On startup (if no chunks exist yet in Cosmos DB)
- On-demand via CLI or API endpoint

Design: Batch embeddings to minimize API calls, upsert to avoid duplicates.
"""

import os
import time
import logging
import asyncio
from pathlib import Path

import git

from app.config import get_settings
from app.indexer.chunker import walk_repo, chunk_file
from app.providers import llm_provider, db_provider

logger = logging.getLogger(__name__)

BATCH_SIZE = 20  # Gemini embedding batch size


async def clone_repo() -> str:
    """
    Clone the eShop repository (shallow clone for speed).
    Returns the path to the cloned repo.
    """
    settings = get_settings()
    cache_dir = settings.eshop_cache_dir

    if os.path.exists(os.path.join(cache_dir, ".git")):
        logger.info(f"[indexer] eShop repo already cloned at {cache_dir}")
        return cache_dir

    logger.info(f"[indexer] Cloning eShop from {settings.eshop_repo_url}...")
    os.makedirs(cache_dir, exist_ok=True)

    git.Repo.clone_from(
        settings.eshop_repo_url,
        cache_dir,
        depth=1,  # Shallow clone
        single_branch=True,
        branch="main",
    )

    logger.info(f"[indexer] eShop cloned to {cache_dir}")
    return cache_dir


async def index_repo(force: bool = False) -> dict:
    """
    Full indexing pipeline:
    1. Clone eShop (if not cached)
    2. Walk and chunk source files
    3. Generate embeddings in batches
    4. Upsert to Cosmos DB eshop_chunks

    Returns stats about the indexing process.
    """
    # Check if already indexed
    if not force:
        existing_count = db_provider.count_chunks()
        if existing_count > 0:
            logger.info(f"[indexer] Already indexed ({existing_count} chunks). Skipping. Use force=True to re-index.")
            return {"status": "skipped", "existing_chunks": existing_count}

    start_time = time.time()

    # Step 1: Clone
    repo_path = await clone_repo()

    # Step 2: Walk and chunk
    files = walk_repo(repo_path)
    all_chunks = []
    for rel_path, content in files:
        chunks = chunk_file(rel_path, content)
        all_chunks.extend(chunks)

    logger.info(f"[indexer] Generated {len(all_chunks)} chunks from {len(files)} files")

    # Step 3: Generate embeddings in batches
    total_embedded = 0
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c["chunk_text"] for c in batch]  # Full chunk embedding

        try:
            embeddings = await llm_provider.generate_embeddings_batch(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
            )

            for chunk, embedding in zip(batch, embeddings):
                chunk["embedding"] = embedding

            total_embedded += len(batch)
            logger.info(f"[indexer] Embedded batch {i // BATCH_SIZE + 1}: {total_embedded}/{len(all_chunks)}")

        except Exception as e:
            logger.error(f"[indexer] Embedding batch failed: {e}")
            # Skip this batch but continue
            for chunk in batch:
                chunk["embedding"] = [0.0] * 768  # Zero vector as fallback

    # Step 4: Upsert to Cosmos DB
    total_upserted = 0
    for chunk in all_chunks:
        if not chunk["embedding"] or all(v == 0.0 for v in chunk["embedding"]):
            continue  # Skip chunks without real embeddings

        try:
            db_provider.upsert_chunk(chunk)
            total_upserted += 1
        except Exception as e:
            logger.warning(f"[indexer] Failed to upsert chunk {chunk['id']}: {e}")

    elapsed = time.time() - start_time

    stats = {
        "status": "completed",
        "files_processed": len(files),
        "chunks_generated": len(all_chunks),
        "chunks_embedded": total_embedded,
        "chunks_indexed": total_upserted,
        "elapsed_seconds": round(elapsed, 2),
    }

    logger.info(f"[indexer] ✅ Indexing complete: {stats}")
    return stats
