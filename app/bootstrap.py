from __future__ import annotations

"""
SRE Agent — Production bootstrap helpers.

Creates the required Cosmos DB database/containers and optionally performs the
initial eShop indexing pass using the same application image used in runtime.
"""

import argparse
import asyncio
import json
import logging

from app.providers import db_provider

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Cosmos DB and optional code index")
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Create Cosmos DB resources only and skip the initial code index pass.",
    )
    parser.add_argument(
        "--force-index",
        action="store_true",
        help="Force a full re-index even if chunks already exist.",
    )
    return parser.parse_args()


async def bootstrap(*, skip_index: bool = False, force_index: bool = False) -> dict:
    summary = {
        "bootstrap": db_provider.ensure_database_and_containers(),
    }

    if skip_index:
        summary["index"] = {"status": "skipped"}
        return summary

    from app.indexer.repo_indexer import index_repo

    summary["index"] = await index_repo(force=force_index)
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    args = parse_args()
    result = asyncio.run(
        bootstrap(skip_index=args.skip_index, force_index=args.force_index)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
