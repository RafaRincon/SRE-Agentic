import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from app.providers import db_provider
from app.config import get_settings


def _clear_container(container_name: str, partition_key_field: str = "service_name"):
    """Delete all documents from a Cosmos DB container."""
    print(f"\n{'='*50}")
    print(f"Clearing container: {container_name}")
    print(f"{'='*50}")

    container = db_provider.get_container(container_name)

    items = list(container.query_items(
        query=f"SELECT c.id, c.{partition_key_field} FROM c",
        enable_cross_partition_query=True
    ))

    print(f"Found {len(items)} documents to delete.")

    deleted = 0
    for idx, item in enumerate(items):
        try:
            container.delete_item(
                item=item['id'],
                partition_key=item.get(partition_key_field, "")
            )
            deleted += 1
            if deleted % 100 == 0:
                print(f"  Deleted {deleted}/{len(items)}...")
        except Exception as e:
            print(f"  Error on {item['id']}: {e}")

    print(f"✅ Done: {deleted}/{len(items)} deleted from {container_name}")
    return deleted


def reset_all():
    """Clear both vector containers (required when embedding model changes)."""
    settings = get_settings()
    print("🔄 Resetting ALL vector containers (embedding model changed)")
    print(f"   Cosmos DB: {settings.cosmos_endpoint}")
    print(f"   Database:  {settings.cosmos_database}")

    total = 0
    total += _clear_container(settings.cosmos_container_chunks)
    total += _clear_container(settings.cosmos_container_knowledge)

    print(f"\n{'='*50}")
    print(f"🏁 Total documents deleted: {total}")
    print(f"{'='*50}")

    return total


if __name__ == "__main__":
    reset_all()
