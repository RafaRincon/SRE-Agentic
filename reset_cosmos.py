import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from app.providers import db_provider
from app.config import get_settings

def reset_chunks():
    print("Connecting to Cosmos DB to clear chunks...")
    settings = get_settings()
    
    # We query all items safely
    container = db_provider.get_container(settings.cosmos_container_chunks)
    
    items = list(container.query_items(
        query="SELECT c.id, c.service_name FROM c",
        enable_cross_partition_query=True
    ))

    print(f"Found {len(items)} chunks to delete.")
    
    for idx, item in enumerate(items):
        try:
            container.delete_item(item=item['id'], partition_key=item['service_name'])
            if (idx + 1) % 100 == 0:
                print(f"Deleted {idx + 1}/{len(items)}...")
        except Exception as e:
            print(f"Error on {item['id']}: {e}")

    print("Success! Cosmos DB chunks cleared.")

if __name__ == "__main__":
    reset_chunks()
