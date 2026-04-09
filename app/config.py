"""
SRE Agent — Application Configuration

Centralized settings management using pydantic-settings.
All values are read from environment variables / .env file.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Azure Cosmos DB NoSQL ---
    cosmos_endpoint: str
    cosmos_key: str
    cosmos_database: str = "sre_agent_db"

    # --- Container names ---
    cosmos_container_chunks: str = "eshop_chunks"
    cosmos_container_ledger: str = "incident_ledger"
    cosmos_container_incidents: str = "incidents"
    cosmos_container_checkpoints: str = "agent_checkpoints"
    cosmos_container_knowledge: str = "sre_knowledge"

    # --- Google Gemini ---
    gemini_api_key: str
    gemini_model: str = "gemini-3-flash-preview"
    gemini_embedding_model: str = "gemini-embedding-2-preview"
    gemini_embedding_dimensions: int = 768

    # --- Observability (Langfuse) ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_base_url: str = ""

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    app_port: int = 8000

    # --- eShop Indexer ---
    eshop_repo_url: str = "https://github.com/dotnet/eShop.git"
    eshop_cache_dir: str = ".eshop_cache"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
