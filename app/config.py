"""
SRE Agent — Application Configuration

Centralized settings management using pydantic-settings.
All values are read from environment variables / .env file.
"""

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # gemini-embedding-001: works in Vertex AI Express Mode (api_key auth)
    # gemini-embedding-2-preview: requires full Vertex AI (project + location), NOT express mode
    gemini_embedding_model: str = "gemini-embedding-001"
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
    app_workers: int = 1
    app_cors_origins: list[str] = Field(default_factory=list)
    app_admin_api_key: str = ""
    app_disable_admin_endpoints: bool = False
    app_require_index_ready: bool | None = None
    app_enable_docs: bool | None = None

    # --- Input hardening ---
    app_max_upload_bytes: int = 5 * 1024 * 1024

    # --- eShop Indexer ---
    eshop_repo_url: str = "https://github.com/dotnet/eShop.git"
    eshop_cache_dir: str = ".eshop_cache"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("app_env")
    @classmethod
    def normalize_env(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("app_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> list[str]:
        if value in (None, "", []):
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("[") and raw.endswith("]"):
                import json

                parsed = json.loads(raw)
                return [str(item).strip() for item in parsed if str(item).strip()]
            return [item.strip() for item in raw.split(",") if item.strip()]
        raise TypeError("APP_CORS_ORIGINS must be a comma-separated string or JSON list")

    @model_validator(mode="after")
    def apply_environment_defaults(self) -> "Settings":
        is_production = self.app_env == "production"

        if self.app_workers < 1:
            raise ValueError("APP_WORKERS must be >= 1")
        if self.app_max_upload_bytes < 1024:
            raise ValueError("APP_MAX_UPLOAD_BYTES must be at least 1024 bytes")

        if not self.app_cors_origins:
            self.app_cors_origins = ["*"] if not is_production else []

        if self.app_require_index_ready is None:
            self.app_require_index_ready = is_production

        if self.app_enable_docs is None:
            self.app_enable_docs = not is_production

        if is_production and not self.app_cors_origins:
            raise ValueError("APP_CORS_ORIGINS must be configured in production")

        if is_production and not self.app_disable_admin_endpoints and not self.app_admin_api_key:
            raise ValueError(
                "APP_ADMIN_API_KEY must be configured in production when admin endpoints are enabled"
            )

        return self


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
