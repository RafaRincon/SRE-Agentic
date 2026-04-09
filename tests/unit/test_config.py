import pytest
from pydantic import ValidationError

from app.config import Settings


def base_kwargs():
    return {
        "cosmos_endpoint": "https://cosmos.test",
        "cosmos_key": "key",
        "cosmos_database": "sre",
        "gemini_api_key": "gem-key",
    }


def test_settings_apply_development_defaults():
    settings = Settings(_env_file=None, **base_kwargs())

    assert settings.app_env == "development"
    assert settings.app_cors_origins == ["*"]
    assert settings.app_require_index_ready is False
    assert settings.app_enable_docs is True


def test_settings_require_explicit_prod_cors_and_admin_key():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            app_env="production",
            app_cors_origins=[],
            **base_kwargs(),
        )

    settings = Settings(
        _env_file=None,
        app_env="production",
        app_cors_origins="https://allowed.example.com",
        app_admin_api_key="prod-admin",
        **base_kwargs(),
    )

    assert settings.app_cors_origins == ["https://allowed.example.com"]
    assert settings.app_require_index_ready is True
    assert settings.app_enable_docs is False


def test_settings_require_core_secrets():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cosmos_key="key", cosmos_database="db", gemini_api_key="gem")
