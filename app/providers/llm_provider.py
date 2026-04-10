from __future__ import annotations

"""
SRE Agent — Gemini LLM Provider (Facade)

Centralized facade for all LLM interactions.
Uses the google-genai SDK directly against the Gemini API.

Key design decisions:
- Single client instance (singleton) for connection reuse
- All calls go through this facade for telemetry injection
- Structured outputs via Pydantic schemas
- Embeddings via gemini-embedding-001 (768 dims, MRL)
"""

import os
from typing import Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.config import get_settings

# Import Langfuse before making LLM calls. Environment variables must already be loaded.
if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
    try:
        from langfuse import get_client as _get_langfuse

        _langfuse = _get_langfuse()
        _LANGFUSE_ENABLED = True
    except Exception:
        _langfuse = None  # type: ignore
        _LANGFUSE_ENABLED = False
else:
    _langfuse = None  # type: ignore
    _LANGFUSE_ENABLED = False

T = TypeVar("T", bound=BaseModel)

import logging
_cache_logger = logging.getLogger(__name__ + ".cache")

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return a cached Gemini client configured for Gemini Developer API."""
    global _client
    if _client is None:
        settings = get_settings()
        # Use the Gemini Developer API when authenticating with an AI Studio API key.
        # Vertex AI endpoints require OAuth2/ADC credentials and will reject plain API keys.
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


from contextlib import contextmanager


@contextmanager
def _noop_ctx():
    """No-op context manager used when Langfuse is disabled."""
    yield None


def _log_cache_metrics(response, func_name: str, model: str) -> None:
    """
    Log implicit cache hit metrics from Gemini usage_metadata.

    Gemini 2.5+ and 3.x automatically cache prompt prefixes. When a request
    hits an existing cache, `cached_content_token_count` > 0, meaning those
    tokens were billed at a reduced rate.

    Minimum token thresholds for implicit caching:
    - gemini-3-flash-*: 1,024 tokens
    - gemini-3-pro-* / gemini-3.1-pro-*: 4,096 tokens
    """
    meta = getattr(response, "usage_metadata", None)
    if not meta:
        return

    cached = getattr(meta, "cached_content_token_count", 0) or 0
    total_in = getattr(meta, "prompt_token_count", 0) or 0
    total_out = getattr(meta, "candidates_token_count", 0) or 0

    if cached > 0 and total_in > 0:
        pct = cached / total_in * 100
        _cache_logger.info(
            "\U0001f4b0 cache_hit | func=%s | model=%s | "
            "cached=%d/%d tokens (%.0f%%) | output=%d",
            func_name, model, cached, total_in, pct, total_out,
        )
    else:
        _cache_logger.debug(
            "cache_miss | func=%s | model=%s | input=%d | output=%d",
            func_name, model, total_in, total_out,
        )


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------


async def generate_text(
    prompt: str,
    *,
    system_instruction: str | None = None,
    model: str | None = None,
) -> str:
    """Generate text from a prompt. Returns raw text response."""
    settings = get_settings()
    client = _get_client()
    _model = model or settings.gemini_model

    config = types.GenerateContentConfig(
        temperature=1,
        thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
    )
    if system_instruction:
        config.system_instruction = system_instruction

    obs_ctx = (
        _langfuse.start_as_current_observation(
            as_type="generation",
            name="generate_text",
            model=_model,
            input={"prompt": prompt[:2000]},
        )
        if _LANGFUSE_ENABLED
        else _noop_ctx()
    )
    with obs_ctx as obs:
        response = await client.aio.models.generate_content(
            model=_model,
            contents=prompt,
            config=config,
        )
        _log_cache_metrics(response, "generate_text", _model)
        result = response.text
        if _LANGFUSE_ENABLED and obs:
            obs.update(output=result[:2000])
        return result


# ---------------------------------------------------------------------------
# Structured output (Pydantic)
# ---------------------------------------------------------------------------


async def generate_structured(
    prompt: str,
    response_schema: Type[T],
    *,
    system_instruction: str | None = None,
    model: str | None = None,
) -> T:
    """
    Generate content constrained to a Pydantic schema.
    Uses Gemini's native JSON mode with strict schema enforcement.
    """
    settings = get_settings()
    client = _get_client()
    _model = model or settings.gemini_model

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=1,
        thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
    )
    if system_instruction:
        config.system_instruction = system_instruction

    schema_name = getattr(response_schema, "__name__", "unknown")
    obs_ctx = (
        _langfuse.start_as_current_observation(
            as_type="generation",
            name=f"generate_structured:{schema_name}",
            model=_model,
            input={"prompt": prompt[:2000]},
            metadata={"schema": schema_name},
        )
        if _LANGFUSE_ENABLED
        else _noop_ctx()
    )
    with obs_ctx as obs:
        response = await client.aio.models.generate_content(
            model=_model,
            contents=prompt,
            config=config,
        )
        _log_cache_metrics(response, f"generate_structured:{schema_name}", _model)
        parsed = response_schema.model_validate_json(response.text)
        if _LANGFUSE_ENABLED and obs:
            # Include the optional reasoning field in observability output when present.
            thinking = getattr(parsed, "thinking_process", None)
            obs.update(
                output={
                    "schema": schema_name,
                    "raw": response.text[:2000],
                    **(  # surface thinking_process at the top level for inspection
                        {"thinking_process": thinking}
                        if thinking
                        else {}
                    ),
                }
            )
        return parsed


# ---------------------------------------------------------------------------
# Multimodal generation (text + image/file)
# ---------------------------------------------------------------------------


async def generate_multimodal(
    text_prompt: str,
    image_bytes: bytes | None = None,
    image_mime_type: str = "image/png",
    *,
    system_instruction: str | None = None,
    response_schema: Type[T] | None = None,
    model: str | None = None,
) -> str | T:
    """
    Generate content from text + optional image.
    If response_schema is provided, returns a Pydantic model.
    Otherwise returns raw text.
    """
    settings = get_settings()
    client = _get_client()
    _model = model or settings.gemini_model

    contents = []
    if image_bytes:
        contents.append(
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type)
        )
    contents.append(text_prompt)

    config = types.GenerateContentConfig(
        temperature=1,
        thinking_config=types.ThinkingConfig(thinking_level="MEDIUM"),
    )
    if system_instruction:
        config.system_instruction = system_instruction
    if response_schema:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema

    schema_name = getattr(response_schema, "__name__", "text") if response_schema else "text"
    obs_ctx = (
        _langfuse.start_as_current_observation(
            as_type="generation",
            name=f"generate_multimodal:{schema_name}",
            model=_model,
            input={"prompt": text_prompt[:2000], "has_image": image_bytes is not None},
            metadata={"schema": schema_name},
        )
        if _LANGFUSE_ENABLED
        else _noop_ctx()
    )
    with obs_ctx as obs:
        response = await client.aio.models.generate_content(
            model=_model,
            contents=contents,
            config=config,
        )
        _log_cache_metrics(response, "generate_multimodal", _model)
        if response_schema:
            parsed = response_schema.model_validate_json(response.text)
            if _LANGFUSE_ENABLED and obs:
                thinking = getattr(parsed, "thinking_process", None)
                obs.update(
                    output={
                        "schema": schema_name,
                        "raw": response.text[:2000],
                        **(  {"thinking_process": thinking} if thinking else {}),
                    }
                )
            return parsed
        if _LANGFUSE_ENABLED and obs:
            obs.update(output=response.text[:2000])
        return response.text


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


async def generate_embedding(
    text: str,
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    """
    Generate a 768-dim embedding using gemini-embedding-2-preview with MRL.

    Task types:
    - RETRIEVAL_DOCUMENT: for indexing documents/code
    - RETRIEVAL_QUERY: for search queries
    - CODE_RETRIEVAL_QUERY: for code search queries
    """
    settings = get_settings()
    client = _get_client()

    result = await client.aio.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.gemini_embedding_dimensions,
        ),
    )
    return result.embeddings[0].values


async def generate_embeddings_batch(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    settings = get_settings()
    client = _get_client()

    result = await client.aio.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.gemini_embedding_dimensions,
        ),
    )
    return [e.values for e in result.embeddings]
