from __future__ import annotations

"""
SRE Agent — Gemini LLM Provider (Facade)

Centralized facade for all LLM interactions.
Uses the google-genai SDK (NOT the deprecated google-generativeai).

Key design decisions:
- Single client instance (singleton) for connection reuse
- All calls go through this facade for telemetry injection
- Structured outputs via Pydantic schemas
- Embeddings via gemini-embedding-2-preview with MRL (768 dims)
"""

from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Type, TypeVar

from app.config import get_settings

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return a cached Gemini client."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


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

    config = types.GenerateContentConfig()
    if system_instruction:
        config.system_instruction = system_instruction

    response = client.models.generate_content(
        model=model or settings.gemini_model,
        contents=prompt,
        config=config,
    )
    return response.text


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

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
    )
    if system_instruction:
        config.system_instruction = system_instruction

    response = client.models.generate_content(
        model=model or settings.gemini_model,
        contents=prompt,
        config=config,
    )

    return response_schema.model_validate_json(response.text)


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

    contents = []
    if image_bytes:
        contents.append(
            types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type)
        )
    contents.append(text_prompt)

    config = types.GenerateContentConfig()
    if system_instruction:
        config.system_instruction = system_instruction
    if response_schema:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema

    response = client.models.generate_content(
        model=model or settings.gemini_model,
        contents=contents,
        config=config,
    )

    if response_schema:
        return response_schema.model_validate_json(response.text)
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

    result = client.models.embed_content(
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

    result = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.gemini_embedding_dimensions,
        ),
    )
    return [e.values for e in result.embeddings]
