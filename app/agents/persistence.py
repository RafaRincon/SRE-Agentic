from __future__ import annotations

"""
Helpers for shaping incident documents before they are persisted.

The incident document stored in Cosmos keeps:
- `raw_report` as the source of truth
- `entities` as a compact derived view
- `world_model` as a compact operational projection

Legacy runtime-only fields remain supported in memory for compatibility,
but they are intentionally omitted from newly persisted incident documents.
"""

from collections.abc import Mapping
from typing import Any


PERSISTED_WORLD_MODEL_KEYS = (
    "affected_service",
    "incident_category",
    "blast_radius",
    "epistemic_snapshot",
)


def _normalize_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return dict(payload)


def normalize_entities_for_persistence(entities: Any) -> dict[str, Any]:
    normalized = _normalize_mapping(entities)
    normalized.pop("thinking_process", None)
    return normalized


def normalize_world_model_for_persistence(world_model: Any) -> dict[str, Any]:
    normalized = _normalize_mapping(world_model)
    return {
        key: normalized[key]
        for key in PERSISTED_WORLD_MODEL_KEYS
        if key in normalized
    }
