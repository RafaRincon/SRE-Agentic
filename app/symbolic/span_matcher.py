from __future__ import annotations

"""
SRE Agent — Span Matcher (Symbolic, Pure Python)

100% deterministic. No LLM. Verifies that a code citation (span) actually
exists in the indexed eShop codebase by searching Cosmos DB vector store.

This is the FIRST anti-hallucination barrier.
"""

import difflib
import logging

logger = logging.getLogger(__name__)


def fuzzy_match_span(
    span: str,
    candidate_text: str,
    threshold: float = 0.6,
) -> tuple[bool, float]:
    """
    Check if a span approximately exists in a candidate text.

    Uses SequenceMatcher for fuzzy matching.
    Returns (matched: bool, similarity: float).
    """
    if not span or not candidate_text:
        return False, 0.0

    # Normalize whitespace
    span_normalized = " ".join(span.split())
    candidate_normalized = " ".join(candidate_text.split())

    # Try exact substring match first (fastest)
    if span_normalized in candidate_normalized:
        return True, 1.0

    # Try case-insensitive exact match
    if span_normalized.lower() in candidate_normalized.lower():
        return True, 0.95

    # Fuzzy match using SequenceMatcher
    # Slide a window of span length across the candidate
    span_len = len(span_normalized)
    best_ratio = 0.0

    for i in range(0, max(1, len(candidate_normalized) - span_len + 1), span_len // 4 or 1):
        window = candidate_normalized[i:i + span_len + 20]  # +20 for flexibility
        ratio = difflib.SequenceMatcher(None, span_normalized.lower(), window.lower()).ratio()
        best_ratio = max(best_ratio, ratio)

        if best_ratio >= threshold:
            return True, best_ratio

    return best_ratio >= threshold, best_ratio


def exact_match_span(span: str, candidate_text: str) -> tuple[bool, float]:
    """
    Strict exact substring match (case-insensitive).
    """
    if not span or not candidate_text:
        return False, 0.0

    span_clean = " ".join(span.split()).lower()
    candidate_clean = " ".join(candidate_text.split()).lower()

    if span_clean in candidate_clean:
        return True, 1.0
    return False, 0.0
