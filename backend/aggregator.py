"""
aggregator.py — Fill-then-majority-vote merge with consensus confidence.

Public API:
    merge_product(extractions: list[tuple[dict, dict]]) -> tuple[dict, dict]

Reduces N per-image (fields, confidence) tuples into a single merged pair
using fill-then-majority-vote semantics. Ties resolve to the first-seen value
(alphabetically first filename, guaranteed by the ingestor's sorted() pass).
Aggregated confidence is computed with all images as the denominator.
"""

from __future__ import annotations

from collections import Counter

from .models import IMDB_FIELDS


def merge_product(
    extractions: list[tuple[dict, dict]],
) -> tuple[dict, dict]:
    """Merge N per-image extraction tuples into one (fields, confidence) pair.

    Args:
        extractions: Ordered list of (fields_dict, confidence_dict) tuples,
                     one per image of the same product. Must be non-empty.

    Returns:
        A ``(merged_fields, merged_confidence)`` tuple where both dicts have
        exactly ``IMDB_FIELDS`` as their key set.
    """
    merged_fields: dict[str, str] = {}
    merged_confidence: dict[str, str] = {}

    total = len(extractions)

    for field in IMDB_FIELDS:
        # Collect non-empty values in first-seen (alphabetical) order
        non_empty_values = [
            f[field]
            for f, _ in extractions
            if f.get(field, "").strip()
        ]

        if not non_empty_values:
            merged_fields[field] = ""
            # All extractions agree the field is absent — that IS confident
            merged_confidence[field] = "high"
            continue

        # Case-insensitive majority vote; preserve the casing of the first-seen winner
        counts: Counter = Counter(v.strip().lower() for v in non_empty_values)
        top_key = counts.most_common(1)[0][0]
        # Pick the first original value whose lowercased form matches the winner
        winner = next(v for v in non_empty_values if v.strip().lower() == top_key)
        merged_fields[field] = winner

        # Aggregated confidence — denominator is ALL images, not just non-empty
        agreeing = sum(
            1 for f, _ in extractions
            if f.get(field, "").strip().lower() == top_key
        )
        if agreeing == total:
            merged_confidence[field] = "high"
        elif agreeing >= total - 1:
            merged_confidence[field] = "medium"
        else:
            merged_confidence[field] = "low"

    return merged_fields, merged_confidence
