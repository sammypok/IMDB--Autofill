"""
Unit tests for backend/aggregator.py — merge_product().

Tests cover AGGR-02 (fill from multiple images, majority vote) and
AGGR-03 (one merged dict per product with confidence). No file I/O required —
plain dicts are passed directly.

Run with: python -m pytest tests/test_aggregator.py -v

RED state: These tests will fail with ImportError until backend/aggregator.py is
implemented in Plan 02-02.
"""

from __future__ import annotations

import pytest

from backend.models import IMDB_FIELDS


# ---------------------------------------------------------------------------
# Helper: build a list of (fields, confidence) tuples from a simple spec
# Each item in `specs` is a dict of {field: value}; unspecified fields are ""
# ---------------------------------------------------------------------------

def _make_images(*field_overrides: dict) -> list[tuple[dict, dict]]:
    """
    Build a list of (fields, confidence) pairs.

    Each positional arg is a dict of {field_name: value} for that image.
    Unspecified fields default to "". Confidence defaults to "medium".
    """
    results: list[tuple[dict, dict]] = []
    for overrides in field_overrides:
        fields = {f: "" for f in IMDB_FIELDS}
        fields.update(overrides)
        confidence = {f: "medium" for f in IMDB_FIELDS}
        results.append((fields, confidence))
    return results


# ---------------------------------------------------------------------------
# Test 1: Fill empty field from another image
# ---------------------------------------------------------------------------

def test_fill_empty_field():
    """Image A has BARCODE='', image B has BARCODE='123' → merged BARCODE='123'."""
    from backend.aggregator import merge_product

    images = _make_images(
        {"BARCODE": ""},
        {"BARCODE": "123"},
    )
    merged_fields, merged_confidence = merge_product(images)

    assert merged_fields["BARCODE"] == "123", (
        f"Expected BARCODE='123' after fill, got {repr(merged_fields['BARCODE'])}"
    )


# ---------------------------------------------------------------------------
# Test 2: Majority vote picks winner
# ---------------------------------------------------------------------------

def test_majority_vote():
    """Three images with values 'A', 'B', 'A' → winner is 'A'."""
    from backend.aggregator import merge_product

    images = _make_images(
        {"BRAND": "A"},
        {"BRAND": "B"},
        {"BRAND": "A"},
    )
    merged_fields, _ = merge_product(images)

    assert merged_fields["BRAND"] == "A", (
        f"Expected majority winner 'A', got {repr(merged_fields['BRAND'])}"
    )


# ---------------------------------------------------------------------------
# Test 3: Tie-breaking goes to first seen
# ---------------------------------------------------------------------------

def test_tie_goes_to_first_seen():
    """Two images 'A', 'B' (equal votes) → winner is 'A' (first seen)."""
    from backend.aggregator import merge_product

    images = _make_images(
        {"MANUFACTURER": "A"},
        {"MANUFACTURER": "B"},
    )
    merged_fields, _ = merge_product(images)

    assert merged_fields["MANUFACTURER"] == "A", (
        f"Expected first-seen 'A' on tie, got {repr(merged_fields['MANUFACTURER'])}"
    )


# ---------------------------------------------------------------------------
# Test 4: All empty values stay empty
# ---------------------------------------------------------------------------

def test_all_empty_stays_empty():
    """All images have BARCODE='' → merged BARCODE=''."""
    from backend.aggregator import merge_product

    images = _make_images(
        {"BARCODE": ""},
        {"BARCODE": ""},
        {"BARCODE": ""},
    )
    merged_fields, _ = merge_product(images)

    assert merged_fields["BARCODE"] == "", (
        f"Expected empty BARCODE when all images are empty, got {repr(merged_fields['BARCODE'])}"
    )


# ---------------------------------------------------------------------------
# Test 5: Confidence is "high" when all images agree
# ---------------------------------------------------------------------------

def test_confidence_all_agree_is_high():
    """N images all agree on 'Alpha' → confidence for that field is 'high'."""
    from backend.aggregator import merge_product

    images = _make_images(
        {"ITEM_NAME": "Alpha"},
        {"ITEM_NAME": "Alpha"},
        {"ITEM_NAME": "Alpha"},
    )
    _, merged_confidence = merge_product(images)

    assert merged_confidence["ITEM_NAME"] == "high", (
        f"Expected 'high' confidence when all images agree, got {repr(merged_confidence['ITEM_NAME'])}"
    )


# ---------------------------------------------------------------------------
# Test 6: Confidence is "medium" when N-1 agree, one differs
# ---------------------------------------------------------------------------

def test_confidence_one_disagrees_is_medium():
    """3 images: 2 agree on 'Alpha', 1 says 'Beta' → confidence is 'medium'."""
    from backend.aggregator import merge_product

    images = _make_images(
        {"ITEM_NAME": "Alpha"},
        {"ITEM_NAME": "Alpha"},
        {"ITEM_NAME": "Beta"},
    )
    _, merged_confidence = merge_product(images)

    assert merged_confidence["ITEM_NAME"] == "medium", (
        f"Expected 'medium' confidence when one image disagrees, got {repr(merged_confidence['ITEM_NAME'])}"
    )


# ---------------------------------------------------------------------------
# Test 7: Confidence is "low" when majority disagree
# ---------------------------------------------------------------------------

def test_confidence_most_disagree_is_low():
    """4 images: 2 'Alpha', 1 'Beta', 1 'Gamma' → majority fraction < 0.5 → 'low'."""
    from backend.aggregator import merge_product

    images = _make_images(
        {"BRAND": "Alpha"},
        {"BRAND": "Beta"},
        {"BRAND": "Gamma"},
        {"BRAND": "Delta"},
    )
    _, merged_confidence = merge_product(images)

    assert merged_confidence["BRAND"] == "low", (
        f"Expected 'low' confidence when most images disagree, got {repr(merged_confidence['BRAND'])}"
    )


# ---------------------------------------------------------------------------
# Test 8: Output dicts have exactly IMDB_FIELDS keys
# ---------------------------------------------------------------------------

def test_output_has_all_13_keys():
    """merge_product returns (merged_fields, merged_confidence) each with exactly IMDB_FIELDS keys."""
    from backend.aggregator import merge_product

    images = _make_images(
        {"ITEM_NAME": "Test Product", "BARCODE": "9876543210"},
        {"ITEM_NAME": "Test Product", "WEIGHT": "500g"},
    )
    merged_fields, merged_confidence = merge_product(images)

    expected_keys = set(IMDB_FIELDS)
    assert set(merged_fields.keys()) == expected_keys, (
        f"merged_fields keys mismatch.\nExpected: {expected_keys}\nGot: {set(merged_fields.keys())}"
    )
    assert set(merged_confidence.keys()) == expected_keys, (
        f"merged_confidence keys mismatch.\nExpected: {expected_keys}\nGot: {set(merged_confidence.keys())}"
    )
