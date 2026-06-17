"""
Unit tests for backend/pipeline.py — process_folder().

Tests cover NORM-05 (each output dict has exactly the 13 IMDB_FIELDS keys) and
end-to-end integration. ingest_folder is patched to return controlled input so no
real images or API calls are needed.

Run with: python -m pytest tests/test_pipeline.py -v

RED state: These tests will fail with ImportError until backend/pipeline.py is
implemented in Plan 02-04.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from backend.models import IMDB_FIELDS


# ---------------------------------------------------------------------------
# Helper: build a minimal bucket dict that ingest_folder would return
# Two products, two images each, with distinct ITEM_NAMEs.
# ---------------------------------------------------------------------------

def _make_bucket(item_name: str, count: int = 2) -> list[tuple[dict, dict]]:
    """Return a bucket (list of (fields, confidence) pairs) for one product."""
    result = []
    for i in range(count):
        fields = {f: "" for f in IMDB_FIELDS}
        fields["ITEM_NAME"] = item_name
        fields["BARCODE"] = f"000{i}"
        confidence = {f: "medium" for f in IMDB_FIELDS}
        result.append((fields, confidence))
    return result


def _make_ingest_return() -> dict[str, list]:
    """Return a mock ingest_folder result with 2 products."""
    return {
        "milk powder": _make_bucket("Milk Powder", count=2),
        "dettol soap": _make_bucket("Dettol Soap", count=3),
    }


# ---------------------------------------------------------------------------
# Test 1: Each result dict has "fields" and "confidence"; "fields" has 13 keys
# ---------------------------------------------------------------------------

def test_output_keys_match_imdb_fields():
    """NORM-05: each product result has 'fields' with exactly 13 IMDB_FIELDS keys."""
    from backend.pipeline import process_folder

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("backend.ingestor.ingest_folder", return_value=_make_ingest_return()):
            results = process_folder(tmpdir)

    assert isinstance(results, list), "process_folder must return a list"
    assert len(results) == 2, f"Expected 2 products, got {len(results)}"

    for i, product in enumerate(results):
        assert "fields" in product, f"Product {i} missing 'fields' key"
        assert "confidence" in product, f"Product {i} missing 'confidence' key"

        fields_keys = set(product["fields"].keys())
        expected_keys = set(IMDB_FIELDS)
        assert fields_keys == expected_keys, (
            f"Product {i} 'fields' has wrong keys.\n"
            f"Expected: {expected_keys}\n"
            f"Got: {fields_keys}"
        )


# ---------------------------------------------------------------------------
# Test 2: process_folder returns a list of dicts with expected structure
# ---------------------------------------------------------------------------

def test_returns_list_of_dicts():
    """process_folder returns a list; each element has 'fields' and 'confidence'."""
    from backend.pipeline import process_folder

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("backend.ingestor.ingest_folder", return_value=_make_ingest_return()):
            results = process_folder(tmpdir)

    assert isinstance(results, list), "Return value must be a list"
    for i, item in enumerate(results):
        assert isinstance(item, dict), f"Element {i} must be a dict, got {type(item)}"
        assert "fields" in item, f"Element {i} must have 'fields' key"
        assert "confidence" in item, f"Element {i} must have 'confidence' key"
        assert isinstance(item["fields"], dict), f"Element {i}['fields'] must be a dict"
        assert isinstance(item["confidence"], dict), (
            f"Element {i}['confidence'] must be a dict"
        )


# ---------------------------------------------------------------------------
# Test 3: Integration — normalize_fields and merge_product are applied to output
# ---------------------------------------------------------------------------

def test_pipeline_integration():
    """
    Mock ingest_folder to return 2 products.
    Verify that the output fields dict passes through the normalizer
    (e.g., WEIGHT is uppercased, N/A values are coerced to '').
    """
    from backend.pipeline import process_folder

    # One product with denormalised values that the validator should fix
    buckets = {
        "test product": [
            (
                {**{f: "" for f in IMDB_FIELDS}, "ITEM_NAME": "Test Product",
                 "WEIGHT": "500ml", "BARCODE": "123-456", "BRAND": "N/A"},
                {f: "high" for f in IMDB_FIELDS},
            ),
            (
                {**{f: "" for f in IMDB_FIELDS}, "ITEM_NAME": "Test Product",
                 "WEIGHT": "500ml", "BARCODE": "123-456", "BRAND": "TestBrand"},
                {f: "high" for f in IMDB_FIELDS},
            ),
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("backend.ingestor.ingest_folder", return_value=buckets):
            results = process_folder(tmpdir)

    assert len(results) == 1
    product = results[0]

    # Validator should have uppercased WEIGHT unit
    assert product["fields"]["WEIGHT"] == "500ML", (
        f"Expected normalised '500ML', got {repr(product['fields']['WEIGHT'])}"
    )

    # Validator should have stripped dashes from BARCODE
    assert product["fields"]["BARCODE"] == "123456", (
        f"Expected stripped '123456', got {repr(product['fields']['BARCODE'])}"
    )

    # N/A coercion: 'N/A' in one image and 'TestBrand' in another — winner after
    # merge should be 'TestBrand' (non-empty fills), then normalised (not 'N/A')
    # At minimum, BRAND must not be 'N/A' in the final output
    assert product["fields"]["BRAND"] != "N/A", (
        f"Validator should coerce 'N/A' to ''; got {repr(product['fields']['BRAND'])}"
    )
