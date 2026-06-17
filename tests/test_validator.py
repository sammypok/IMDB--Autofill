"""
Unit tests for backend/validator.py — normalize_fields() and helpers.

Tests cover NORM-01 (WEIGHT normalisation), NORM-02 (BARCODE sanitisation),
NORM-03 (PACKAGING TYPE uppercased), NORM-04 (N/A coercion to empty string).
No file I/O required — plain dicts are passed directly.

Run with: python -m pytest tests/test_validator.py -v

RED state: These tests will fail with ImportError until backend/validator.py is
implemented in Plan 02-03.
"""

from __future__ import annotations

import pytest

from backend.models import IMDB_FIELDS


# ---------------------------------------------------------------------------
# Helper: build a fields dict with one field overridden
# ---------------------------------------------------------------------------

def _fields(**overrides) -> dict:
    """Return an IMDB_FIELDS dict with all values '' except the given overrides."""
    result = {f: "" for f in IMDB_FIELDS}
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# NORM-01: WEIGHT normalisation (uppercase unit, preserve value)
# ---------------------------------------------------------------------------

def test_weight_no_space():
    """'500ml' → '500ML'"""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(WEIGHT="500ml"))
    assert out["WEIGHT"] == "500ML", f"Expected '500ML', got {repr(out['WEIGHT'])}"


def test_weight_with_space():
    """'1.5 kg' → '1.5 KG'"""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(WEIGHT="1.5 kg"))
    assert out["WEIGHT"] == "1.5 KG", f"Expected '1.5 KG', got {repr(out['WEIGHT'])}"


def test_weight_decimal():
    """'250.5g' → '250.5G'"""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(WEIGHT="250.5g"))
    assert out["WEIGHT"] == "250.5G", f"Expected '250.5G', got {repr(out['WEIGHT'])}"


def test_weight_empty():
    """Empty WEIGHT stays empty."""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(WEIGHT=""))
    assert out["WEIGHT"] == "", f"Expected '', got {repr(out['WEIGHT'])}"


# ---------------------------------------------------------------------------
# NORM-02: BARCODE sanitisation (strip dashes, spaces, letters)
# ---------------------------------------------------------------------------

def test_barcode_strip_dashes():
    """'1234-567' → '1234567'"""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(BARCODE="1234-567"))
    assert out["BARCODE"] == "1234567", f"Expected '1234567', got {repr(out['BARCODE'])}"


def test_barcode_strip_spaces():
    """'123 456 789' → '123456789'"""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(BARCODE="123 456 789"))
    assert out["BARCODE"] == "123456789", f"Expected '123456789', got {repr(out['BARCODE'])}"


def test_barcode_strip_letters():
    """'ABC123' → '123' (letters removed, digits kept)"""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(BARCODE="ABC123"))
    assert out["BARCODE"] == "123", f"Expected '123', got {repr(out['BARCODE'])}"


# ---------------------------------------------------------------------------
# NORM-03: PACKAGING TYPE uppercased
# ---------------------------------------------------------------------------

def test_packaging_type_uppercased():
    """'bottle' → 'BOTTLE'"""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(**{"PACKAGING  TYPE": "bottle"}))
    assert out["PACKAGING  TYPE"] == "BOTTLE", (
        f"Expected 'BOTTLE', got {repr(out['PACKAGING  TYPE'])}"
    )


# ---------------------------------------------------------------------------
# NORM-04: N/A coercion — "N/A", "none", "null" → ""
# ---------------------------------------------------------------------------

def test_na_coercion_na_string():
    """Field value 'N/A' → ''"""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(ITEM_NAME="N/A"))
    assert out["ITEM_NAME"] == "", f"Expected '' for 'N/A', got {repr(out['ITEM_NAME'])}"


def test_na_coercion_none_string():
    """Field value 'none' (case-insensitive) → ''"""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(BRAND="none"))
    assert out["BRAND"] == "", f"Expected '' for 'none', got {repr(out['BRAND'])}"


def test_na_coercion_null_string():
    """Field value 'null' (case-insensitive) → ''"""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(MANUFACTURER="null"))
    assert out["MANUFACTURER"] == "", (
        f"Expected '' for 'null', got {repr(out['MANUFACTURER'])}"
    )


# ---------------------------------------------------------------------------
# Additional: ITEM_NAME is not blanket-uppercased or lowercased
# ---------------------------------------------------------------------------

def test_item_name_not_lowercased():
    """ITEM_NAME 'Dettol Soap' must not be lowercased or uppercased."""
    from backend.validator import normalize_fields
    out = normalize_fields(_fields(ITEM_NAME="Dettol Soap"))
    assert out["ITEM_NAME"] == "Dettol Soap", (
        f"Expected 'Dettol Soap' unchanged, got {repr(out['ITEM_NAME'])}"
    )


# ---------------------------------------------------------------------------
# Additional: normalize_fields returns a new dict (does not mutate input)
# ---------------------------------------------------------------------------

def test_output_is_new_dict():
    """normalize_fields must not mutate the input dict."""
    from backend.validator import normalize_fields
    original = _fields(WEIGHT="500ml", BARCODE="123-456")
    original_copy = dict(original)

    out = normalize_fields(original)

    # Input must be unchanged
    assert original == original_copy, (
        "normalize_fields mutated the input dict"
    )
    # Output must be a different object
    assert out is not original, (
        "normalize_fields must return a new dict, not the same object"
    )
