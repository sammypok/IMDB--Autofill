"""
test_evaluate.py — Unit tests for scripts/evaluate.py core logic.

Tests cover the `evaluate` function directly using inline fixture data.
File I/O functions (load_ground_truth / load_predictions) are not tested here
because they depend on external files; integration tests are out of scope.
"""
import sys
import os

# Allow imports from project root without package install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import evaluate
from backend.models import IMDB_FIELDS


def _make_row(**overrides) -> dict:
    """Return a minimal IMDB row with all fields set to empty string.

    Keyword arguments override specific fields.
    """
    row = {f: "" for f in IMDB_FIELDS}
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Test 1: Perfect match
# ---------------------------------------------------------------------------

def test_evaluate_perfect_match():
    """Ground truth and predictions identical on one product — all fields 1/1."""
    gt = [_make_row(ITEM_NAME="TESTPROD", BARCODE="1234567890", BRAND="BrandX")]
    preds = [_make_row(ITEM_NAME="TESTPROD", BARCODE="1234567890", BRAND="BrandX")]

    per_col, matched, gt_total = evaluate(gt, preds)

    assert matched == 1
    assert gt_total == 1
    for field in IMDB_FIELDS:
        assert per_col[field]["total"] == 1, f"{field}: expected total=1"
        assert per_col[field]["correct"] == 1, f"{field}: expected correct=1"


# ---------------------------------------------------------------------------
# Test 2: Case-insensitive join on ITEM_NAME
# ---------------------------------------------------------------------------

def test_evaluate_case_insensitive_join():
    """ITEM_NAME join must be case-insensitive: 'Palmolive' == 'PALMOLIVE'."""
    gt = [_make_row(ITEM_NAME="Palmolive", BARCODE="111")]
    preds = [_make_row(ITEM_NAME="PALMOLIVE", BARCODE="111")]

    per_col, matched, gt_total = evaluate(gt, preds)

    # Must match — not be counted as unmatched
    assert matched == 1, "Expected 1 matched product after case-insensitive join"
    assert gt_total == 1


# ---------------------------------------------------------------------------
# Test 3: Field value normalization (strip + upper)
# ---------------------------------------------------------------------------

def test_evaluate_field_value_normalization():
    """WEIGHT '250g' vs '250G' — after .strip().upper() these are equal."""
    gt = [_make_row(ITEM_NAME="SOAP", WEIGHT="250g")]
    preds = [_make_row(ITEM_NAME="SOAP", WEIGHT="250G")]

    per_col, matched, gt_total = evaluate(gt, preds)

    assert matched == 1
    weight_stats = per_col["WEIGHT"]
    assert weight_stats["total"] == 1
    assert weight_stats["correct"] == 1, (
        "WEIGHT '250g' vs '250G' should normalize to equal after .strip().upper()"
    )


# ---------------------------------------------------------------------------
# Test 4: No match returns zero totals
# ---------------------------------------------------------------------------

def test_evaluate_no_match_returns_zero_totals():
    """Prediction ITEM_NAME not in ground truth — totals stay 0, matched = 0."""
    gt = [_make_row(ITEM_NAME="KNOWN_PROD", BARCODE="999")]
    preds = [_make_row(ITEM_NAME="UNKNOWN_PROD", BARCODE="999")]

    per_col, matched, gt_total = evaluate(gt, preds)

    assert matched == 0, "No rows should match when ITEM_NAME differs"
    assert gt_total == 1
    for field in IMDB_FIELDS:
        assert per_col[field]["total"] == 0, f"{field}: expected total=0"
        assert per_col[field]["correct"] == 0, f"{field}: expected correct=0"


# ---------------------------------------------------------------------------
# Test 5: Partial match
# ---------------------------------------------------------------------------

def test_evaluate_partial_match():
    """2 products in ground truth, 1 matched, 1 unmatched — matched count = 1."""
    gt = [
        _make_row(ITEM_NAME="PROD_A", BARCODE="001"),
        _make_row(ITEM_NAME="PROD_B", BARCODE="002"),
    ]
    preds = [
        _make_row(ITEM_NAME="PROD_A", BARCODE="001"),
        _make_row(ITEM_NAME="PROD_C", BARCODE="003"),  # unmatched
    ]

    per_col, matched, gt_total = evaluate(gt, preds)

    assert matched == 1, "Only PROD_A should match"
    assert gt_total == 2
    # Per-field totals should reflect only the 1 matched product
    for field in IMDB_FIELDS:
        assert per_col[field]["total"] == 1, f"{field}: expected total=1"


# ---------------------------------------------------------------------------
# Test 6: PACKAGING  TYPE two-space key — no KeyError
# ---------------------------------------------------------------------------

def test_evaluate_packaging_type_two_space_key():
    """'PACKAGING  TYPE' (two spaces) must be accessed without KeyError."""
    two_space_key = "PACKAGING  TYPE"

    # Confirm the key is actually in IMDB_FIELDS (guards against silent regression)
    assert two_space_key in IMDB_FIELDS, (
        f"'{two_space_key}' not found in IMDB_FIELDS — check models.py"
    )

    gt = [_make_row(ITEM_NAME="TESTPKG", **{two_space_key: "Bottle"})]
    preds = [_make_row(ITEM_NAME="TESTPKG", **{two_space_key: "Bottle"})]

    # Should not raise KeyError
    per_col, matched, gt_total = evaluate(gt, preds)

    assert matched == 1
    pkg_stats = per_col[two_space_key]
    assert pkg_stats["total"] == 1
    assert pkg_stats["correct"] == 1, (
        "PACKAGING  TYPE 'Bottle' vs 'Bottle' should match"
    )
