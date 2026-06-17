"""
Unit tests for backend/ingestor.py — ingest_folder().

All tests use unittest.mock.patch to mock backend.vlm_caller.extract_from_image.
No live API calls are made; minimal JPEG fixtures created with tempfile.

Run with: python -m pytest tests/test_ingestor.py -v

RED state: These tests will fail with ImportError until backend/ingestor.py is
implemented in Plan 02-02.
"""

from __future__ import annotations

import logging
import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image

from backend.models import IMDB_FIELDS


# ---------------------------------------------------------------------------
# Helper: create a minimal JPEG file and return its path
# ---------------------------------------------------------------------------

def _make_jpeg(directory: str, filename: str) -> str:
    """Create a small JPEG file in the given directory and return its path."""
    path = os.path.join(directory, filename)
    img = Image.new("RGB", (50, 50), color="white")
    img.save(path, format="JPEG")
    return path


def _make_extraction(item_name: str) -> tuple[dict, dict]:
    """Return a (fields, confidence) tuple with the given ITEM_NAME."""
    fields = {f: "" for f in IMDB_FIELDS}
    fields["ITEM_NAME"] = item_name
    confidence = {f: "low" for f in IMDB_FIELDS}
    return fields, confidence


# ---------------------------------------------------------------------------
# Test 1: Images are grouped into buckets by ITEM_NAME
# ---------------------------------------------------------------------------

def test_groups_images_by_item_name():
    """ingest_folder groups images by their ITEM_NAME extraction result."""
    from backend.ingestor import ingest_folder

    with tempfile.TemporaryDirectory() as tmpdir:
        img_a1 = _make_jpeg(tmpdir, "a1.jpg")
        img_a2 = _make_jpeg(tmpdir, "a2.jpg")
        img_b1 = _make_jpeg(tmpdir, "b1.jpg")

        side_effects = [
            _make_extraction("Milk Powder"),
            _make_extraction("Milk Powder"),
            _make_extraction("Dettol Soap"),
        ]

        with patch("backend.vlm_caller.extract_from_image", side_effect=side_effects):
            buckets = ingest_folder(tmpdir)

    # Two distinct products
    assert len(buckets) == 2
    # Each bucket must have the correct image count
    milk_key = next(k for k in buckets if "milk" in k)
    soap_key = next(k for k in buckets if "dettol" in k)
    assert len(buckets[milk_key]) == 2
    assert len(buckets[soap_key]) == 1


# ---------------------------------------------------------------------------
# Test 2: Grouping key is lowercase + stripped
# ---------------------------------------------------------------------------

def test_grouping_key_is_lowercase_stripped():
    """'  Milk Powder  ' and 'milk powder' must map to the same bucket."""
    from backend.ingestor import ingest_folder

    with tempfile.TemporaryDirectory() as tmpdir:
        img1 = _make_jpeg(tmpdir, "img1.jpg")
        img2 = _make_jpeg(tmpdir, "img2.jpg")

        side_effects = [
            _make_extraction("  Milk Powder  "),
            _make_extraction("milk powder"),
        ]

        with patch("backend.vlm_caller.extract_from_image", side_effect=side_effects):
            buckets = ingest_folder(tmpdir)

    # Both images must map to the same bucket
    assert len(buckets) == 1, (
        f"Expected 1 bucket for normalised 'milk powder', got {len(buckets)}: {list(buckets.keys())}"
    )
    only_bucket = next(iter(buckets.values()))
    assert len(only_bucket) == 2


# ---------------------------------------------------------------------------
# Test 3: Image with empty ITEM_NAME is skipped; logger.warning called
# ---------------------------------------------------------------------------

def test_empty_name_tag_skipped(caplog):
    """Images where ITEM_NAME=='' are not added to any bucket."""
    from backend.ingestor import ingest_folder

    with tempfile.TemporaryDirectory() as tmpdir:
        img_ok = _make_jpeg(tmpdir, "ok.jpg")
        img_bad = _make_jpeg(tmpdir, "bad.jpg")

        side_effects = [
            _make_extraction("Milk Powder"),
            _make_extraction(""),  # empty name → must be skipped
        ]

        with caplog.at_level(logging.WARNING, logger="backend.ingestor"):
            with patch("backend.vlm_caller.extract_from_image", side_effect=side_effects):
                buckets = ingest_folder(tmpdir)

    # Only the valid product forms a bucket
    assert len(buckets) == 1
    # A warning must have been issued
    assert any("ITEM_NAME" in record.message or "empty" in record.message.lower()
               for record in caplog.records), (
        "Expected a logger.warning for the empty ITEM_NAME image"
    )


# ---------------------------------------------------------------------------
# Test 4: A single-image product forms a bucket of size 1
# ---------------------------------------------------------------------------

def test_single_image_product():
    """Product with one image forms a correctly-sized bucket."""
    from backend.ingestor import ingest_folder

    with tempfile.TemporaryDirectory() as tmpdir:
        img = _make_jpeg(tmpdir, "solo.jpg")

        side_effects = [_make_extraction("Solo Product")]

        with patch("backend.vlm_caller.extract_from_image", side_effect=side_effects):
            buckets = ingest_folder(tmpdir)

    assert len(buckets) == 1
    only_bucket = next(iter(buckets.values()))
    assert len(only_bucket) == 1


# ---------------------------------------------------------------------------
# Test 5: Bucket count != 45 triggers logger.warning (not exception)
# ---------------------------------------------------------------------------

def test_bucket_count_warning(caplog):
    """If the number of distinct products != 45, a warning is logged."""
    from backend.ingestor import ingest_folder

    with tempfile.TemporaryDirectory() as tmpdir:
        # 3 products != 45 → should trigger warning
        images_and_names = [
            ("prod1.jpg", "Product Alpha"),
            ("prod2.jpg", "Product Beta"),
            ("prod3.jpg", "Product Gamma"),
        ]
        side_effects = []
        for filename, name in images_and_names:
            _make_jpeg(tmpdir, filename)
            side_effects.append(_make_extraction(name))

        with caplog.at_level(logging.WARNING, logger="backend.ingestor"):
            with patch("backend.vlm_caller.extract_from_image", side_effect=side_effects):
                buckets = ingest_folder(tmpdir)

    # Buckets are returned regardless
    assert len(buckets) == 3
    # A warning about the unexpected bucket count must have been issued
    assert any("45" in record.message or "bucket" in record.message.lower()
               for record in caplog.records), (
        "Expected a logger.warning when bucket count != 45"
    )
