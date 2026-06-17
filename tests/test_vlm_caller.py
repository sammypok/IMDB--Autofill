"""
Unit tests for backend/vlm_caller.py.

All tests use mocked anthropic client — no live API calls are made.
Run with: python -m pytest tests/test_vlm_caller.py -v
"""

from __future__ import annotations

import base64
import tempfile
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import anthropic

from backend.models import IMDB_FIELDS
from backend.vlm_caller import EXTRACTION_TOOL, encode_image_for_claude, extract_from_image


# ---------------------------------------------------------------------------
# Helper: build a fake tool-use response that the mocked client returns
# ---------------------------------------------------------------------------

def _make_fake_response(tool_input: dict) -> MagicMock:
    """Return a MagicMock mimicking anthropic.types.Message with one tool_use block."""
    content_block = MagicMock()
    content_block.input = tool_input
    response = MagicMock()
    response.content = [content_block]
    return response


def _full_tool_input() -> dict:
    """All 13 IMDB fields populated with dummy values + all 13 confidence = 'high'."""
    data: dict = {}
    for field in IMDB_FIELDS:
        data[field] = f"value_for_{field}"
        data[f"{field}_confidence"] = "high"
    return data


# ---------------------------------------------------------------------------
# Test 1: Tool schema covers all IMDB_FIELDS
# ---------------------------------------------------------------------------

def test_tool_schema_covers_all_fields():
    props = EXTRACTION_TOOL["input_schema"]["properties"]

    # All 13 IMDB_FIELDS keys must be present
    for field in IMDB_FIELDS:
        assert field in props, f"Field '{field}' missing from tool schema properties"

    # All 13 confidence keys must be present
    for field in IMDB_FIELDS:
        conf_key = f"{field}_confidence"
        assert conf_key in props, f"Confidence key '{conf_key}' missing from tool schema"

    # Exactly 26 properties total (13 fields + 13 confidence)
    assert len(props) == 26, f"Expected 26 properties, got {len(props)}"

    # strict mode enabled
    assert EXTRACTION_TOOL["strict"] is True, "EXTRACTION_TOOL must have strict=True"

    # No extra properties allowed
    assert EXTRACTION_TOOL["input_schema"]["additionalProperties"] is False

    # Verify 'PACKAGING  TYPE' with two spaces is present (not one space, not underscore)
    assert "PACKAGING  TYPE" in props, (
        "'PACKAGING  TYPE' (two spaces) must be a key in tool schema properties"
    )


# ---------------------------------------------------------------------------
# Test 2: extract_from_image returns all 13 keys
# ---------------------------------------------------------------------------

def test_extract_returns_all_13_keys():
    fake_response = _make_fake_response(_full_tool_input())
    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    with patch("backend.vlm_caller._get_client", return_value=mock_client):
        # Use a dummy path — encode_image_for_claude is called inside, so we need a real image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(tmp_path, format="JPEG")

        try:
            fields, confidence = extract_from_image(tmp_path)
        finally:
            os.unlink(tmp_path)

    # Exactly the 13 IMDB_FIELDS keys — no more, no less
    assert set(fields.keys()) == set(IMDB_FIELDS), (
        f"fields keys mismatch. Got: {set(fields.keys())}"
    )
    assert set(confidence.keys()) == set(IMDB_FIELDS), (
        f"confidence keys mismatch. Got: {set(confidence.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 3: Missing fields in tool_input become empty string (never None or "N/A")
# ---------------------------------------------------------------------------

def test_unknown_fields_become_empty_string():
    # Only 5 fields populated in the fake API response; rest are absent
    partial_input: dict = {}
    populated_fields = IMDB_FIELDS[:5]
    for field in populated_fields:
        partial_input[field] = f"val_{field}"
        partial_input[f"{field}_confidence"] = "medium"
    # No entries for the other 8 fields

    fake_response = _make_fake_response(partial_input)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = fake_response

    with patch("backend.vlm_caller._get_client", return_value=mock_client):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            img = Image.new("RGB", (50, 50), color="green")
            img.save(tmp_path, format="JPEG")

        try:
            fields, confidence = extract_from_image(tmp_path)
        finally:
            os.unlink(tmp_path)

    # All 13 keys must be present
    assert set(fields.keys()) == set(IMDB_FIELDS)

    # Missing fields must be empty string, not None, not "N/A"
    for field in IMDB_FIELDS[5:]:
        val = fields[field]
        assert val == "", f"Missing field '{field}' should be '' but got {repr(val)}"
        assert val is not None, f"Missing field '{field}' must not be None"


# ---------------------------------------------------------------------------
# Test 4: encode_image_for_claude resizes to <=1568px (real Pillow, no API)
# ---------------------------------------------------------------------------

def test_encode_image_for_claude():
    # Large image that must be downsized
    img = Image.new("RGB", (3000, 2000), color="red")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        img.save(tmp_path, format="JPEG")

    try:
        b64_data, media_type = encode_image_for_claude(tmp_path)
    finally:
        os.unlink(tmp_path)

    assert media_type == "image/jpeg"

    # Decode and open to verify dimensions
    raw_bytes = base64.b64decode(b64_data)
    decoded_img = Image.open(BytesIO(raw_bytes))
    assert max(decoded_img.size) <= 1568, (
        f"Long edge {max(decoded_img.size)} exceeds 1568px limit"
    )


# ---------------------------------------------------------------------------
# Test 5: anthropic.BadRequestError returns fallback dicts (no exception raised)
# ---------------------------------------------------------------------------

def test_bad_request_error_returns_fallback():
    mock_response = MagicMock()
    mock_response.status_code = 400

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = anthropic.BadRequestError(
        "test error",
        response=mock_response,
        body={},
    )

    with patch("backend.vlm_caller._get_client", return_value=mock_client):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            img = Image.new("RGB", (50, 50), color="white")
            img.save(tmp_path, format="JPEG")

        try:
            # Must NOT raise any exception
            fields, confidence = extract_from_image(tmp_path)
        finally:
            os.unlink(tmp_path)

    expected_fields = {f: "" for f in IMDB_FIELDS}
    expected_confidence = {f: "low" for f in IMDB_FIELDS}

    assert fields == expected_fields, (
        f"BadRequestError fallback fields mismatch. Got: {fields}"
    )
    assert confidence == expected_confidence, (
        f"BadRequestError fallback confidence mismatch. Got: {confidence}"
    )
