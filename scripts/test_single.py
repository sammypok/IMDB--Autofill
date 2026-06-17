#!/usr/bin/env python3
"""Smoke test: extract IMDB fields from a single product image via Claude Vision."""
import sys
import json
from pathlib import Path

# Allow running from project root: python scripts/test_single.py path/to/image.jpg
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.vlm_caller import extract_from_image
from backend.models import IMDB_FIELDS


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_single.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not Path(image_path).exists():
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    print(f"\nExtracting fields from: {image_path}\n{'='*60}")
    fields, confidence = extract_from_image(image_path)

    # Verify schema compliance
    assert set(fields.keys()) == set(IMDB_FIELDS), (
        f"Key mismatch!\nExpected: {IMDB_FIELDS}\nGot: {list(fields.keys())}"
    )
    assert all(v is not None for v in fields.values()), "None value in fields -- must be empty string"
    assert all(v != "N/A" for v in fields.values()), "'N/A' found -- must be empty string"

    # Print results
    print(f"{'Field':<20} {'Value':<40} Confidence")
    print("-" * 75)
    for field in IMDB_FIELDS:
        value = fields[field] or "(empty)"
        conf = confidence[field]
        marker = " <-- NAME TAG" if field == "ITEM_NAME" else ""
        print(f"{field:<20} {value:<40} {conf}{marker}")

    empty_count = sum(1 for v in fields.values() if not v)
    print(f"\n{'='*60}")
    print(f"Fields populated: {13 - empty_count}/13")
    print(f"Fields empty: {empty_count}/13")
    print("\nSchema compliance: PASSED (all 13 keys, no None, no N/A)")
    print(f"\nRaw JSON:\n{json.dumps(fields, indent=2)}")


if __name__ == "__main__":
    main()
