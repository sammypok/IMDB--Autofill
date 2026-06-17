"""
models.py — Authoritative field definitions for IMDB extraction.

IMDB_FIELDS is the single source of truth for all 13 product label fields.
Every downstream module (vlm_caller, exporter, scorer) imports from here.

Field names are copied verbatim from output_results.xlsx header row.
Do NOT rename, snake_case, or otherwise transform these names.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# IMDB_FIELDS — authoritative ordered list of the 13 extraction target fields
# Copied verbatim from output_results.xlsx header row (verified at module init).
# NOTE: 'PACKAGING  TYPE' contains TWO spaces — this is the exact xlsx value.
# Use PACKAGING_TYPE_KEY whenever you need this string to avoid silent typos.
# ---------------------------------------------------------------------------
IMDB_FIELDS: list[str] = [
    "ITEM_NAME",
    "BARCODE",
    "MANUFACTURER",
    "BRAND",
    "WEIGHT",
    "PACKAGING  TYPE",
    "COUNTRY",
    "VARIANT",
    "TYPE",
    "FRAGRANCE_FLAVOR",
    "PROMOTION",
    "ADDONS",
    "TAGLINE",
]

# Safe constant for the two-space field name — use this instead of the literal
PACKAGING_TYPE_KEY: str = "PACKAGING  TYPE"


# ---------------------------------------------------------------------------
# Verification — run `python -m backend.models` to confirm IMDB_FIELDS is
# still in sync with the ground-truth Excel file.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import openpyxl

    XLSX_PATH = "output_results.xlsx"

    try:
        wb = openpyxl.load_workbook(XLSX_PATH)
    except FileNotFoundError:
        print(f"ERROR: {XLSX_PATH} not found. Run from project root.", file=sys.stderr)
        sys.exit(1)

    ws = wb.active
    actual_headers: list[str] = [
        cell.value for cell in next(ws.iter_rows(max_row=1))
        if cell.value is not None
    ]

    print("Ground truth headers from xlsx:")
    for i, h in enumerate(actual_headers):
        print(f"  {i:2d}: {repr(h)}")

    print()
    print("IMDB_FIELDS constant:")
    for i, h in enumerate(IMDB_FIELDS):
        print(f"  {i:2d}: {repr(h)}")

    print()

    if actual_headers == IMDB_FIELDS:
        print("IMDB_FIELDS verified.")
        sys.exit(0)
    else:
        mismatches = []
        if len(actual_headers) != len(IMDB_FIELDS):
            mismatches.append(
                f"Length mismatch: xlsx={len(actual_headers)}, constant={len(IMDB_FIELDS)}"
            )
        for i, (a, b) in enumerate(zip(actual_headers, IMDB_FIELDS)):
            if a != b:
                mismatches.append(f"  Index {i}: xlsx={repr(a)}, constant={repr(b)}")
        print("ERROR: IMDB_FIELDS does NOT match xlsx header row!", file=sys.stderr)
        for m in mismatches:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)
