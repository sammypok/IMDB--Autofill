"""
validator.py — Field normalization rules for IMDB extraction output.

Normalizes raw Claude extraction values to ground-truth format before
downstream export. All functions are pure (no side effects).

Requirements covered:
  NORM-01: WEIGHT normalized to uppercase unit, original spacing preserved
  NORM-02: BARCODE contains only digits (spaces, dashes, letters stripped)
  NORM-03: PACKAGING TYPE fully uppercased
  NORM-04: None / N/A / none / null coerced to empty string
"""

from __future__ import annotations

import re

from .models import IMDB_FIELDS, PACKAGING_TYPE_KEY

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_WEIGHT_RE = re.compile(r'^(\d+(?:\.\d+)?)\s*([a-zA-Z]+)$')

# All string values considered "missing" — compared case-insensitively after strip()
_BAD_VALUES = {"n/a", "na", "none", "null", "n.a.", "n.a"}


# ---------------------------------------------------------------------------
# normalize_weight
# ---------------------------------------------------------------------------

def normalize_weight(value: str) -> str:
    """Return WEIGHT with the unit uppercased; original spacing preserved.

    Examples:
        '500ml'   → '500ML'    (no space, unit uppercased)
        '500 ml'  → '500 ML'   (space preserved, unit uppercased)
        '1.5 kg'  → '1.5 KG'   (decimal + space)
        '250g'    → '250G'     (no space)
        ''        → ''
    """
    stripped = value.strip()
    if not stripped:
        return ""
    m = _WEIGHT_RE.match(stripped)
    if m:
        num, unit = m.group(1), m.group(2)
        # Detect whether the original had whitespace between number and unit
        has_space = bool(re.search(r'\d\s+[a-zA-Z]', stripped))
        sep = " " if has_space else ""
        return f"{num}{sep}{unit.upper()}"
    # Unrecognized format — return as-is to avoid data loss
    return stripped


# ---------------------------------------------------------------------------
# normalize_barcode
# ---------------------------------------------------------------------------

def normalize_barcode(value: str) -> str:
    """Strip all non-digit characters from BARCODE (spaces, dashes, letters)."""
    return re.sub(r'\D', '', value)


# ---------------------------------------------------------------------------
# normalize_packaging_type
# ---------------------------------------------------------------------------

def normalize_packaging_type(value: str) -> str:
    """Return PACKAGING TYPE fully uppercased."""
    return value.upper()


# ---------------------------------------------------------------------------
# normalize_fields — main entry point
# ---------------------------------------------------------------------------

def normalize_fields(fields: dict[str, str]) -> dict[str, str]:
    """Normalize all IMDB fields according to ground-truth format rules.

    Returns a NEW dict — the input is never mutated.

    Normalization applied:
      - WEIGHT: uppercase unit, preserve spacing
      - BARCODE: digits only
      - PACKAGING  TYPE: fully uppercased  (two spaces — verified against IMDB_FIELDS)
      - VARIANT, TYPE, FRAGRANCE_FLAVOR: fully uppercased to match ground truth
      - All 13 fields: None / 'N/A' / 'none' / 'null' → ''
    """
    result = dict(fields)  # COPY — never mutate input

    result["WEIGHT"] = normalize_weight(result.get("WEIGHT", ""))
    result["BARCODE"] = normalize_barcode(result.get("BARCODE", ""))
    result[PACKAGING_TYPE_KEY] = normalize_packaging_type(result.get(PACKAGING_TYPE_KEY, ""))
    # Ground truth uses uppercase for these fields
    for _uc_field in ("VARIANT", "TYPE", "FRAGRANCE_FLAVOR"):
        if result.get(_uc_field):
            result[_uc_field] = result[_uc_field].upper()

    # Global N/A / null coercion — applies to ALL 13 fields, not just the three above.
    # Iterates IMDB_FIELDS so field names are never hardcoded here.
    for key in IMDB_FIELDS:
        val = result.get(key)
        if val is None or str(val).strip().lower() in _BAD_VALUES:
            result[key] = ""

    return result
