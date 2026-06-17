"""
exporter.py — Pure functions to convert pipeline results into file bytes.

Both functions are stateless and produce output entirely in memory.
They are imported by main.py export endpoints and are independently testable.

All column ordering is driven by IMDB_FIELDS — field names are NEVER hardcoded here.
"""

from __future__ import annotations

import csv
import io

import openpyxl

from backend.models import IMDB_FIELDS


def build_csv_bytes(results: list[dict]) -> bytes:
    """Return UTF-8 CSV bytes for the given pipeline results.

    The header row is exactly IMDB_FIELDS (including 'PACKAGING  TYPE' with
    two spaces). Each data row writes empty string for any missing field —
    never None or null.

    Args:
        results: List of result dicts with shape
                 {"fields": {field: value, ...}, "confidence": {...}}.
                 An empty list produces a header-only CSV without crashing.

    Returns:
        UTF-8 encoded bytes suitable for an HTTP response or file write.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)

    # Header row — exact column order from constant (never literals)
    writer.writerow(IMDB_FIELDS)

    # Data rows — empty string fallback, never None
    for r in results:
        writer.writerow([r["fields"].get(f, "") or "" for f in IMDB_FIELDS])

    return buf.getvalue().encode("utf-8-sig")  # BOM makes Excel auto-detect UTF-8


def build_xlsx_bytes(results: list[dict]) -> bytes:
    """Return XLSX bytes for the given pipeline results.

    The header row is exactly IMDB_FIELDS. Each data row writes empty string
    for any missing field — never None or null. The returned bytes can be
    opened directly by openpyxl.load_workbook(io.BytesIO(...)).

    Args:
        results: List of result dicts with shape
                 {"fields": {field: value, ...}, "confidence": {...}}.
                 An empty list produces a header-only workbook without crashing.

    Returns:
        Raw XLSX bytes suitable for an HTTP response or file write.
    """
    wb = openpyxl.Workbook()
    ws = wb.active

    # Header row — exact column order from constant (never literals)
    ws.append(list(IMDB_FIELDS))

    # Data rows — empty string fallback, never None
    for r in results:
        ws.append([r["fields"].get(f, "") or "" for f in IMDB_FIELDS])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.read()
