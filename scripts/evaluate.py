#!/usr/bin/env python3
"""
evaluate.py — Standalone evaluation CLI for IMDB extraction results.

Usage:
    python scripts/evaluate.py <predictions_csv> [ground_truth_xlsx]

Reads a predictions CSV and the ground truth output_results.xlsx, joins rows
by ITEM_NAME (case-insensitive), and reports per-column accuracy percentages
plus an overall score.

Also importable as a module:
    from scripts.evaluate import load_ground_truth, load_predictions, evaluate, print_report
"""
import csv
import os
import sys

# Allow running from project root without package install — consistent with
# scripts/test_single.py pattern.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openpyxl

from backend.models import IMDB_FIELDS


def load_ground_truth(xlsx_path: str) -> list[dict]:
    """Load ground truth data from an xlsx file.

    Args:
        xlsx_path: Path to the ground truth Excel file.

    Returns:
        List of dicts, one per data row. Headers are taken from the first row.
        Trailing empty columns (None headers) are skipped. Entirely-empty rows
        are also skipped. Values are NOT stripped or uppercased here — callers
        handle normalization.
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active

    rows_iter = ws.iter_rows(values_only=True)
    header_row = next(rows_iter, None)
    if header_row is None:
        return []

    # Skip None header cells (handles trailing empty columns in xlsx)
    headers = [h for h in header_row if h is not None]

    result = []
    for row in rows_iter:
        row_values = row[: len(headers)]
        # Skip entirely-empty rows
        if all(v is None for v in row_values):
            continue
        result.append(dict(zip(headers, row_values)))

    return result


def load_predictions(csv_path: str) -> list[dict]:
    """Load prediction rows from a CSV file.

    Args:
        csv_path: Path to the predictions CSV file.

    Returns:
        List of dicts directly from csv.DictReader.
    """
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def evaluate(
    ground_truth: list[dict], predictions: list[dict]
) -> tuple[dict, int, int]:
    """Compare predictions against ground truth and compute per-field accuracy.

    Join is performed on ITEM_NAME, case-insensitively (both sides stripped and
    uppercased). Unmatched prediction rows produce a stderr warning — they do
    NOT crash the script.

    Args:
        ground_truth: List of ground truth dicts (from load_ground_truth).
        predictions:  List of prediction dicts (from load_predictions).

    Returns:
        Tuple of:
          - per_col: dict mapping each IMDB_FIELDS name to {'correct': int, 'total': int}
          - matched: number of products successfully joined
          - gt_total: total number of ground truth products
    """
    gt_by_name = {
        str(r.get("ITEM_NAME", "") or "").strip().upper(): r for r in ground_truth
    }
    pred_by_name = {
        str(r.get("ITEM_NAME", "") or "").strip().upper(): r for r in predictions
    }

    matched_keys = set(gt_by_name) & set(pred_by_name)
    unmatched_keys = set(pred_by_name) - set(gt_by_name)

    for key in sorted(unmatched_keys):
        print(f"WARNING: no ground truth match for '{key}'", file=sys.stderr)

    per_col: dict[str, dict[str, int]] = {
        f: {"correct": 0, "total": 0} for f in IMDB_FIELDS
    }

    for key in matched_keys:
        gt_row = gt_by_name[key]
        pred_row = pred_by_name[key]
        for field in IMDB_FIELDS:
            gt_val = str(gt_row.get(field, "") or "").strip().upper()
            pred_val = str(pred_row.get(field, "") or "").strip().upper()
            per_col[field]["total"] += 1
            if gt_val == pred_val:
                per_col[field]["correct"] += 1

    return per_col, len(matched_keys), len(ground_truth)


def print_report(per_col: dict, matched: int, gt_total: int) -> None:
    """Print a human-readable accuracy report to stdout.

    Args:
        per_col:   Per-field accuracy dict from evaluate().
        matched:   Number of products matched.
        gt_total:  Total number of ground truth products.
    """
    print(f"Matched {matched}/{gt_total} products")

    total_correct = 0
    total_all = 0

    for field in IMDB_FIELDS:
        correct = per_col[field]["correct"]
        total = per_col[field]["total"]
        acc = correct / total * 100 if total > 0 else 0.0
        print(f"  {field:<20} {acc:5.1f}%")
        total_correct += correct
        total_all += total

    overall = total_correct / total_all * 100 if total_all > 0 else 0.0
    print(f"\nOverall accuracy: {overall:.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/evaluate.py <predictions_csv> [ground_truth_xlsx]",
            file=sys.stderr,
        )
        sys.exit(1)

    predictions_csv = sys.argv[1]
    ground_truth_xlsx = sys.argv[2] if len(sys.argv) >= 3 else "output_results.xlsx"

    gt = load_ground_truth(ground_truth_xlsx)
    preds = load_predictions(predictions_csv)
    per_col, matched, gt_total = evaluate(gt, preds)
    print_report(per_col, matched, gt_total)
