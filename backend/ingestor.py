"""
ingestor.py -- Two-stage product extraction pipeline.

Stage 1 -- Visual grouping:
    Images are grouped by filename prefix (e.g. S221234199_*.jpg).
    Files that don't match the convention fall back to CLIP visual similarity.

Stage 2 -- Field extraction (vlm_caller.py):
    GPT-4o extracts the 13 IMDB fields from each product group in parallel.
    Up to MAX_PARALLEL_EXTRACTIONS groups are processed simultaneously.

Public API:
    ingest_folder(images_dir, on_total_known=None, on_group_done=None)
        -> tuple[dict[str, list[tuple[dict, dict]]], float]
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .models import IMDB_FIELDS  # noqa: F401
from . import embedder, vlm_caller

logger = logging.getLogger(__name__)

MAX_PARALLEL_EXTRACTIONS = 8  # simultaneous GPT-4o calls

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _group_key(extractions: list[tuple[dict, dict]], paths: list[str]) -> str:
    """Pick the most informative bucket key from a group's extractions.

    Priority: BRAND+TYPE -> BRAND alone -> ITEM_NAME -> filename stem.
    """
    for fields, _ in extractions:
        brand = fields.get("BRAND", "").strip().lower()
        type_ = fields.get("TYPE", "").strip().lower()
        if brand and type_:
            return f"{brand}_{type_}"

    for fields, _ in extractions:
        brand = fields.get("BRAND", "").strip().lower()
        if brand:
            return brand

    for fields, _ in extractions:
        name = fields.get("ITEM_NAME", "").strip().lower()
        if name:
            return name

    return Path(paths[0]).stem.lower()


def _filename_prefix(path: str) -> str | None:
    """Return the product-ID prefix from filenames like 'S221234199_550719011.jpg'.

    Returns the part before the first underscore if it looks like an ID
    (alphanumeric, len >= 4). Returns None for files that don't follow
    this convention.
    """
    stem = Path(path).stem  # e.g. 'S221234199_550719011'
    if "_" not in stem:
        return None
    prefix = stem.split("_")[0]
    if len(prefix) >= 4 and prefix.replace("-", "").isalnum():
        return prefix
    return None


def _group_by_filename(image_paths: list[str]) -> tuple[list[list[str]], list[str]]:
    """Group images by filename prefix. Returns (groups, ungrouped_paths)."""
    prefix_map: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for path in image_paths:
        prefix = _filename_prefix(path)
        if prefix:
            prefix_map.setdefault(prefix, []).append(path)
        else:
            ungrouped.append(path)
    return list(prefix_map.values()), ungrouped


def _extract_one(
    group_idx: int,
    group_paths: list[str],
) -> tuple[int, list[str], dict, dict]:
    """Worker: extract fields for one product group (runs in a thread pool)."""
    fields, confidence = vlm_caller.extract_from_group(group_paths)
    if not fields.get("ITEM_NAME", "").strip():
        stem = Path(group_paths[0]).stem
        fields["ITEM_NAME"] = stem
        logger.warning("Empty ITEM_NAME for group %d -- using filename: %s", group_idx, stem)
    return group_idx, group_paths, fields, confidence


def ingest_folder(
    images_dir: str,
    on_total_known=None,
    on_group_done=None,
) -> tuple[dict[str, list[tuple[dict, dict]]], float]:
    """Run the two-stage pipeline on a flat directory of product images.

    Args:
        on_total_known: optional callback(n: int) fired once Stage 1 finishes,
                        before any extraction starts. Use to set job.total.
        on_group_done:  optional callback() fired each time one group finishes
                        extraction. Use to increment job.processed.

    Returns:
        (buckets, elapsed_seconds) where buckets maps product key to
        list of (fields, confidence) tuples.
    """
    image_paths = sorted(
        str(p) for p in Path(images_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        logger.warning("No images found in %s", images_dir)
        return {}, 0.0

    # ------------------------------------------------------------------
    # Stage 1: Group by filename prefix (e.g. 'S221234199_*.jpg')
    #          Fall back to CLIP for files without the convention.
    # ------------------------------------------------------------------
    prefix_groups, ungrouped = _group_by_filename(image_paths)
    logger.info(
        "Stage 1 (filename): %d image(s) -> %d prefix group(s), %d ungrouped.",
        len(image_paths), len(prefix_groups), len(ungrouped),
    )

    if ungrouped:
        logger.info("Stage 1 (CLIP): embedding %d ungrouped image(s)...", len(ungrouped))
        embeddings = embedder.embed_images(ungrouped)
        clip_groups = embedder.cluster_images(ungrouped, embeddings)
        groups = prefix_groups + clip_groups
    else:
        groups = prefix_groups

    logger.info(
        "Stage 1 complete: %d image(s) -> %d product group(s).",
        len(image_paths), len(groups),
    )

    if on_total_known is not None:
        on_total_known(len(groups))

    # ------------------------------------------------------------------
    # Stage 2: Parallel GPT-4o extraction
    # All groups are submitted at once; up to MAX_PARALLEL_EXTRACTIONS
    # run simultaneously. Reduces wall-clock time from O(N) to roughly
    # O(ceil(N / MAX_PARALLEL_EXTRACTIONS)).
    # ------------------------------------------------------------------
    results_map: dict[int, tuple[list[str], dict, dict]] = {}
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_EXTRACTIONS) as pool:
        futures = {
            pool.submit(_extract_one, idx, paths): idx
            for idx, paths in enumerate(groups)
        }
        for future in as_completed(futures):
            group_idx, group_paths, fields, confidence = future.result()
            results_map[group_idx] = (group_paths, fields, confidence)
            if on_group_done is not None:
                on_group_done()

    elapsed = time.monotonic() - t0
    logger.info(
        "Stage 2 complete: %d group(s) extracted in %.1fs (parallel, max_workers=%d).",
        len(groups), elapsed, MAX_PARALLEL_EXTRACTIONS,
    )

    # ------------------------------------------------------------------
    # Build ordered buckets (preserve original group order)
    # ------------------------------------------------------------------
    buckets: dict[str, list[tuple[dict, dict]]] = {}
    for group_idx in sorted(results_map):
        group_paths, fields, confidence = results_map[group_idx]
        extractions: list[tuple[dict, dict]] = [(fields, confidence)]
        key = _group_key(extractions, group_paths)
        if key in buckets:
            key = f"{key}_{group_idx}"
        buckets[key] = extractions
        logger.info(
            "Group %d -> %r (%d image(s)): BRAND=%r TYPE=%r",
            group_idx, key, len(group_paths),
            fields.get("BRAND"), fields.get("TYPE"),
        )

    return buckets, elapsed
