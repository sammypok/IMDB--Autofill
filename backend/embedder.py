"""
embedder.py — CLIP-based visual similarity grouping.

Stage 1 of the two-stage pipeline:
    embed_images(paths)          → (N, D) normalized embedding matrix
    cluster_images(paths, embs)  → list of groups (each group = list of paths)

Uses openai/clip-vit-base-patch32 via HuggingFace transformers.
Model is ~600MB and downloads automatically on first run.
Embeddings are L2-normalized so cosine similarity = dot product.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        logger.info("Loading CLIP model (first run downloads ~600 MB — please wait)...")
        from transformers import CLIPModel, CLIPProcessor
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _model.eval()
        logger.info("CLIP model ready.")
    return _model, _processor


def embed_images(image_paths: list[str]) -> np.ndarray:
    """Encode each image with CLIP and return an (N, D) normalized float32 array."""
    import torch

    model, processor = _get_model()
    vectors = []

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            vision_out = model.vision_model(**inputs)
            features = model.visual_projection(vision_out.pooler_output)
        # L2-normalize so dot product == cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)
        vectors.append(features.squeeze().cpu().numpy())

    return np.array(vectors, dtype=np.float32)


def cluster_images(
    image_paths: list[str],
    embeddings: np.ndarray,
    threshold: float = 0.82,
) -> list[list[str]]:
    """Group image paths by visual similarity using greedy centroid matching.

    Each image is compared to all existing group centroids. If the best
    cosine similarity exceeds *threshold*, the image joins that group and the
    centroid is updated. Otherwise a new group is created.

    Args:
        image_paths: Ordered list of image file paths.
        embeddings:  Corresponding (N, D) normalized embedding matrix.
        threshold:   Cosine similarity cutoff. 0.82 works well for product
                     packaging across angles; lower if products are very similar.

    Returns:
        List of groups — each group is a list of image path strings.
    """
    groups: list[list[str]] = []
    centroids: list[np.ndarray] = []

    for path, emb in zip(image_paths, embeddings):
        best_sim = -1.0
        best_idx = -1

        for i, centroid in enumerate(centroids):
            sim = float(np.dot(emb, centroid))  # cosine sim (both normalized)
            logger.info("  compare %s vs group %d: sim=%.4f", Path(path).name, i, sim)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim >= threshold:
            groups[best_idx].append(path)
            # Running average centroid, renormalized
            n = len(groups[best_idx])
            updated = centroids[best_idx] * (n - 1) / n + emb / n
            norm = np.linalg.norm(updated)
            centroids[best_idx] = updated / norm if norm > 0 else updated
            logger.info(
                "CLIP: %s → group %d (sim=%.3f, size=%d)",
                Path(path).name, best_idx, best_sim, n,
            )
        else:
            groups.append([path])
            centroids.append(emb.copy())
            logger.info(
                "CLIP: %s → new group %d",
                Path(path).name, len(groups) - 1,
            )

    return groups
