"""
main.py — FastAPI server that wraps the Phase 2 pipeline as an async background job.

Endpoints:
    POST /upload                  — accepts images, returns {job_id} immediately
    GET  /status/{job_id}         — returns {status, processed, total}
    GET  /results/{job_id}        — returns full results array when done
    GET  /export/{job_id}/csv     — streams predictions.csv when done; 409 while running
    GET  /export/{job_id}/xlsx    — streams predictions.xlsx when done; 409 while running
    GET  /                        — serves frontend/index.html (StaticFiles mount)

Critical design rules:
  - asyncio.create_task used (NOT BackgroundTasks) for concurrent job execution
  - StaticFiles mounted LAST — mounting before API routes intercepts /status/, /export/ etc.
  - Task reference stored on JobState to prevent asyncio GC before completion
  - ingest_folder wrapped in asyncio.to_thread — sync call, must not block event loop
"""

from __future__ import annotations

import asyncio
import io
import csv
import json
import logging
import logging.config
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Literal

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
        }
    },
    "formatters": {
        "simple": {"format": "%(levelname)s %(name)s: %(message)s"}
    },
    "root": {"level": "WARNING", "handlers": ["console"]},
    "loggers": {
        "backend": {"level": "INFO", "propagate": True},
    },
})

import openai as _openai
import openpyxl
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.models import IMDB_FIELDS
from backend.exporter import build_csv_bytes, build_xlsx_bytes
from backend import ingestor as _ingestor
from backend import embedder as _embedder
from backend.aggregator import merge_product
from backend.validator import normalize_fields


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    status: Literal["queued", "running", "done", "error"] = "queued"
    processed: int = 0
    total: int = 0
    failed: list[str] = field(default_factory=list)
    results: list[dict] | None = None
    error: str | None = None
    images_dir: str | None = None
    elapsed_seconds: float = 0.0
    task: asyncio.Task | None = None  # holds reference to prevent GC
    completed_at: float | None = None  # monotonic timestamp when job reached done/error
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def increment_processed(self) -> None:
        with self._lock:
            self.processed += 1


# ---------------------------------------------------------------------------
# Module-level job store and app
# ---------------------------------------------------------------------------

JOBS: dict[str, JobState] = {}
app = FastAPI(title="IMDB AutoFill")

# ---------------------------------------------------------------------------
# Module-level OpenAI client for chat (re-used across requests)
# ---------------------------------------------------------------------------
_CHAT_CLIENT = _openai.OpenAI()

# ---------------------------------------------------------------------------
# Optional API key authentication
# AUTOFILL_API_KEY env var — if set, every request must include
# the header  X-API-Key: <value>.  Unset = auth disabled (local dev).
# ---------------------------------------------------------------------------
_REQUIRED_API_KEY: str | None = os.environ.get("AUTOFILL_API_KEY") or None


def _check_auth(x_api_key: str | None) -> None:
    if _REQUIRED_API_KEY and x_api_key != _REQUIRED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# ---------------------------------------------------------------------------
# Job TTL pruning — jobs older than JOB_TTL_SECONDS are evicted from JOBS
# ---------------------------------------------------------------------------
JOB_TTL_SECONDS = 3600  # 1 hour

# ---------------------------------------------------------------------------
# Startup: warm up the CLIP model in a background thread so the first
# upload with ungrouped images does not block the thread pool.
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    await asyncio.to_thread(_embedder._get_model)


def _prune_old_jobs() -> None:
    now = time.monotonic()
    stale = [
        jid for jid, j in JOBS.items()
        if j.completed_at is not None and (now - j.completed_at) > JOB_TTL_SECONDS
    ]
    for jid in stale:
        JOBS.pop(jid, None)
    if stale:
        logging.getLogger("backend").info("Pruned %d stale job(s).", len(stale))


# ---------------------------------------------------------------------------
# Background job coroutine
# ---------------------------------------------------------------------------

async def run_job(job_id: str, images_dir: str) -> None:
    """Run the full pipeline in the background for the given job."""
    job = JOBS[job_id]
    try:
        job.status = "running"

        # Callbacks let the progress bar update during parallel extraction
        def on_total_known(n: int) -> None:
            job.total = n

        def on_group_done() -> None:
            job.increment_processed()

        # Wrap synchronous ingest_folder so event loop stays responsive
        buckets, elapsed = await asyncio.to_thread(
            _ingestor.ingest_folder, images_dir, on_total_known, on_group_done
        )

        job.elapsed_seconds = elapsed
        results: list[dict] = []

        for product_key, extractions in buckets.items():
            try:
                merged_fields, merged_confidence = merge_product(extractions)
                normalized = normalize_fields(merged_fields)
                results.append({
                    "fields": normalized,
                    "confidence": merged_confidence,
                })
            except Exception:
                # Record fallback row — ITEM_NAME=product_key, all others empty
                job.failed.append(product_key)
                fallback = {f: "" for f in IMDB_FIELDS}
                fallback["ITEM_NAME"] = product_key
                results.append({
                    "fields": fallback,
                    "confidence": {f: "low" for f in IMDB_FIELDS},
                })

        job.results = results
        job.status = "done"
        job.completed_at = time.monotonic()

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        job.completed_at = time.monotonic()
    finally:
        # Always clean up temp dir
        shutil.rmtree(images_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

MAX_FILES_PER_UPLOAD = 50
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB per file


@app.post("/upload")
async def upload(
    files: Annotated[list[UploadFile], File()],
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict:
    """Accept uploaded images, start background processing, return job_id immediately."""
    _check_auth(x_api_key)
    _prune_old_jobs()

    if len(files) > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=422,
            detail=f"Too many files. Maximum is {MAX_FILES_PER_UPLOAD} per upload.",
        )

    job_id = str(uuid.uuid4())
    images_dir = tempfile.mkdtemp(prefix=f"imdb_job_{job_id}_")

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    for i, f in enumerate(files):
        # Filter non-image files — check content_type first, fall back to extension
        # (Windows file picker often sends empty content_type)
        filename = f.filename or f"img_{i}.jpg"
        ext = os.path.splitext(filename)[1].lower()
        is_image = (f.content_type or "").startswith("image/") or ext in IMAGE_EXTS
        if not is_image:
            continue
        content = await f.read()
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"File '{filename}' exceeds the 20 MB size limit.",
            )
        dest = os.path.join(images_dir, filename)
        with open(dest, "wb") as fh:
            fh.write(content)

    JOBS[job_id] = JobState(images_dir=images_dir)
    task = asyncio.create_task(run_job(job_id, images_dir))
    JOBS[job_id].task = task  # hold reference — prevents asyncio GC before completion

    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# GET /status/{job_id}
# ---------------------------------------------------------------------------

@app.get("/status/{job_id}")
async def get_status(
    job_id: str,
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict:
    """Return job progress: {status, processed, total}."""
    _check_auth(x_api_key)
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]
    response: dict = {
        "status": job.status,
        "processed": job.processed,
        "total": job.total,
    }
    if job.status == "error":
        response["error"] = job.error
    if job.status == "done" and job.failed:
        response["failed"] = job.failed

    return response


# ---------------------------------------------------------------------------
# GET /results/{job_id}
# ---------------------------------------------------------------------------

@app.get("/results/{job_id}")
async def get_results(
    job_id: str,
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict:
    """Return the full results array when job is done."""
    _check_auth(x_api_key)
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]
    if job.status != "done":
        raise HTTPException(
            status_code=409,
            detail={"error": "Job not complete", "status": job.status},
        )

    return {"results": job.results, "elapsed_seconds": job.elapsed_seconds}


# ---------------------------------------------------------------------------
# GET /export/{job_id}/csv
# ---------------------------------------------------------------------------

@app.get("/export/{job_id}/csv")
async def export_csv(
    job_id: str,
    x_api_key: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Stream predictions.csv when done; 409 while job is still running."""
    _check_auth(x_api_key)
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]
    if job.status != "done":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Job not complete",
                "status": job.status,
                "processed": job.processed,
                "total": job.total,
            },
        )

    data = build_csv_bytes(job.results)
    return StreamingResponse(
        iter([data]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )


# ---------------------------------------------------------------------------
# GET /export/{job_id}/xlsx
# ---------------------------------------------------------------------------

@app.get("/export/{job_id}/xlsx")
async def export_xlsx(
    job_id: str,
    x_api_key: Annotated[str | None, Header()] = None,
) -> StreamingResponse:
    """Stream predictions.xlsx when done; 409 while job is still running."""
    _check_auth(x_api_key)
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]
    if job.status != "done":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Job not complete",
                "status": job.status,
                "processed": job.processed,
                "total": job.total,
            },
        )

    data = build_xlsx_bytes(job.results)
    return StreamingResponse(
        iter([data]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="predictions.xlsx"'},
    )


# ---------------------------------------------------------------------------
# POST /chat — product-scoped conversational assistant
# ---------------------------------------------------------------------------

_CHAT_SYSTEM = (
    "You are a product catalog assistant. You help users understand and analyse "
    "product data that has been extracted from product images.\n\n"
    "You ONLY answer questions about:\n"
    "- Product names, brands, manufacturers and their relationships\n"
    "- Product specifications: weight, packaging type, country of origin, variant, "
    "type, fragrance or flavor\n"
    "- Promotions, taglines, add-ons found on products\n"
    "- Catalog analysis: counts by manufacturer, brand, country; comparisons; groupings\n"
    "- General product or brand knowledge\n\n"
    "If asked anything unrelated to products, politely say you can only help with "
    "product-related questions and suggest something they could ask instead.\n"
    "Be concise, factual, and friendly. When counting or grouping, be precise."
)

MAX_CHAT_PRODUCTS = 500
MAX_CHAT_MSG_LENGTH = 2000


class _ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=MAX_CHAT_MSG_LENGTH)


class _ChatRequest(BaseModel):
    messages: list[_ChatMessage] = Field(default_factory=list, max_length=40)
    job_id: str | None = None  # preferred: look up catalog server-side
    products: list[dict] = Field(default_factory=list, max_length=MAX_CHAT_PRODUCTS)


@app.post("/chat")
async def chat_endpoint(
    body: _ChatRequest,
    x_api_key: Annotated[str | None, Header()] = None,
) -> dict:
    """Product-scoped chat. Accepts {messages, job_id?, products?} and returns {reply}."""
    _check_auth(x_api_key)

    # Prefer server-side catalog lookup via job_id (avoids sending full catalog every turn)
    products: list[dict] = []
    if body.job_id and body.job_id in JOBS:
        job = JOBS[body.job_id]
        if job.status == "done" and job.results:
            products = [r["fields"] for r in job.results]
    elif body.products:
        products = body.products[:MAX_CHAT_PRODUCTS]

    if products:
        catalog_ctx = (
            f"\n\nCurrent product catalog ({len(products)} product(s)):\n"
            + json.dumps(products, ensure_ascii=False)
        )
    else:
        catalog_ctx = (
            "\n\nNo products have been loaded yet — the user has not uploaded any "
            "images. Remind them to upload product images first."
        )

    system_content = _CHAT_SYSTEM + catalog_ctx
    messages = [m.model_dump() for m in body.messages[-20:]]

    def _call() -> str:
        resp = _CHAT_CLIENT.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_content}] + messages,
            max_tokens=512,
            temperature=0.3,
            timeout=30,
        )
        return resp.choices[0].message.content

    reply = await asyncio.to_thread(_call)
    return {"reply": reply}


# ---------------------------------------------------------------------------
# StaticFiles mount — MUST be last (after all route definitions)
# StaticFiles with html=True catches unmatched paths and serves index.html
# ---------------------------------------------------------------------------
# NOTE: This mount is registered after the frontend/index.html is created (Task 3).
# If frontend/ does not exist, this line raises at import time.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
