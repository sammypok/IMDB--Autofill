"""
test_main.py — FastAPI TestClient tests for all HTTP API endpoints.

Tests use direct JOBS dict manipulation to pre-seed done jobs for export tests,
so no live Anthropic API calls are made during testing.

Endpoints tested:
    POST /upload
    GET  /status/{job_id}
    GET  /results/{job_id}
    GET  /export/{job_id}/csv
    GET  /export/{job_id}/xlsx
"""

from __future__ import annotations

import csv
import io
import uuid

import openpyxl
import pytest
from fastapi.testclient import TestClient

from backend.main import JOBS, JobState, app
from backend.models import IMDB_FIELDS

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_jobs():
    """Clear JOBS after each test to prevent state bleed."""
    yield
    JOBS.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_done_job(n_results: int = 2) -> str:
    """Insert a done job into JOBS and return its job_id."""
    job_id = str(uuid.uuid4())
    results = [
        {
            "fields": {f: f"val_{i}_{j}" for i, f in enumerate(IMDB_FIELDS)},
            "confidence": {},
        }
        for j in range(n_results)
    ]
    JOBS[job_id] = JobState(
        status="done",
        processed=n_results,
        total=n_results,
        results=results,
    )
    return job_id


# ---------------------------------------------------------------------------
# Upload tests
# ---------------------------------------------------------------------------


def test_upload_returns_job_id():
    """POST /upload with a fake JPEG returns 200 with a job_id."""
    resp = client.post(
        "/upload",
        files=[("files", ("img.jpg", b"fake_bytes", "image/jpeg"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert isinstance(body["job_id"], str)


def test_upload_non_image_filtered():
    """POST /upload with a text file returns 200 — non-image silently filtered."""
    resp = client.post(
        "/upload",
        files=[("files", ("readme.txt", b"hello world", "text/plain"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body


# ---------------------------------------------------------------------------
# Status tests
# ---------------------------------------------------------------------------


def test_status_unknown_job_returns_404():
    """GET /status/{unknown_id} returns 404."""
    resp = client.get("/status/nonexistent-id")
    assert resp.status_code == 404


def test_status_queued_job():
    """POST /upload then immediately GET /status returns a valid status."""
    upload_resp = client.post(
        "/upload",
        files=[("files", ("img.jpg", b"fake_bytes", "image/jpeg"))],
    )
    job_id = upload_resp.json()["job_id"]

    status_resp = client.get(f"/status/{job_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert "status" in body
    assert body["status"] in {"queued", "running", "done", "error"}


# ---------------------------------------------------------------------------
# Export tests — 409 before job done
# ---------------------------------------------------------------------------


def test_export_csv_before_done_returns_409():
    """GET /export/{job_id}/csv while job is running returns 409."""
    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobState(status="running", processed=1, total=5)
    resp = client.get(f"/export/{job_id}/csv")
    assert resp.status_code == 409
    # FastAPI wraps HTTPException detail under a "detail" key
    assert "status" in resp.json()["detail"]


def test_export_xlsx_before_done_returns_409():
    """GET /export/{job_id}/xlsx while job is running returns 409."""
    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobState(status="running", processed=1, total=5)
    resp = client.get(f"/export/{job_id}/xlsx")
    assert resp.status_code == 409
    # FastAPI wraps HTTPException detail under a "detail" key
    assert "status" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Export tests — done job
# ---------------------------------------------------------------------------


def test_export_csv_done_returns_file():
    """GET /export/{job_id}/csv on a done job streams valid CSV."""
    job_id = _make_done_job(2)
    resp = client.get(f"/export/{job_id}/csv")

    assert resp.status_code == 200
    assert "predictions.csv" in resp.headers.get("content-disposition", "")

    reader = csv.reader(io.StringIO(resp.text))
    rows = list(reader)
    assert len(rows) >= 3  # header + 2 data rows
    assert rows[0] == list(IMDB_FIELDS)
    assert len(rows) - 1 == 2  # exactly 2 data rows


def test_export_xlsx_done_returns_file():
    """GET /export/{job_id}/xlsx on a done job streams valid XLSX bytes."""
    job_id = _make_done_job(2)
    resp = client.get(f"/export/{job_id}/xlsx")

    assert resp.status_code == 200
    assert "predictions.xlsx" in resp.headers.get("content-disposition", "")

    wb = openpyxl.load_workbook(io.BytesIO(resp.content))
    ws = wb.active
    first_row = [cell.value for cell in next(ws.iter_rows(max_row=1))]
    assert first_row == list(IMDB_FIELDS)


# ---------------------------------------------------------------------------
# Results tests
# ---------------------------------------------------------------------------


def test_results_unknown_job_returns_404():
    """GET /results/{unknown_id} returns 404."""
    resp = client.get("/results/nonexistent")
    assert resp.status_code == 404


def test_results_done_job():
    """GET /results/{job_id} on a done job returns results array of correct length."""
    job_id = _make_done_job(3)
    resp = client.get(f"/results/{job_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 3


# ---------------------------------------------------------------------------
# Export unknown job returns 404
# ---------------------------------------------------------------------------


def test_export_unknown_job_returns_404():
    """GET /export/{unknown_id}/csv returns 404."""
    resp = client.get("/export/nonexistent/csv")
    assert resp.status_code == 404
