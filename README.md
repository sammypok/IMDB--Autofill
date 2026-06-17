# IMDB AutoFill

AI-powered product catalog autofill — extracts IMDB fields from product images using vision AI and CLIP grouping.

## What it does

Upload product images and the system automatically extracts all IMDB catalog fields:

- Item name, brand, manufacturer, country of origin
- Weight/volume, packaging type, variant, type
- Fragrance/flavor, promotions, taglines, add-ons

Multiple images of the same product are grouped automatically using CLIP visual similarity, then merged into a single catalog entry.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Vision AI | GPT-4o Vision — 13-field extraction via function calling |
| Image Grouping | CLIP (openai/clip-vit-base-patch32) via HuggingFace |
| Frontend | Single-page HTML/JS (no framework) |
| Export | CSV (UTF-8 BOM) and XLSX |

## Setup

**Requirements:** Python 3.10+

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Environment Variables

```env
OPENAI_API_KEY=sk-proj-...        # Required — OpenAI API key
AUTOFILL_API_KEY=                 # Optional — protect the API with a key
```

## Running

```bash
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Usage

1. Click **Upload Images** and select one or more product photos
2. The system groups images by product using CLIP visual similarity
3. Vision AI extracts all IMDB fields from each group
4. Results appear in a table — edit any field inline if needed
5. Export as **CSV** or **XLSX**
6. Use the **chat assistant** to ask questions about the extracted catalog

## Project Structure

```
backend/
  main.py        # FastAPI app — upload, status, results, export, chat endpoints
  ingestor.py    # Orchestrates CLIP grouping + VLM extraction pipeline
  embedder.py    # CLIP-based image embedding and clustering
  vlm_caller.py  # GPT-4o Vision API calls with retry/backoff
  aggregator.py  # Merges multi-image extractions into one product record
  validator.py   # Normalizes and validates extracted field values
  exporter.py    # Builds CSV and XLSX bytes from results
  models.py      # IMDB_FIELDS constant and shared types
frontend/
  index.html     # Single-page app — upload, progress, results table, chat
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Upload images, returns `{job_id}` |
| `GET` | `/status/{job_id}` | Poll job progress |
| `GET` | `/results/{job_id}` | Fetch full results when done |
| `GET` | `/export/{job_id}/csv` | Download predictions.csv |
| `GET` | `/export/{job_id}/xlsx` | Download predictions.xlsx |
| `POST` | `/chat` | Product catalog chat assistant |
