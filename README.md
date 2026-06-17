# IMDB AutoFill

AI-powered product catalog autofill — extracts IMDB fields from any product image in real time using vision AI and CLIP grouping.

## What it does

Upload any product photo — taken right now with your phone, downloaded from the web, or from an existing dataset — and the system automatically extracts all IMDB catalog fields:

- Item name, brand, manufacturer, country of origin
- Weight/volume, packaging type, variant, type
- Fragrance/flavor, promotions, taglines, add-ons

Multiple images of the same product are grouped automatically using CLIP visual similarity, then merged into a single catalog entry.

## Key Features

### Real-Time Image Processing
There is no fixed dataset. Upload product images at any time — photos taken on the spot, scanned packaging, or bulk image folders — and the system extracts structured catalog data instantly. New uploads start processing immediately in the background while you continue working.

### AI Chat Assistant
Once images are processed, a built-in chat assistant lets you ask questions about your extracted product catalog:
- *"How many products are from Ghana?"*
- *"List all products by Unilever"*
- *"Which products have promotions?"*
- *"Compare the weights of these two products"*

The assistant has full context of your catalog and answers in real time.

### Smart Product Grouping
Multiple photos of the same product (different angles, lighting, packaging sides) are automatically detected and merged into one catalog entry using CLIP visual similarity — no manual grouping needed.

### Export Anywhere
Download your completed catalog as **CSV** or **XLSX**, ready to import into any spreadsheet or database.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| Vision AI | Vision API — 13-field extraction via function calling |
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

1. Click **Upload Images** and select any product photos (taken live or from files)
2. The system groups images by product using CLIP visual similarity
3. Vision AI extracts all IMDB fields from each group in real time
4. Results appear in a table — edit any field inline if needed
5. Export as **CSV** or **XLSX**
6. Use the **chat assistant** to ask questions about the extracted catalog

## Project Structure

```
backend/
  main.py        # FastAPI app — upload, status, results, export, chat endpoints
  ingestor.py    # Orchestrates CLIP grouping + VLM extraction pipeline
  embedder.py    # CLIP-based image embedding and clustering
  vlm_caller.py  # Vision API calls with retry/backoff
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
