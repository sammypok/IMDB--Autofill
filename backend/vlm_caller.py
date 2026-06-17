"""
vlm_caller.py — OpenAI Vision extraction module.

Encodes a product image and calls GPT-4o with strict function calling,
returning two dicts (field_values, confidence_values) with exactly 13 IMDB_FIELDS keys.

Exports:
    extract_from_image   — main extraction entry point
    encode_image_for_claude — image pre-processing utility
    EXTRACTION_TOOL      — tool schema constant (26 properties, strict=True)
"""

from __future__ import annotations

import base64
import json
import logging
import time
from io import BytesIO

import openai
from dotenv import load_dotenv
from PIL import Image

from .models import IMDB_FIELDS, PACKAGING_TYPE_KEY

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy client initialisation
# ---------------------------------------------------------------------------
_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI()  # reads OPENAI_API_KEY from env
    return _client


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
MAX_EDGE_PX = 1568


def encode_image_for_claude(image_path: str) -> tuple[str, str]:
    """Resize to <=1568px long edge, convert to JPEG, return (base64_str, media_type)."""
    img = Image.open(image_path)
    if max(img.size) > MAX_EDGE_PX:
        img.thumbnail((MAX_EDGE_PX, MAX_EDGE_PX), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

def _schema_key(field: str) -> str:
    """Convert field name to valid API property key (spaces → underscores)."""
    return field.replace(" ", "_")


def _build_extraction_tool() -> dict:
    """Build OpenAI function-calling tool schema from IMDB_FIELDS."""
    _field_descriptions: dict[str, str] = {
        "ITEM_NAME": "Full product name as shown on label or product name tag at bottom of image",
        "BARCODE": "Numeric digits only, no spaces or dashes. Empty string if not visible.",
        "MANUFACTURER": "Company that manufactured the product",
        "BRAND": "Brand name (may differ from manufacturer)",
        "WEIGHT": "Weight or volume with unit (e.g. '500g', '1L', '200ml'). Empty if not shown.",
        PACKAGING_TYPE_KEY: (
            "Packaging form in UPPERCASE (e.g. 'BOTTLE', 'BOX', 'BAG', 'TUBE', 'CAN')"
        ),
        "COUNTRY": "Country of origin or manufacture",
        "VARIANT": (
            "Product variant or size variant (e.g. 'Original', 'Extra Strong'). "
            "Empty if not applicable."
        ),
        "TYPE": "Product category or type (e.g. 'Shampoo', 'Detergent', 'Biscuit')",
        "FRAGRANCE_FLAVOR": "Scent or flavor name. Empty string if not applicable.",
        "PROMOTION": (
            "Any promotional text on pack (e.g. '20% Extra Free'). Empty string if none."
        ),
        "ADDONS": "Any free item or bundle included. Empty string if none.",
        "TAGLINE": "Marketing tagline or slogan on the product. Empty string if not visible.",
    }

    properties: dict = {}
    for field in IMDB_FIELDS:
        key = _schema_key(field)
        properties[key] = {
            "type": "string",
            "description": _field_descriptions[field],
        }
        properties[f"{key}_confidence"] = {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": f"Your confidence in the {field} extraction",
        }

    required = [_schema_key(f) for f in IMDB_FIELDS] + [
        f"{_schema_key(f)}_confidence" for f in IMDB_FIELDS
    ]

    return {
        "type": "function",
        "function": {
            "name": "extract_product_fields",
            "description": (
                "Extract structured product catalog data from a retail product image. "
                "Read all label text carefully. Use empty string for any field not clearly legible. "
                "Do NOT guess or infer values you cannot read directly."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


EXTRACTION_TOOL: dict = _build_extraction_tool()

EXTRACTION_PROMPT = (
    "You are a retail product recognition expert. You identify products using BOTH visual recognition "
    "AND text reading — exactly like a human who works in a supermarket and can recognise any product "
    "from its colors, logo shape, packaging design, and label text, even when looking at the back or side."
    "\n\n"
    "IMAGE TAG AT THE BOTTOM — READ THIS FIRST:"
    "\nEvery product image has a descriptive text tag/label at the very bottom of the image. "
    "This tag contains the authoritative product name for the catalog. "
    "Read this bottom tag FIRST and use it as the primary source for ITEM_NAME. "
    "It overrides anything else you read from the packaging itself."
    "\n\n"
    "VISUAL RECOGNITION: Look at the overall visual identity — dominant colors, logo style, "
    "bottle/box shape, and design patterns. These are consistent on ALL faces of the product. "
    "Use this to anchor BRAND and fill gaps when a back or side image has less text."
    "\n\n"
    "CRITICAL DISTINCTIONS:"
    "\n- BRAND vs MANUFACTURER: These are DIFFERENT fields and must NEVER be confused."
    "\n  BRAND = the consumer-facing product brand printed prominently on the front label "
    "(e.g. 'Sunsilk', 'Ariel', 'Coca-Cola', 'Milo'). This is what a customer asks for in a shop."
    "\n  MANUFACTURER = the corporate/legal entity that made it, usually in small text on the back "
    "('Manufactured by Unilever Ghana Ltd', 'Procter & Gamble', 'Nestlé S.A.'). "
    "If you see 'Unilever' or 'P&G' or 'Nestlé' as prominent back-label text, "
    "put it in MANUFACTURER — NOT in BRAND or ITEM_NAME."
    "\n- ITEM_NAME = the full commercial product name. Read it from the bottom image tag first. "
    "Never use a manufacturer company name as ITEM_NAME."
    "\n\nFIELD GUIDANCE:"
    "\n- BARCODE: numeric digits only, no spaces or dashes. Empty if not visible."
    "\n- WEIGHT: weight or volume with unit (e.g. '500G', '1L', '200ML'). Empty if not shown."
    "\n- PACKAGING TYPE: container form in UPPERCASE (BOTTLE, BOX, BAG, TUBE, CAN, SACHET, JAR, TUB, GLASS JAR, etc.)."
    "\n- COUNTRY: country of origin or manufacture."
    "\n- VARIANT: variant in UPPERCASE (e.g. 'ORIGINAL', 'EXTRA STRONG'). Empty if none."
    "\n- TYPE: product category in UPPERCASE (e.g. 'SHAMPOO', 'DETERGENT', 'BISCUIT', 'MARGARINE')."
    "\n- FRAGRANCE_FLAVOR: scent or flavor in UPPERCASE (e.g. 'ROSE', 'ORIGINAL'). Empty if none."
    "\n- PROMOTION: promotional text on pack ('20% EXTRA FREE', 'NEW IMPROVED'). Empty if none."
    "\n- ADDONS: free item bundled with product (e.g. 'SPOON INCLUDED'). Empty if none."
    "\n- TAGLINE: marketing slogan on pack. Empty if none."
    "\n\nUse empty string only when a field is genuinely absent. Never leave BRAND or ITEM_NAME empty."
    "\n\nCONFIDENCE RULES — be decisive, not hedging:"
    "\n- Mark confidence 'high' whenever you can clearly read or visually confirm the value."
    "\n- Mark confidence 'high' when you are certain a field does NOT appear on the product (empty string + high = confident absence)."
    "\n- Mark confidence 'medium' only when the text is partially legible or you are inferring."
    "\n- Mark confidence 'low' ONLY when you genuinely cannot determine the value at all."
    "\n- Do NOT default to 'medium' out of caution — if you can read it, say 'high'."
)


# ---------------------------------------------------------------------------
# Primary extraction functions
# ---------------------------------------------------------------------------

def _parse_tool_response(tool_input: dict) -> tuple[dict, dict]:
    """Parse raw GPT-4o function-call output into (fields, confidence) dicts."""
    fields = {f: str(tool_input.get(_schema_key(f), "")).strip() for f in IMDB_FIELDS}
    confidence = {
        f: tool_input.get(f"{_schema_key(f)}_confidence", "low") for f in IMDB_FIELDS
    }
    return fields, confidence


def extract_from_group(
    image_paths: list[str],
    max_retries: int = 3,
) -> tuple[dict, dict]:
    """Call GPT-4o Vision with ALL images of a product in a single request.

    Sending all angles at once lets GPT-4o reason across images — e.g. read
    the barcode from the back, the brand from the front, and the weight from
    the side — in one pass, just like a human would.

    Returns:
        (field_values, confidence_values) — both dicts have exactly IMDB_FIELDS keys.
    """
    client = _get_client()

    # Build content blocks: one per image, then the prompt text
    content: list[dict] = []
    for path in image_paths:
        b64_data, _ = encode_image_for_claude(path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_data}",
                "detail": "high",
            },
        })

    n = len(image_paths)
    batch_prompt = (
        f"You are looking at {n} image(s) of the SAME product taken from different angles. "
        "Use ALL images together to fill the fields — read the barcode from whichever image shows it, "
        "the weight from whichever side has it, the brand from the front, etc. "
        "The bottom image tag in any of these images contains the authoritative product name.\n\n"
        + EXTRACTION_PROMPT
    )
    content.append({"type": "text", "text": batch_prompt})

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=1024,
                temperature=0,
                seed=42,
                timeout=60,
                tools=[EXTRACTION_TOOL],
                tool_choice={
                    "type": "function",
                    "function": {"name": "extract_product_fields"},
                },
                messages=[{"role": "user", "content": content}],
            )

            tool_call = response.choices[0].message.tool_calls[0]
            tool_input: dict = json.loads(tool_call.function.arguments)

            logger.info(
                "Batch extraction (%d image(s)): ITEM_NAME=%r BRAND=%r WEIGHT=%r",
                n,
                tool_input.get("ITEM_NAME"),
                tool_input.get("BRAND"),
                tool_input.get("WEIGHT"),
            )

            return _parse_tool_response(tool_input)

        except openai.RateLimitError as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("Rate limit hit (attempt %d/%d). Retrying in %ds.", attempt + 1, max_retries, wait)
                time.sleep(wait)
                continue
            logger.error("Rate limit exhausted after %d attempt(s) for batch of %d image(s): %s", max_retries, n, exc)
        except openai.APIError as exc:
            logger.error("OpenAI API error (batch %d images): %s", n, exc)
            break

    logger.error("Returning empty extraction for batch of %d image(s) after all retries.", n)
    return {f: "" for f in IMDB_FIELDS}, {f: "low" for f in IMDB_FIELDS}


def extract_from_image(
    image_path: str,
    max_retries: int = 3,
) -> tuple[dict, dict]:
    """Call GPT-4o Vision on a single product image (fallback / legacy)."""
    return extract_from_group([image_path], max_retries=max_retries)
