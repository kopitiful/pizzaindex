"""
Use Claude to extract the Margherita price from a menu text.
Falls back here when the fast regex path in menu_extractor.py finds nothing.

Uses structured outputs + prompt caching on the system prompt.
"""

import logging
from typing import Optional

import anthropic

import config

log = logging.getLogger(__name__)

_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


_SYSTEM = (
    "Du bist ein Datenextraktions-Assistent für Speisekarten-Daten. "
    "Analysiere den übergebenen Speisekarten-Text und extrahiere den Preis "
    "der NORMALEN Pizza Margherita (Vor-Ort-Preis, Standard-Größe). "
    "WICHTIG – setze gefunden=false und preis=null wenn es sich handelt um: "
    "Party-Pizza, Partyblech, Blechpizza, Familienpizza, Meterpizza, "
    "Mini-Pizza, kleine Pizza, halbe Pizza oder einen einzelnen Pizza-Slice. "
    "Wenn mehrere Größen angeboten werden, wähle die mittlere/normale Größe "
    "(typisch ~28–36 cm), NICHT die kleinste. "
    "Ignoriere Allergen-Kennziffern in Klammern (z.B. '(1,7,24)') – "
    "das sind keine Preise. "
    "Antworte immer als JSON gemäß dem vorgegebenen Schema."
)

_PRICE_SCHEMA = {
    "type": "object",
    "properties": {
        "preis": {
            "type": ["number", "null"],
            "description": "Preis der Pizza Margherita in Euro, z.B. 8.9",
        },
        "groesse_cm": {
            "type": ["integer", "null"],
            "description": "Durchmesser in cm, falls angegeben",
        },
        "groesse_label": {
            "type": ["string", "null"],
            "description": "Größenbeschreibung, z.B. 'klein', 'groß', '30cm'",
        },
        "gefunden": {
            "type": "boolean",
            "description": "True wenn Pizza Margherita im Text vorkommt",
        },
    },
    "required": ["preis", "groesse_cm", "groesse_label", "gefunden"],
    "additionalProperties": False,
}


async def extract_price_from_image(image_data: bytes, media_type: str = "image/jpeg") -> Optional[dict]:
    """
    Ask Claude to find the Margherita price in a menu photo.
    image_data: raw image bytes; media_type: image/jpeg, image/png, etc.
    """
    import base64
    b64 = base64.standard_b64encode(image_data).decode()
    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=256,
            system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
            output_config={"format": {"type": "json_schema", "schema": _PRICE_SCHEMA}},
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {
                        "type": "text",
                        "text": "Extrahiere den Margherita-Preis aus diesem Speisekarten-Foto:",
                    },
                ],
            }],
        )
    except anthropic.APIError as e:
        log.warning("Claude Vision API error: %s", e)
        return None

    import json
    text_block = next((b for b in response.content if b.type == "text"), None)
    if not text_block:
        return None
    try:
        return json.loads(text_block.text)
    except json.JSONDecodeError as e:
        log.warning("JSON parse error: %s – raw: %s", e, text_block.text[:200])
        return None


async def extract_price(menu_text: str) -> Optional[dict]:
    """
    Ask Claude to find the Margherita price in the given menu text.
    Returns a dict with keys: preis, groesse_cm, groesse_label, gefunden.
    Returns None on API error.
    """
    # Truncate to keep costs reasonable (~2500 tokens ~ 10 KB of text)
    text = menu_text[:10_000]

    try:
        client = _get_client()
        response = await client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=256,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM,
                    "cache_control": {"type": "ephemeral"},  # cache stable system prompt
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _PRICE_SCHEMA,
                }
            },
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Extrahiere den Margherita-Preis aus diesem Speisekarten-Text:\n\n"
                        f"{text}"
                    ),
                }
            ],
        )
    except anthropic.APIError as e:
        log.warning("Claude API error: %s", e)
        return None

    import json

    text_block = next(
        (b for b in response.content if b.type == "text"), None
    )
    if not text_block:
        return None

    try:
        return json.loads(text_block.text)
    except json.JSONDecodeError as e:
        log.warning("JSON parse error: %s – raw: %s", e, text_block.text[:200])
        return None
