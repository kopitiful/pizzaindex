"""
Fetch Google Maps ratings for priced pizzerias via the Places API (Text Search).
Requires GOOGLE_API_KEY in .env.

Usage:
    python pipeline.py fetch-ratings --city köln
"""

import asyncio
import logging
from typing import Optional

import httpx
from tqdm import tqdm

import config
import database as db

log = logging.getLogger(__name__)

_PLACES_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"


async def _fetch_one(
    pizzeria_id: int,
    name: str,
    city: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> bool:
    async with semaphore:
        query = f"{name} {city}"
        try:
            resp = await client.get(
                _PLACES_URL,
                params={"query": query, "key": config.GOOGLE_API_KEY, "language": "de"},
                timeout=10,
            )
            data = resp.json()
        except Exception as e:
            log.debug("Places API failed for %r: %s", name, e)
            return False

        results = data.get("results", [])
        if not results:
            log.debug("No Places result for %r", name)
            return False

        place = results[0]
        rating = place.get("rating")
        review_count = place.get("user_ratings_total")
        if rating is None:
            return False

        db.update_google_rating(pizzeria_id, rating, review_count)
        log.debug("[%s] rating=%.1f (%d reviews)", name, rating, review_count or 0)
        return True


async def run(
    city: Optional[str] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
):
    if not config.GOOGLE_API_KEY:
        print("No GOOGLE_API_KEY set in .env — skipping ratings.")
        return

    db.init_db()
    rows = db.get_priced_pizzerias(city=city, bbox=bbox)
    if not rows:
        print("No priced pizzerias found.")
        return

    print(f"Fetching Google ratings for {len(rows)} pizzerias…")
    semaphore = asyncio.Semaphore(5)
    found = 0

    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [
            _fetch_one(r["id"], r["name"], r.get("city") or city or "", client, semaphore)
            for r in rows
        ]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Ratings"):
            if await coro:
                found += 1

    print(f"Fetched ratings for {found}/{len(rows)} pizzerias.")
