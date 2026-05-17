"""
Google integration for the pizza pipeline:

  1. Places API  — find missing website URLs + ratings for any country
  2. Custom Search API — find the right menu page directly via Google search

Requires in .env:
  GOOGLE_API_KEY=...
  GOOGLE_CSE_ID=...   (programmablesearchengine.google.com, "search entire web")
"""

import asyncio
import logging
from typing import Optional
from urllib.parse import urlparse

import httpx

import config
import database as db

log = logging.getLogger(__name__)

_PLACES_SEARCH_URL  = "https://maps.googleapis.com/maps/api/place/textsearch/json"
_PLACES_DETAIL_URL  = "https://maps.googleapis.com/maps/api/place/details/json"
_SEARCH_URL         = "https://www.googleapis.com/customsearch/v1"

# Pricing (per 1,000 requests, after free $200/month credit)
_PLACES_COST_PER_1K  = 32.0   # Text Search basic data
_SEARCH_COST_PER_DAY_FREE = 100  # Custom Search: 100/day free, then $5/1k

_semaphore = asyncio.Semaphore(5)


_PLACES_FREE_MONTHLY = 200.0  # $200 free credit per month

def _cost_estimate(n_places: int, n_search: int = 0) -> tuple[float, str]:
    # 2 calls per pizzeria: Text Search + Place Details
    gross = (n_places * 2 / 1000) * _PLACES_COST_PER_1K
    lines = [f"  Places API:  {n_places} Pizzerien × 2 Calls  ≈ ${gross:.2f} (aus ${_PLACES_FREE_MONTHLY:.0f}/Monat Guthaben)"]
    if n_search:
        search_free = min(n_search, _SEARCH_COST_PER_DAY_FREE)
        search_paid = max(0, n_search - _SEARCH_COST_PER_DAY_FREE)
        search_cost = (search_paid / 1000) * 5.0
        lines.append(f"  Custom Search: {n_search} Anfragen  ({search_free} kostenlos + {search_paid} kostenpflichtig ≈ ${search_cost:.2f})")
        gross += search_cost
    over_budget = max(0.0, gross - _PLACES_FREE_MONTHLY)
    lines.append(f"  Über Guthaben: ${over_budget:.2f}" if over_budget > 0 else "  Innerhalb des kostenlosen Guthabens ✓")
    return over_budget, "\n".join(lines)


def _confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [j/N] ").strip().lower() in ("j", "ja", "y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ── Places API ────────────────────────────────────────────────────────────────

async def find_via_places(
    name: str,
    city: str,
    client: httpx.AsyncClient,
) -> Optional[dict]:
    """
    Search Google Places for a pizzeria by name+city.
    Step 1: Text Search → place_id + rating
    Step 2: Place Details → website
    Returns dict with website, rating, review_count — or None.
    """
    if not config.GOOGLE_API_KEY:
        return None
    async with _semaphore:
        try:
            resp = await client.get(
                _PLACES_SEARCH_URL,
                params={
                    "query": f"{name} {city}",
                    "key": config.GOOGLE_API_KEY,
                    "language": "de",
                    "type": "restaurant",
                },
                timeout=10,
            )
            results = resp.json().get("results", [])
        except Exception as e:
            log.debug("Places Text Search error for %r: %s", name, e)
            return None

        if not results:
            return None

        place = results[0]
        place_id = place.get("place_id")
        rating = place.get("rating")
        review_count = place.get("user_ratings_total")

        # Step 2: fetch website + business_status via Place Details
        website = None
        business_status = None
        if place_id:
            try:
                detail_resp = await client.get(
                    _PLACES_DETAIL_URL,
                    params={
                        "place_id": place_id,
                        "fields": "website,business_status",
                        "key": config.GOOGLE_API_KEY,
                    },
                    timeout=10,
                )
                result_data = detail_resp.json().get("result", {})
                website = result_data.get("website")
                business_status = result_data.get("business_status")
            except Exception as e:
                log.debug("Places Detail error for %r: %s", name, e)

    return {
        "website":         website,
        "rating":          rating,
        "review_count":    review_count,
        "business_status": business_status,
    }


# ── Custom Search API ─────────────────────────────────────────────────────────

async def find_menu_url(
    name: str,
    city: str,
    website: Optional[str],
    client: httpx.AsyncClient,
) -> Optional[str]:
    """
    Use Google Custom Search to find the best menu/Speisekarte URL.
    If website is known: search within that domain.
    Otherwise: search the whole web for the restaurant's menu.
    Returns a URL string or None.
    """
    if not config.GOOGLE_API_KEY or not config.GOOGLE_CSE_ID:
        return None

    if website:
        domain = urlparse(website).netloc.lstrip("www.")
        query = f"site:{domain} speisekarte OR menu OR karte OR pizza"
    else:
        query = f'"{name}" {city} speisekarte OR menu'

    async with _semaphore:
        try:
            resp = await client.get(
                _SEARCH_URL,
                params={
                    "q":   query,
                    "key": config.GOOGLE_API_KEY,
                    "cx":  config.GOOGLE_CSE_ID,
                    "num": 3,
                    "hl":  "de",
                },
                timeout=10,
            )
            items = resp.json().get("items", [])
        except Exception as e:
            log.debug("Custom Search error for %r: %s", name, e)
            return None

    if not items:
        return None

    # Prefer pages whose URL or title contains menu-related keywords
    menu_kws = {"speisekarte", "menu", "karte", "pizza", "angebot"}
    for item in items:
        link  = item.get("link", "")
        title = item.get("title", "").lower()
        if any(kw in link.lower() or kw in title for kw in menu_kws):
            log.debug("Google menu hit for %r: %s", name, link)
            return link

    return items[0].get("link")  # fallback: first result


# ── Bulk website + rating fetcher ─────────────────────────────────────────────

async def run_find_websites(
    city: Optional[str] = None,
    bbox: Optional[tuple] = None,
    dry_run: bool = False,
    yes: bool = False,
):
    """
    For every pizzeria in the bbox/city that has no website:
    query Google Places, store website + rating if found.
    """
    if not config.GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not set in .env — skipping.")
        return

    db.init_db()
    with db.get_conn() as conn:
        sql = """
            SELECT id, name, city FROM pizzerias
            WHERE website IS NULL
        """
        params: list = []
        if bbox:
            s, w, n, e = bbox
            sql += " AND lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?"
            params += [s, n, w, e]
        elif city:
            sql += " AND LOWER(city) LIKE ?"
            params.append(f"%{city.lower()}%")
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    if not rows:
        print("No pizzerias without website found.")
        return

    cost, breakdown = _cost_estimate(len(rows))
    print(f"\nCost estimate for {len(rows)} Places API requests:")
    print(breakdown)
    if cost > 0.50 and not dry_run and not yes:
        if not _confirm("Fortfahren?"):
            print("Abgebrochen.")
            return

    print(f"\nSearching Google Places for {len(rows)} pizzerias without website…")
    found = 0

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def process(row):
            nonlocal found
            result = await find_via_places(row["name"], row.get("city") or city or "", client)
            if not result:
                return
            website = result.get("website")
            rating  = result.get("rating")
            rc      = result.get("review_count")
            if website:
                log.info("[%s] → %s  ⭐ %s", row["name"], website, rating or "—")
                if not dry_run:
                    with db.get_conn() as conn:
                        conn.execute(
                            "UPDATE pizzerias SET website=?, google_rating=?, google_review_count=? WHERE id=?",
                            (website, rating, rc, row["id"]),
                        )
                found += 1
            elif rating:
                if not dry_run:
                    db.update_google_rating(row["id"], rating, rc)

        from tqdm import tqdm
        tasks = [process(r) for r in rows]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Places"):
            await coro

    action = "Would update" if dry_run else "Updated"
    print(f"{action} {found}/{len(rows)} pizzerias with a website from Google.")


async def run_fetch_ratings(
    city: Optional[str] = None,
    bbox: Optional[tuple] = None,
    yes: bool = False,
):
    """Fetch/refresh Google ratings for all priced pizzerias in the area."""
    if not config.GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not set in .env — skipping.")
        return

    db.init_db()
    rows = db.get_priced_pizzerias(city=city, bbox=bbox)
    if not rows:
        print("No priced pizzerias found.")
        return

    cost, breakdown = _cost_estimate(len(rows))
    print(f"\nCost estimate for {len(rows)} Places API requests:")
    print(breakdown)
    if cost > 0.50 and not yes:
        if not _confirm("Fortfahren?"):
            print("Abgebrochen.")
            return

    print(f"\nFetching Google ratings for {len(rows)} pizzerias…")
    found = 0

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def process(row):
            nonlocal found
            result = await find_via_places(row["name"], row.get("city") or city or "", client)
            if result and result.get("rating"):
                db.update_google_rating(row["id"], result["rating"], result["review_count"])
                found += 1

        from tqdm import tqdm
        tasks = [process(r) for r in rows]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Ratings"):
            await coro

    print(f"Fetched ratings for {found}/{len(rows)} pizzerias.")


async def run_check_status(
    city: Optional[str] = None,
    bbox: Optional[tuple] = None,
    yes: bool = False,
):
    """Check Google Places business_status for all priced pizzerias."""
    if not config.GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not set in .env — skipping.")
        return

    db.init_db()
    rows = db.get_priced_pizzerias(city=city, bbox=bbox)
    if not rows:
        print("No priced pizzerias found.")
        return

    cost, breakdown = _cost_estimate(len(rows))
    print(f"\nCost estimate for {len(rows)} Places API requests:")
    print(breakdown)
    if cost > 0.50 and not yes:
        if not _confirm("Fortfahren?"):
            print("Abgebrochen.")
            return

    print(f"\nChecking business status for {len(rows)} pizzerias…")
    counts = {"OPERATIONAL": 0, "CLOSED_TEMPORARILY": 0, "CLOSED_PERMANENTLY": 0, "unknown": 0}

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def process(row):
            result = await find_via_places(row["name"], row.get("city") or city or "", client)
            status = result.get("business_status") if result else None
            db.update_business_status(row["id"], status)
            key = status if status in counts else "unknown"
            counts[key] += 1
            if status and status != "OPERATIONAL":
                log.info("[%s] status: %s", row["name"], status)

        from tqdm import tqdm
        tasks = [process(r) for r in rows]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Status"):
            await coro

    print(f"\nErgebnis:")
    print(f"  🟢 Geöffnet (OPERATIONAL):          {counts['OPERATIONAL']}")
    print(f"  🟡 Vorübergehend geschlossen:        {counts['CLOSED_TEMPORARILY']}")
    print(f"  🔴 Dauerhaft geschlossen:            {counts['CLOSED_PERMANENTLY']}")
    print(f"  ❓ Status unbekannt:                 {counts['unknown']}")
