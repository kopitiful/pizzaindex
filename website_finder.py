"""
Find missing website URLs for pizzerias in the DB via Gelbe Seiten.

For each pizzeria without a website, searches Gelbe Seiten by name+city,
fetches the best matching detail page, and extracts the restaurant website URL.
"""

import asyncio
import logging
import re
from typing import Optional
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

import config
import database as db

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
}

_BLOCKLIST = re.compile(
    r"facebook\.com|instagram\.com|twitter\.com|gelbeseiten\.de"
    r"|dastelefonbuch|dasoertliche|meinungsmeister|lieferando"
    r"|tripadvisor|yelp|google\.com|apple\.com|schwannverlag",
    re.I,
)

# Stop words stripped before name comparison
_STOP = re.compile(
    r"\b(pizzeria|pizzaria|ristorante|restaurant|trattoria|"
    r"grill|pizza|gaststätte|gasthaus|cafe|café|bar|kebap|kebab|"
    r"la|da|il|le|di)\b",
    re.I,
)


def _normalize(name: str) -> str:
    name = _STOP.sub("", name).lower()
    name = re.sub(r"[^a-z0-9äöüß]", " ", name)
    tokens = [t for t in name.split() if len(t) >= 3]
    return " ".join(tokens)


def _slug(text: str) -> str:
    """Convert name to Gelbe Seiten URL segment."""
    text = text.lower().strip()
    text = re.sub(r"[äÄ]", "ae", text)
    text = re.sub(r"[öÖ]", "oe", text)
    text = re.sub(r"[üÜ]", "ue", text)
    text = re.sub(r"ß", "ss", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


_GS_FOOD_CATEGORIES = re.compile(
    r"pizzeria|pizza|restaurant|gaststätte|trattoria|ristorante|"
    r"cafe|café|bistro|imbiss|schnellrestaurant|kebab|grill",
    re.I,
)


def _similarity(a: str, b: str) -> float:
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


async def _search_gelbeseiten(
    name: str, city: str, client: httpx.AsyncClient
) -> Optional[str]:
    """
    Search Gelbe Seiten for this specific pizzeria.
    Only returns a detail URL if the result looks like a food business
    AND has a reasonable name similarity to our search term.
    """
    search_name = _slug(name)
    search_city = _slug(city)
    url = f"https://www.gelbeseiten.de/suche/{search_name}/{search_city}"
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=12)
        if resp.status_code != 200:
            return None
    except Exception as e:
        log.debug("GS search failed for %r: %s", name, e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    for article in soup.select("article.mod-Treffer"):
        link = article.select_one("a[href*='gsbiz']")
        if not link:
            continue
        gs_name_el = article.select_one("h2.mod-Treffer__name")
        gs_name = gs_name_el.text.strip() if gs_name_el else ""
        category_el = article.select_one(".mod-Treffer--besteBranche, [class*='Branche']")
        category = category_el.text.strip() if category_el else ""

        # Must be a food/restaurant category
        if not _GS_FOOD_CATEGORIES.search(category) and not _GS_FOOD_CATEGORIES.search(gs_name):
            log.debug("Skipping non-food GS result '%s' (cat: %s)", gs_name, category)
            continue

        # Name must be reasonably similar
        sim = _similarity(name, gs_name)
        if sim < 0.3:
            log.debug("Low similarity %.2f: '%s' vs '%s'", sim, name, gs_name)
            continue

        return link["href"]
    return None


async def _get_website_from_detail(
    detail_url: str, client: httpx.AsyncClient
) -> Optional[str]:
    """Fetch a Gelbe Seiten detail page and extract the restaurant website URL."""
    try:
        resp = await client.get(detail_url, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
    except Exception as e:
        log.debug("GS detail failed %s: %s", detail_url, e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        if not href.startswith("http"):
            continue
        if _BLOCKLIST.search(href):
            continue
        if "website" in text or "webseite" in text or href in text:
            return href
    return None


async def find_website(
    name: str,
    city: str,
    client: httpx.AsyncClient,
) -> Optional[str]:
    detail_url = await _search_gelbeseiten(name, city, client)
    if not detail_url:
        return None
    return await _get_website_from_detail(detail_url, client)


async def run(
    city: Optional[str] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
):
    db.init_db()
    city = city or "düsseldorf"

    with db.get_conn() as conn:
        sql = """
            SELECT id, name, city, postcode
            FROM pizzerias
            WHERE website IS NULL
        """
        params: list = []
        if city:
            sql += " AND LOWER(city) LIKE ?"
            params.append(f"%{city.lower()}%")
        db_rows = [dict(r) for r in conn.execute(sql, params).fetchall()]

    if not db_rows:
        print("No pizzerias without website found.")
        return

    if limit:
        db_rows = db_rows[:limit]

    print(f"Searching Gelbe Seiten for {len(db_rows)} pizzerias in {city}…")

    found = 0
    semaphore = asyncio.Semaphore(6)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def process(row: dict):
            nonlocal found
            async with semaphore:
                name = row["name"]
                city_val = row["city"] or city
                website = await find_website(name, city_val, client)
                if website:
                    log.info("[%s] → %s", name, website)
                    if not dry_run:
                        with db.get_conn() as conn:
                            conn.execute(
                                "UPDATE pizzerias SET website = ? WHERE id = ?",
                                (website, row["id"]),
                            )
                    return True
                else:
                    log.debug("Not found: %s", name)
                    return False

        tasks = [process(row) for row in db_rows]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Searching"):
            if await coro:
                found += 1

    action = "Would update" if dry_run else "Updated"
    print(f"\n{action} {found}/{len(db_rows)} pizzerias with a website URL.")
    if not dry_run:
        print(db.stats())
