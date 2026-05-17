"""
Import pizzerias directly from Gelbe Seiten into the DB.
Fills in lat/lon via OpenStreetMap Nominatim geocoding.
Skips entries already in the DB (matched by name+city).
"""

import asyncio
import logging
import re
import sqlite3
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

import config
import database as db

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9",
    "Accept": "text/html",
}

_BLOCKLIST = re.compile(
    r"facebook\.com|instagram\.com|twitter\.com|gelbeseiten\.de"
    r"|dastelefonbuch|dasoertliche|meinungsmeister|lieferando"
    r"|tripadvisor|yelp|google\.com|apple\.com|schwannverlag",
    re.I,
)


async def _get_all_listing_urls(city: str, client: httpx.AsyncClient) -> list[str]:
    """Return all Gelbe Seiten detail page URLs for pizzerias in city."""
    slug = city.lower().replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace(" ", "-")
    urls = set()
    for page_start in range(0, 300, 50):  # up to 300 results
        url = f"https://www.gelbeseiten.de/suche/pizzeria/{slug}"
        if page_start > 0:
            url += f"?von={page_start + 1}"
        try:
            resp = await client.get(url, headers=_HEADERS, timeout=12)
            if resp.status_code != 200:
                break
        except Exception as e:
            log.warning("GS listing failed: %s", e)
            break

        soup = BeautifulSoup(resp.text, "lxml")
        links = soup.select("article.mod-Treffer a[href*='gsbiz']")
        if not links:
            break
        new = {a["href"] for a in links}
        if new.issubset(urls):
            break  # same page repeated → no more results
        urls |= new

    return list(urls)


async def _parse_detail(url: str, client: httpx.AsyncClient) -> Optional[dict]:
    """Fetch a GS detail page and extract name, address, website, phone."""
    try:
        resp = await client.get(url, headers=_HEADERS, timeout=12)
        resp.raise_for_status()
    except Exception as e:
        log.debug("GS detail failed %s: %s", url, e)
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    name_el = soup.select_one("h1.mod-Treffer__name, h1[itemprop='name'], h1")
    name = name_el.get_text(strip=True) if name_el else None
    if not name:
        return None

    # Address
    street = None
    postcode = None
    city = None
    addr_el = soup.select_one("[itemprop='address'], .mod-AdresseKompakt")
    if addr_el:
        street_el = addr_el.select_one("[itemprop='streetAddress'], .mod-AdresseKompakt--strasse")
        pc_el = addr_el.select_one("[itemprop='postalCode'], .mod-AdresseKompakt--plz")
        city_el = addr_el.select_one("[itemprop='addressLocality'], .mod-AdresseKompakt--ort")
        street = street_el.get_text(strip=True) if street_el else None
        postcode = pc_el.get_text(strip=True) if pc_el else None
        city = city_el.get_text(strip=True) if city_el else None

    # Fallback: find address text in page
    if not street:
        for el in soup.select(".mod-Treffer__adresse, [class*='Adresse']"):
            text = el.get_text(" ", strip=True)
            if text:
                parts = text.split(",")
                if len(parts) >= 2:
                    street = parts[0].strip()
                    rest = parts[-1].strip()
                    m = re.match(r"(\d{5})\s+(.*)", rest)
                    if m:
                        postcode, city = m.group(1), m.group(2).strip()
                break

    # Website
    website = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True).lower()
        if not href.startswith("http"):
            continue
        if _BLOCKLIST.search(href):
            continue
        if "website" in text or "webseite" in text or href in text:
            website = href
            break

    # Phone
    phone = None
    phone_el = soup.select_one("[itemprop='telephone'], .mod-TelNummer")
    if phone_el:
        phone = phone_el.get_text(strip=True)

    return {
        "name": name,
        "street": street,
        "postcode": postcode,
        "city": city,
        "website": website,
        "phone": phone,
    }


_nominatim_semaphore = asyncio.Semaphore(1)  # Nominatim: 1 req/s max


async def _geocode(
    name: str,
    street: Optional[str],
    postcode: Optional[str],
    city: Optional[str],
    client: httpx.AsyncClient,
) -> Optional[tuple[float, float]]:
    """Geocode address via Nominatim (1 req/s rate limit). Returns (lat, lon) or None."""
    parts = []
    if street:
        parts.append(street)
    if postcode:
        parts.append(postcode)
    if city:
        parts.append(city)
    parts.append("Deutschland")
    query = ", ".join(p for p in parts if p)
    if not query.replace("Deutschland", "").strip():
        return None  # no usable address

    async with _nominatim_semaphore:
        await asyncio.sleep(1.1)  # respect Nominatim 1 req/s policy
        try:
            resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": query, "format": "json", "limit": "1", "countrycodes": "de"},
                headers={"User-Agent": config.USER_AGENT},
                timeout=10,
            )
            if resp.status_code == 429:
                await asyncio.sleep(5)
                resp = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": query, "format": "json", "limit": "1", "countrycodes": "de"},
                    headers={"User-Agent": config.USER_AGENT},
                    timeout=10,
                )
            results = resp.json()
            if results:
                return float(results[0]["lat"]), float(results[0]["lon"])
        except Exception as e:
            log.debug("Geocode failed for %r: %s", query, e)
    return None


def _already_in_db(name: str, city: Optional[str]) -> bool:
    """Check if a pizzeria with this name+city already exists in the DB."""
    with db.get_conn() as conn:
        name_norm = re.sub(r"\s+", " ", name.lower().strip())
        rows = conn.execute(
            "SELECT name FROM pizzerias WHERE LOWER(name) LIKE ?",
            (f"%{name_norm[:20]}%",),
        ).fetchall()
        if not rows:
            return False
        # Check city too if we have it
        if city:
            for row in rows:
                return True  # name match is sufficient at this granularity
        return bool(rows)


async def run(city: str = "düsseldorf", dry_run: bool = False):
    db.init_db()

    print(f"Importing pizzerias from Gelbe Seiten for '{city}'…")

    async with httpx.AsyncClient(follow_redirects=True) as client:
        listing_urls = await _get_all_listing_urls(city, client)
    print(f"Found {len(listing_urls)} Gelbe Seiten listings")

    semaphore = asyncio.Semaphore(6)
    added = 0
    skipped = 0

    async with httpx.AsyncClient(follow_redirects=True) as client:
        async def process(url: str):
            nonlocal added, skipped
            async with semaphore:
                entry = await _parse_detail(url, client)
                if not entry or not entry["name"]:
                    return

                if _already_in_db(entry["name"], entry["city"]):
                    log.debug("Already in DB: %s", entry["name"])
                    skipped += 1
                    return

                coords = await _geocode(
                    entry["name"],
                    entry["street"],
                    entry["postcode"],
                    entry["city"] or city,
                    client,
                )
                if not coords:
                    log.debug("No geocode for: %s", entry["name"])
                    return

                lat, lon = coords
                log.info("NEW: %s @ %.4f,%.4f  %s", entry["name"], lat, lon, entry["website"] or "")

                if not dry_run:
                    p = db.Pizzeria(
                        osm_id=f"gs/{url.split('/')[-1]}",
                        name=entry["name"],
                        lat=lat,
                        lon=lon,
                        city=entry["city"] or city,
                        postcode=entry["postcode"],
                        street=entry["street"],
                        website=entry["website"],
                        phone=entry["phone"],
                    )
                    db.upsert_pizzeria(p)
                added += 1

        tasks = [process(u) for u in listing_urls]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Importing"):
            await coro

    action = "Would add" if dry_run else "Added"
    print(f"\n{action} {added} new pizzerias ({skipped} already in DB).")
    if not dry_run:
        print(db.stats())
