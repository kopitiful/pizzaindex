"""
Given a pizzeria's homepage URL, find the most likely menu/Speisekarte page.
Returns the URL of the page most likely to contain prices.
"""

import re
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

import config

_MENU_RE = re.compile(
    r"(speisekarte|speise-karte|karte|menu|men%C3%BC|men\xfc|pizza|angebot|preisliste)",
    re.I,
)


def _score_link(href: str, text: str) -> int:
    score = 0
    combined = (href + " " + text).lower()
    for kw in config.MENU_KEYWORDS:
        if kw in combined:
            score += 1
    return score


def _is_same_domain(base: str, href: str) -> bool:
    base_host = urlparse(base).netloc
    href_host = urlparse(href).netloc
    return not href_host or href_host == base_host


async def find_menu_url(
    homepage: str,
    client: httpx.AsyncClient,
) -> Optional[str]:
    """
    Fetch homepage, score all <a> links, return the best candidate.
    Falls back to the homepage itself if nothing better is found.
    """
    try:
        resp = await client.get(
            homepage,
            follow_redirects=True,
            timeout=config.REQUEST_TIMEOUT,
        )
        if resp.status_code >= 400:
            return None
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type:
            # Homepage is already a PDF or something – treat as menu directly
            return str(resp.url)
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    base_url = str(resp.url)

    best_url, best_score = base_url, 0

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Skip non-navigable links
        if href.lower().startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        text = a.get_text(strip=True)
        full_url = urljoin(base_url, href)

        if not _is_same_domain(base_url, full_url):
            continue
        if full_url.rstrip("/") == base_url.rstrip("/"):
            continue  # skip self-links — they'd lock in the homepage score
        if not _MENU_RE.search(href + text):
            continue

        score = _score_link(href, text)
        if score > best_score:
            best_score = score
            best_url = full_url

    return best_url
