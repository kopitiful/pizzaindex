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

_JS_THRESHOLD = 2000  # chars — below this, try Playwright for link discovery


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


def _extract_best_link(html: str, base_url: str) -> tuple[str, int]:
    """Parse HTML and return (best_menu_url, best_score)."""
    soup = BeautifulSoup(html, "lxml")
    best_url, best_score = base_url, 0

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().startswith(("javascript:", "mailto:", "tel:", "#")):
            continue
        text = a.get_text(strip=True)
        full_url = urljoin(base_url, href)

        is_pdf = full_url.lower().endswith(".pdf")
        # PDFs may live on CDN domains — allow them; only restrict same-domain for HTML links
        if not is_pdf and not _is_same_domain(base_url, full_url):
            continue
        if full_url.rstrip("/") == base_url.rstrip("/"):
            continue
        if not _MENU_RE.search(href + text) and not is_pdf:
            continue

        score = _score_link(href, text) + (2 if is_pdf else 0)
        if score > best_score:
            best_score = score
            best_url = full_url

    return best_url, best_score


async def _playwright_links(url: str) -> list[str]:
    """Render page with Playwright and return all href values."""
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=20000)
            links = await page.eval_on_selector_all(
                "a[href]", "els => els.map(e => e.href)"
            )
            await browser.close()
            return links
    except Exception:
        return []


async def find_menu_url(
    homepage: str,
    client: httpx.AsyncClient,
) -> Optional[str]:
    """
    Fetch homepage, score all <a> links, return the best candidate.
    Falls back to Playwright link extraction when static HTML is too sparse.
    Falls back to the homepage itself if nothing better is found.
    """
    resp_text = ""
    base_url = homepage
    try:
        resp = await client.get(
            homepage,
            follow_redirects=True,
            timeout=config.REQUEST_TIMEOUT,
        )
        if resp.status_code >= 400:
            # Try Playwright even on 4xx — page may still render via JS
            links = await _playwright_links(homepage)
            pdf_links = [l for l in links if l.lower().endswith(".pdf")]
            if pdf_links:
                return pdf_links[0]
            return None
        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type:
            return str(resp.url)
        resp_text = resp.text
        base_url = str(resp.url)
    except Exception:
        return None

    best_url, best_score = _extract_best_link(resp_text, base_url)

    # If static HTML is sparse or found nothing useful, try Playwright for JS-rendered links
    if len(resp_text) < _JS_THRESHOLD or best_url == base_url:
        pw_links = await _playwright_links(base_url)
        # Prefer PDF links with menu keywords; fall back to any PDF
        menu_pdfs = [l for l in pw_links if l.lower().endswith(".pdf")
                     and _MENU_RE.search(l)]
        any_pdfs = [l for l in pw_links if l.lower().endswith(".pdf")]
        if menu_pdfs:
            return menu_pdfs[0]
        if any_pdfs:
            return any_pdfs[0]
        # Re-score with Playwright-rendered links (build fake HTML for reuse)
        for link in pw_links:
            is_pdf = link.lower().endswith(".pdf")
            if not _is_same_domain(base_url, link) and not is_pdf:
                continue
            if link.rstrip("/") == base_url.rstrip("/"):
                continue
            score = _score_link(link, "") + (2 if is_pdf else 0)
            if score > best_score:
                best_score = score
                best_url = link

    return best_url
