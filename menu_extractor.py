"""
Download a menu URL and extract raw text.
Handles: HTML, text-based PDF, image PDF (OCR fallback).
JS-heavy sites (Wix, Squarespace, …) get an async Playwright fallback.
If the page only has images (photo menu), blobs are returned for Claude Vision.
"""

import asyncio
import io
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx
from bs4 import BeautifulSoup

import config

# Minimum chars from plain HTML before we try the JS renderer
_JS_FALLBACK_THRESHOLD = 500

log = logging.getLogger(__name__)


@dataclass
class MenuText:
    text: str
    method: str                          # html | pdf | ocr | playwright | playwright_images
    images: list[bytes] = field(default_factory=list)  # blobs for Claude Vision


def _extract_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(["nav", "header", "footer", "script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


async def _playwright_fetch(url: str) -> tuple[str, list[bytes]]:
    """
    Render URL with async Playwright; return (visible_text, image_blobs).
    Runs safely inside an existing asyncio event loop.
    """
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page(user_agent=config.USER_AGENT)
        await page.goto(url, wait_until="load", timeout=25_000)
        await page.wait_for_timeout(3_000)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2_000)

        text = (await page.evaluate("document.body.innerText") or "").strip()

        img_srcs: list[str] = await page.evaluate("""() => {
            const srcs = new Set();
            for (const el of document.querySelectorAll('img, [data-src]')) {
                const src = el.src
                    || el.getAttribute('data-src')
                    || el.getAttribute('data-lazy')
                    || '';
                if (!src.startsWith('http')) continue;
                if (el.naturalWidth > 0 && (el.naturalWidth < 300 || el.naturalHeight < 300)) continue;
                if (/logo|icon|avatar|thumb|favicon/i.test(src)) continue;
                srcs.add(src);
            }
            return Array.from(srcs);
        }""")

        screenshot: Optional[bytes] = None
        if not img_srcs and len(text) < _JS_FALLBACK_THRESHOLD:
            screenshot = await page.screenshot(full_page=True)

        await browser.close()

    image_blobs: list[bytes] = []
    async with httpx.AsyncClient(timeout=10, follow_redirects=True) as img_client:
        for src in img_srcs[:6]:
            try:
                resp = await img_client.get(src)
                image_blobs.append(resp.content)
            except Exception:
                pass
    if screenshot:
        image_blobs.append(screenshot)

    return text, image_blobs


def _extract_pdf(data: bytes) -> Optional[MenuText]:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages_text = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(pages_text).strip()
        if text:
            return MenuText(text=text, method="pdf")
    except Exception as e:
        log.debug("pdfplumber failed: %s", e)
    return _ocr_pdf(data)


def _ocr_pdf(data: bytes) -> Optional[MenuText]:
    try:
        import pytesseract
        from PIL import Image
        import pdfplumber

        texts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages[:6]:
                img = page.to_image(resolution=200).original
                texts.append(pytesseract.image_to_string(img, lang="deu"))
        text = "\n".join(texts).strip()
        if text:
            return MenuText(text=text, method="ocr")
    except Exception as e:
        log.debug("OCR failed: %s", e)
    return None


async def fetch_menu_text(
    url: str,
    client: httpx.AsyncClient,
) -> Optional[MenuText]:
    try:
        resp = await client.get(url, follow_redirects=True, timeout=config.REQUEST_TIMEOUT)
        if resp.status_code >= 400:
            return None
    except Exception as e:
        log.debug("fetch %s failed: %s", url, e)
        return None

    content_type = resp.headers.get("content-type", "")

    if "pdf" in content_type or url.lower().endswith(".pdf"):
        return _extract_pdf(resp.content)

    if "html" in content_type:
        text = _extract_html(resp.text)
        if len(text) < _JS_FALLBACK_THRESHOLD:
            log.debug("Short HTML (%d chars) at %s — trying Playwright", len(text), url)
            try:
                pw_text, images = await _playwright_fetch(url)
                if images:
                    log.debug("Playwright found %d image(s) at %s", len(images), url)
                    return MenuText(text=pw_text, method="playwright_images", images=images)
                if pw_text and len(pw_text) > len(text):
                    return MenuText(text=pw_text, method="playwright")
            except Exception as e:
                log.debug("Playwright failed for %s: %s", url, e)
        return MenuText(text=text, method="html") if text else None

    # Unknown content type — treat as text
    return MenuText(text=resp.text[:20_000], method="html")


# ── Quick regex pre-check before calling the LLM ──────────────────────────────

_MARG_RE = re.compile(config.MARGHERITA_PATTERN, re.I)
_PRICE_RE = re.compile(config.PRICE_PATTERN)
_NON_STANDARD_RE = re.compile(
    r'party[\s\-]?pizza|party[\s\-]?blech|partyblech'
    r'|\bblech\b'
    r'|familien[\s\-]?(pizza|format)'
    r'|meter[\s\-]?pizza|meterpizza'
    r'|pizza[\s\-]?slice|slice[\s\-]?pizza'
    r'|halbe\s+pizza|andere\s+h[äa]lfte'
    r'|mini[\s\-]?pizza|pizza[\s\-]?mini',
    re.I,
)


def regex_extract_price(text: str) -> Optional[tuple[float, str]]:
    """
    Fast path: search for 'margherita' and a price on the same or adjacent line.
    Returns (price_float, raw_snippet) or None.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if _MARG_RE.search(line):
            window = "\n".join(lines[i : i + 7])
            if _NON_STANDARD_RE.search(window):
                continue
            for m in _PRICE_RE.finditer(window):
                raw = m.group(1).replace(",", ".")
                try:
                    price = float(raw)
                except ValueError:
                    continue
                if not (3.5 <= price <= 30.0):
                    continue
                pre = window[:m.start(1)]
                suf = window[m.end(1):]
                # Reject if embedded in allergen code sequence "2,3,20,28"
                if re.search(r'\d[,;]\s*$', pre) and re.search(r'^[,;]\s*\d', suf):
                    continue
                # Reject if enclosed in parens as allergen code "(16,24)"
                pre_ch = pre[-1] if pre else ' '
                suf_ch = suf[0] if suf else ' '
                if pre_ch in ('(', ',') and suf_ch in (')', ',') \
                        and not re.search(r'[€EeUu]', suf[:6]):
                    continue
                return price, window.strip()
    return None
