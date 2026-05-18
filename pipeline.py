"""
Main pipeline: fetches pizzeria URLs, scrapes menus, extracts prices.

Usage:
    # Step 1 – collect URLs from OpenStreetMap (run once):
    python pipeline.py collect

    # Step 2 – scrape menus and extract prices:
    python pipeline.py scrape [--limit N] [--concurrency N]
"""

import argparse
import asyncio
import logging
import re
import statistics
from typing import Optional

import httpx
from tqdm import tqdm

import config
import database as db
import menu_finder
import menu_extractor
import llm_extractor
import google_finder

log = logging.getLogger(__name__)


async def process_pizzeria(
    row: dict,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    no_llm: bool = False,
) -> Optional[db.PriceRecord]:
    """
    Full pipeline for one pizzeria:
      1. Find menu URL on their website
      2. Extract raw text (HTML / PDF / OCR)
      3. Try regex → fall back to Claude
      4. Return a PriceRecord or None
    """
    async with semaphore:
        pizzeria_id = row["id"]
        homepage = row["website"]
        name = row["name"]
        city = row.get("city") or ""

        # 1. Find menu page via homepage link scan
        menu_url = await menu_finder.find_menu_url(homepage, client)

        # 1b. Google Custom Search fallback — when homepage scan found nothing better
        if (not menu_url or menu_url.rstrip("/") == (homepage or "").rstrip("/")) \
                and config.GOOGLE_CSE_ID and config.GOOGLE_API_KEY:
            google_url = await google_finder.find_menu_url(name, city, homepage, client)
            if google_url:
                log.debug("[%s] Google menu URL: %s", name, google_url)
                menu_url = google_url

        if not menu_url:
            log.debug("[%s] no menu URL found", name)
            return None

        # 2. Fetch text
        result = await menu_extractor.fetch_menu_text(menu_url, client)
        if not result or not result.text.strip():
            log.debug("[%s] empty menu text", name)
            return None

        # 3a. Fast regex path
        regex_result = menu_extractor.regex_extract_price(result.text)
        if regex_result:
            price, snippet = regex_result
            log.debug("[%s] regex hit: %.2f€", name, price)
            return db.PriceRecord(
                pizzeria_id=pizzeria_id,
                price=price,
                menu_url=menu_url,
                extraction_method=result.method,
                raw_snippet=snippet[:500],
            )

        # 3b. LLM text fallback (skipped when --no-llm)
        if no_llm:
            return None
        llm_result = await llm_extractor.extract_price(result.text)
        if llm_result and llm_result.get("gefunden") and llm_result.get("preis"):
            price = llm_result["preis"]
            if 3.0 <= price <= 30.0:
                log.debug("[%s] LLM hit: %.2f€", name, price)
                return db.PriceRecord(
                    pizzeria_id=pizzeria_id,
                    price=price,
                    size_cm=llm_result.get("groesse_cm"),
                    size_label=llm_result.get("groesse_label"),
                    menu_url=menu_url,
                    extraction_method=f"llm_{result.method}",
                    raw_snippet=result.text[:500],
                )

        # 3c. Claude Vision fallback for photo menus
        if result.method == "playwright_images" and result.images:
            for img_blob in result.images:
                # Detect media type from magic bytes
                mt = "image/png" if img_blob[:4] == b"\x89PNG" else "image/jpeg"
                vision_result = await llm_extractor.extract_price_from_image(img_blob, mt)
                if vision_result and vision_result.get("gefunden") and vision_result.get("preis"):
                    price = vision_result["preis"]
                    if 3.0 <= price <= 30.0:
                        log.debug("[%s] Vision hit: %.2f€", name, price)
                        return db.PriceRecord(
                            pizzeria_id=pizzeria_id,
                            price=price,
                            size_cm=vision_result.get("groesse_cm"),
                            size_label=vision_result.get("groesse_label"),
                            menu_url=menu_url,
                            extraction_method="llm_vision",
                            raw_snippet=None,
                        )

        log.debug("[%s] no price found", name)
        return None


async def scrape(limit: Optional[int] = None, concurrency: int = config.MAX_CONCURRENT, city: Optional[str] = None, no_llm: bool = False):
    db.init_db()
    from osm_collector import CITY_BBOXES
    bbox = None
    if city:
        key = city.lower()
        if key in CITY_BBOXES:
            raw = CITY_BBOXES[key][0][1]  # e.g. "51.10,6.65,51.38,6.95"
            parts = [float(x) for x in raw.split(",")]
            bbox = (parts[0], parts[1], parts[2], parts[3])
    rows = db.get_pizzerias_without_price(bbox=bbox, city=(None if bbox else city))
    if limit:
        rows = rows[:limit]

    print(f"Scraping {len(rows)} pizzerias (concurrency={concurrency})…")

    semaphore = asyncio.Semaphore(concurrency)
    found = 0

    async with httpx.AsyncClient(
        headers={"User-Agent": config.USER_AGENT},
        follow_redirects=True,
        timeout=config.REQUEST_TIMEOUT,
    ) as client:
        tasks = [process_pizzeria(row, client, semaphore, no_llm=no_llm) for row in rows]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            record = await coro
            if record:
                db.insert_price(record)
                found += 1

    print(f"\nDone. Found prices for {found}/{len(rows)} pizzerias.")
    print(db.stats())


async def collect(city: Optional[str] = None):
    from osm_collector import run as osm_run, CITY_BBOXES
    regions = None
    if city:
        key = city.lower()
        if key not in CITY_BBOXES:
            print(f"Unknown city '{city}'. Known cities: {', '.join(CITY_BBOXES)}")
            return
        regions = CITY_BBOXES[key]
    await osm_run(regions)


_PLZ_STADTTEIL: dict[str, str] = {
    # Köln
    "50667": "Altstadt-Nord", "50668": "Altstadt-Nord",
    "50670": "Neustadt-Nord", "50672": "Neustadt-Nord",
    "50674": "Neustadt-Süd", "50676": "Altstadt-Süd",
    "50677": "Altstadt-Süd", "50678": "Altstadt-Süd",
    "50679": "Deutz", "50733": "Nippes", "50735": "Nippes",
    "50737": "Nippes", "50739": "Bocklemünd",
    "50765": "Chorweiler", "50767": "Chorweiler",
    "50769": "Roggendorf", "50823": "Ehrenfeld",
    "50825": "Ehrenfeld", "50827": "Bickendorf",
    "50829": "Weiden", "50858": "Junkersdorf",
    "50859": "Lövenich", "50931": "Lindenthal",
    "50933": "Braunsfeld", "50935": "Lindenthal",
    "50937": "Sülz", "50939": "Klettenberg",
    "50968": "Bayenthal", "50969": "Marienburg",
    "50996": "Rodenkirchen", "50997": "Sürth",
    "50999": "Godorf", "51061": "Mülheim",
    "51063": "Mülheim", "51065": "Buchheim",
    "51067": "Holweide", "51069": "Dünnwald",
    "51103": "Kalk", "51105": "Kalk", "51107": "Vingst",
    "51109": "Brück", "51143": "Porz",
    # Den Haag (4-digit prefix of NL postcodes)
    "2511": "Centrum", "2512": "Centrum", "2513": "Centrum",
    "2514": "Centrum", "2515": "Centrum", "2516": "Centrum",
    "2517": "Centrum", "2518": "Centrum",
    "2521": "Bezuidenhout", "2522": "Bezuidenhout", "2523": "Bezuidenhout",
    "2524": "Bezuidenhout", "2525": "Bezuidenhout",
    "2531": "Laak", "2532": "Laak", "2533": "Laak",
    "2534": "Laak", "2535": "Laak",
    "2541": "Loosduinen", "2542": "Loosduinen", "2543": "Loosduinen",
    "2544": "Loosduinen", "2545": "Loosduinen",
    "2551": "Escamp", "2552": "Escamp", "2553": "Escamp",
    "2554": "Escamp", "2555": "Escamp",
    "2561": "Escamp", "2562": "Escamp", "2563": "Escamp",
    "2564": "Escamp", "2565": "Escamp",
    "2571": "Leidschenveen-Ypenburg", "2572": "Leidschenveen-Ypenburg",
    "2581": "Scheveningen", "2582": "Scheveningen", "2583": "Scheveningen",
    "2584": "Scheveningen", "2585": "Scheveningen",
    "2591": "Segbroek", "2592": "Segbroek", "2593": "Segbroek",
    "2594": "Segbroek", "2595": "Segbroek",
    # Düsseldorf
    "40210": "Stadtmitte", "40211": "Stadtmitte",
    "40212": "Carlstadt", "40213": "Altstadt",
    "40215": "Friedrichstadt", "40217": "Oberbilk",
    "40219": "Hafen", "40221": "Hafen",
    "40225": "Bilk", "40227": "Oberbilk",
    "40229": "Flingern Süd", "40231": "Eller",
    "40233": "Flingern Nord", "40235": "Düsseltal",
    "40237": "Düsseltal", "40239": "Düsseltal",
    "40468": "Rath", "40470": "Mörsenbroich",
    "40472": "Rath", "40474": "Golzheim",
    "40476": "Pempelfort", "40477": "Pempelfort",
    "40479": "Pempelfort", "40545": "Oberkassel",
    "40547": "Oberkassel", "40549": "Heerdt",
    "40589": "Wersten", "40591": "Reisholz",
    "40593": "Benrath", "40595": "Benrath",
    "40597": "Benrath", "40599": "Urdenbach",
    "40625": "Eller", "40627": "Lierenfeld",
    "40629": "Vennhausen",
    # Berlin
    "10115": "Mitte", "10117": "Mitte", "10119": "Mitte",
    "10178": "Mitte", "10179": "Mitte",
    "10243": "Friedrichshain", "10245": "Friedrichshain",
    "10247": "Friedrichshain", "10249": "Friedrichshain",
    "10315": "Lichtenberg", "10317": "Lichtenberg",
    "10318": "Lichtenberg", "10319": "Lichtenberg",
    "10365": "Lichtenberg", "10367": "Lichtenberg", "10369": "Lichtenberg",
    "10405": "Prenzlauer Berg", "10407": "Prenzlauer Berg",
    "10409": "Prenzlauer Berg", "10435": "Prenzlauer Berg",
    "10437": "Prenzlauer Berg", "10439": "Prenzlauer Berg",
    "10551": "Tiergarten", "10553": "Tiergarten",
    "10555": "Tiergarten", "10557": "Tiergarten", "10559": "Tiergarten",
    "10585": "Charlottenburg", "10587": "Charlottenburg",
    "10589": "Charlottenburg", "10623": "Charlottenburg",
    "10625": "Charlottenburg", "10627": "Charlottenburg",
    "10629": "Wilmersdorf",
    "10707": "Wilmersdorf", "10709": "Wilmersdorf",
    "10711": "Wilmersdorf", "10713": "Wilmersdorf",
    "10715": "Wilmersdorf", "10717": "Wilmersdorf", "10719": "Wilmersdorf",
    "10777": "Schöneberg", "10779": "Schöneberg",
    "10781": "Schöneberg", "10783": "Schöneberg",
    "10785": "Tiergarten", "10787": "Tiergarten",
    "10789": "Schöneberg", "10823": "Schöneberg",
    "10825": "Schöneberg", "10827": "Schöneberg", "10829": "Tempelhof",
    "10961": "Kreuzberg", "10963": "Kreuzberg",
    "10965": "Kreuzberg", "10967": "Kreuzberg",
    "10969": "Kreuzberg", "10997": "Kreuzberg", "10999": "Kreuzberg",
    "12043": "Neukölln", "12045": "Neukölln", "12047": "Neukölln",
    "12049": "Neukölln", "12051": "Neukölln", "12053": "Neukölln",
    "12055": "Neukölln", "12057": "Neukölln", "12059": "Neukölln",
    "12099": "Tempelhof", "12101": "Tempelhof",
    "12103": "Tempelhof", "12105": "Tempelhof",
    "12107": "Tempelhof", "12109": "Tempelhof",
    "12157": "Steglitz", "12159": "Steglitz", "12161": "Steglitz",
    "12163": "Steglitz", "12165": "Steglitz",
    "12167": "Steglitz", "12169": "Steglitz",
    "12277": "Mariendorf", "12279": "Mariendorf",
    "12347": "Britz", "12349": "Britz",
    "12351": "Buckow", "12353": "Buckow",
    "12355": "Buckow", "12357": "Buckow", "12359": "Buckow",
    "12435": "Treptow", "12437": "Treptow",
    "12439": "Treptow", "12487": "Treptow",
    "12459": "Köpenick", "12527": "Köpenick",
    "12555": "Köpenick", "12557": "Köpenick", "12559": "Köpenick",
    "12489": "Adlershof",
    "12619": "Marzahn", "12621": "Marzahn", "12623": "Marzahn",
    "12627": "Marzahn", "12629": "Hellersdorf",
    "12679": "Marzahn", "12681": "Marzahn", "12683": "Marzahn",
    "12685": "Marzahn", "12687": "Marzahn", "12689": "Marzahn",
    "13051": "Weißensee", "13053": "Weißensee",
    "13055": "Lichtenberg", "13057": "Lichtenberg", "13059": "Hohenschönhausen",
    "13083": "Pankow", "13086": "Pankow",
    "13088": "Weißensee", "13089": "Weißensee",
    "13125": "Pankow", "13127": "Pankow", "13129": "Pankow",
    "13156": "Pankow", "13158": "Pankow", "13159": "Pankow",
    "13187": "Pankow", "13189": "Pankow",
    "13347": "Wedding", "13349": "Wedding", "13351": "Wedding",
    "13353": "Wedding", "13355": "Wedding",
    "13357": "Wedding", "13359": "Wedding",
    "13403": "Reinickendorf", "13405": "Reinickendorf",
    "13407": "Reinickendorf", "13409": "Reinickendorf",
    "13435": "Reinickendorf", "13437": "Reinickendorf", "13439": "Reinickendorf",
    "13581": "Spandau", "13583": "Spandau", "13585": "Spandau",
    "13587": "Spandau", "13589": "Spandau", "13591": "Spandau",
    "13593": "Spandau", "13595": "Spandau",
    "13597": "Spandau", "13599": "Spandau",
    "14050": "Charlottenburg", "14052": "Charlottenburg",
    "14053": "Charlottenburg", "14055": "Charlottenburg",
    "14057": "Charlottenburg", "14059": "Charlottenburg",
    "14089": "Spandau", "14109": "Spandau",
    "14129": "Zehlendorf", "14163": "Zehlendorf",
    "14165": "Zehlendorf", "14167": "Steglitz", "14169": "Steglitz",
    "14193": "Wilmersdorf", "14195": "Dahlem",
    "14197": "Wilmersdorf", "14199": "Wilmersdorf",
    # München
    "80331": "Altstadt-Lehel", "80333": "Maxvorstadt",
    "80335": "Maxvorstadt", "80336": "Ludwigsvorstadt",
    "80337": "Isarvorstadt", "80339": "Schwanthalerhöhe",
    "80469": "Isarvorstadt", "80538": "Altstadt-Lehel",
    "80539": "Maxvorstadt",
    "80634": "Neuhausen", "80636": "Neuhausen", "80637": "Neuhausen",
    "80638": "Nymphenburg", "80639": "Nymphenburg",
    "80686": "Nymphenburg", "80687": "Nymphenburg",
    "80796": "Maxvorstadt", "80797": "Schwabing-West",
    "80798": "Schwabing-West", "80799": "Maxvorstadt",
    "80800": "Schwabing", "80801": "Schwabing",
    "80802": "Schwabing", "80803": "Schwabing",
    "80804": "Schwabing", "80805": "Schwabing",
    "80809": "Milbertshofen",
    "80933": "Feldmoching", "80935": "Feldmoching",
    "80937": "Feldmoching", "80939": "Schwabing-Nord",
    "80992": "Moosach", "80993": "Moosach", "80995": "Moosach",
    "80997": "Allach", "80999": "Allach",
    "81241": "Pasing", "81243": "Pasing",
    "81245": "Pasing", "81247": "Pasing", "81249": "Aubing",
    "81369": "Sendling", "81371": "Sendling",
    "81373": "Sendling-Westpark",
    "81375": "Hadern", "81377": "Hadern",
    "81379": "Obersendling", "81476": "Forstenried",
    "81477": "Solln", "81479": "Solln",
    "81539": "Au-Haidhausen", "81541": "Obergiesing",
    "81543": "Obergiesing", "81545": "Giesing",
    "81547": "Giesing", "81549": "Giesing",
    "81667": "Au", "81669": "Au",
    "81671": "Berg am Laim", "81673": "Berg am Laim",
    "81675": "Bogenhausen", "81677": "Bogenhausen",
    "81679": "Bogenhausen",
    "81737": "Perlach", "81739": "Perlach",
    "81825": "Trudering", "81827": "Trudering",
    "81829": "Trudering-Riem",
    "81925": "Bogenhausen", "81927": "Bogenhausen", "81929": "Bogenhausen",
    # Hannover
    "30159": "Mitte", "30161": "Oststadt", "30163": "Nordstadt",
    "30165": "Nordstadt", "30167": "Nordstadt", "30169": "Südstadt",
    "30171": "Südstadt", "30173": "Döhren", "30175": "Südstadt",
    "30177": "Oststadt", "30179": "List", "30449": "Linden-Nord",
    "30451": "Linden-Mitte", "30453": "Linden-Süd",
    "30455": "Ahlem", "30457": "Ricklingen", "30459": "Ricklingen",
    "30519": "Döhren", "30521": "Döhren", "30539": "Mittelfeld",
    "30559": "Kleefeld", "30625": "Kleefeld", "30627": "Bothfeld",
    "30629": "Bothfeld", "30655": "Vahrenheide", "30657": "Vahrenheide",
    "30659": "Sahlkamp", "30669": "Marienwerder",
    # Bonn
    "53111": "Innenstadt", "53113": "Südstadt",
    "53115": "Südstadt", "53117": "Nordstadt",
    "53119": "Nordstadt", "53121": "Endenich",
    "53123": "Dransdorf", "53125": "Lessenich",
    "53127": "Poppelsdorf", "53129": "Kessenich",
    "53173": "Bad Godesberg", "53175": "Bad Godesberg",
    "53177": "Bad Godesberg", "53179": "Bad Godesberg",
    "53225": "Beuel", "53227": "Beuel", "53229": "Beuel",
    # Bremen
    "28195": "Mitte", "28197": "Walle",
    "28199": "Neustadt", "28201": "Neustadt",
    "28203": "Östliche Vorstadt", "28205": "Östliche Vorstadt",
    "28207": "Östliche Vorstadt", "28209": "Schwachhausen",
    "28211": "Schwachhausen", "28213": "Vahr",
    "28215": "Vahr", "28217": "Walle",
    "28219": "Findorff", "28355": "Horn-Lehe",
    "28357": "Schwachhausen", "28359": "Hastedt",
    "28717": "Blumenthal", "28719": "Burglesum",
    "28755": "Vegesack", "28757": "Vegesack",
    "28759": "Blumenthal", "28779": "Blumenthal",
    # Stuttgart
    "70173": "Mitte", "70174": "Mitte",
    "70176": "West", "70178": "Süd",
    "70180": "Süd", "70182": "Ost",
    "70184": "Ost", "70186": "Ost",
    "70188": "Ost", "70190": "Bad Cannstatt",
    "70191": "Nord", "70192": "Nord",
    "70193": "West", "70195": "West",
    "70197": "West", "70199": "Süd",
    "70327": "Bad Cannstatt", "70329": "Bad Cannstatt",
    "70372": "Bad Cannstatt", "70374": "Bad Cannstatt",
    "70376": "Bad Cannstatt", "70378": "Bad Cannstatt",
    "70435": "Zuffenhausen", "70437": "Zuffenhausen",
    "70439": "Zuffenhausen", "70469": "Feuerbach",
    "70499": "Weilimdorf", "70563": "Vaihingen",
    "70565": "Vaihingen", "70567": "Vaihingen",
    "70569": "Vaihingen", "70597": "Degerloch",
    "70599": "Degerloch",
    # Hamburg
    "20095": "Altstadt", "20097": "Hammerbrook",
    "20099": "St. Georg",
    "20144": "Harvestehude", "20146": "Harvestehude",
    "20148": "Harvestehude", "20149": "Harvestehude",
    "20249": "Eppendorf", "20251": "Eppendorf",
    "20253": "Eimsbüttel", "20255": "Eimsbüttel",
    "20257": "Eimsbüttel", "20259": "Eimsbüttel",
    "20354": "Neustadt", "20355": "Neustadt",
    "20357": "St. Pauli", "20359": "St. Pauli",
    "20457": "HafenCity", "20459": "Neustadt",
    "21073": "Harburg", "21075": "Harburg",
    "21077": "Harburg", "21079": "Harburg",
    "21107": "Wilhelmsburg", "21109": "Wilhelmsburg",
    "22041": "Wandsbek", "22043": "Wandsbek",
    "22045": "Wandsbek", "22047": "Wandsbek", "22049": "Wandsbek",
    "22083": "Uhlenhorst", "22085": "Uhlenhorst",
    "22087": "Hohenfelde", "22089": "Eilbek",
    "22111": "Horn", "22113": "Horn",
    "22115": "Billstedt", "22117": "Billstedt", "22119": "Billstedt",
    "22297": "Winterhude", "22299": "Winterhude",
    "22301": "Winterhude", "22303": "Barmbek-Süd",
    "22305": "Barmbek-Süd", "22307": "Barmbek-Nord", "22309": "Barmbek-Nord",
    "22335": "Fuhlsbüttel", "22337": "Ohlsdorf",
    "22391": "Poppenbüttel", "22393": "Poppenbüttel",
    "22395": "Rahlstedt",
    "22525": "Stellingen", "22527": "Eidelstedt",
    "22529": "Lokstedt", "22547": "Lurup", "22549": "Lurup",
    "22559": "Rissen", "22587": "Blankenese",
    "22607": "Othmarschen", "22609": "Othmarschen",
    "22761": "Altona-Altstadt", "22763": "Altona-Altstadt",
    "22765": "Altona-Nord", "22767": "St. Pauli", "22769": "Altona-Nord",
}


def _stadtteil(row: dict) -> str:
    plz = (row.get("postcode") or "").strip()
    # Try full postcode, then 4-digit prefix (for NL "2511 AB" format)
    if plz in _PLZ_STADTTEIL:
        return _PLZ_STADTTEIL[plz]
    if plz[:4] in _PLZ_STADTTEIL:
        return _PLZ_STADTTEIL[plz[:4]]
    city = (row.get("city") or "").strip()
    return city or plz or "—"


_NON_STANDARD_SNIPPET_RE = re.compile(
    r'party[\s\-]?pizza|party[\s\-]?blech|partyblech'
    r'|\bblech\b'
    r'|familien[\s\-]?(pizza|format)'
    r'|meter[\s\-]?pizza|meterpizza'
    r'|pizza[\s\-]?slice|slice[\s\-]?pizza'
    r'|halbe\s+pizza|andere\s+h[äa]lfte'
    r'|mini[\s\-]?pizza|pizza[\s\-]?mini',
    re.I,
)
_MINI_LABEL_RE = re.compile(r'\b(mini|kleine?\s*pizza|small)\b', re.I)


def _is_valid_margherita(r: dict) -> bool:
    price = r.get("price", 0)
    if not (3.5 <= price <= 20.0):
        return False
    snippet = r.get("raw_snippet") or ""
    if _NON_STANDARD_SNIPPET_RE.search(snippet):
        return False
    # Reject allergen code false positives: price appears inside a parenthesised
    # integer list with no € symbol, e.g. "(7,8,16,24,28)"
    price_str = f"{price:.2f}".replace(".", ",")   # "16,24"
    if re.search(r'\([\d,\s]*' + re.escape(price_str) + r'[\d,\s]*\)', snippet):
        return False
    label = r.get("size_label") or ""
    if _MINI_LABEL_RE.search(label):
        return False
    size = r.get("size_cm") or 0
    if size and (size < 24 or size > 50):
        return False
    return True


_MARG_RE_COMPILED = re.compile(config.MARGHERITA_PATTERN, re.I)
_PRICE_RE_COMPILED = re.compile(config.PRICE_PATTERN)


def _detect_uncertain(rows: list[dict]) -> list[dict]:
    """
    Return subset of rows that look strittig (uncertain) with reasons attached.
    Signals checked:
      1. Price > 15 € or < 5 € (unusual range)
      2. Another price appears *before* 'margherita' on the same snippet line
         → typical symptom of multi-column PDF merging
      3. Multiple different prices on the margherita line
    """
    uncertain = []
    for r in rows:
        reasons = []
        price = r.get("price", 0)
        snippet = r.get("raw_snippet") or ""

        if price > 15.0:
            reasons.append(f"Preis {price:.2f} € ungewöhnlich hoch")
        elif price < 5.0:
            reasons.append(f"Preis {price:.2f} € ungewöhnlich niedrig")

        for line in snippet.splitlines():
            if not _MARG_RE_COMPILED.search(line):
                continue
            marg_match = _MARG_RE_COMPILED.search(line)
            before = line[:marg_match.start()]
            after  = line[marg_match.end():]
            prices_before = _PRICE_RE_COMPILED.findall(before)
            prices_after  = _PRICE_RE_COMPILED.findall(after)
            if prices_before:
                reasons.append(
                    f"Preis(e) vor 'Margherita' in selber Zeile: {prices_before} "
                    f"(möglicherweise mehrspaltige PDF)"
                )
            all_prices = prices_before + prices_after
            if len(all_prices) > 1:
                reasons.append(f"Mehrere Preise in Margherita-Zeile: {all_prices}")
            break  # only check first margherita line

        if reasons:
            entry = dict(r)
            entry["_review_reasons"] = reasons
            uncertain.append(entry)
    return uncertain


_DELIVERY_DOMAIN_RE = re.compile(
    r"lieferando|liefersoft|lieferheld|pizza\.de|mjam|foodpanda|"
    r"uber.{0,5}eats|wolt|deliveroo|just.?eat|dominos\.de|"
    r"bestellsystem|order-online|lieferservice",
    re.I,
)
_DELIVERY_NAME_RE = re.compile(r"lieferservice|lieferdienst|lieferung|delivery", re.I)


def _classify_type(row: dict) -> str:
    menu_url = row.get("menu_url") or ""
    website = row.get("website") or ""
    name = row.get("name") or ""
    via_delivery = _DELIVERY_DOMAIN_RE.search(menu_url) or _DELIVERY_DOMAIN_RE.search(website)
    name_delivery = _DELIVERY_NAME_RE.search(name)
    if via_delivery and not name_delivery:
        # Has own site + delivery platform menu → both
        return "Restaurant & Lieferung"
    if via_delivery or name_delivery:
        return "Lieferdienst"
    return "Restaurant"


def export_map(city: Optional[str] = None, output: str = "pizza_map.html"):
    import json, base64
    db.init_db()
    from osm_collector import CITY_BBOXES
    bbox = None
    if city:
        key = city.lower()
        if key in CITY_BBOXES:
            raw = CITY_BBOXES[key][0][1]
            parts = [float(x) for x in raw.split(",")]
            bbox = (parts[0], parts[1], parts[2], parts[3])
    rows = db.get_priced_pizzerias(bbox=bbox, city=(None if bbox else city))
    before = len(rows)
    rows = [r for r in rows if _is_valid_margherita(r)]
    if not rows:
        print("No priced pizzerias found — run scrape first.")
        return
    if before != len(rows):
        print(f"Filtered {before - len(rows)} non-standard entries (party/blech/mini/slice), {len(rows)} remaining.")

    city_title = (city or "").capitalize()
    display_title = city_title or "Alle Städte"
    multi_city = not bool(city_title)
    def _norm_city(s: str) -> str:
        return s.lower().replace("ü","ue").replace("ö","oe").replace("ä","ae").replace("ß","ss")
    data = []
    for r in sorted(rows, key=lambda x: x["price"]):
        street_line = " ".join(p for p in [r.get("street") or "", r.get("housenumber") or ""] if p).strip()
        city_line = f"{r.get('postcode', '') or ''} {r.get('city', '') or ''}".strip()
        if street_line and city_line:
            address_html = f"{street_line}<br>{city_line}"
        else:
            address_html = street_line or city_line
        label = r.get("size_label") or (f"{r['size_cm']} cm" if r.get("size_cm") else "")
        price_str = f"{r['price']:.2f} €" + (f" ({label})" if label else "")
        rating = r.get("google_rating")
        review_count = r.get("google_review_count")
        rating_str = f"⭐ {rating:.1f} ({review_count} Bew.)" if rating else ""
        bstatus = r.get("business_status") or ""
        status_icon = {"OPERATIONAL": "🟢", "CLOSED_TEMPORARILY": "🟡",
                       "CLOSED_PERMANENTLY": "🔴"}.get(bstatus, "")
        popup = f"<b>{r['name']}</b><br>{address_html}<br>Margherita: {price_str}"
        if rating_str:
            popup += f"<br>{rating_str}"
        if status_icon:
            status_label = {"OPERATIONAL": "Geöffnet", "CLOSED_TEMPORARILY": "Vorübergehend geschlossen",
                            "CLOSED_PERMANENTLY": "Dauerhaft geschlossen"}.get(bstatus, "")
            popup += f"<br>{status_icon} {status_label}"
        if r.get("website"):
            popup += f"<br><a href='{r['website']}' target='_blank'>Website</a>"
        _city_key = ""
        for _cn2, _regions2 in CITY_BBOXES.items():
            _s2, _w2, _n2, _e2 = [float(x) for x in _regions2[0][1].split(",")]
            if _s2 <= r["lat"] <= _n2 and _w2 <= r["lon"] <= _e2:
                _city_key = _norm_city(_cn2)
                break
        data.append({
            "lat": r["lat"],
            "lon": r["lon"],
            "popup": popup,
            "name": r["name"],
            "website": r.get("website") or "",
            "price": r["price"],
            "price_str": price_str,
            "ort": _stadtteil(r),
            "typ": _classify_type(r),
            "rating": rating,
            "rating_str": rating_str,
            "city_key": _city_key,
            "status_icon": status_icon,
        })

    center_lat = sum(d["lat"] for d in data) / len(data)
    center_lon = sum(d["lon"] for d in data) / len(data)
    data_json = json.dumps(data, ensure_ascii=False)
    has_ratings = any(d["rating"] for d in data)

    # City views + bounds for intro navigation
    _city_views: dict = {}
    _city_bounds: dict = {}
    _cities_shown: list = []
    for _cn, _regions in CITY_BBOXES.items():
        _s, _w, _n, _e = [float(x) for x in _regions[0][1].split(",")]
        if any(_s <= d["lat"] <= _n and _w <= d["lon"] <= _e for d in data):
            _city_views[_norm_city(_cn)] = [(_s + _n) / 2, (_w + _e) / 2]
            _city_bounds[_norm_city(_cn)] = [_s, _w, _n, _e]
            _cities_shown.append(_regions[0][0])
    city_views_json = json.dumps(_city_views)
    city_bounds_json = json.dumps(_city_bounds)

    # City average stats for comparison table
    from collections import defaultdict
    _city_price_map: dict = defaultdict(list)
    for d in data:
        if d["city_key"]:
            _city_price_map[d["city_key"]].append(d["price"])
    _key_to_display = {_norm_city(k): v[0][0] for k, v in CITY_BBOXES.items()}
    city_stats = sorted([
        {"key": k, "name": _key_to_display.get(k, k.capitalize()),
         "avg": sum(v) / len(v), "median": statistics.median(v),
         "count": len(v), "min": min(v), "max": max(v)}
        for k, v in _city_price_map.items()
    ], key=lambda x: x["avg"])
    city_stats_json = json.dumps(city_stats, ensure_ascii=False)

    _cheapest = min(data, key=lambda d: d["price"])
    _priciest = max(data, key=lambda d: d["price"])
    _cheapest_city = next(
        (v[0][0] for k, v in CITY_BBOXES.items() if _norm_city(k) == _cheapest["city_key"]), ""
    )
    _priciest_city = next(
        (v[0][0] for k, v in CITY_BBOXES.items() if _norm_city(k) == _priciest["city_key"]), ""
    )
    stat_cheapest = f"{_cheapest['price']:.2f} € · {_cheapest['name']}" + (f", {_cheapest_city}" if _cheapest_city else "")
    stat_priciest = f"{_priciest['price']:.2f} € · {_priciest['name']}" + (f", {_priciest_city}" if _priciest_city else "")
    total_count = len(data)

    intro_hint = "Stadt eingeben"
    intro_placeholder = "Stadtname …"

    _i18n = {
        "de": {
            "eyebrow": "Preisvergleich",
            "desc": "Was kostet eine Margherita in deiner Stadt? Echte Preise, direkt von den Speisekarten.",
            "hint": "Stadt eingeben", "placeholder": "Stadtname …",
            "browse": "Alle Städte anzeigen",
            "error": "× Unbekannte Stadt – versuch es noch mal",
            "countSuffix": "Pizzerien · sortiert nach Preis",
            "btnStats": "\U0001f4ca Städtevergleich", "btnTop10": "\U0001f3c6 Top 10",
            "thRestaurant": "Restaurant", "thPrice": "Preis",
            "thDistrict": "Stadtteil", "thRating": "Bewertung",
            "statsCity": "Stadt", "statsAvg": "Ø Preis", "statsMedian": "Median",
            "statsMin": "Min", "statsMax": "Max",
            "top10Cheap": "\U0001f49a Top 10 Günstigste",
            "top10Pricey": "\U0001f534 Top 10 Teuerste",
        },
        "en": {
            "eyebrow": "Price comparison",
            "desc": "What does a Margherita cost in your city? Real prices, straight from the menus.",
            "hint": "Enter a city", "placeholder": "City name …",
            "browse": "Browse all cities",
            "error": "× Unknown city – try again",
            "countSuffix": "pizzerias · sorted by price",
            "btnStats": "\U0001f4ca City comparison", "btnTop10": "\U0001f3c6 Top 10",
            "thRestaurant": "Restaurant", "thPrice": "Price",
            "thDistrict": "District", "thRating": "Rating",
            "statsCity": "City", "statsAvg": "Avg price", "statsMedian": "Median",
            "statsMin": "Min", "statsMax": "Max",
            "top10Cheap": "\U0001f49a Top 10 Cheapest",
            "top10Pricey": "\U0001f534 Top 10 Most Expensive",
        },
        "nl": {
            "eyebrow": "Prijsvergelijking",
            "desc": "Wat kost een Margherita in jouw stad? Echte prijzen, rechtstreeks van de menukaarten.",
            "hint": "Stad invoeren", "placeholder": "Stadsnaam …",
            "browse": "Alle steden bekijken",
            "error": "× Onbekende stad – probeer opnieuw",
            "countSuffix": "pizzeria’s · gesorteerd op prijs",
            "btnStats": "\U0001f4ca Stadsoverzicht", "btnTop10": "\U0001f3c6 Top 10",
            "thRestaurant": "Restaurant", "thPrice": "Prijs",
            "thDistrict": "Wijk", "thRating": "Beoordeling",
            "statsCity": "Stad", "statsAvg": "Gem. prijs", "statsMedian": "Mediaan",
            "statsMin": "Min", "statsMax": "Max",
            "top10Cheap": "\U0001f49a Top 10 Goedkoopste",
            "top10Pricey": "\U0001f534 Top 10 Duurste",
        },
    }
    i18n_json = json.dumps(_i18n, ensure_ascii=False)

    # Map initialisation
    if multi_city:
        _lats = [d["lat"] for d in data]
        _lons = [d["lon"] for d in data]
        _bounds = [[min(_lats) - 0.05, min(_lons) - 0.1],
                   [max(_lats) + 0.05, max(_lons) + 0.1]]
        map_init_js = f"var map = L.map('map'); map.fitBounds({json.dumps(_bounds)});"
    else:
        map_init_js = f"var map = L.map('map').setView([{center_lat}, {center_lon}], 13);"

    # ── Pizza intro animation — real photo ─────────────────────────────────
    import os as _os
    _img_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "1890071_6_articlelarge_iStock-451865971.jpg")
    with open(_img_path, "rb") as _f:
        _pizza_uri = "data:image/jpeg;base64," + base64.b64encode(_f.read()).decode()

    intro_css = f"""
/* ── Intro: full-screen photo splits apart ── */
#intro {{
  position: fixed; inset: 0; z-index: 9999;
  /* no background — slices ARE the background */
}}
/* Slices fill entire screen, each clips a wedge of the photo */
#pizza-fs {{
  position: absolute; inset: 0; z-index: 1;
}}
.slice {{
  position: absolute; inset: 0;
  background-image: url("{_pizza_uri}");
  background-size: cover;
  background-position: center center;
  transition: transform 0.9s cubic-bezier(0.6,0,0.9,0.4);
  will-change: transform;
}}
#s-n  {{ clip-path: polygon(50% 50%, 29% 0%,   71% 0%); }}
#s-ne {{ clip-path: polygon(50% 50%, 71% 0%,   100% 0%,  100% 29%); }}
#s-e  {{ clip-path: polygon(50% 50%, 100% 29%, 100% 71%); }}
#s-se {{ clip-path: polygon(50% 50%, 100% 71%, 100% 100%, 71% 100%); }}
#s-s  {{ clip-path: polygon(50% 50%, 71% 100%, 29% 100%); }}
#s-sw {{ clip-path: polygon(50% 50%, 29% 100%, 0%   100%, 0%  71%); }}
#s-w  {{ clip-path: polygon(50% 50%, 0%  71%,  0%   29%); }}
#s-nw {{ clip-path: polygon(50% 50%, 0%  29%,  0%   0%,   29% 0%); }}
.fly-n  {{ transform: translateY(-110vh) !important; }}
.fly-ne {{ transform: translate(110vw,-110vh) !important; }}
.fly-e  {{ transform: translateX(110vw) !important; }}
.fly-se {{ transform: translate(110vw,110vh) !important; }}
.fly-s  {{ transform: translateY(110vh) !important; }}
.fly-sw {{ transform: translate(-110vw,110vh) !important; }}
.fly-w  {{ transform: translateX(-110vw) !important; }}
.fly-nw {{ transform: translate(-110vw,-110vh) !important; }}
/* UI panel floats above the photo */
#intro-stage {{
  position: absolute; inset: 0; z-index: 2;
  display: flex; align-items: center; justify-content: center;
  pointer-events: none;
}}
.intro-panel {{
  pointer-events: auto;
  background: rgba(8,5,3,0.58);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  border: 1px solid rgba(220,175,90,.22);
  border-radius: 20px;
  padding: clamp(22px,4vh,44px) clamp(28px,6vw,60px);
  display: flex; flex-direction: column; align-items: center;
  gap: clamp(14px,2.5vh,26px);
  max-width: min(88vw,460px); width: 100%;
  transition: opacity 0.3s ease;
}}
.intro-eyebrow {{
  color: rgba(220,175,90,.65);
  font-size: clamp(.6rem,1.2vw,.78rem);
  letter-spacing: .28em; text-transform: uppercase;
  text-align: center;
}}
.intro-title {{
  color: #fff;
  font-size: clamp(1.6rem,4.5vw,3rem);
  font-weight: 300; letter-spacing: .06em;
  text-align: center; line-height: 1.08; margin: 0;
  text-shadow: 0 2px 16px rgba(0,0,0,.5);
}}
.intro-title em {{
  font-style: italic; color: #e8b860; font-weight: 500;
}}
.intro-ui {{
  display: flex; flex-direction: column; align-items: center;
  gap: 10px; width: 100%;
}}
.intro-desc {{
  color: rgba(220,190,130,.60);
  font-size: clamp(.72rem,1.3vw,.84rem);
  text-align: center; line-height: 1.5;
  margin: -4px 0 4px;
}}
.intro-stats {{
  width: 100%; display: flex; flex-direction: column; gap: 5px;
  border-top: 1px solid rgba(220,175,90,.15);
  border-bottom: 1px solid rgba(220,175,90,.15);
  padding: 10px 0;
}}
.intro-stat {{
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 10px;
}}
.stat-label {{
  color: rgba(220,175,90,.55);
  font-size: clamp(.6rem,1.1vw,.72rem);
  letter-spacing: .12em; text-transform: uppercase;
  white-space: nowrap; flex-shrink: 0;
}}
.stat-val {{
  color: rgba(255,240,210,.75);
  font-size: clamp(.68rem,1.2vw,.78rem);
  text-align: right;
}}
.intro-hint {{
  color: rgba(220,190,130,.55);
  font-size: clamp(.7rem,1.4vw,.86rem);
  letter-spacing: .07em; text-transform: uppercase;
  text-align: center;
}}
.input-wrap {{
  position: relative; width: 100%;
  border-bottom: 1.5px solid rgba(220,175,90,.38);
  transition: border-color .25s;
}}
.input-wrap:focus-within {{ border-bottom-color: rgba(220,175,90,.95); }}
#city-input {{
  background: transparent; border: none; outline: none;
  padding: 10px 32px 10px 0; width: 100%;
  color: #fff; font-size: clamp(1rem,2.6vw,1.35rem);
  font-style: italic; letter-spacing: .04em;
  caret-color: #e8b860;
}}
#city-input::placeholder {{ color: rgba(220,190,130,.28); font-style: italic; }}
.input-arrow {{
  position: absolute; right: 4px; top: 50%; transform: translateY(-50%);
  color: rgba(220,175,90,.45); font-size: .95rem;
  pointer-events: none; transition: color .25s;
}}
.input-wrap:focus-within .input-arrow {{ color: rgba(220,175,90,.95); }}
.input-error {{
  color: #ff8a7a; font-size: .78rem;
  letter-spacing: .04em; min-height: 1.1em; text-align: center;
}}
.intro-browse-btn {{
  background: transparent; border: none; outline: none; cursor: pointer;
  color: rgba(220,175,90,.45); font-size: clamp(.65rem,1.1vw,.75rem);
  letter-spacing: .1em; text-transform: uppercase; padding: 4px 0;
  transition: color .2s;
}}
.intro-browse-btn:hover {{ color: rgba(220,175,90,.85); }}"""

    intro_html = f"""
<div id="intro">
  <div id="pizza-fs">
    <div class="slice" id="s-n"></div>
    <div class="slice" id="s-ne"></div>
    <div class="slice" id="s-e"></div>
    <div class="slice" id="s-se"></div>
    <div class="slice" id="s-s"></div>
    <div class="slice" id="s-sw"></div>
    <div class="slice" id="s-w"></div>
    <div class="slice" id="s-nw"></div>
  </div>
  <div id="intro-stage">
    <div class="intro-panel" id="intro-panel">
      <div id="lang-sw-intro">
        <button class="lang-btn active" data-lang="de">DE</button>
        <button class="lang-btn" data-lang="en">EN</button>
        <button class="lang-btn" data-lang="nl">NL</button>
      </div>
      <p class="intro-eyebrow" id="intro-eyebrow">Preisvergleich</p>
      <p class="intro-title">Pizza<br><em>Margherita</em></p>
      <p class="intro-desc" id="intro-desc">Was kostet eine Margherita in deiner Stadt? Echte Preise, direkt von den Speisekarten.</p>
      <div class="intro-ui">
        <p class="intro-hint" id="intro-hint">Stadt eingeben</p>
        <div class="input-wrap">
          <input type="text" id="city-input" placeholder="Stadtname …"
                 autocomplete="off" spellcheck="false"/>
          <span class="input-arrow">→</span>
        </div>
        <p class="input-error" id="input-error"></p>
        <button class="intro-browse-btn" id="browse-btn">Alle Städte anzeigen</button>
      </div>
    </div>
  </div>
</div>"""
    intro_js = f"""
(function() {{
  var singleCity = '{city_title}'.toLowerCase();
  var CITY_VIEWS  = {city_views_json};
  var CITY_BOUNDS = {city_bounds_json};
  function norm(s) {{
    return s.toLowerCase()
      .replace(/ü/g,'ue').replace(/ö/g,'oe').replace(/ä/g,'ae').replace(/ß/g,'ss');
  }}
  var SLICES = [
    ['s-n','fly-n',0],  ['s-ne','fly-ne',40], ['s-e','fly-e',80],  ['s-se','fly-se',120],
    ['s-s','fly-s',160],['s-sw','fly-sw',120],['s-w','fly-w',80],  ['s-nw','fly-nw',40]
  ];
  var ALL_BOUNDS = {json.dumps(_bounds) if multi_city else 'null'};
  var input = document.getElementById('city-input');
  var error = document.getElementById('input-error');
  var intro = document.getElementById('intro');
  var panel = document.getElementById('intro-panel');
  if (!input) return;
  function _animateOut(cb) {{
    if (panel) {{ panel.style.transition = 'opacity 0.35s'; panel.style.opacity = '0'; }}
    SLICES.forEach(function(s) {{
      setTimeout(function() {{
        var el = document.getElementById(s[0]);
        if (el) el.classList.add(s[1]);
      }}, s[2]);
    }});
    setTimeout(cb, 1100);
  }}
  function doFly(key) {{
    var b = CITY_BOUNDS[key];
    _animateOut(function() {{
      intro.remove();
      if (b) {{ map.flyToBounds([[b[0],b[1]],[b[2],b[3]]], {{padding:[30,30]}}); }}
      else   {{ map.invalidateSize(); }}
      filterCity(key);
    }});
  }}
  function doFlyAll() {{
    _animateOut(function() {{
      intro.remove();
      if (ALL_BOUNDS) {{ map.fitBounds(ALL_BOUNDS, {{padding:[20,20]}}); }}
      else {{ map.invalidateSize(); }}
    }});
  }}
  var browseBtn = document.getElementById('browse-btn');
  if (browseBtn) browseBtn.addEventListener('click', doFlyAll);
  input.addEventListener('keydown', function(e) {{
    if (e.key !== 'Enter') return;
    var key = norm(input.value.trim());
    var ok = singleCity ? (key === norm(singleCity)) : !!CITY_VIEWS[key];
    if (ok) {{
      error.textContent = '';
      doFly(singleCity ? norm(singleCity) : key);
    }} else {{
      error.textContent = i18n[currentLang].error;
      input.select();
    }}
  }});
  input.focus();
}})();"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pizza Margherita – {display_title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🍕</text></svg>">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ display: flex; height: 100vh; font-family: system-ui, sans-serif; }}
#map {{ flex: 0 0 62%; height: 100vh; }}
#sidebar {{ flex: 1; height: 100vh; overflow-y: auto; background: #fafafa; display: flex; flex-direction: column; }}
#sidebar-header {{ padding: 12px 16px; background: #400b1b; color: #fff; }}
#sidebar-header h2 {{ font-size: 1em; font-weight: 600; }}
#sidebar-header p {{ font-size: 0.78em; opacity: .75; margin-top: 3px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
thead th {{
    position: sticky; top: 0; z-index: 1;
    background: #6b1428; color: #fff;
    padding: 7px 10px; text-align: left; font-weight: 500;
}}
th.preis, td.preis {{ text-align: right; }}
tbody tr {{ border-bottom: 1px solid #eee; cursor: pointer; transition: background .1s; }}
tbody tr:hover, tbody tr.active {{ background: #fff3cd; }}
td {{ padding: 5px 8px; vertical-align: middle; }}
td.rank {{ color: #999; font-size: .75em; width: 22px; }}
td.name a {{ color: #222; text-decoration: none; font-weight: 500; }}
td.name a:hover {{ text-decoration: underline; color: #400b1b; }}
td.name .typ {{ display: block; font-size: .72em; color: #888; margin-top: 1px; }}
td.preis {{ font-weight: 700; color: #400b1b; white-space: nowrap; }}
td.ort {{ color: #555; font-size: .8em; }}
td.rating {{ font-size: .8em; white-space: nowrap; }}
#city-stats {{ border-bottom: 2px solid #e8d5b0; background: #fff8f0; }}
.stats-table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
.stats-table thead th {{ background: #f5e6cc; color: #400b1b; padding: 6px 8px; font-weight: 600; }}
.stats-table thead th.r {{ text-align: right; }}
.stats-table tbody tr {{ border-bottom: 1px solid #ede0c8; cursor: pointer; transition: background .1s; }}
.stats-table tbody tr:hover {{ background: #fff3cd; }}
.stats-table td {{ padding: 5px 8px; }}
.stats-table td.r {{ text-align: right; }}
.stats-table td.fw {{ font-weight: 700; color: #400b1b; }}
#stats-btn {{
  font-size: .72em; margin-left: 8px; padding: 2px 7px;
  border: 1px solid rgba(255,255,255,.35); border-radius: 4px;
  background: rgba(255,255,255,.12); color: #fff; cursor: pointer; vertical-align: middle;
}}
#stats-btn:hover {{ background: rgba(255,255,255,.25); }}
#top10-btn {{
  font-size: .72em; margin-left: 6px; padding: 2px 7px;
  border: 1px solid rgba(255,255,255,.35); border-radius: 4px;
  background: rgba(255,255,255,.12); color: #fff; cursor: pointer; vertical-align: middle;
}}
#top10-btn:hover {{ background: rgba(255,255,255,.25); }}
.lang-sw {{ display: inline-flex; gap: 3px; margin-left: 8px; vertical-align: middle; }}
.lang-btn {{
  background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.2);
  border-radius: 3px; color: rgba(255,255,255,.5); font-size: .65em;
  padding: 1px 5px; cursor: pointer; transition: all .15s; font-family: inherit;
}}
.lang-btn:hover {{ background: rgba(255,255,255,.22); color: #fff; }}
.lang-btn.active {{ background: rgba(255,255,255,.28); color: #fff;
  border-color: rgba(255,255,255,.55); font-weight: 700; }}
#lang-sw-intro {{
  position: absolute; top: 14px; right: 18px; display: flex; gap: 5px; z-index: 3;
  pointer-events: auto;
}}
#lang-sw-intro .lang-btn {{
  background: rgba(8,5,3,.5); border: 1px solid rgba(220,175,90,.25);
  color: rgba(220,175,90,.45); font-size: .68em; padding: 2px 6px;
}}
#lang-sw-intro .lang-btn:hover {{ color: rgba(220,175,90,.9); border-color: rgba(220,175,90,.6); }}
#lang-sw-intro .lang-btn.active {{ color: rgba(220,175,90,.95);
  border-color: rgba(220,175,90,.7); font-weight: 700; }}
/* Custom map pins */
.pin-cheap {{
  background: #22a855; color: #fff; border-radius: 50% 50% 50% 0;
  transform: rotate(-45deg); width: 26px; height: 26px;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 800; border: 2px solid #fff;
  box-shadow: 0 2px 6px rgba(0,0,0,.45);
}}
.pin-cheap-inner {{ transform: rotate(45deg); }}
.pin-pricey {{
  background: #d63031; color: #fff; border-radius: 50% 50% 50% 0;
  transform: rotate(-45deg); width: 26px; height: 26px;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 800; border: 2px solid #fff;
  box-shadow: 0 2px 6px rgba(0,0,0,.45);
}}
.pin-pricey-inner {{ transform: rotate(45deg); }}
/* Top-10 panel */
#top10-panel {{ border-bottom: 2px solid #e8d5b0; background: #fff8f0; }}
.top10-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
.top10-col {{ padding: 6px 8px; }}
.top10-col h4 {{ font-size: .7em; font-weight: 700; text-transform: uppercase;
  letter-spacing: .08em; padding: 3px 0 5px; border-bottom: 1px solid #eee; margin-bottom: 4px; }}
.top10-col.cheap h4 {{ color: #22a855; }}
.top10-col.pricey h4 {{ color: #d63031; }}
.top10-item {{ display: flex; align-items: baseline; gap: 4px; padding: 2px 2px;
  cursor: pointer; border-radius: 3px; }}
.top10-item:hover {{ background: #fff3cd; }}
.t10-rank {{ font-size: .68em; color: #aaa; width: 14px; flex-shrink: 0; text-align: right; }}
.t10-name {{ font-size: .74em; color: #333; flex: 1; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; }}
.t10-price {{ font-size: .74em; font-weight: 700; white-space: nowrap; }}
.top10-col.cheap .t10-price {{ color: #22a855; }}
.top10-col.pricey .t10-price {{ color: #d63031; }}
{intro_css}
</style>
</head>
<body>
{intro_html}
<div id="map"></div>
<div id="sidebar">
  <div id="sidebar-header">
    <h2>🍕 Pizza Margherita — {display_title}</h2>
    <p><span id="sidebar-count">{len(data)}</span> <span id="count-suffix">Pizzerien · sortiert nach Preis</span> <button id="stats-btn" onclick="toggleStats()">📊 Städtevergleich</button><button id="top10-btn" onclick="toggleTop10()">🏆 Top 10</button><span class="lang-sw"><button class="lang-btn active" data-lang="de">DE</button><button class="lang-btn" data-lang="en">EN</button><button class="lang-btn" data-lang="nl">NL</button></span></p>
  </div>
  <div id="city-stats" style="display:none"></div>
  <div id="top10-panel" style="display:none"></div>
  <table>
    <thead>
      <tr>
        <th class="rank">#</th>
        <th id="th-restaurant">Restaurant</th>
        <th class="preis" id="th-price">Preis</th>
        <th id="th-district">Stadtteil</th>
        {'<th id="th-rating">Bewertung</th>' if has_ratings else ''}
      </tr>
    </thead>
    <tbody id="tbl"></tbody>
  </table>
</div>
<script>
{map_init_js}
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    attribution: '&copy; OpenStreetMap contributors'
}}).addTo(map);

var data = {data_json};
var hasRatings = {'true' if has_ratings else 'false'};
var markers = [];
var activeRow = null;

var cheapMap = {{}};
var priceyMap = {{}};
var _n = data.length;
for (var _ci = 0; _ci < Math.min(10, _n); _ci++) {{
  cheapMap[_ci] = _ci + 1;
}}
for (var _pi = 0; _pi < Math.min(10, _n); _pi++) {{
  var _pidx = _n - 1 - _pi;
  if (!(_pidx in cheapMap)) priceyMap[_pidx] = _pi + 1;
}}
function makePinIcon(rank, cls) {{
  return L.divIcon({{
    className: '',
    html: '<div class="' + cls + '"><span class="' + cls + '-inner">' + rank + '</span></div>',
    iconSize: [26, 26], iconAnchor: [13, 26], popupAnchor: [0, -28]
  }});
}}
data.forEach(function(d, i) {{
    var m;
    if (i in cheapMap) {{
        m = L.marker([d.lat, d.lon], {{ icon: makePinIcon(cheapMap[i], 'pin-cheap'), zIndexOffset: 200 }}).addTo(map);
    }} else if (i in priceyMap) {{
        m = L.marker([d.lat, d.lon], {{ icon: makePinIcon(priceyMap[i], 'pin-pricey'), zIndexOffset: 100 }}).addTo(map);
    }} else {{
        m = L.marker([d.lat, d.lon]).addTo(map);
    }}
    m.bindPopup(d.popup);
    markers.push(m);
}});

var tbody = document.getElementById('tbl');
data.forEach(function(d, i) {{
    var tr = document.createElement('tr');
    var nameCell = d.website
        ? '<a href="' + d.website + '" target="_blank" onclick="event.stopPropagation()">' + d.name + '</a>'
        : d.name;
    nameCell += '<span class="typ">' + d.typ + '</span>';
    var ratingCell = hasRatings
        ? '<td class="rating">' + (d.rating_str || '—') + '</td>'
        : '';
    var statusBadge = d.status_icon ? '<span title="' + d.status_icon + '" style="margin-left:3px">' + d.status_icon + '</span>' : '';
    tr.innerHTML =
        '<td class="rank">' + (i + 1) + '</td>' +
        '<td class="name">' + nameCell + statusBadge + '</td>' +
        '<td class="preis">' + d.price_str + '</td>' +
        '<td class="ort">' + d.ort + '</td>' +
        ratingCell;
    tr.addEventListener('mouseenter', function() {{
        if (activeRow) activeRow.classList.remove('active');
        activeRow = tr;
        tr.classList.add('active');
        markers[i].openPopup();
        map.panTo(markers[i].getLatLng());
    }});
    tbody.appendChild(tr);
}});

var cityStats = {city_stats_json};
var CITY_VIEWS_G = {city_views_json};
var CITY_BOUNDS_G = {city_bounds_json};
var activeCityKey = null;

var i18n = {i18n_json};
var currentLang = 'de';

function renderTop10Panel() {{
  var panel = document.getElementById('top10-panel');
  if (!panel) return;
  var t = i18n[currentLang];
  var cheapHtml2 = '';
  for (var _ci2 = 0; _ci2 < Math.min(10, data.length); _ci2++) {{
    var _d = data[_ci2];
    cheapHtml2 += '<div class="top10-item" data-idx="' + _ci2 + '">'
      + '<span class="t10-rank">' + (_ci2 + 1) + '</span>'
      + '<span class="t10-name" title="' + _d.name + '">' + _d.name + '</span>'
      + '<span class="t10-price">' + _d.price_str + '</span></div>';
  }}
  var priceyHtml2 = '';
  for (var _pi2 = 0; _pi2 < Math.min(10, data.length); _pi2++) {{
    var _idx2 = data.length - 1 - _pi2;
    var _d2 = data[_idx2];
    priceyHtml2 += '<div class="top10-item" data-idx="' + _idx2 + '">'
      + '<span class="t10-rank">' + (_pi2 + 1) + '</span>'
      + '<span class="t10-name" title="' + _d2.name + '">' + _d2.name + '</span>'
      + '<span class="t10-price">' + _d2.price_str + '</span></div>';
  }}
  panel.innerHTML = '<div class="top10-grid">'
    + '<div class="top10-col cheap"><h4>' + t.top10Cheap + '</h4>' + cheapHtml2 + '</div>'
    + '<div class="top10-col pricey"><h4>' + t.top10Pricey + '</h4>' + priceyHtml2 + '</div>'
    + '</div>';
  panel.querySelectorAll('.top10-item').forEach(function(el) {{
    el.addEventListener('click', function() {{
      var idx = parseInt(this.dataset.idx);
      markers[idx].openPopup();
      map.panTo(markers[idx].getLatLng());
    }});
  }});
}}

function renderStatsTable() {{
  var el = document.getElementById('city-stats');
  if (!el) return;
  var t = i18n[currentLang];
  var h = '<table class="stats-table"><thead><tr>'
    + '<th>' + t.statsCity + '</th><th class="r">' + t.statsAvg + '</th>'
    + '<th class="r">' + t.statsMedian + '</th><th class="r">' + t.statsMin + '</th>'
    + '<th class="r">' + t.statsMax + '</th><th class="r">n</th>'
    + '</tr></thead><tbody>';
  cityStats.forEach(function(s) {{
    h += '<tr data-key="' + s.key + '" onclick="filterAndGo(this.dataset.key)" title="Nur ' + s.name + ' anzeigen">'
      + '<td>' + s.name + '</td>'
      + '<td class="r fw">' + s.avg.toFixed(2) + ' €</td>'
      + '<td class="r">' + s.median.toFixed(2) + ' €</td>'
      + '<td class="r">' + s.min.toFixed(2) + ' €</td>'
      + '<td class="r">' + s.max.toFixed(2) + ' €</td>'
      + '<td class="r">' + s.count + '</td></tr>';
  }});
  h += '</tbody></table>';
  el.innerHTML = h;
}}

function setLang(lang) {{
  if (!i18n[lang]) return;
  currentLang = lang;
  try {{ localStorage.setItem('pizza-lang', lang); }} catch(e) {{}}
  var t = i18n[lang];
  document.querySelectorAll('.lang-btn').forEach(function(b) {{
    b.classList.toggle('active', b.dataset.lang === lang);
  }});
  var _el;
  _el = document.getElementById('intro-eyebrow'); if (_el) _el.textContent = t.eyebrow;
  _el = document.getElementById('intro-desc');    if (_el) _el.textContent = t.desc;
  _el = document.getElementById('intro-hint');    if (_el) _el.textContent = t.hint;
  _el = document.getElementById('city-input');    if (_el) _el.placeholder = t.placeholder;
  _el = document.getElementById('browse-btn');    if (_el) _el.textContent = t.browse;
  _el = document.getElementById('count-suffix');  if (_el) _el.textContent = t.countSuffix;
  _el = document.getElementById('stats-btn');     if (_el) _el.textContent = t.btnStats;
  _el = document.getElementById('top10-btn');     if (_el) _el.textContent = t.btnTop10;
  _el = document.getElementById('th-restaurant'); if (_el) _el.textContent = t.thRestaurant;
  _el = document.getElementById('th-price');      if (_el) _el.textContent = t.thPrice;
  _el = document.getElementById('th-district');   if (_el) _el.textContent = t.thDistrict;
  _el = document.getElementById('th-rating');     if (_el) _el.textContent = t.thRating;
  renderStatsTable();
  renderTop10Panel();
}}

function updateSidebarFromBounds() {{
  var bounds = map.getBounds();
  var trs = document.getElementById('tbl').querySelectorAll('tr');
  var cnt = 0;
  trs.forEach(function(tr, i) {{
    var d = data[i];
    var cityOk = !activeCityKey || d.city_key === activeCityKey;
    var show = cityOk && bounds.contains([d.lat, d.lon]);
    tr.style.display = show ? '' : 'none';
    if (show) cnt++;
  }});
  var cntEl = document.getElementById('sidebar-count');
  if (cntEl) cntEl.textContent = cnt;
}}
map.on('moveend', updateSidebarFromBounds);

function filterCity(key) {{
  activeCityKey = key || null;
  updateSidebarFromBounds();
}}
function toggleStats() {{
  var el = document.getElementById('city-stats');
  if (!el) return;
  var showing = el.style.display === 'none';
  el.style.display = showing ? 'block' : 'none';
  if (showing) renderStatsTable();
}}
function toggleTop10() {{
  var el = document.getElementById('top10-panel');
  if (!el) return;
  var showing = el.style.display === 'none';
  el.style.display = showing ? 'block' : 'none';
  if (showing) renderTop10Panel();
}}
function filterAndGo(key) {{
  activeCityKey = key || null;
  var b = CITY_BOUNDS_G[key];
  if (b) map.flyToBounds([[b[0], b[1]], [b[2], b[3]]], {{padding: [30, 30]}});
}}
document.querySelectorAll('.lang-btn').forEach(function(b) {{
  b.addEventListener('click', function() {{ setLang(this.dataset.lang); }});
}});
(function() {{
  var saved = null;
  try {{ saved = localStorage.getItem('pizza-lang'); }} catch(e) {{}}
  var detected = saved || (navigator.language || '').slice(0,2).toLowerCase();
  if (detected === 'nl') setLang('nl');
  else if (detected === 'en') setLang('en');
}})();
{intro_js}
</script>
</body>
</html>"""

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Map saved to {output} ({len(data)} pizzerias)")

    # ── Review: strittige Einträge ────────────────────────────────────────────
    uncertain = _detect_uncertain(rows)
    if uncertain:
        print(f"\n{'─'*60}")
        print(f"⚠️  {len(uncertain)} strittige Einträge – bitte prüfen:")
        print(f"{'─'*60}")
        for u in uncertain:
            print(f"\n  [{u['name']}] ({u.get('city','?')}) → {u['price']:.2f} €")
            for reason in u["_review_reasons"]:
                print(f"    • {reason}")
            # Show the relevant snippet line(s)
            snippet_lines = (u.get("raw_snippet") or "").splitlines()
            for line in snippet_lines:
                if _MARG_RE_COMPILED.search(line):
                    print(f'    Zeile: „{line.strip()}“')
                    break
        print(f"\n{'─'*60}")
        print("Tipp: 'python3 pipeline.py set-price <name> <preis>' zum Korrigieren.")
        print(f"{'─'*60}\n")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Pizza Price Index Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    collect_p = sub.add_parser("collect", help="Collect pizzeria URLs from OpenStreetMap")
    collect_p.add_argument("--city", default=None,
                           help="Limit to a city (e.g. düsseldorf, berlin, münchen, hamburg, köln)")

    scrape_p = sub.add_parser("scrape", help="Scrape menus and extract prices")
    scrape_p.add_argument("--limit", type=int, default=None,
                          help="Max number of pizzerias to process")
    scrape_p.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENT,
                          help="Parallel requests")
    scrape_p.add_argument("--city", default=None,
                          help="Only scrape pizzerias in this city")
    scrape_p.add_argument("--no-llm", action="store_true",
                          help="Skip Claude LLM fallback (regex only, $0 API cost)")

    sub.add_parser("stats", help="Show current database statistics")

    map_p = sub.add_parser("map", help="Export an interactive HTML map of priced pizzerias")
    map_p.add_argument("--city", default=None, help="Filter by city")
    map_p.add_argument("--output", default="pizza_map.html", help="Output HTML file")

    fw_p = sub.add_parser("find-websites",
                          help="Search DuckDuckGo for website URLs of pizzerias missing one")
    fw_p.add_argument("--city", default=None, help="Limit to city")
    fw_p.add_argument("--limit", type=int, default=None, help="Max pizzerias to search")
    fw_p.add_argument("--dry-run", action="store_true",
                      help="Show results without writing to DB")

    gs_p = sub.add_parser("import-gelbeseiten",
                          help="Import pizzerias from Gelbe Seiten (with geocoding)")
    gs_p.add_argument("--city", default="düsseldorf")
    gs_p.add_argument("--dry-run", action="store_true")

    sw_p = sub.add_parser("set-website",
                          help="Manually set a website URL for a pizzeria by name")
    sw_p.add_argument("name", help="Partial name to match (case-insensitive)")
    sw_p.add_argument("url", help="Website URL to set")

    sp_p = sub.add_parser("set-price",
                          help="Manually correct the Margherita price for a pizzeria by name")
    sp_p.add_argument("name", help="Partial name to match (case-insensitive)")
    sp_p.add_argument("price", type=float, help="Correct price in Euro, e.g. 10.90")

    dp_p = sub.add_parser("delete-price",
                          help="Delete the price record for a pizzeria (removes it from the map)")
    dp_p.add_argument("name", help="Partial name to match (case-insensitive)")

    fr_p = sub.add_parser("fetch-ratings",
                          help="Fetch Google Maps ratings via Places API (needs GOOGLE_API_KEY in .env)")
    fr_p.add_argument("--city", default=None, help="Filter by city")
    fr_p.add_argument("--yes", action="store_true", help="Skip cost confirmation")

    fwg_p = sub.add_parser("find-websites-google",
                            help="Find missing website URLs via Google Places API (works internationally)")
    fwg_p.add_argument("--city", default=None, help="Filter by city")
    fwg_p.add_argument("--dry-run", action="store_true")
    fwg_p.add_argument("--yes", action="store_true", help="Skip cost confirmation")

    cs_p = sub.add_parser("check-status",
                           help="Check Google Places business_status for priced pizzerias (open/closed)")
    cs_p.add_argument("--city", default=None, help="Filter by city")
    cs_p.add_argument("--yes", action="store_true", help="Skip cost confirmation")

    args = parser.parse_args()

    if args.cmd == "collect":
        asyncio.run(collect(city=args.city))
    elif args.cmd == "scrape":
        asyncio.run(scrape(limit=args.limit, concurrency=args.concurrency, city=args.city, no_llm=args.no_llm))
    elif args.cmd == "stats":
        db.init_db()
        print(db.stats())
    elif args.cmd == "map":
        export_map(city=args.city, output=args.output)
    elif args.cmd == "find-websites":
        from website_finder import run as fw_run
        asyncio.run(fw_run(city=args.city, limit=args.limit, dry_run=args.dry_run))
    elif args.cmd == "import-gelbeseiten":
        from gelbeseiten_importer import run as gs_run
        asyncio.run(gs_run(city=args.city or "düsseldorf", dry_run=args.dry_run))
    elif args.cmd == "set-website":
        db.init_db()
        import sqlite3 as _sql
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT id, name, city FROM pizzerias WHERE LOWER(name) LIKE ?",
                (f"%{args.name.lower()}%",)
            ).fetchall()
        if not rows:
            print(f"No pizzeria found matching '{args.name}'")
        elif len(rows) > 1:
            print(f"Multiple matches — be more specific:")
            for r in rows:
                print(f"  id={r['id']}  {r['name']}  ({r['city']})")
        else:
            r = rows[0]
            with db.get_conn() as conn:
                conn.execute("UPDATE pizzerias SET website = ? WHERE id = ?",
                             (args.url, r["id"]))
            print(f"Updated '{r['name']}' → {args.url}")
    elif args.cmd == "set-price":
        db.init_db()
        with db.get_conn() as conn:
            hits = conn.execute(
                "SELECT p.id, p.name, p.city FROM pizzerias p "
                "JOIN prices pr ON pr.pizzeria_id = p.id "
                "WHERE LOWER(p.name) LIKE ?",
                (f"%{args.name.lower()}%",)
            ).fetchall()
        if not hits:
            print(f"Kein Eintrag mit Preis gefunden für '{args.name}'")
        elif len(hits) > 1:
            print("Mehrere Treffer — bitte genauer angeben:")
            for h in hits:
                print(f"  id={h['id']}  {h['name']}  ({h['city']})")
        else:
            h = hits[0]
            with db.get_conn() as conn:
                conn.execute("UPDATE prices SET price = ? WHERE pizzeria_id = ?",
                             (args.price, h["id"]))
            print(f"✓ Preis für '{h['name']}' korrigiert → {args.price:.2f} €")
    elif args.cmd == "delete-price":
        db.init_db()
        with db.get_conn() as conn:
            hits = conn.execute(
                "SELECT p.id, p.name, p.city FROM pizzerias p "
                "JOIN prices pr ON pr.pizzeria_id = p.id "
                "WHERE LOWER(p.name) LIKE ?",
                (f"%{args.name.lower()}%",)
            ).fetchall()
        if not hits:
            print(f"Kein Eintrag mit Preis gefunden für '{args.name}'")
        elif len(hits) > 1:
            print("Mehrere Treffer — bitte genauer angeben:")
            for h in hits:
                print(f"  id={h['id']}  {h['name']}  ({h['city']})")
        else:
            h = hits[0]
            with db.get_conn() as conn:
                conn.execute("DELETE FROM prices WHERE pizzeria_id = ?", (h["id"],))
            print(f"✓ Preiseintrag für '{h['name']}' gelöscht (nicht mehr auf der Karte)")
    elif args.cmd == "fetch-ratings":
        from osm_collector import CITY_BBOXES
        bbox = None
        if args.city:
            key = args.city.lower()
            if key in CITY_BBOXES:
                raw = CITY_BBOXES[key][0][1]
                parts = [float(x) for x in raw.split(",")]
                bbox = (parts[0], parts[1], parts[2], parts[3])
        asyncio.run(google_finder.run_fetch_ratings(
            city=(None if bbox else args.city),
            bbox=bbox,
            yes=args.yes,
        ))
    elif args.cmd == "find-websites-google":
        from osm_collector import CITY_BBOXES
        bbox = None
        if args.city:
            key = args.city.lower()
            if key in CITY_BBOXES:
                raw = CITY_BBOXES[key][0][1]
                parts = [float(x) for x in raw.split(",")]
                bbox = (parts[0], parts[1], parts[2], parts[3])
        asyncio.run(google_finder.run_find_websites(
            city=(None if bbox else args.city),
            bbox=bbox,
            dry_run=args.dry_run,
            yes=args.yes,
        ))
    elif args.cmd == "check-status":
        from osm_collector import CITY_BBOXES
        bbox = None
        if args.city:
            key = args.city.lower()
            if key in CITY_BBOXES:
                raw = CITY_BBOXES[key][0][1]
                parts = [float(x) for x in raw.split(",")]
                bbox = (parts[0], parts[1], parts[2], parts[3])
        asyncio.run(google_finder.run_check_status(
            city=(None if bbox else args.city),
            bbox=bbox,
            yes=args.yes,
        ))


if __name__ == "__main__":
    main()
