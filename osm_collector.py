"""
Collect pizzeria data from OpenStreetMap via Overpass API.
Writes results directly into the local SQLite database.
"""

import asyncio
import logging
from typing import Optional

import httpx
from tqdm import tqdm

import config
import database as db

log = logging.getLogger(__name__)

# Overpass query: all pizza restaurants in Germany with any useful tags
# Germany split into 4 roughly equal strips (south→north)
_DE_REGIONS = [
    ("Süd",      "47.2,5.9,50.0,15.1"),
    ("Mitte",    "50.0,5.9,52.0,15.1"),
    ("Nord-West","52.0,5.9,55.1,11.0"),
    ("Nord-Ost", "52.0,11.0,55.1,15.1"),
]

# Predefined city bounding boxes  (south, west, north, east)
CITY_BBOXES: dict[str, list[tuple[str, str]]] = {
    "düsseldorf": [("Düsseldorf", "51.10,6.65,51.38,6.95")],
    "berlin":     [("Berlin",     "52.34,13.09,52.68,13.76")],
    "münchen":    [("München",    "48.06,11.36,48.25,11.72")],
    "hamburg":    [("Hamburg",    "53.39,9.73,53.74,10.33")],
    "köln":       [("Köln",       "50.83,6.77,51.08,7.16")],
    "den haag":   [("Den Haag",   "52.00,4.15,52.18,4.45")],
    "hannover":   [("Hannover",   "52.30,9.62,52.44,9.85")],
    "bonn":       [("Bonn",       "50.65,7.02,50.78,7.18")],
    "bremen":     [("Bremen",     "53.02,8.69,53.19,8.96")],
    "stuttgart":  [("Stuttgart",  "48.69,9.10,48.83,9.29")],
    "amsterdam":  [("Amsterdam",  "52.28,4.72,52.43,5.07")],
    "rotterdam":  [("Rotterdam",  "51.86,4.36,51.97,4.60")],
    "leipzig":    [("Leipzig",    "51.27,12.27,51.42,12.52")],
    "magdeburg":  [("Magdeburg",  "52.05,11.55,52.20,11.75")],
    "haarlem":    [("Haarlem",    "52.35,4.58,52.42,4.67")],
    "frankfurt":  [("Frankfurt",  "50.02,8.47,50.23,8.80")],
    "dortmund":   [("Dortmund",   "51.44,7.34,51.60,7.63")],
    "essen":      [("Essen",      "51.38,6.90,51.53,7.12")],
    "dresden":    [("Dresden",    "51.00,13.60,51.18,13.95")],
    "nürnberg":   [("Nürnberg",   "49.38,10.97,49.52,11.17")],
    "duisburg":   [("Duisburg",   "51.36,6.68,51.48,6.87")],
    "utrecht":    [("Utrecht",    "52.04,5.04,52.13,5.18")],
    "eindhoven":  [("Eindhoven",  "51.39,5.42,51.50,5.55")],
    "groningen":  [("Groningen",  "53.18,6.52,53.26,6.64")],
    "tilburg":    [("Tilburg",    "51.53,5.02,51.60,5.17")],
    "almere":     [("Almere",     "52.32,5.16,52.43,5.34")],
    "breda":      [("Breda",      "51.56,4.74,51.64,4.88")],
    "nijmegen":   [("Nijmegen",   "51.82,5.82,51.88,5.95")],
    "apeldoorn":       [("Apeldoorn",       "52.17,5.89,52.25,6.05")],
    "enschede":        [("Enschede",        "52.18,6.83,52.27,6.96")],
    "arnhem":          [("Arnhem",          "51.95,5.85,52.01,5.98")],
    "amersfoort":      [("Amersfoort",      "52.13,5.34,52.20,5.45")],
    "zaanstad":        [("Zaanstad",        "52.43,4.78,52.50,4.89")],
    "'s-hertogenbosch":[("'s-Hertogenbosch","51.66,5.25,51.73,5.36")],
    "haarlemmermeer":  [("Haarlemmermeer",  "52.28,4.58,52.38,4.80")],
    "zwolle":          [("Zwolle",          "52.49,6.05,52.53,6.15")],
    "rostock":         [("Rostock",         "54.06,12.04,54.13,12.22")],
    "kassel":          [("Kassel",          "51.28,9.44,51.35,9.54")],
    "hagen":           [("Hagen",           "51.34,7.42,51.41,7.54")],
    "potsdam":         [("Potsdam",         "52.36,13.00,52.44,13.14")],
    "hamm":            [("Hamm",            "51.65,7.77,51.72,7.89")],
    "mülheim":         [("Mülheim an der Ruhr","51.40,6.85,51.46,6.94")],
    "ludwigshafen":    [("Ludwigshafen",    "49.46,8.39,49.51,8.49")],
    "oldenburg":       [("Oldenburg",       "53.11,8.17,53.18,8.28")],
    "leverkusen":      [("Leverkusen",      "51.01,6.94,51.07,7.04")],
    "darmstadt":       [("Darmstadt",       "49.84,8.60,49.91,8.71")],
    "solingen":        [("Solingen",        "51.14,7.03,51.21,7.18")],
    "heidelberg":      [("Heidelberg",      "49.38,8.65,49.43,8.74")],
    "herne":           [("Herne",           "51.51,7.18,51.57,7.30")],
    "neuss":           [("Neuss",           "51.17,6.63,51.24,6.74")],
    "regensburg":      [("Regensburg",      "48.99,12.05,49.05,12.16")],
    "pforzheim":       [("Pforzheim",       "48.87,8.66,48.92,8.74")],
    "ingolstadt":      [("Ingolstadt",      "48.74,11.39,48.79,11.46")],
    "würzburg":        [("Würzburg",        "49.77,9.91,49.82,9.98")],
    "fürth":           [("Fürth",           "49.46,10.97,49.51,11.03")],
    "wolfsburg":       [("Wolfsburg",       "52.40,10.75,52.46,10.83")],
    "offenbach":       [("Offenbach",       "50.09,8.75,50.13,8.81")],
    "ulm":             [("Ulm",             "48.38,9.97,48.42,10.03")],
    "heilbronn":       [("Heilbronn",       "49.13,9.20,49.17,9.25")],
    "göttingen":       [("Göttingen",       "51.52,9.92,51.56,9.98")],
    "paderborn":       [("Paderborn",       "51.71,8.73,51.74,8.78")],
    "recklinghausen":  [("Recklinghausen",  "51.60,7.19,51.64,7.24")],
    "bottrop":         [("Bottrop",         "51.52,6.91,51.56,6.97")],
    "salzgitter":      [("Salzgitter",      "52.14,10.33,52.18,10.40")],
    "bremerhaven":     [("Bremerhaven",     "53.54,8.57,53.58,8.63")],
    "reutlingen":      [("Reutlingen",      "48.48,9.20,48.52,9.26")],
    "moers":           [("Moers",           "51.45,6.62,51.49,6.68")],
    "koblenz":         [("Koblenz",         "50.34,7.55,50.38,7.62")],
    "bergischgladbach":[("Bergisch Gladbach","50.98,7.10,51.03,7.18")],
    "remscheid":       [("Remscheid",       "51.17,7.17,51.21,7.25")],
    "erlangen":        [("Erlangen",        "49.57,10.97,49.62,11.05")],
    "trier":           [("Trier",           "49.74,6.62,49.77,6.67")],
    "jena":            [("Jena",            "50.91,11.55,50.95,11.62")],
    "siegen":          [("Siegen",          "50.87,7.99,50.91,8.05")],
    "hildesheim":      [("Hildesheim",      "52.14,9.93,52.17,9.99")],
    "cottbus":         [("Cottbus",         "51.74,14.31,51.78,14.37")],
    "gütersloh":       [("Gütersloh",       "51.89,8.37,51.93,8.43")],
    "kaiserslautern":  [("Kaiserslautern",  "49.44,7.74,49.47,7.80")],
    "witten":          [("Witten",          "51.43,7.32,51.47,7.38")],
    "hanau":           [("Hanau",           "50.12,8.90,50.16,8.96")],
    "schwerin":        [("Schwerin",        "53.62,11.39,53.66,11.45")],
    "esslingen":       [("Esslingen",       "48.73,9.30,48.76,9.35")],
    "ludwigsburg":     [("Ludwigsburg",     "48.89,9.18,48.92,9.23")],
    "düren":           [("Düren",           "50.79,6.47,50.82,6.52")],
    "iserlohn":        [("Iserlohn",        "51.35,7.65,51.41,7.76")],
    "tübingen":        [("Tübingen",        "48.50,9.02,48.54,9.10")],
    "flensburg":       [("Flensburg",       "54.77,9.40,54.82,9.49")],
    "zwickau":         [("Zwickau",         "50.70,12.46,50.75,12.54")],
    "ratingen":        [("Ratingen",        "51.27,6.81,51.33,6.90")],
    "gießen":          [("Gießen",          "50.56,8.63,50.60,8.71")],
    "konstanz":        [("Konstanz",        "47.64,9.14,47.68,9.22")],
    "lünen":           [("Lünen",           "51.60,7.49,51.64,7.57")],
    "marl":            [("Marl",            "51.63,7.05,51.68,7.14")],
    "worms":           [("Worms",           "49.61,8.32,49.65,8.40")],
    "velbert":         [("Velbert",         "51.31,7.00,51.36,7.09")],
    "minden":          [("Minden",          "52.27,8.87,52.31,8.95")],
    "neumünster":      [("Neumünster",      "54.05,9.95,54.09,10.03")],
    "norderstedt":     [("Norderstedt",     "53.69,9.96,53.73,10.03")],
    "delmenhorst":     [("Delmenhorst",     "53.03,8.59,53.07,8.67")],
    "viersen":         [("Viersen",         "51.24,6.35,51.28,6.44")],
    "aschaffenburg":   [("Aschaffenburg",   "49.96,9.11,50.00,9.19")],
    "marburg":         [("Marburg",         "50.79,8.73,50.83,8.81")],
    "wilhelmshaven":   [("Wilhelmshaven",   "53.51,8.07,53.55,8.15")],
    "landshut":        [("Landshut",        "48.52,12.11,48.56,12.19")],
    "gladbeck":        [("Gladbeck",        "51.55,6.95,51.59,7.04")],
    "dorsten":         [("Dorsten",         "51.64,6.93,51.68,7.02")],
    "castrop-rauxel":  [("Castrop-Rauxel",  "51.53,7.27,51.57,7.36")],
    "troisdorf":       [("Troisdorf",       "50.79,7.11,50.83,7.20")],
    "arnsberg":        [("Arnsberg",        "51.38,8.03,51.42,8.12")],
    "rheine":          [("Rheine",          "52.26,7.40,52.30,7.49")],
    "bocholt":         [("Bocholt",         "51.82,6.57,51.86,6.66")],
    "lüneburg":        [("Lüneburg",        "53.23,10.37,53.27,10.45")],
    "lippstadt":       [("Lippstadt",       "51.65,8.30,51.69,8.39")],
    "dinslaken":       [("Dinslaken",       "51.54,6.69,51.58,6.78")],
    "herford":         [("Herford",         "52.09,8.63,52.13,8.71")],
    "kerpen":          [("Kerpen",          "50.85,6.66,50.89,6.75")],
    "plauen":          [("Plauen",          "50.48,12.10,50.52,12.18")],
    # Round 8
    "neubrandenburg":  [("Neubrandenburg",  "53.54,13.24,53.58,13.32")],
    "weimar":          [("Weimar",          "50.94,11.27,51.03,11.42")],
    "sindelfingen":    [("Sindelfingen",    "48.67,8.95,48.75,9.10")],
    "herten":          [("Herten",          "51.55,7.08,51.64,7.23")],
    "fulda":           [("Fulda",           "50.53,9.65,50.58,9.73")],
    "greifswald":      [("Greifswald",      "54.07,13.35,54.12,13.43")],
    "dormagen":        [("Dormagen",        "51.07,6.80,51.12,6.89")],
    "passau":          [("Passau",          "48.55,13.42,48.59,13.50")],
    "freising":        [("Freising",        "48.37,11.70,48.44,11.82")],
    "bamberg":         [("Bamberg",         "49.85,10.85,49.93,10.98")],
    "straubing":       [("Straubing",       "48.84,12.52,48.92,12.65")],
    "hof":             [("Hof",             "50.27,11.87,50.35,12.00")],
    "kempten":         [("Kempten",         "47.70,10.28,47.74,10.36")],
    "weiden":          [("Weiden i.d.OPf.", "49.63,12.11,49.72,12.24")],
    "kaufbeuren":      [("Kaufbeuren",      "47.86,10.60,47.90,10.68")],
    "amberg":          [("Amberg",          "49.42,11.84,49.47,11.92")],
    "schwabach":       [("Schwabach",       "49.30,10.97,49.36,11.11")],
    "coburg":          [("Coburg",          "50.23,10.92,50.30,11.06")],
    "ansbach":         [("Ansbach",         "49.27,10.55,49.32,10.63")],
    "memmingen":       [("Memmingen",       "47.97,10.16,48.00,10.23")],
    "rosenheim":       [("Rosenheim",       "47.84,12.10,47.88,12.18")],
    "bayreuth":        [("Bayreuth",        "49.93,11.55,49.97,11.63")],
    # ── neue Städterunde ────────────────────────────────────────────────────
    "bochum":              [("Bochum",              "51.43,7.17,51.52,7.27")],
    "wuppertal":           [("Wuppertal",           "51.20,7.04,51.30,7.28")],
    "bielefeld":           [("Bielefeld",           "51.97,8.47,52.06,8.60")],
    "münster":             [("Münster",             "51.91,7.57,52.00,7.70")],
    "mannheim":            [("Mannheim",            "49.44,8.43,49.54,8.53")],
    "karlsruhe":           [("Karlsruhe",           "48.97,8.35,49.04,8.46")],
    "augsburg":            [("Augsburg",            "48.33,10.85,48.40,10.94")],
    "wiesbaden":           [("Wiesbaden",           "49.99,8.19,50.07,8.31")],
    "gelsenkirchen":       [("Gelsenkirchen",       "51.49,7.04,51.55,7.15")],
    "mönchengladbach":     [("Mönchengladbach",     "51.12,6.33,51.27,6.52")],
    "braunschweig":        [("Braunschweig",        "52.23,10.47,52.30,10.57")],
    "kiel":                [("Kiel",                "54.30,10.09,54.37,10.17")],
    "chemnitz":            [("Chemnitz",            "50.81,12.86,50.85,12.96")],
    "halle (saale)":       [("Halle (Saale)",       "51.47,11.94,51.52,12.01")],
    "freiburg im breisgau":[("Freiburg im Breisgau","47.97,7.79,48.02,7.89")],
    "krefeld":             [("Krefeld",             "51.30,6.54,51.38,6.65")],
    "mainz":               [("Mainz",               "49.97,8.20,50.02,8.30")],
    "lübeck":              [("Lübeck",              "53.85,10.66,53.89,10.75")],
    "erfurt":              [("Erfurt",              "50.96,11.00,51.02,11.07")],
    "oberhausen":          [("Oberhausen",          "51.45,6.82,51.50,6.90")],
    "saarbrücken":         [("Saarbrücken",         "49.22,6.96,49.27,7.04")],
    "osnabrück":           [("Osnabrück",           "52.26,8.03,52.31,8.10")],
    "offenbach am main":   [("Offenbach am Main",   "50.09,8.74,50.14,8.82")],
    "gera":                [("Gera",                "50.87,12.07,50.92,12.14")],
    "villingen-schwenningen":[("Villingen-Schwenningen","47.99,8.45,48.07,8.56")],
    "dessau-roßlau":       [("Dessau-Roßlau",       "51.82,12.22,51.86,12.28")],
    "detmold":             [("Detmold",             "51.92,8.87,51.96,8.93")],
    "celle":               [("Celle",               "52.60,10.07,52.64,10.12")],
    "aalen":               [("Aalen",               "48.82,10.08,48.86,10.13")],
    "rüsselsheim am main": [("Rüsselsheim am Main", "49.99,8.38,50.02,8.43")],
    "neuwied":             [("Neuwied",             "50.42,7.45,50.46,7.52")],
}


def _make_query(bbox: str) -> str:
    return f"""
[out:json][timeout:120];
(
  node["amenity"="restaurant"]["cuisine"~"pizza",i]({bbox});
  node["amenity"="fast_food"]["cuisine"~"pizza",i]({bbox});
  way["amenity"="restaurant"]["cuisine"~"pizza",i]({bbox});
  way["amenity"="fast_food"]["cuisine"~"pizza",i]({bbox});
  node["amenity"~"restaurant|fast_food"]["name"~"pizza|pizzeria",i]({bbox});
  way["amenity"~"restaurant|fast_food"]["name"~"pizza|pizzeria",i]({bbox});
);
out center body;
"""


def _parse_node(node: dict) -> Optional[db.Pizzeria]:
    tags = node.get("tags", {})
    name = tags.get("name") or tags.get("brand")
    if not name:
        return None

    # nodes have lat/lon directly; ways use center
    lat = node.get("lat") or (node.get("center") or {}).get("lat")
    lon = node.get("lon") or (node.get("center") or {}).get("lon")
    if lat is None or lon is None:
        return None

    website = (
        tags.get("website")
        or tags.get("contact:website")
        or tags.get("url")
    )
    if website and not website.startswith("http"):
        website = "https://" + website

    element_type = node.get("type", "node")
    return db.Pizzeria(
        osm_id=f"{element_type}/{node['id']}",
        name=name,
        lat=lat,
        lon=lon,
        city=tags.get("addr:city"),
        postcode=tags.get("addr:postcode"),
        street=tags.get("addr:street"),
        housenumber=tags.get("addr:housenumber"),
        website=website,
        phone=tags.get("phone") or tags.get("contact:phone"),
    )


_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


async def _query_region(
    client: httpx.AsyncClient,
    name: str,
    bbox: str,
) -> list[dict]:
    query = _make_query(bbox)
    for url in _MIRRORS:
        try:
            resp = await client.post(
                url,
                data={"data": query},
                headers={
                    "User-Agent": config.USER_AGENT,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            log.info("Region %s: %d elements (via %s)", name, len(elements), url)
            return elements
        except Exception as e:
            log.warning("Region %s / %s failed: %s", name, url, e)
    log.error("Region %s: all mirrors failed", name)
    return []


async def fetch_from_osm(regions: list[tuple[str, str]] | None = None) -> list[db.Pizzeria]:
    if regions is None:
        regions = _DE_REGIONS
    log.info("Querying Overpass API in %d regions…", len(regions))
    all_elements: list[dict] = []

    async with httpx.AsyncClient(timeout=150) as client:
        for name, bbox in regions:
            elements = await _query_region(client, name, bbox)
            all_elements.extend(elements)

    log.info("OSM total: %d elements", len(all_elements))

    seen: set[str] = set()
    pizzerias = []
    for el in all_elements:
        p = _parse_node(el)
        if p and p.osm_id not in seen:
            seen.add(p.osm_id)
            pizzerias.append(p)

    log.info("Parsed %d unique pizzerias", len(pizzerias))
    return pizzerias


def save_to_db(pizzerias: list[db.Pizzeria]) -> dict:
    counts = {"total": 0, "with_website": 0}
    for p in tqdm(pizzerias, desc="Saving to DB"):
        db.upsert_pizzeria(p)
        counts["total"] += 1
        if p.website:
            counts["with_website"] += 1
    return counts


async def run(regions: list[tuple[str, str]] | None = None):
    db.init_db()
    pizzerias = await fetch_from_osm(regions)
    counts = save_to_db(pizzerias)
    print(f"\nDone. {counts['total']} pizzerias saved, "
          f"{counts['with_website']} have a website URL.")
    print(db.stats())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(run())
