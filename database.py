import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import config


@dataclass
class Pizzeria:
    osm_id: str
    name: str
    lat: float
    lon: float
    city: Optional[str] = None
    postcode: Optional[str] = None
    street: Optional[str] = None
    housenumber: Optional[str] = None
    website: Optional[str] = None
    phone: Optional[str] = None


@dataclass
class PriceRecord:
    pizzeria_id: int
    price: float
    size_cm: Optional[int] = None
    size_label: Optional[str] = None
    menu_url: Optional[str] = None
    extraction_method: Optional[str] = None  # html | pdf | ocr | llm
    raw_snippet: Optional[str] = None
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pizzerias (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                osm_id      TEXT UNIQUE,
                name        TEXT NOT NULL,
                lat         REAL NOT NULL,
                lon         REAL NOT NULL,
                city        TEXT,
                postcode    TEXT,
                street      TEXT,
                housenumber TEXT,
                website     TEXT,
                phone       TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS prices (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                pizzeria_id         INTEGER NOT NULL REFERENCES pizzerias(id),
                price               REAL NOT NULL,
                size_cm             INTEGER,
                size_label          TEXT,
                menu_url            TEXT,
                extraction_method   TEXT,
                raw_snippet         TEXT,
                scraped_at          TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_pizzerias_osm ON pizzerias(osm_id);
            CREATE INDEX IF NOT EXISTS idx_prices_pizzeria ON prices(pizzeria_id);
        """)
        # Migrate existing DBs
        for col in ("street TEXT", "housenumber TEXT",
                    "google_rating REAL", "google_review_count INTEGER",
                    "business_status TEXT"):
            try:
                conn.execute(f"ALTER TABLE pizzerias ADD COLUMN {col}")
            except sqlite3.OperationalError:
                pass


def upsert_pizzeria(p: Pizzeria) -> int:
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO pizzerias (osm_id, name, lat, lon, city, postcode, street, housenumber, website, phone)
            VALUES (:osm_id, :name, :lat, :lon, :city, :postcode, :street, :housenumber, :website, :phone)
            ON CONFLICT(osm_id) DO UPDATE SET
                name        = excluded.name,
                website     = excluded.website,
                phone       = excluded.phone,
                city        = excluded.city,
                postcode    = excluded.postcode,
                street      = excluded.street,
                housenumber = excluded.housenumber
        """, vars(p))
        row = conn.execute(
            "SELECT id FROM pizzerias WHERE osm_id = ?", (p.osm_id,)
        ).fetchone()
        return row["id"]


def insert_price(r: PriceRecord):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO prices
                (pizzeria_id, price, size_cm, size_label, menu_url,
                 extraction_method, raw_snippet, scraped_at)
            VALUES
                (:pizzeria_id, :price, :size_cm, :size_label, :menu_url,
                 :extraction_method, :raw_snippet, :scraped_at)
        """, vars(r))


def get_pizzerias_without_price(
    city: Optional[str] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,  # south,west,north,east
) -> list[dict]:
    with get_conn() as conn:
        sql = """
            SELECT p.id, p.name, p.website, p.city, p.postcode,
                   p.street, p.housenumber, p.lat, p.lon
            FROM pizzerias p
            LEFT JOIN prices pr ON pr.pizzeria_id = p.id
            WHERE pr.id IS NULL
              AND p.website IS NOT NULL
        """
        params: list = []
        if bbox:
            south, west, north, east = bbox
            sql += " AND p.lat BETWEEN ? AND ? AND p.lon BETWEEN ? AND ?"
            params += [south, north, west, east]
        elif city:
            sql += " AND LOWER(p.city) LIKE ?"
            params.append(f"%{city.lower()}%")
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def update_google_rating(pizzeria_id: int, rating: Optional[float], review_count: Optional[int]):
    with get_conn() as conn:
        conn.execute(
            "UPDATE pizzerias SET google_rating = ?, google_review_count = ? WHERE id = ?",
            (rating, review_count, pizzeria_id),
        )


def update_business_status(pizzeria_id: int, status: Optional[str]):
    with get_conn() as conn:
        conn.execute(
            "UPDATE pizzerias SET business_status = ? WHERE id = ?",
            (status, pizzeria_id),
        )


def get_priced_pizzerias(
    city: Optional[str] = None,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> list[dict]:
    with get_conn() as conn:
        sql = """
            SELECT p.id, p.name, p.lat, p.lon, p.city, p.postcode,
                   p.street, p.housenumber, p.phone, p.website,
                   p.google_rating, p.google_review_count, p.business_status,
                   pr.price, pr.size_label, pr.size_cm, pr.menu_url, pr.raw_snippet
            FROM pizzerias p
            JOIN prices pr ON pr.pizzeria_id = p.id
        """
        params: list = []
        if bbox:
            south, west, north, east = bbox
            sql += " WHERE p.lat BETWEEN ? AND ? AND p.lon BETWEEN ? AND ?"
            params += [south, north, west, east]
        elif city:
            sql += " WHERE LOWER(p.city) LIKE ?"
            params.append(f"%{city.lower()}%")
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM pizzerias").fetchone()[0]
        with_site = conn.execute(
            "SELECT COUNT(*) FROM pizzerias WHERE website IS NOT NULL"
        ).fetchone()[0]
        priced = conn.execute(
            "SELECT COUNT(DISTINCT pizzeria_id) FROM prices"
        ).fetchone()[0]
        avg_price = conn.execute("SELECT AVG(price) FROM prices").fetchone()[0]
        return {
            "total_pizzerias": total,
            "with_website": with_site,
            "priced": priced,
            "avg_margherita_price": round(avg_price, 2) if avg_price else None,
        }
