"""Async SQLite metadata store for market data."""

from __future__ import annotations

import json
import aiosqlite
import structlog
from pathlib import Path
from typing import Any

logger = structlog.get_logger()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    title TEXT,
    slug TEXT,
    active INTEGER,
    closed INTEGER,
    category TEXT,
    updated_at INTEGER
);

CREATE TABLE IF NOT EXISTS markets (
    id TEXT PRIMARY KEY,
    event_id TEXT,
    question TEXT,
    slug TEXT,
    condition_id TEXT,
    outcomes TEXT,
    clob_token_ids TEXT,
    active INTEGER,
    closed INTEGER,
    tick_size REAL,
    min_order_size REAL,
    category TEXT,
    tags TEXT,
    end_date TEXT,
    start_date TEXT,
    volume_num REAL,
    liquidity_num REAL,
    updated_at INTEGER,
    FOREIGN KEY (event_id) REFERENCES events(id)
);

CREATE TABLE IF NOT EXISTS asset_market_map (
    asset_id TEXT PRIMARY KEY,
    market_id TEXT,
    outcome TEXT,
    FOREIGN KEY (market_id) REFERENCES markets(id)
);
"""


class MetadataDB:
    """Async SQLite wrapper for market metadata."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info("db_initialized", path=str(self.db_path))

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _conn(self) -> aiosqlite.Connection:
        assert self._db is not None, "DB not initialized"
        return self._db

    async def upsert_event(self, event: dict) -> None:
        db = self._conn()
        await db.execute(
            """INSERT OR REPLACE INTO events (id, title, slug, active, closed, category, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (event.get("id"), event.get("title"), event.get("slug"),
             int(event.get("active", False)), int(event.get("closed", False)),
             event.get("category"), event.get("updated_at", 0)),
        )
        await db.commit()

    async def upsert_market(self, market: dict) -> None:
        db = self._conn()
        await db.execute(
            """INSERT OR REPLACE INTO markets
               (id, event_id, question, slug, condition_id, outcomes, clob_token_ids,
                active, closed, tick_size, min_order_size, category, tags,
                end_date, start_date, volume_num, liquidity_num, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (market.get("id"), market.get("event_id"), market.get("question"),
             market.get("slug"), market.get("condition_id"),
             market.get("outcomes"), market.get("clob_token_ids"),
             int(market.get("active", False)), int(market.get("closed", False)),
             market.get("tick_size"), market.get("min_order_size"),
             market.get("category"), market.get("tags"),
             market.get("end_date"), market.get("start_date"),
             market.get("volume_num"), market.get("liquidity_num"),
             market.get("updated_at", 0)),
        )
        await db.commit()

    async def upsert_asset_mapping(self, asset_id: str, market_id: str,
                                   outcome: str) -> None:
        db = self._conn()
        await db.execute(
            "INSERT OR REPLACE INTO asset_market_map (asset_id, market_id, outcome) VALUES (?, ?, ?)",
            (asset_id, market_id, outcome),
        )
        await db.commit()

    async def upsert_market_batch(self, markets: list[dict]) -> None:
        db = self._conn()
        rows = []
        for m in markets:
            rows.append((
                m.get("id"), m.get("event_id"), m.get("question"),
                m.get("slug"), m.get("condition_id"),
                m.get("outcomes"), m.get("clob_token_ids"),
                int(m.get("active", False)), int(m.get("closed", False)),
                m.get("tick_size"), m.get("min_order_size"),
                m.get("category"), m.get("tags"),
                m.get("end_date"), m.get("start_date"),
                m.get("volume_num"), m.get("liquidity_num"),
                m.get("updated_at", 0)),
            )
        await db.executemany(
            """INSERT OR REPLACE INTO markets
               (id, event_id, question, slug, condition_id, outcomes, clob_token_ids,
                active, closed, tick_size, min_order_size, category, tags,
                end_date, start_date, volume_num, liquidity_num, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        await db.commit()

    async def upsert_asset_batch(self, mappings: list[tuple[str, str, str]]) -> None:
        db = self._conn()
        await db.executemany(
            "INSERT OR REPLACE INTO asset_market_map (asset_id, market_id, outcome) VALUES (?, ?, ?)",
            mappings,
        )
        await db.commit()

    async def get_active_markets(self, tag_filter: list[str] | None = None) -> list[dict]:
        db = self._conn()
        query = "SELECT * FROM markets WHERE active = 1 AND closed = 0"
        params: list = []
        if tag_filter:
            conditions = " OR ".join(["tags LIKE ?" for _ in tag_filter])
            query += f" AND ({conditions})"
            params.extend([f"%{t}%" for t in tag_filter])
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_asset_ids_for_subscription(self, tag_filter: list[str] | None = None) -> list[str]:
        markets = await self.get_active_markets(tag_filter)
        asset_ids = []
        for m in markets:
            token_ids = json.loads(m.get("clob_token_ids", "[]"))
            asset_ids.extend(token_ids)
        return asset_ids

    async def get_market_by_asset_id(self, asset_id: str) -> dict | None:
        db = self._conn()
        cursor = await db.execute(
            """SELECT m.* FROM markets m
               JOIN asset_market_map a ON a.market_id = m.id
               WHERE a.asset_id = ?""",
            (asset_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
