"""Polymarket market metadata discovery and enrichment."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Iterable

import aiohttp
import orjson
import polars as pl
import structlog

from poly.storage.recover import iter_recovered_gzip_jsonl_lines

logger = structlog.get_logger()

UPDOWN_MARKETS = [
    ("btc", "5m", 300),
    ("btc", "15m", 900),
    ("eth", "5m", 300),
    ("eth", "15m", 900),
]


@dataclass(frozen=True)
class MetadataFetchConfig:
    data_dir: Path
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    dates: tuple[str, ...] = ()
    concurrency: int = 4
    timeout_seconds: float = 12.0


def metadata_path(data_dir: Path, date: str) -> Path:
    return data_dir / "normalized" / date / "poly_market_metadata.parquet"


def updown_slugs_for_date(date: str) -> list[str]:
    """Generate expected BTC/ETH UpDown market slugs for a UTC date."""
    day = datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc)
    end = day + timedelta(days=1)
    slugs: list[str] = []
    for symbol, period_label, period_seconds in UPDOWN_MARKETS:
        ts = int(day.timestamp())
        while ts < int(end.timestamp()):
            slugs.append(f"{symbol}-updown-{period_label}-{ts}")
            ts += period_seconds
    return slugs


def updown_slugs_for_ns_range(start_ns: int, end_ns: int, buffer_seconds: int = 3600) -> list[str]:
    start = max(0, start_ns // 1_000_000_000 - buffer_seconds)
    end = end_ns // 1_000_000_000 + buffer_seconds
    slugs: list[str] = []
    for symbol, period_label, period_seconds in UPDOWN_MARKETS:
        ts = start - (start % period_seconds)
        while ts <= end:
            slugs.append(f"{symbol}-updown-{period_label}-{ts}")
            ts += period_seconds
    return slugs


def discover_slugs_from_local_data(data_dir: Path, date: str) -> list[str]:
    """Use existing normalized timestamps/source hints to avoid fetching a full day when possible."""
    slugs: set[str] = set()
    min_ns: int | None = None
    max_ns: int | None = None
    for table in ["poly_l2_book", "poly_trades", "poly_best_bid_ask"]:
        path = data_dir / "normalized" / date / f"{table}.parquet"
        if not path.exists():
            continue
        try:
            schema = pl.read_parquet_schema(str(path))
            scan = pl.scan_parquet(str(path))
        except Exception:
            continue
        if "source" in schema:
            try:
                sources = scan.select(pl.col("source").drop_nulls().unique()).collect()
            except Exception:
                sources = pl.DataFrame()
            for source in sources.get_column("source").to_list() if "source" in sources.columns else []:
                if isinstance(source, str) and source.startswith("polymarket:"):
                    slug = source.split(":", 1)[1]
                    if slug:
                        slugs.add(slug)
        if "recv_ns" in schema:
            try:
                bounds = scan.select(
                    pl.col("recv_ns").min().alias("min_ns"),
                    pl.col("recv_ns").max().alias("max_ns"),
                ).collect()
            except Exception:
                bounds = pl.DataFrame()
            table_min = bounds["min_ns"][0] if "min_ns" in bounds.columns and not bounds.is_empty() else None
            table_max = bounds["max_ns"][0] if "max_ns" in bounds.columns and not bounds.is_empty() else None
            if table_min is not None:
                min_ns = table_min if min_ns is None else min(min_ns, table_min)
            if table_max is not None:
                max_ns = table_max if max_ns is None else max(max_ns, table_max)
    if min_ns is not None and max_ns is not None:
        slugs.update(updown_slugs_for_ns_range(int(min_ns), int(max_ns)))
    return sorted(slugs)


def parse_json_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def iso_to_ns(value: object) -> int | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(dt.timestamp() * 1_000_000_000)


def parse_updown_slug(slug: str) -> tuple[str | None, str | None, int | None, int | None]:
    parts = slug.split("-")
    if len(parts) < 4 or parts[1] != "updown":
        return None, None, None, None
    symbol = parts[0].lower()
    period = parts[2].lower()
    try:
        start = int(parts[3])
    except ValueError:
        return symbol, period, None, None
    seconds = 300 if period == "5m" else 900 if period == "15m" else None
    expiry = start + seconds if seconds is not None else None
    return symbol, period, start * 1_000_000_000, expiry * 1_000_000_000 if expiry else None


def updown_expiry_ns(slug_expiry_ns: int | None, end_date: object) -> int | None:
    """Prefer slug-derived expiry for UpDown markets.

    Some Polymarket API responses expose an `endDate` rounded to the UTC date
    rather than the 5m/15m market expiry. The slug timestamp is the start time,
    so `parse_updown_slug` gives the exact short-market expiry.
    """
    return slug_expiry_ns or iso_to_ns(end_date)


def market_to_asset_rows(market: dict) -> list[dict[str, object]]:
    token_ids = parse_json_list(market.get("clobTokenIds"))
    outcomes = parse_json_list(market.get("outcomes"))
    if not token_ids:
        return []

    slug = str(market.get("slug") or "")
    symbol, period, start_ns, slug_expiry_ns = parse_updown_slug(slug)
    end_date = market.get("endDate")
    expiry_ns = updown_expiry_ns(slug_expiry_ns, end_date)
    outcomes_json = json.dumps(outcomes, ensure_ascii=False)
    token_ids_json = json.dumps(token_ids, ensure_ascii=False)

    rows: list[dict[str, object]] = []
    for idx, token_id in enumerate(token_ids):
        outcome = outcomes[idx] if idx < len(outcomes) else None
        rows.append(
            {
                "asset_id": str(token_id),
                "outcome": str(outcome) if outcome is not None else "",
                "outcome_index": idx,
                "market_id": str(market.get("id") or ""),
                "condition_id": str(market.get("conditionId") or ""),
                "slug": slug,
                "symbol": symbol or "",
                "period": period or "",
                "question": str(market.get("question") or ""),
                "outcomes": outcomes_json,
                "clob_token_ids": token_ids_json,
                "active": bool(market.get("active", False)),
                "closed": bool(market.get("closed", False)),
                "accepting_orders": bool(market.get("acceptingOrders", False)),
                "start_date": market.get("startDate"),
                "end_date": end_date,
                "start_ns": start_ns,
                "expiry_ns": expiry_ns,
                "tick_size": as_float(market.get("orderPriceMinTickSize") or market.get("minimumTickSize")),
                "min_order_size": as_float(market.get("orderMinSize") or market.get("minimumOrderSize")),
                "maker_base_fee": as_float(market.get("makerBaseFee")),
                "taker_base_fee": as_float(market.get("takerBaseFee")),
                "volume": as_float(market.get("volumeNum") or market.get("volume")),
                "volume_24h": as_float(market.get("volume24hr") or market.get("volume24hrClob")),
                "liquidity": as_float(market.get("liquidityNum") or market.get("liquidity")),
                "best_bid_snapshot": as_float(market.get("bestBid")),
                "best_ask_snapshot": as_float(market.get("bestAsk")),
                "spread_snapshot": as_float(market.get("spread")),
                "updated_at": market.get("updatedAt"),
            }
        )
    return rows


def clob_market_to_asset_rows(market: dict) -> list[dict[str, object]]:
    tokens = market.get("tokens") or []
    if not isinstance(tokens, list) or not tokens:
        return []

    slug = str(market.get("market_slug") or "")
    symbol, period, start_ns, slug_expiry_ns = parse_updown_slug(slug)
    end_date = market.get("end_date_iso")
    expiry_ns = updown_expiry_ns(slug_expiry_ns, end_date)
    token_ids = [str(t.get("token_id") or "") for t in tokens if isinstance(t, dict)]
    outcomes = [str(t.get("outcome") or "") for t in tokens if isinstance(t, dict)]

    rows: list[dict[str, object]] = []
    for idx, token in enumerate(tokens):
        if not isinstance(token, dict):
            continue
        token_id = str(token.get("token_id") or "")
        if not token_id:
            continue
        rows.append(
            {
                "asset_id": token_id,
                "outcome": str(token.get("outcome") or ""),
                "outcome_index": idx,
                "market_id": str(market.get("condition_id") or ""),
                "condition_id": str(market.get("condition_id") or ""),
                "slug": slug,
                "symbol": symbol or "",
                "period": period or "",
                "question": str(market.get("question") or ""),
                "outcomes": json.dumps(outcomes, ensure_ascii=False),
                "clob_token_ids": json.dumps(token_ids, ensure_ascii=False),
                "active": bool(market.get("active", False)),
                "closed": bool(market.get("closed", False)),
                "accepting_orders": bool(market.get("accepting_orders", False)),
                "start_date": market.get("accepting_order_timestamp"),
                "end_date": end_date,
                "start_ns": start_ns,
                "expiry_ns": expiry_ns,
                "tick_size": as_float(market.get("minimum_tick_size")),
                "min_order_size": as_float(market.get("minimum_order_size")),
                "maker_base_fee": as_float(market.get("maker_base_fee")),
                "taker_base_fee": as_float(market.get("taker_base_fee")),
                "volume": None,
                "volume_24h": None,
                "liquidity": None,
                "best_bid_snapshot": as_float(token.get("price")),
                "best_ask_snapshot": None,
                "spread_snapshot": None,
                "updated_at": None,
            }
        )
    return rows


def as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_condition_ids_from_local_data(data_dir: Path, date: str) -> list[str]:
    condition_ids: set[str] = set()
    for table in ["poly_l2_book", "poly_trades", "poly_best_bid_ask"]:
        path = data_dir / "normalized" / date / f"{table}.parquet"
        if not path.exists():
            continue
        try:
            schema = pl.read_parquet_schema(str(path))
            if "market" not in schema:
                continue
            df = pl.scan_parquet(str(path)).select(pl.col("market").drop_nulls().unique()).collect()
        except Exception:
            continue
        for value in df["market"].drop_nulls().unique().to_list():
            if isinstance(value, str) and value.startswith("0x"):
                condition_ids.add(value)

    raw_path = data_dir / "raw_feed" / date / "polymarket_market.jsonl.gz"
    if raw_path.exists():
        for line in iter_recovered_gzip_jsonl_lines(raw_path):
            try:
                outer = orjson.loads(line)
                raw = outer.get("raw", {})
                if isinstance(raw, str):
                    raw = orjson.loads(raw)
            except Exception:
                continue
            messages = raw if isinstance(raw, list) else [raw]
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                market = msg.get("market")
                if isinstance(market, str) and market.startswith("0x"):
                    condition_ids.add(market)
    return sorted(condition_ids)


async def fetch_market_by_slug(
    session: aiohttp.ClientSession,
    gamma_url: str,
    slug: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        for attempt in range(4):
            try:
                async with session.get(f"{gamma_url.rstrip('/')}/markets", params={"slug": slug}) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        return None
                    payload = await resp.json()
                    break
            except Exception:
                await asyncio.sleep(0.25 * (attempt + 1))
        else:
            return None
    if isinstance(payload, list) and payload:
        first = payload[0]
        return first if isinstance(first, dict) else None
    return payload if isinstance(payload, dict) and payload.get("id") else None


async def fetch_market_by_condition_id(
    session: aiohttp.ClientSession,
    clob_url: str,
    condition_id: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    async with semaphore:
        for attempt in range(4):
            try:
                async with session.get(f"{clob_url.rstrip('/')}/markets/{condition_id}") as resp:
                    if resp.status == 429:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    if resp.status != 200:
                        return None
                    payload = await resp.json()
                    return payload if isinstance(payload, dict) and payload.get("condition_id") else None
            except Exception:
                await asyncio.sleep(0.25 * (attempt + 1))
    return None


async def fetch_metadata(config: MetadataFetchConfig) -> pl.DataFrame:
    slugs: list[str] = []
    condition_ids: list[str] = []
    for date in config.dates:
        condition_ids.extend(discover_condition_ids_from_local_data(config.data_dir, date))
        local_slugs = discover_slugs_from_local_data(config.data_dir, date)
        slugs.extend(local_slugs or updown_slugs_for_date(date))
    slugs = sorted(set(slugs))
    condition_ids = sorted(set(condition_ids))
    if not slugs and not condition_ids:
        return pl.DataFrame()

    timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
    semaphore = asyncio.Semaphore(max(config.concurrency, 1))
    rows: list[dict[str, object]] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        condition_tasks = [
            fetch_market_by_condition_id(session, config.clob_url, condition_id, semaphore)
            for condition_id in condition_ids
        ]
        for market in await asyncio.gather(*condition_tasks):
            if market:
                rows.extend(clob_market_to_asset_rows(market))

        known_assets = {str(row.get("asset_id")) for row in rows}
        slug_tasks = [fetch_market_by_slug(session, config.gamma_url, slug, semaphore) for slug in slugs]
        for market in await asyncio.gather(*slug_tasks):
            if market:
                for row in market_to_asset_rows(market):
                    if str(row.get("asset_id")) not in known_assets:
                        rows.append(row)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).unique(subset=["asset_id"], keep="last").sort(["symbol", "period", "expiry_ns", "outcome_index"])


def write_metadata_by_date(frame: pl.DataFrame, data_dir: Path, dates: Iterable[str]) -> list[Path]:
    if frame.is_empty():
        return []
    paths: list[Path] = []
    for date in dates:
        start = int(datetime.strptime(date, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1_000_000_000)
        end = start + 24 * 60 * 60 * 1_000_000_000
        date_frame = frame.filter(
            ((pl.col("expiry_ns").is_null()) | ((pl.col("expiry_ns") >= start) & (pl.col("expiry_ns") <= end + 900 * 1_000_000_000)))
        )
        if date_frame.is_empty():
            continue
        path = metadata_path(data_dir, date)
        path.parent.mkdir(parents=True, exist_ok=True)
        date_frame.write_parquet(str(path))
        paths.append(path)
    return paths


def load_metadata(data_dir: Path, date: str) -> pl.DataFrame:
    path = metadata_path(data_dir, date)
    if path.exists():
        return pl.read_parquet(str(path))
    return pl.DataFrame()


def enrich_with_metadata(frame: pl.DataFrame, metadata: pl.DataFrame) -> pl.DataFrame:
    """Left join metadata by asset_id and coalesce common fields."""
    if frame is None or frame.is_empty() or metadata is None or metadata.is_empty() or "asset_id" not in frame.columns:
        return frame

    right_cols = [
        c
        for c in [
            "asset_id",
            "market_id",
            "condition_id",
            "slug",
            "outcome",
            "outcome_index",
            "symbol",
            "period",
            "question",
            "start_ns",
            "expiry_ns",
            "tick_size",
            "min_order_size",
            "maker_base_fee",
            "taker_base_fee",
            "volume_24h",
            "liquidity",
        ]
        if c in metadata.columns
    ]
    right = metadata.select(right_cols).unique(subset=["asset_id"], keep="last")
    joined = frame.join(right, on="asset_id", how="left", suffix="_meta")

    for col in right_cols:
        if col == "asset_id":
            continue
        meta_col = f"{col}_meta"
        if meta_col not in joined.columns:
            continue
        if col in frame.columns:
            if frame.schema.get(col) == pl.String:
                joined = joined.with_columns(
                    pl.when(pl.col(col).is_null() | (pl.col(col) == ""))
                    .then(pl.col(meta_col))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            else:
                joined = joined.with_columns(pl.coalesce([pl.col(col), pl.col(meta_col)]).alias(col))
        else:
            joined = joined.rename({meta_col: col})
        joined = joined.drop(meta_col, strict=False)
    return joined
