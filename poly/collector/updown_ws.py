"""Up/Down market collector with auto-rotation.

Tracks btc-updown-5m, btc-updown-15m, eth-updown-5m, eth-updown-15m markets.
Automatically subscribes to upcoming markets and rotates when they expire.
"""

from __future__ import annotations

import asyncio
import time
import json
import structlog
import orjson
import aiohttp

import websockets

from poly.config import Config
from poly.storage.raw import RawWriter
from poly.storage.normalized import ParquetWriter, L2_BOOK_SCHEMA, TRADE_SCHEMA, BEST_BID_ASK_SCHEMA
from poly.engine.orderbook import OrderBookEngine

logger = structlog.get_logger()

# Market definitions: (base_slug, period_seconds, subscribe_ahead_seconds)
MARKET_DEFS = [
    ("btc-updown-5m",  300, 60),   # 5-min BTC, subscribe 60s early
    ("btc-updown-15m", 900, 120),  # 15-min BTC, subscribe 120s early
    ("eth-updown-5m",  300, 60),
    ("eth-updown-15m", 900, 120),
]


def _align_ts(now: int, period: int) -> int:
    """Align timestamp to the start of the current period."""
    return now - (now % period)


def _upcoming_timestamps(period: int, ahead: int, now: int) -> list[int]:
    """Return current + next period start timestamps within `ahead` seconds."""
    current = _align_ts(now, period)
    result = [current]
    t = current + period
    while t <= now + ahead:
        result.append(t)
        t += period
    return result


async def fetch_market_tokens(session: aiohttp.ClientSession, slug: str) -> dict | None:
    """Fetch market info from Gamma API by slug. Returns {slug, token_ids, condition_id} or None."""
    try:
        async with session.get(
            "https://gamma-api.polymarket.com/markets",
            params={"slug": slug},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            if not data:
                return None
            m = data[0] if isinstance(data, list) else data
            if not isinstance(m, dict) or not m.get("id"):
                return None
            tokens_str = m.get("clobTokenIds", "[]")
            try:
                token_ids = json.loads(tokens_str) if isinstance(tokens_str, str) else tokens_str
            except json.JSONDecodeError:
                return None
            return {
                "slug": slug,
                "question": m.get("question", ""),
                "token_ids": token_ids,
                "condition_id": m.get("conditionId", ""),
                "active": m.get("active", False),
            }
    except Exception as e:
        logger.warning("fetch_market_error", slug=slug, error=str(e))
        return None


class UpDownCollector:
    """Collects Polymarket Up/Down markets with automatic rotation."""

    def __init__(self, config: Config, raw_writer: RawWriter,
                 book_writer: ParquetWriter, trade_writer: ParquetWriter,
                 bba_writer: ParquetWriter, engine: OrderBookEngine) -> None:
        self.config = config
        self.raw_writer = raw_writer
        self.book_writer = book_writer
        self.trade_writer = trade_writer
        self.bba_writer = bba_writer
        self.engine = engine
        self._ws = None
        self._subscribed_assets: dict[str, str] = {}  # asset_id -> slug
        self._known_slugs: set[str] = set()
        self._msg_count = 0

    async def run(self) -> None:
        """Main loop: manage subscriptions and WebSocket connection."""
        backoff = 1.0
        while True:
            try:
                await self._ws_loop()
                backoff = 1.0
            except Exception as e:
                logger.warning("updown_ws_error", error=str(e), backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _ws_loop(self) -> None:
        """Connect, manage subscriptions, and process messages."""
        url = self.config.poly_market_ws_url
        logger.info("updown_ws_connecting", url=url)

        async with websockets.connect(url, ping_interval=None) as ws:
            self._ws = ws
            heartbeat_task = asyncio.create_task(self._heartbeat(ws))
            rotation_task = asyncio.create_task(self._rotation_loop())

            try:
                async for message in ws:
                    import poly_core
                    recv_ns = poly_core.now_ns()
                    self._msg_count += 1

                    if isinstance(message, bytes):
                        text = message.decode("utf-8", errors="replace")
                    else:
                        text = message
                    text = text.strip()
                    if not text or text in ("PONG", "PING"):
                        continue

                    await self.raw_writer.write(message if isinstance(message, bytes) else message.encode(), recv_ns)

                    try:
                        parsed = orjson.loads(text)
                    except Exception:
                        continue

                    messages = parsed if isinstance(parsed, list) else [parsed]
                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        event_type = msg.get("event_type")
                        if not event_type:
                            continue
                        self._dispatch(msg, recv_ns)
            finally:
                heartbeat_task.cancel()
                rotation_task.cancel()
                self._ws = None

    async def _rotation_loop(self) -> None:
        """Periodically check for new markets to subscribe and expired ones to drop."""
        async with aiohttp.ClientSession() as session:
            while True:
                await self._refresh_subscriptions(session)
                await asyncio.sleep(10)  # check every 10 seconds

    async def _refresh_subscriptions(self, session: aiohttp.ClientSession) -> None:
        """Discover and subscribe to current + upcoming markets."""
        now = int(time.time())
        new_slugs: list[str] = []

        for base_slug, period, ahead in MARKET_DEFS:
            timestamps = _upcoming_timestamps(period, ahead, now)
            for ts in timestamps:
                slug = f"{base_slug}-{ts}"
                # Only subscribe if not expired yet
                if ts + period <= now:
                    continue  # already expired
                if slug not in self._known_slugs:
                    new_slugs.append(slug)

        if not new_slugs:
            return

        # Fetch market info for new slugs
        new_assets: list[str] = []
        for slug in new_slugs:
            info = await fetch_market_tokens(session, slug)
            if info is None or not info["token_ids"]:
                logger.debug("updown_market_not_found", slug=slug)
                self._known_slugs.add(slug)  # don't retry constantly
                continue

            self._known_slugs.add(slug)
            for tid in info["token_ids"]:
                self._subscribed_assets[tid] = slug
                new_assets.append(tid)

            logger.info("updown_new_market",
                       slug=slug,
                       question=info["question"][:60],
                       tokens=len(info["token_ids"]))

        if not new_assets or self._ws is None:
            return

        # Subscribe to new assets
        sub_msg = orjson.dumps({
            "assets_ids": new_assets,
            "operation": "subscribe",
            "type": "market",
            "custom_feature_enabled": True,
        })
        await self._ws.send(sub_msg)
        logger.info("updown_subscribed", new_assets=len(new_assets),
                   total_assets=len(self._subscribed_assets))

    def _dispatch(self, msg: dict, recv_ns: int) -> None:
        event_type = msg.get("event_type")
        if event_type == "book":
            self._handle_book(msg, recv_ns)
        elif event_type == "price_change":
            self._handle_price_change(msg, recv_ns)
        elif event_type == "last_trade_price":
            self._handle_trade(msg, recv_ns)
        elif event_type == "best_bid_ask":
            self._handle_bba(msg, recv_ns)

    def _handle_book(self, msg: dict, recv_ns: int) -> None:
        asset_id = msg.get("asset_id", "")
        market = msg.get("market", "")
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        exchange_ts = int(msg.get("timestamp", "0"))

        features = self.engine.handle_book(asset_id, bids, asks, exchange_ts)
        if features:
            slug = self._subscribed_assets.get(asset_id, "")
            row = self._book_row(features, recv_ns, exchange_ts, market, slug)
            self.book_writer.append(row)

    def _handle_price_change(self, msg: dict, recv_ns: int) -> None:
        market = msg.get("market", "")
        exchange_ts = int(msg.get("timestamp", "0"))
        for change in msg.get("price_changes", []):
            asset_id = change.get("asset_id", "")
            side = change.get("side", "BUY").lower()
            price = change.get("price", "0")
            size = change.get("size", "0")
            features = self.engine.handle_price_change(asset_id, side, price, size, exchange_ts)
            if features:
                slug = self._subscribed_assets.get(asset_id, "")
                row = self._book_row(features, recv_ns, exchange_ts, market, slug)
                self.book_writer.append(row)

    def _handle_trade(self, msg: dict, recv_ns: int) -> None:
        asset_id = msg.get("asset_id", "")
        slug = self._subscribed_assets.get(asset_id, "")
        self.trade_writer.append({
            "source": "polymarket",
            "asset_id": asset_id,
            "market": msg.get("market", ""),
            "recv_ns": recv_ns,
            "exchange_ts": int(msg.get("timestamp", "0")),
            "price": float(msg.get("price", "0")),
            "size": float(msg.get("size", "0")),
            "side": msg.get("side", ""),
            "fee_rate_bps": float(msg.get("fee_rate_bps", "0")),
        })

    def _handle_bba(self, msg: dict, recv_ns: int) -> None:
        self.bba_writer.append({
            "source": "polymarket",
            "asset_id": msg.get("asset_id", ""),
            "recv_ns": recv_ns,
            "exchange_ts": int(msg.get("timestamp", "0")),
            "best_bid": float(msg.get("best_bid", "0")),
            "best_ask": float(msg.get("best_ask", "0")),
            "spread": float(msg.get("spread", "0")),
        })

    async def _heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(self.config.ws_ping_interval)
            try:
                await ws.send("PING")
            except Exception:
                break

    @staticmethod
    def _book_row(features: dict, recv_ns: int, exchange_ts: int,
                  market: str = "", slug: str = "") -> dict:
        return {
            "source": f"polymarket:{slug}" if slug else "polymarket",
            "asset_id": str(features.get("asset_id", "")),
            "market": market,
            "recv_ns": recv_ns,
            "exchange_ts": exchange_ts,
            "best_bid": float(features.get("best_bid") or 0),
            "best_ask": float(features.get("best_ask") or 0),
            "spread": float(features.get("spread") or 0),
            "midpoint": float(features.get("midpoint") or 0),
            "microprice": float(features.get("microprice") or 0),
            "imbalance": float(features.get("imbalance") or 0),
            "total_bid_levels": int(features.get("total_bid_levels") or 0),
            "total_ask_levels": int(features.get("total_ask_levels") or 0),
        }
