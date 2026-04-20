"""Polymarket market WebSocket collector — the most critical data source."""

from __future__ import annotations

import asyncio
import time
import structlog
import orjson

import websockets

from poly.config import Config
from poly.storage.raw import RawWriter
from poly.storage.normalized import ParquetWriter, TRADE_SCHEMA, BEST_BID_ASK_SCHEMA
from poly.engine.orderbook import OrderBookEngine

logger = structlog.get_logger()


class DataWatchdog:
    """Detects silent WebSocket freezes (issue #292)."""

    def __init__(self, timeout_seconds: float = 120.0) -> None:
        self.timeout = timeout_seconds
        self._last_message_time: float = time.monotonic()
        self._triggered = False

    def feed(self) -> None:
        self._last_message_time = time.monotonic()
        self._triggered = False

    def is_expired(self) -> bool:
        if time.monotonic() - self._last_message_time > self.timeout:
            if not self._triggered:
                self._triggered = True
                return True
        return False


class PolymarketMarketWS:
    """WebSocket client for Polymarket market data channel."""

    def __init__(self, config: Config, raw_writer: RawWriter,
                 engine: OrderBookEngine, norm_writer: ParquetWriter,
                 trade_writer: ParquetWriter, bba_writer: ParquetWriter) -> None:
        self.config = config
        self.raw_writer = raw_writer
        self.engine = engine
        self.norm_writer = norm_writer
        self.trade_writer = trade_writer
        self.bba_writer = bba_writer
        self._ws = None
        self._msg_count = 0

    async def run(self, asset_ids: list[str]) -> None:
        """Main loop: connect, subscribe, receive, reconnect."""
        backoff = 1.0
        while True:
            try:
                await self._connect_and_subscribe(asset_ids)
                backoff = 1.0  # reset on clean disconnect
            except Exception as e:
                logger.warning("market_ws_error", error=str(e), backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _connect_and_subscribe(self, asset_ids: list[str]) -> None:
        url = self.config.poly_market_ws_url
        logger.info("market_ws_connecting", url=url, num_assets=len(asset_ids))

        async with websockets.connect(url, ping_interval=None) as ws:
            self._ws = ws
            # Subscribe (limit to 100 per connection for stability)
            sub_ids = asset_ids[:100]
            sub_msg = orjson.dumps({
                "assets_ids": sub_ids,
                "type": "market",
                "custom_feature_enabled": True,
            })
            await ws.send(sub_msg)
            logger.info("market_ws_subscribed", num_assets=len(sub_ids))

            # Start background tasks
            heartbeat_task = asyncio.create_task(self._heartbeat(ws))
            watchdog = DataWatchdog(self.config.watchdog_timeout)
            watchdog_task = asyncio.create_task(self._watchdog_check(ws, watchdog))

            try:
                async for message in ws:
                    # Import poly_core lazily (may not be available during dev)
                    import poly_core
                    recv_ns = poly_core.now_ns()
                    watchdog.feed()
                    self._msg_count += 1

                    # Persist raw
                    await self.raw_writer.write(message, recv_ns)

                    # Parse and dispatch — skip non-JSON (e.g. PONG)
                    if isinstance(message, bytes):
                        text = message.decode("utf-8", errors="replace")
                    else:
                        text = message
                    text = text.strip()
                    if not text or text in ("PONG", "PING"):
                        continue
                    try:
                        parsed = orjson.loads(text)
                    except Exception:
                        continue

                    # WS may return a list of events or a single dict
                    messages = parsed if isinstance(parsed, list) else [parsed]

                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        event_type = msg.get("event_type")
                        if not event_type:
                            continue
                        if event_type == "book":
                            self._handle_book(msg, recv_ns)
                        elif event_type == "price_change":
                            self._handle_price_change(msg, recv_ns)
                        elif event_type == "last_trade_price":
                            self._handle_trade(msg, recv_ns)
                        elif event_type == "best_bid_ask":
                            self._handle_best_bid_ask(msg, recv_ns)
                        elif event_type == "tick_size_change":
                            logger.info("tick_size_change", asset_id=msg.get("asset_id"),
                                       old=msg.get("old_tick_size"), new=msg.get("new_tick_size"))

            finally:
                heartbeat_task.cancel()
                watchdog_task.cancel()
                self._ws = None

    async def _heartbeat(self, ws) -> None:
        while True:
            await asyncio.sleep(self.config.ws_ping_interval)
            try:
                await ws.send("PING")
            except Exception:
                break

    async def _watchdog_check(self, ws, watchdog: DataWatchdog) -> None:
        while True:
            await asyncio.sleep(15.0)
            if watchdog.is_expired():
                logger.warning("watchdog_expired — forcing reconnect")
                await ws.close()
                return

    def _handle_book(self, msg: dict, recv_ns: int) -> None:
        asset_id = msg.get("asset_id", "")
        market = msg.get("market", "")
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        exchange_ts = int(msg.get("timestamp", "0"))

        features = self.engine.handle_book(asset_id, bids, asks, exchange_ts)
        if features:
            row = self._book_row(features, recv_ns, exchange_ts, market)
            self.norm_writer.append(row)

    def _handle_price_change(self, msg: dict, recv_ns: int) -> None:
        market = msg.get("market", "")
        exchange_ts = int(msg.get("timestamp", "0"))

        for change in msg.get("price_changes", []):
            asset_id = change.get("asset_id", "")
            side = change.get("side", "BUY").lower()
            price = change.get("price", "0")
            size = change.get("size", "0")

            features = self.engine.handle_price_change(
                asset_id, side, price, size, exchange_ts
            )
            if features:
                row = self._book_row(features, recv_ns, exchange_ts, market)
                self.norm_writer.append(row)

    def _handle_trade(self, msg: dict, recv_ns: int) -> None:
        self.trade_writer.append({
            "source": "polymarket",
            "asset_id": msg.get("asset_id", ""),
            "market": msg.get("market", ""),
            "recv_ns": recv_ns,
            "exchange_ts": int(msg.get("timestamp", "0")),
            "price": float(msg.get("price", "0")),
            "size": float(msg.get("size", "0")),
            "side": msg.get("side", ""),
            "fee_rate_bps": float(msg.get("fee_rate_bps", "0")),
        })

    def _handle_best_bid_ask(self, msg: dict, recv_ns: int) -> None:
        self.bba_writer.append({
            "source": "polymarket",
            "asset_id": msg.get("asset_id", ""),
            "recv_ns": recv_ns,
            "exchange_ts": int(msg.get("timestamp", "0")),
            "best_bid": float(msg.get("best_bid", "0")),
            "best_ask": float(msg.get("best_ask", "0")),
            "spread": float(msg.get("spread", "0")),
        })

    @staticmethod
    def _book_row(features: dict, recv_ns: int, exchange_ts: int,
                  market: str = "") -> dict:
        """Build a Parquet-compatible row from orderbook features."""
        from poly.engine.orderbook import extract_depth_features
        row: dict = {
            "source": "polymarket",
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
        row.update(extract_depth_features(features))
        return row
