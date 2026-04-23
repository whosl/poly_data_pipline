"""Polymarket authenticated user WebSocket for order/trade lifecycle."""

from __future__ import annotations

import asyncio
import structlog
import orjson

import websockets

from poly.config import Config
from poly.storage.raw import RawWriter
from poly.storage.normalized import ParquetWriter

logger = structlog.get_logger()


class PolymarketUserWS:
    """Authenticated WebSocket for own order and trade events."""

    def __init__(self, config: Config, raw_writer: RawWriter,
                 order_writer: ParquetWriter, trade_writer: ParquetWriter) -> None:
        self.config = config
        self.raw_writer = raw_writer
        self.order_writer = order_writer
        self.trade_writer = trade_writer
        self._orders: dict[str, dict] = {}
        self._msg_count = 0

    async def run(self, condition_ids: list[str] | None = None) -> None:
        """Connect with auth and process order/trade events."""
        if not self.config.api_key:
            logger.warning("user_ws_skipped — no API credentials")
            return

        backoff = 1.0
        while True:
            try:
                await self._connect_and_listen(condition_ids)
                backoff = 1.0
            except Exception as e:
                logger.warning("user_ws_error", error=str(e), backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _connect_and_listen(self, condition_ids: list[str] | None) -> None:
        url = self.config.poly_user_ws_url
        logger.info("user_ws_connecting", url=url)

        async with websockets.connect(url, ping_interval=None) as ws:
            auth_msg = orjson.dumps({
                "auth": {
                    "apiKey": self.config.api_key,
                    "secret": self.config.api_secret,
                    "passphrase": self.config.api_passphrase,
                },
                "type": "user",
            })
            await ws.send(auth_msg)
            logger.info("user_ws_auth_sent")

            if condition_ids:
                sub = orjson.dumps({
                    "markets": condition_ids,
                    "operation": "subscribe",
                })
                await ws.send(sub)

            try:
                async for message in ws:
                    import poly_core
                    recv_ns = poly_core.now_ns()
                    self._msg_count += 1
                    await self.raw_writer.write(message, recv_ns)

                    if isinstance(message, bytes):
                        text = message.decode("utf-8", errors="replace")
                    else:
                        text = message
                    text = text.strip()
                    if not text or text == "PONG":
                        continue

                    try:
                        msg = orjson.loads(text)
                    except Exception:
                        continue

                    event_type = msg.get("event_type")
                    if event_type == "order":
                        self._handle_order(msg, recv_ns)
                    elif event_type == "trade":
                        self._handle_trade(msg, recv_ns)
            finally:
                pass

    def _handle_order(self, msg: dict, recv_ns: int) -> None:
        order_id = msg.get("id", "")
        exchange_ts = int(msg.get("timestamp", "0"))

        self._orders[order_id] = {
            "order_id": order_id,
            "market": msg.get("market", ""),
            "asset_id": msg.get("asset_id", ""),
            "side": msg.get("side", ""),
            "price": msg.get("price", "0"),
            "original_size": msg.get("original_size", "0"),
            "size_matched": msg.get("size_matched", "0"),
            "order_type": msg.get("type", ""),
            "status": msg.get("type", ""),
        }

        self.order_writer.append({
            "order_id": order_id,
            "market": msg.get("market", ""),
            "asset_id": msg.get("asset_id", ""),
            "side": msg.get("side", ""),
            "price": float(msg.get("price", "0")),
            "original_size": float(msg.get("original_size", "0")),
            "size_matched": float(msg.get("size_matched", "0")),
            "order_type": msg.get("type", ""),
            "recv_ns": recv_ns,
            "exchange_ts": exchange_ts,
        })

    def _handle_trade(self, msg: dict, recv_ns: int) -> None:
        self.trade_writer.append({
            "trade_id": msg.get("id", ""),
            "market": msg.get("market", ""),
            "asset_id": msg.get("asset_id", ""),
            "side": msg.get("side", ""),
            "price": float(msg.get("price", "0")),
            "size": float(msg.get("size", "0")),
            "status": msg.get("status", ""),
            "recv_ns": recv_ns,
            "exchange_ts": int(msg.get("timestamp", "0")),
        })
