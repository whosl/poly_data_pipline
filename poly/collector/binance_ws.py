"""Binance WebSocket collector for BTC reference market data."""

from __future__ import annotations

import asyncio
import time
import structlog
import orjson
import aiohttp
import re

import websockets

from poly.config import Config
from poly.collector.binance_depth import depth_features
from poly.storage.raw import RawWriter
from poly.storage.normalized import ParquetWriter

logger = structlog.get_logger()

DEPTH_STREAM_RE = re.compile(r"@depth(?P<levels>\d+)")


class BinanceWS:
    """WebSocket client for Binance combined streams."""

    def __init__(self, config: Config, raw_writer: RawWriter,
                 bba_writer: ParquetWriter | None, trade_writer: ParquetWriter | None,
                 book_writer: ParquetWriter | None,
                 raw_only: bool = False) -> None:
        self.config = config
        self.raw_writer = raw_writer
        self.bba_writer = bba_writer
        self.trade_writer = trade_writer
        self.book_writer = book_writer
        self.raw_only = raw_only
        self._clock_offset_ns: int = 0
        self._msg_count = 0

    async def run(self, symbols: list[str]) -> None:
        """Connect and process Binance streams."""
        # Initial clock sync
        await self._sync_clock()

        backoff = 1.0
        while True:
            try:
                await self._connect_and_listen(symbols)
                backoff = 1.0
            except Exception as e:
                logger.warning("binance_ws_error", error=str(e), backoff=backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _connect_and_listen(self, symbols: list[str]) -> None:
        streams = []
        for sym in symbols:
            streams.extend([
                f"{sym}@bookTicker",
                f"{sym}@aggTrade",
                f"{sym}@depth20@100ms",
            ])

        url = f"{self.config.binance_ws_url}/stream?streams={'/'.join(streams)}"
        logger.info("binance_ws_connecting", url=url)

        # Start clock sync background task
        clock_task = asyncio.create_task(self._clock_sync_loop())

        try:
            async with websockets.connect(url) as ws:
                logger.info("binance_ws_connected")
                async for message in ws:
                    import poly_core
                    recv_ns = poly_core.now_ns()
                    self._msg_count += 1

                    try:
                        envelope = orjson.loads(message)
                    except Exception:
                        continue

                    stream = envelope.get("stream", "")
                    data = envelope.get("data")
                    if data is None:
                        continue

                    if stream.endswith("@bookTicker"):
                        await self.raw_writer.write_obj(envelope, recv_ns)
                        if not self.raw_only:
                            self._handle_book_ticker(data, recv_ns)
                    elif stream.endswith("@aggTrade"):
                        await self.raw_writer.write_obj(envelope, recv_ns)
                        if not self.raw_only:
                            self._handle_agg_trade(data, recv_ns)
                    elif "@depth" in stream:
                        if self.raw_only:
                            await self.raw_writer.write_obj(envelope, recv_ns)
                        else:
                            row = self._handle_depth(data, recv_ns, stream)
                            if row is not None:
                                await self.raw_writer.write_obj(compact_depth_envelope(stream, row), recv_ns)
        finally:
            clock_task.cancel()

    def _handle_book_ticker(self, data: dict, recv_ns: int) -> None:
        self.bba_writer.append({
            "source": "binance",
            "asset_id": data.get("s", "").lower(),
            "recv_ns": recv_ns,
            "exchange_ts": data.get("u", 0),
            "best_bid": float(data.get("b", "0")),
            "best_ask": float(data.get("a", "0")),
            "spread": float(data.get("a", "0")) - float(data.get("b", "0")),
        })

    def _handle_agg_trade(self, data: dict, recv_ns: int) -> None:
        # m=True means buyer is maker => sell taker initiated
        side = "SELL" if data.get("m", False) else "BUY"
        self.trade_writer.append({
            "source": "binance",
            "asset_id": data.get("s", "").lower(),
            "market": "",
            "recv_ns": recv_ns,
            "exchange_ts": data.get("T", 0) * 1_000_000,  # ms -> ns
            "price": float(data.get("p", "0")),
            "size": float(data.get("q", "0")),
            "side": side,
            "fee_rate_bps": 0.0,
        })

    def _handle_depth(self, data: dict, recv_ns: int, stream: str) -> dict | None:
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not bids or not asks:
            return None

        # Extract symbol from stream name
        symbol = stream.split("@")[0]
        max_depth = depth_levels_from_stream(stream)

        row = {
            "source": "binance",
            "asset_id": symbol,
            "market": "",
            "recv_ns": recv_ns,
            "exchange_ts": data.get("lastUpdateId", 0),
            "best_bid": float(bids[0][0]),
            "best_ask": float(asks[0][0]),
            "spread": float(asks[0][0]) - float(bids[0][0]),
            "midpoint": (float(bids[0][0]) + float(asks[0][0])) / 2,
            "microprice": None,
            "imbalance": None,
            "total_bid_levels": len(bids),
            "total_ask_levels": len(asks),
        }
        row.update(depth_features(bids, asks, max_depth=max_depth))
        self.book_writer.append(row)
        return row

    async def _sync_clock(self) -> None:
        """Measure clock offset against Binance server time."""
        try:
            async with aiohttp.ClientSession() as session:
                local_before = time.time()
                async with session.get("https://api.binance.com/api/v3/time") as resp:
                    local_after = time.time()
                    if resp.status == 200:
                        data = await resp.json()
                        server_ms = data.get("serverTime", 0)
                        local_ms = (local_before + local_after) / 2 * 1000
                        self._clock_offset_ns = int((server_ms - local_ms) * 1_000_000)
                        logger.info("clock_sync", offset_ms=self._clock_offset_ns / 1_000_000)
        except Exception as e:
            logger.warning("clock_sync_failed", error=str(e))

    async def _clock_sync_loop(self) -> None:
        """Periodically re-sync clock."""
        while True:
            await asyncio.sleep(600)
            await self._sync_clock()


def depth_levels_from_stream(stream: str) -> int:
    match = DEPTH_STREAM_RE.search(stream)
    if match:
        return int(match.group("levels"))
    return 20


def compact_depth_envelope(stream: str, row: dict) -> dict:
    data = {
        key: value
        for key, value in row.items()
        if key not in {"source", "market", "recv_ns"}
    }
    data["compact_depth_features"] = True
    return {"stream": stream, "data": data}
