"""Binance raw data normalizer — JSONL to Parquet."""

from __future__ import annotations

import structlog
import orjson
import polars as pl
import re
from pathlib import Path
from poly.collector.binance_depth import depth_features
from poly.storage.recover import iter_recovered_gzip_jsonl_lines

logger = structlog.get_logger()
DEPTH_STREAM_RE = re.compile(r"@depth(?P<levels>\d+)")


class BinanceNormalizer:
    """Normalize raw Binance JSONL into analysis-ready Parquet."""

    def run(self, data_dir: Path, date: str) -> None:
        raw_path = data_dir / "raw_feed" / date / "binance_spot.jsonl.gz"
        if not raw_path.exists():
            logger.warning("binance_norm_no_raw", path=str(raw_path))
            return

        out_dir = data_dir / "normalized" / date
        out_dir.mkdir(parents=True, exist_ok=True)

        bba_rows: list[dict] = []
        trade_rows: list[dict] = []
        book_rows: list[dict] = []
        count = 0

        for line in iter_recovered_gzip_jsonl_lines(raw_path):
            try:
                outer = orjson.loads(line)
            except Exception:
                continue

            recv_ns = outer.get("recv_ns", 0)
            raw = outer.get("raw", {})
            if isinstance(raw, str):
                try:
                    raw = orjson.loads(raw)
                except Exception:
                    continue

            stream = raw.get("stream", "")
            data = raw.get("data")
            if data is None:
                continue

            if stream.endswith("@bookTicker"):
                bba_rows.append({
                    "source": "binance",
                    "asset_id": data.get("s", "").lower(),
                    "recv_ns": recv_ns,
                    "exchange_ts": data.get("u", 0),
                    "best_bid": float(data.get("b", "0")),
                    "best_ask": float(data.get("a", "0")),
                    "spread": float(data.get("a", "0")) - float(data.get("b", "0")),
                })

            elif stream.endswith("@aggTrade"):
                side = "SELL" if data.get("m", False) else "BUY"
                trade_rows.append({
                    "source": "binance",
                    "asset_id": data.get("s", "").lower(),
                    "market": "",
                    "recv_ns": recv_ns,
                    "exchange_ts": data.get("T", 0) * 1_000_000,
                    "price": float(data.get("p", "0")),
                    "size": float(data.get("q", "0")),
                    "side": side,
                    "fee_rate_bps": 0.0,
                })

            elif "@depth" in stream:
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                if bids and asks:
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
                    book_rows.append(row)
                elif data.get("compact_depth_features"):
                    book_rows.append(compact_depth_row(data, recv_ns, stream))

            count += 1

        if bba_rows:
            pl.DataFrame(bba_rows).write_parquet(str(out_dir / "binance_best_bid_ask.parquet"))
            logger.info("binance_norm_bba", rows=len(bba_rows))

        if trade_rows:
            pl.DataFrame(trade_rows).write_parquet(str(out_dir / "binance_trades.parquet"))
            logger.info("binance_norm_trades", rows=len(trade_rows))

        if book_rows:
            pl.DataFrame(book_rows).write_parquet(str(out_dir / "binance_l2_book.parquet"))
            logger.info("binance_norm_book", rows=len(book_rows))

        logger.info("binance_norm_done", date=date, total_messages=count)


def depth_levels_from_stream(stream: str) -> int:
    match = DEPTH_STREAM_RE.search(stream)
    if match:
        return int(match.group("levels"))
    return 20


def compact_depth_row(data: dict, recv_ns: int, stream: str) -> dict:
    symbol = data.get("asset_id") or stream.split("@")[0]
    row = {
        "source": "binance",
        "asset_id": str(symbol).lower(),
        "market": data.get("market", ""),
        "recv_ns": recv_ns,
        "exchange_ts": data.get("exchange_ts", 0),
        "best_bid": data.get("best_bid"),
        "best_ask": data.get("best_ask"),
        "spread": data.get("spread"),
        "midpoint": data.get("midpoint"),
        "microprice": data.get("microprice"),
        "imbalance": data.get("imbalance"),
        "total_bid_levels": data.get("total_bid_levels"),
        "total_ask_levels": data.get("total_ask_levels"),
    }
    for key, value in data.items():
        if key.startswith(("depth_", "cum_", "bid_depth_", "ask_depth_", "near_touch_")):
            row[key] = value
    return row
