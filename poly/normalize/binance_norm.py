"""Binance raw data normalizer — JSONL to Parquet."""

from __future__ import annotations

import structlog
import orjson
import polars as pl
from pathlib import Path
from poly.collector.binance_depth import depth_features
from poly.storage.recover import iter_recovered_gzip_jsonl_lines

logger = structlog.get_logger()


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
                    row.update(depth_features(bids, asks))
                    book_rows.append(row)

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
