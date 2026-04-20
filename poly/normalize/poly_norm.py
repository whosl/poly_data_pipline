"""Polymarket raw data normalizer — JSONL to Parquet."""

from __future__ import annotations

import structlog
import orjson
import polars as pl
from pathlib import Path
from poly.metadata.polymarket import enrich_with_metadata, load_metadata
from poly.storage.recover import iter_recovered_gzip_jsonl_lines

logger = structlog.get_logger()


class PolyNormalizer:
    """Normalize raw Polymarket JSONL into analysis-ready Parquet."""

    def run(self, data_dir: Path, date: str) -> None:
        raw_path = data_dir / "raw_feed" / date / "polymarket_market.jsonl.gz"
        if not raw_path.exists():
            logger.warning("poly_norm_no_raw", path=str(raw_path))
            return

        out_dir = data_dir / "normalized" / date
        out_dir.mkdir(parents=True, exist_ok=True)

        l2_rows: list[dict] = []
        trade_rows: list[dict] = []
        bba_rows: list[dict] = []

        import poly_core
        books: dict[str, poly_core.OrderBook] = {}

        count = 0
        for line in iter_recovered_gzip_jsonl_lines(raw_path):
            try:
                outer = orjson.loads(line)
            except Exception:
                continue

            recv_ns = outer.get("recv_ns", 0)
            raw_payload = outer.get("raw", {})
            if isinstance(raw_payload, str):
                try:
                    raw_payload = orjson.loads(raw_payload)
                except Exception:
                    continue

            # Raw may be a single dict or list of events
            if isinstance(raw_payload, list):
                msgs = raw_payload
            else:
                msgs = [raw_payload]

            for msg in msgs:
                if not isinstance(msg, dict):
                    continue
                self._process_msg(msg, recv_ns, books, l2_rows, trade_rows, bba_rows)
                count += 1

        if l2_rows:
            self._enrich(pl.DataFrame(l2_rows), data_dir, date).write_parquet(str(out_dir / "poly_l2_book.parquet"))
            logger.info("poly_norm_l2", rows=len(l2_rows))

        if trade_rows:
            self._enrich(pl.DataFrame(trade_rows), data_dir, date).write_parquet(str(out_dir / "poly_trades.parquet"))
            logger.info("poly_norm_trades", rows=len(trade_rows))

        if bba_rows:
            self._enrich(pl.DataFrame(bba_rows), data_dir, date).write_parquet(str(out_dir / "poly_best_bid_ask.parquet"))
            logger.info("poly_norm_bba", rows=len(bba_rows))

        logger.info("poly_norm_done", date=date, total_messages=count)

    def _process_msg(self, msg: dict, recv_ns: int, books: dict,
                     l2_rows: list, trade_rows: list, bba_rows: list) -> None:
        import poly_core
        event_type = msg.get("event_type")
        if not event_type:
            return

        if event_type == "book":
            asset_id = msg.get("asset_id", "")
            if asset_id not in books:
                books[asset_id] = poly_core.OrderBook(asset_id)
            bids = [(b["price"], b["size"]) for b in msg.get("bids", [])]
            asks = [(a["price"], a["size"]) for a in msg.get("asks", [])]
            exchange_ts = int(msg.get("timestamp", "0"))
            books[asset_id].apply_snapshot(bids, asks, exchange_ts)
            features = books[asset_id].depth_summary()
            if features:
                l2_rows.append(self._book_features(features, recv_ns, exchange_ts, msg.get("market", "")))

        elif event_type == "price_change":
            exchange_ts = int(msg.get("timestamp", "0"))
            for change in msg.get("price_changes", []):
                asset_id = change.get("asset_id", "")
                if asset_id not in books:
                    books[asset_id] = poly_core.OrderBook(asset_id)
                books[asset_id].apply_delta(
                    change.get("side", "BUY").lower(),
                    change.get("price", "0"),
                    change.get("size", "0"),
                    exchange_ts,
                )
                features = books[asset_id].depth_summary()
                if features:
                    l2_rows.append(self._book_features(features, recv_ns, exchange_ts, msg.get("market", "")))

        elif event_type == "last_trade_price":
            trade_rows.append({
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

        elif event_type == "best_bid_ask":
            bba_rows.append({
                "source": "polymarket",
                "asset_id": msg.get("asset_id", ""),
                "market": msg.get("market", ""),
                "recv_ns": recv_ns,
                "exchange_ts": int(msg.get("timestamp", "0")),
                "best_bid": float(msg.get("best_bid", "0")),
                "best_ask": float(msg.get("best_ask", "0")),
                "spread": float(msg.get("spread", "0")),
            })

    @staticmethod
    def _book_features(features: dict, recv_ns: int, exchange_ts: int, market: str = "") -> dict:
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

    @staticmethod
    def _enrich(frame: pl.DataFrame, data_dir: Path, date: str) -> pl.DataFrame:
        metadata = load_metadata(data_dir, date)
        if metadata.is_empty():
            return frame
        enriched = enrich_with_metadata(frame, metadata)
        if "source" in enriched.columns and "slug" in enriched.columns:
            enriched = enriched.with_columns(
                pl.when((pl.col("source") == "polymarket") & pl.col("slug").is_not_null() & (pl.col("slug") != ""))
                .then(pl.concat_str([pl.lit("polymarket:"), pl.col("slug")]))
                .otherwise(pl.col("source"))
                .alias("source")
            )
        return enriched
