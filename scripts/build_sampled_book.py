#!/usr/bin/env python3
"""Build filtered/downsampled Polymarket book tables for training.

This keeps raw/normalized data intact and writes a lighter
`poly_sampled_book.parquet` per date. Training prefers this table when present.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl
import pyarrow.parquet as pq
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.training.io import discover_dates

structlog.configure(processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()])
logger = structlog.get_logger()

NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--sample-interval-ms", type=int, default=250)
    parser.add_argument("--horizon-seconds", type=int, default=10)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--min-mid", type=float, default=0.02)
    parser.add_argument("--max-mid", type=float, default=0.98)
    parser.add_argument("--max-spread", type=float, default=0.25)
    parser.add_argument("--require-metadata", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dates = discover_dates(args.data_dir, args.dates)
    if not dates:
        raise SystemExit(f"No dates found under {args.data_dir}/normalized")
    for date in dates:
        build_date(args.data_dir, date, args)


def build_date(data_dir: Path, date: str, args: argparse.Namespace) -> None:
    norm_dir = data_dir / "normalized" / date
    book_path = norm_dir / "poly_l2_book.parquet"
    metadata_path = norm_dir / "poly_market_metadata.parquet"
    output_path = norm_dir / "poly_sampled_book.parquet"
    if output_path.exists() and not args.overwrite:
        logger.info("sampled_book_exists", date=date, path=str(output_path))
        return
    if not book_path.exists():
        logger.warning("missing_poly_l2_book", date=date, path=str(book_path))
        return
    if not metadata_path.exists():
        logger.warning("missing_poly_market_metadata", date=date, path=str(metadata_path))
        if args.require_metadata:
            return

    interval_ns = args.sample_interval_ms * NS_PER_MS
    horizon_ns = args.horizon_seconds * NS_PER_SECOND
    warmup_ns = int(args.warmup_seconds * NS_PER_SECOND)

    lf = pl.scan_parquet(str(book_path))
    if metadata_path.exists():
        metadata = (
            pl.scan_parquet(str(metadata_path))
            .unique(subset=["asset_id"], keep="last")
            .select(
                [
                    "asset_id",
                    "market_id",
                    "condition_id",
                    "slug",
                    "outcome",
                    "symbol",
                    "period",
                    "start_ns",
                    "expiry_ns",
                    "tick_size",
                    "min_order_size",
                    "maker_base_fee",
                    "taker_base_fee",
                    *[
                        c
                        for c in ["volume_24h", "liquidity"]
                        if c in pl.read_parquet_schema(str(metadata_path))
                    ],
                ]
            )
        )
        lf = lf.join(metadata, on="asset_id", how="left", suffix="_meta")

    lf = ensure_book_columns(lf)
    lf = lf.with_columns(
        [
            ((pl.col("recv_ns") // interval_ns) * interval_ns).alias("_sample_bucket_ns"),
            pl.when(pl.col("midpoint").is_null())
            .then((pl.col("best_bid") + pl.col("best_ask")) / 2)
            .otherwise(pl.col("midpoint"))
            .alias("midpoint"),
            pl.when(pl.col("spread").is_null())
            .then(pl.col("best_ask") - pl.col("best_bid"))
            .otherwise(pl.col("spread"))
            .alias("spread"),
        ]
    )
    lf = lf.filter(
        pl.col("recv_ns").is_not_null()
        & pl.col("asset_id").is_not_null()
        & pl.col("best_bid").is_not_null()
        & pl.col("best_ask").is_not_null()
        & (pl.col("best_bid") > 0)
        & (pl.col("best_ask") > 0)
        & (pl.col("best_bid") < pl.col("best_ask"))
        & pl.col("midpoint").is_between(args.min_mid, args.max_mid, closed="both")
        & (pl.col("spread") >= 0)
        & (pl.col("spread") <= args.max_spread)
    )
    if "start_ns" in lf.collect_schema() and "expiry_ns" in lf.collect_schema():
        lf = lf.filter(
            pl.col("start_ns").is_not_null()
            & pl.col("expiry_ns").is_not_null()
            & (pl.col("recv_ns") >= pl.col("start_ns") + warmup_ns)
            & (pl.col("recv_ns") + horizon_ns <= pl.col("expiry_ns"))
        )

    lf = (
        lf.sort(["asset_id", "_sample_bucket_ns", "recv_ns"])
        .unique(subset=["asset_id", "_sample_bucket_ns"], keep="last", maintain_order=False)
        .drop("_sample_bucket_ns")
        .sort(["asset_id", "recv_ns"])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(str(output_path))
    rows = pq.ParquetFile(output_path).metadata.num_rows
    logger.info(
        "sampled_book_done",
        date=date,
        rows=rows,
        path=str(output_path),
        sample_interval_ms=args.sample_interval_ms,
        horizon_seconds=args.horizon_seconds,
        warmup_seconds=args.warmup_seconds,
    )


def ensure_book_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    exprs = []
    for col, dtype, default in [
        ("market", pl.String, ""),
        ("source", pl.String, "polymarket"),
        ("market_id", pl.String, ""),
        ("condition_id", pl.String, ""),
        ("slug", pl.String, ""),
        ("outcome", pl.String, ""),
        ("symbol", pl.String, ""),
        ("period", pl.String, ""),
        ("start_ns", pl.Int64, None),
        ("expiry_ns", pl.Int64, None),
        ("tick_size", pl.Float64, None),
        ("min_order_size", pl.Float64, None),
        ("maker_base_fee", pl.Float64, None),
        ("taker_base_fee", pl.Float64, None),
        ("midpoint", pl.Float64, None),
        ("spread", pl.Float64, None),
    ]:
        if col not in schema:
            exprs.append(pl.lit(default, dtype=dtype).alias(col))
    if exprs:
        lf = lf.with_columns(exprs)
    return lf


if __name__ == "__main__":
    main()
