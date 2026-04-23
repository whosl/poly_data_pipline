#!/usr/bin/env python3
"""Build filtered/downsampled Polymarket book tables for training.

This keeps raw/normalized data intact and writes a lighter
`poly_sampled_book.parquet` per date. Training prefers this table when present.

Two modes:
  --mode time-bucket  (default): take last row per (asset, 250ms bucket)
  --mode event-driven: keep rows where any event column changed vs previous row
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

# --- Tiered event columns ---
# Tier 1 (critical): always keep on change — these are the core price signal.
TIER1_COLUMNS = [
    "best_bid",
    "best_ask",
    "total_bid_levels",
    "total_ask_levels",
]

# Tier 2 (secondary): only keep if abs(delta) > magnitude threshold.
TIER2_COLUMNS = [
    "depth_top1_imbalance",
    "depth_top3_imbalance",
    "depth_top5_imbalance",
    "depth_top10_imbalance",
    "depth_top20_imbalance",
    "cum_bid_depth_top10",
    "cum_ask_depth_top10",
    "cum_bid_depth_top20",
    "cum_ask_depth_top20",
]

# Default magnitude thresholds per column (abs delta must exceed this).
# Imbalance features are ~[0,1], cum_depth features are counts.
DEFAULT_MAGNITUDE_THRESHOLDS = {
    "depth_top1_imbalance": 0.001,
    "depth_top3_imbalance": 0.001,
    "depth_top5_imbalance": 0.001,
    "depth_top10_imbalance": 0.001,
    "depth_top20_imbalance": 0.001,
    "cum_bid_depth_top10": 10.0,
    "cum_ask_depth_top10": 10.0,
    "cum_bid_depth_top20": 10.0,
    "cum_ask_depth_top20": 10.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--mode", choices=["time-bucket", "event-driven"], default="time-bucket")
    parser.add_argument("--sample-interval-ms", type=int, default=250,
                        help="Bucket width for time-bucket mode (ignored in event-driven mode)")
    parser.add_argument("--event-columns", nargs="*", default=None,
                        help="Override tier1+tier2 columns for event-driven mode.")
    parser.add_argument("--tier1-columns", nargs="*", default=None,
                        help="Critical columns: change always triggers an event. "
                             "Defaults to best_bid/ask + total_levels.")
    parser.add_argument("--tier2-columns", nargs="*", default=None,
                        help="Secondary columns: change triggers only if abs(delta) > threshold. "
                             "Defaults to depth/imbalance columns.")
    parser.add_argument("--magnitude", type=float, default=None,
                        help="Uniform magnitude threshold for all tier2 columns. "
                             "Overrides per-column defaults.")
    parser.add_argument("--min-gap-ms", type=float, default=0,
                        help="Minimum gap between consecutive events in event-driven mode. "
                             "Events closer than this are dropped (debounce). 0 = no debounce.")
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

    if args.mode == "time-bucket":
        lf = _downsample_time_bucket(lf, args)
    else:
        lf = _downsample_event_driven(lf, args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lf.sink_parquet(str(output_path))
    rows = pq.ParquetFile(output_path).metadata.num_rows
    logger.info(
        "sampled_book_done",
        date=date,
        rows=rows,
        path=str(output_path),
        mode=args.mode,
        horizon_seconds=args.horizon_seconds,
        warmup_seconds=args.warmup_seconds,
    )


def _downsample_time_bucket(lf: pl.LazyFrame, args: argparse.Namespace) -> pl.LazyFrame:
    """Original time-bucket mode: take last row per (asset, N ms bucket)."""
    interval_ns = args.sample_interval_ms * NS_PER_MS
    lf = lf.with_columns(
        ((pl.col("recv_ns") // interval_ns) * interval_ns).alias("_sample_bucket_ns")
    )
    return (
        lf.sort(["asset_id", "_sample_bucket_ns", "recv_ns"])
        .unique(subset=["asset_id", "_sample_bucket_ns"], keep="last", maintain_order=False)
        .drop("_sample_bucket_ns")
        .sort(["asset_id", "recv_ns"])
    )


def _downsample_event_driven(lf: pl.LazyFrame, args: argparse.Namespace) -> pl.LazyFrame:
    """Event-driven mode with tiered magnitude filtering.

    Tier 1 (critical columns): any change triggers an event.
    Tier 2 (secondary columns): change triggers only if abs(delta) > threshold.

    Always keeps the first row per asset.
    """
    schema = lf.collect_schema()

    # Resolve columns: --event-columns overrides tier1+tier2 if given
    if args.event_columns:
        tier1 = [c for c in args.event_columns if c in schema]
        tier2: list[str] = []
    else:
        tier1 = [c for c in (args.tier1_columns or TIER1_COLUMNS) if c in schema]
        tier2 = [c for c in (args.tier2_columns or TIER2_COLUMNS) if c in schema]

    all_cols = tier1 + tier2
    if not all_cols:
        logger.warning("no_event_columns_found", available=schema.names())
        return lf.sort(["asset_id", "recv_ns"])

    # Build magnitude thresholds for tier2
    mag_thresholds = dict(DEFAULT_MAGNITUDE_THRESHOLDS)
    if args.magnitude is not None:
        # Uniform override: apply to all tier2 columns
        mag_thresholds = {c: args.magnitude for c in tier2}
    else:
        # Only keep thresholds for columns actually in tier2
        mag_thresholds = {c: mag_thresholds.get(c, 0.0) for c in tier2}

    min_gap_ns = int(args.min_gap_ms * NS_PER_MS)
    logger.info(
        "event_driven_mode",
        tier1=tier1,
        tier2=tier2,
        magnitude_thresholds=mag_thresholds,
        min_gap_ms=args.min_gap_ms,
    )

    # Collect the filtered + sorted frame, then process per-asset.
    df_all = lf.sort(["asset_id", "recv_ns"]).collect()
    total_in = df_all.height
    logger.info("event_driven_rows", rows=total_in)

    partitions = df_all.partition_by("asset_id", as_dict=False)
    parts: list[pl.DataFrame] = []
    total_out = 0

    for chunk in partitions:
        if chunk.height == 0:
            continue

        # Tier 1: any change triggers
        changed = pl.lit(False)
        for col in tier1:
            prev = pl.col(col).shift(1)
            changed = changed | (pl.col(col) != prev) | (pl.col(col).is_not_null() != prev.is_not_null())

        # Tier 2: only if abs(delta) > threshold
        for col in tier2:
            thresh = mag_thresholds.get(col, 0.0)
            delta = (pl.col(col) - pl.col(col).shift(1)).abs()
            changed = changed | (delta > thresh)

        chunk = chunk.with_columns(changed.alias("_changed"))
        chunk = chunk.filter(pl.col("_changed")).drop("_changed")

        # Debounce: drop events closer than min_gap_ns to the previous kept event
        if min_gap_ns > 0 and chunk.height > 0:
            gap = pl.col("recv_ns") - pl.col("recv_ns").shift(1)
            keep = gap.is_null() | (gap >= min_gap_ns)
            chunk = chunk.filter(keep)

        total_out += chunk.height
        parts.append(chunk)

    result = pl.concat(parts) if parts else df_all.head(0)
    logger.info(
        "event_driven_done",
        rows_in=total_in,
        rows_out=total_out,
        ratio=f"{total_out / total_in * 100:.1f}%" if total_in else "N/A",
    )
    return result.lazy()


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
