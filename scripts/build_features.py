#!/usr/bin/env python3
"""Build engineered alpha-training features from normalized/research parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.training.config import DatasetConfig
from poly.training.features import build_training_dataset, write_dataset_artifacts
from poly.training.io import discover_dates

structlog.configure(processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--dates", nargs="*", default=None, help="YYYYMMDD dates. Defaults to all discovered dates.")
    parser.add_argument("--sample-interval-ms", type=int, default=100)
    parser.add_argument("--horizon-seconds", type=int, default=10)
    parser.add_argument("--classification-theta-bps", type=float, default=5.0)
    parser.add_argument("--entry-threshold-bps", type=float, default=8.0)
    parser.add_argument("--strategy-entry-threshold-price", type=float, default=0.04)
    parser.add_argument("--two-leg-max-total-price", type=float, default=0.96)
    parser.add_argument("--two-leg-no-fill-edge", type=float, default=-1.0)
    parser.add_argument("--two-leg-maker-fill-trade-side", default="SELL")
    parser.add_argument("--fee-rate", type=float, default=0.072)
    parser.add_argument("--price-buffer", type=float, default=0.01)
    parser.add_argument("--taker-cost-bps", type=float, default=0.0)
    parser.add_argument("--slippage-buffer-bps", type=float, default=2.0)
    parser.add_argument("--safety-margin-bps", type=float, default=1.0)
    parser.add_argument("--join-tolerance-ms", type=int, default=500)
    parser.add_argument("--basename", default="alpha_dataset")
    parser.add_argument("--no-csv", action="store_true", help="Only write parquet + metadata artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dates = discover_dates(args.data_dir, args.dates)
    if not dates:
        raise SystemExit(f"No dates found under {args.data_dir}/normalized or {args.data_dir}/research")

    config = DatasetConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dates=dates,
        sample_interval_ms=args.sample_interval_ms,
        horizon_seconds=args.horizon_seconds,
        classification_theta_bps=args.classification_theta_bps,
        entry_threshold_bps=args.entry_threshold_bps,
        strategy_entry_threshold_price=args.strategy_entry_threshold_price,
        two_leg_max_total_price=args.two_leg_max_total_price,
        two_leg_no_fill_edge=args.two_leg_no_fill_edge,
        two_leg_maker_fill_trade_side=args.two_leg_maker_fill_trade_side,
        fee_rate=args.fee_rate,
        price_buffer=args.price_buffer,
        taker_cost_bps=args.taker_cost_bps,
        slippage_buffer_bps=args.slippage_buffer_bps,
        safety_margin_bps=args.safety_margin_bps,
        join_tolerance_ms=args.join_tolerance_ms,
    )
    result = build_training_dataset(config)
    paths = write_dataset_artifacts(result, args.output_dir, args.basename, write_csv=not args.no_csv)
    print(f"rows={result.dataset.height}")
    for key, value in paths.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
