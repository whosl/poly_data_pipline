#!/usr/bin/env python3
"""Build labels-only artifacts for alpha and entry-worthiness inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.training.config import DatasetConfig, save_json
from poly.training.features import build_training_dataset
from poly.training.io import discover_dates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=None, help="Existing feature dataset parquet.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--dates", nargs="*", default=None)
    parser.add_argument("--horizon-seconds", type=int, default=10)
    parser.add_argument("--classification-theta-bps", type=float, default=5.0)
    parser.add_argument("--entry-threshold-bps", type=float, default=8.0)
    parser.add_argument("--strategy-entry-threshold-price", type=float, default=0.04)
    parser.add_argument("--two-leg-max-total-price", type=float, default=0.96)
    parser.add_argument("--two-leg-no-fill-edge", type=float, default=-1.0)
    parser.add_argument("--two-leg-maker-fill-trade-side", default="SELL")
    parser.add_argument("--final-profit-success-price", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset:
        df = pl.read_parquet(str(args.dataset))
        metadata = {"source_dataset": str(args.dataset)}
    else:
        dates = discover_dates(args.data_dir, args.dates)
        config = DatasetConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            dates=dates,
            horizon_seconds=args.horizon_seconds,
            classification_theta_bps=args.classification_theta_bps,
            entry_threshold_bps=args.entry_threshold_bps,
            strategy_entry_threshold_price=args.strategy_entry_threshold_price,
            two_leg_max_total_price=args.two_leg_max_total_price,
            two_leg_no_fill_edge=args.two_leg_no_fill_edge,
            two_leg_maker_fill_trade_side=args.two_leg_maker_fill_trade_side,
            final_profit_success_price=args.final_profit_success_price,
        )
        result = build_training_dataset(config)
        df = result.dataset
        metadata = result.metadata

    label_cols = [
        c
        for c in df.columns
        if c in {"recv_ns", "market_id", "asset_id", "symbol", "date", "current_mid", "y_entry", "y_entry_after_cost"}
        or c.startswith("future_mid_")
        or c.startswith("markout_")
        or c.startswith("y_")
    ]
    labels = df.select(label_cols)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "alpha_labels.parquet"
    labels.write_parquet(str(out))
    labels.write_csv(str(args.output_dir / "alpha_labels.csv"))
    save_json({"rows": labels.height, "columns": label_cols, "metadata": metadata}, args.output_dir / "alpha_labels_metadata.json")
    print(f"rows={labels.height}")
    print(f"parquet={out}")


if __name__ == "__main__":
    main()
