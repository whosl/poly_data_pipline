#!/usr/bin/env python3
"""Evaluate baseline alpha models with prediction and trading-usefulness metrics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.training.evaluation import evaluate_dataset_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training/evaluation"))
    parser.add_argument("--target-reg", default="y_reg_10s")
    parser.add_argument("--target-cls", default="y_cls_10s")
    parser.add_argument("--taker-cost-bps", type=float, default=0.0)
    parser.add_argument("--prediction-thresholds", nargs="*", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_dataset_models(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        target_reg=args.target_reg,
        target_cls=args.target_cls,
        taker_cost_bps=args.taker_cost_bps,
        prediction_thresholds=args.prediction_thresholds,
    )
    print(f"models={len(report['models'])}")
    print(f"summary={args.output_dir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
