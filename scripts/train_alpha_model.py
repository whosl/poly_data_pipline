#!/usr/bin/env python3
"""Train baseline alpha models on an engineered feature dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.training.config import TrainConfig
from poly.training.models import train_baselines

structlog.configure(processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Feature dataset parquet from build_features.py")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training/models"))
    parser.add_argument("--target-reg", default="y_reg_10s")
    parser.add_argument("--target-cls", default="y_cls_10s")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model names.")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--split-purge-ms", type=int, default=0)
    parser.add_argument("--split-embargo-ms", type=int, default=0)
    parser.add_argument("--sample-weight-col", default=None)
    parser.add_argument("--winsorize-lower", type=float, default=None,
                        help="Lower quantile for winsorization (e.g. 0.005 for 0.5th percentile)")
    parser.add_argument("--winsorize-upper", type=float, default=None,
                        help="Upper quantile for winsorization (e.g. 0.995 for 99.5th percentile)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        target_reg=args.target_reg,
        target_cls=args.target_cls,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        split_purge_ms=args.split_purge_ms,
        split_embargo_ms=args.split_embargo_ms,
        sample_weight_col=args.sample_weight_col,
        winsorize_lower=args.winsorize_lower,
        winsorize_upper=args.winsorize_upper,
    )
    if args.models is not None:
        config.models = args.models
    artifacts = train_baselines(config)
    print(f"features={len(artifacts.feature_columns)}")
    for name, path in artifacts.model_paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
