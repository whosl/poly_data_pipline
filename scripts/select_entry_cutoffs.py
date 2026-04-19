#!/usr/bin/env python3
"""Select entry probability cutoffs on validation and apply them to test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.training.config import save_json
from poly.training.evaluation import choose_positive_class
from poly.training.splits import chronological_split, split_ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=4,
        action="append",
        metavar=("NAME", "VALIDATION_DATASET", "TEST_DATASET", "MODEL_DIR"),
        required=True,
        help=(
            "Run definition. Validation dataset supplies the validation split used to select the cutoff; "
            "test dataset supplies the final test split. Use the same path for within-asset runs."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-profit", default="final_profit_10s")
    parser.add_argument("--target-cls", default="y_final_profit_entry_10s")
    parser.add_argument("--positive-class", default="enter")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--split-purge-ms", type=int, default=0)
    parser.add_argument("--split-embargo-ms", type=int, default=0)
    parser.add_argument("--threshold-start", type=float, default=0.40)
    parser.add_argument("--threshold-stop", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--top-k-candidates", nargs="*", type=int, default=[50, 100, 250, 500, 1000, 2500, 5000])
    parser.add_argument("--min-validation-entries", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    threshold_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "target_profit": args.target_profit,
        "target_cls": args.target_cls,
        "positive_class": args.positive_class,
        "split_purge_ms": args.split_purge_ms,
        "split_embargo_ms": args.split_embargo_ms,
        "runs": [],
    }

    for run_name, validation_dataset, test_dataset, model_dir in args.run:
        validation_path = Path(validation_dataset)
        test_path = Path(test_dataset)
        model_path = Path(model_dir)
        val_split, val_ranges = load_named_split(validation_path, "validation", args)
        test_split, test_ranges = load_named_split(test_path, "test", args)
        metadata["runs"].append(
            {
                "name": run_name,
                "validation_dataset": str(validation_path),
                "test_dataset": str(test_path),
                "model_dir": str(model_path),
                "validation_split_ranges": val_ranges,
                "test_split_ranges": test_ranges,
            }
        )
        for path in sorted(model_path.glob("*.joblib")):
            payload = joblib.load(path)
            if payload.get("task") != "classification":
                continue
            model_name = path.stem
            val_scored = score_classifier(val_split, payload, args.positive_class)
            test_scored = score_classifier(test_split, payload, args.positive_class)
            thresholds = candidate_thresholds(
                val_scored["p_enter"].to_numpy(),
                start=args.threshold_start,
                stop=args.threshold_stop,
                step=args.threshold_step,
                top_k=args.top_k_candidates,
            )
            model_threshold_rows = []
            for threshold in thresholds:
                stats = selection_stats(val_scored, args.target_profit, args.target_cls, threshold, "val")
                row = {"run": run_name, "model": model_name, "threshold": threshold, **stats}
                threshold_rows.append(row)
                model_threshold_rows.append(row)
            best = select_best_threshold(model_threshold_rows, args.min_validation_entries)
            if best is None:
                selected_rows.append(
                    {
                        "run": run_name,
                        "model": model_name,
                        "selected_threshold": None,
                        "reason": "no validation threshold met min entry constraint",
                    }
                )
                continue
            threshold = float(best["threshold"])
            test_stats = selection_stats(test_scored, args.target_profit, args.target_cls, threshold, "test")
            selected_rows.append(
                {
                    "run": run_name,
                    "model": model_name,
                    "selected_threshold": threshold,
                    "reason": "selected_on_validation_avg_profit",
                    **{k: v for k, v in best.items() if k.startswith("val_")},
                    **test_stats,
                }
            )

    if threshold_rows:
        pl.DataFrame(threshold_rows).write_csv(str(args.output_dir / "validation_threshold_grid.csv"))
    if selected_rows:
        pl.DataFrame(selected_rows).sort(
            ["test_avg_profit", "test_total_profit", "test_entries"],
            descending=[True, True, True],
            nulls_last=True,
        ).write_csv(str(args.output_dir / "validation_selected_cutoffs_test_results.csv"))
    save_json(metadata, args.output_dir / "cutoff_selection_metadata.json")
    print(f"runs={len(args.run)}")
    print(f"threshold_grid={args.output_dir / 'validation_threshold_grid.csv'}")
    print(f"selected={args.output_dir / 'validation_selected_cutoffs_test_results.csv'}")


def load_named_split(path: Path, name: str, args: argparse.Namespace) -> tuple[pl.DataFrame, dict[str, dict[str, int | None]]]:
    dataset = pl.read_parquet(str(path))
    splits = chronological_split(
        dataset,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        purge_ms=args.split_purge_ms,
        embargo_ms=args.split_embargo_ms,
    )
    return splits[name], split_ranges(splits)


def score_classifier(df: pl.DataFrame, payload: dict[str, Any], preferred_positive_class: object) -> pl.DataFrame:
    model = payload["model"]
    feature_columns = payload["feature_columns"]
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing model feature columns: {missing[:10]}")
    x = df.select(feature_columns).to_pandas()
    proba = model.predict_proba(x)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["model"].classes_)
    positive_class = preferred_positive_class if preferred_positive_class in classes else choose_positive_class(classes)
    if positive_class not in classes:
        raise ValueError(f"could not find positive class {positive_class!r} in model classes {classes}")
    positive_idx = classes.index(positive_class)
    return df.with_columns(pl.Series("p_enter", proba[:, positive_idx]))


def candidate_thresholds(
    proba: np.ndarray,
    start: float,
    stop: float,
    step: float,
    top_k: list[int],
) -> list[float]:
    grid = list(np.round(np.arange(start, stop + step / 2, step), 6))
    clean = proba[np.isfinite(proba)]
    derived = []
    if clean.size:
        descending = np.sort(clean)[::-1]
        for k in top_k:
            if 0 < k <= descending.size:
                derived.append(float(np.round(descending[k - 1], 6)))
    return sorted(set(float(x) for x in grid + derived))


def select_best_threshold(rows: list[dict[str, Any]], min_entries: int) -> dict[str, Any] | None:
    eligible = [
        row
        for row in rows
        if row.get("val_entries", 0) >= min_entries and row.get("val_avg_profit") is not None
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            row["val_avg_profit"],
            row["val_total_profit"],
            row["val_entries"],
            row["threshold"],
        ),
    )


def selection_stats(
    df: pl.DataFrame,
    target_profit: str,
    target_cls: str,
    threshold: float,
    prefix: str,
) -> dict[str, float | int | None]:
    required = [target_profit, "p_enter"]
    if target_cls in df.columns:
        required.append(target_cls)
    selected = df.filter(pl.col("p_enter") >= threshold).drop_nulls(required)
    if selected.is_empty():
        return {
            f"{prefix}_entries": 0,
            f"{prefix}_avg_profit": None,
            f"{prefix}_median_profit": None,
            f"{prefix}_total_profit": 0.0,
            f"{prefix}_success_rate": None,
            f"{prefix}_avg_fail_unwind_loss": None,
            f"{prefix}_avg_p_enter": None,
        }
    profit = selected[target_profit]
    failed = selected.filter(pl.col(target_profit) < 0)
    if target_cls in selected.columns:
        success_rate = float((selected[target_cls] == "enter").mean())
    else:
        success_rate = float((profit > 0).mean())
    avg_fail_loss = None
    if not failed.is_empty():
        avg_fail_loss = float((-failed[target_profit]).mean())
    return {
        f"{prefix}_entries": selected.height,
        f"{prefix}_avg_profit": float(profit.mean()),
        f"{prefix}_median_profit": float(profit.median()),
        f"{prefix}_total_profit": float(profit.sum()),
        f"{prefix}_success_rate": success_rate,
        f"{prefix}_avg_fail_unwind_loss": avg_fail_loss,
        f"{prefix}_avg_p_enter": float(selected["p_enter"].mean()),
    }


if __name__ == "__main__":
    main()
