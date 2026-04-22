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
    parser.add_argument(
        "--selection-mode",
        choices=["row", "live"],
        default="row",
        help="row selects every row above threshold; live applies threshold-crossing, cooldown, and entry filters.",
    )
    parser.add_argument("--signal-cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--max-entries-per-signal-key", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--min-entry-ask", type=float, default=0.05)
    parser.add_argument("--max-entry-ask", type=float, default=0.95)
    parser.add_argument("--min-time-to-expiry", type=float, default=20.0)
    parser.add_argument("--max-spread", type=float, default=0.05)
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
        "selection_mode": args.selection_mode,
        "signal_cooldown_seconds": args.signal_cooldown_seconds,
        "max_entries_per_signal_key": args.max_entries_per_signal_key,
        "entry_filters": {
            "min_entry_ask": args.min_entry_ask,
            "max_entry_ask": args.max_entry_ask,
            "min_time_to_expiry": args.min_time_to_expiry,
            "max_spread": args.max_spread,
        },
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
                stats = selection_stats(val_scored, args.target_profit, args.target_cls, threshold, "val", args)
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
            test_stats = selection_stats(test_scored, args.target_profit, args.target_cls, threshold, "test", args)
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
    args: argparse.Namespace,
) -> dict[str, float | int | None]:
    required = [target_profit, "p_enter"]
    if target_cls in df.columns:
        required.append(target_cls)
    selected = select_rows(df, threshold, args).drop_nulls(required)
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


def select_rows(df: pl.DataFrame, threshold: float, args: argparse.Namespace) -> pl.DataFrame:
    if args.selection_mode == "row":
        return df.filter(pl.col("p_enter") >= threshold)
    return live_like_select_rows(df, threshold, args)


def live_like_select_rows(df: pl.DataFrame, threshold: float, args: argparse.Namespace) -> pl.DataFrame:
    """Mirror live signal gating for offline cutoff selection.

    This intentionally uses simple row iteration because the gating state is
    threshold-dependent and keyed by asset plus market/outcome.
    """
    if df.is_empty():
        return df
    rows = df.sort("recv_ns").with_row_index("_row_idx")
    cooldown_ns = int(args.signal_cooldown_seconds * 1_000_000_000)
    last_proba_by_asset: dict[str, float] = {}
    last_signal_ns_by_key: dict[str, int] = {}
    signal_count_by_key: dict[str, int] = {}
    selected_indices: list[int] = []
    max_entries = max(0, int(args.max_entries_per_signal_key))

    needed = {
        "_row_idx",
        "recv_ns",
        "asset_id",
        "market_id",
        "slug",
        "outcome",
        "opposite_asset_id",
        "best_bid",
        "best_ask",
        "current_spread",
        "time_to_expiry_seconds",
        "p_enter",
    }
    missing = needed - set(rows.columns)
    if missing:
        raise ValueError(f"live selection missing required columns: {sorted(missing)}")

    for row in rows.select(sorted(needed)).iter_rows(named=True):
        asset_id = str(row["asset_id"])
        recv_ns = int(row["recv_ns"])
        proba = row["p_enter"]
        if proba is None or not np.isfinite(proba):
            last_proba_by_asset[asset_id] = float("-inf")
            continue

        prev_proba = last_proba_by_asset.get(asset_id)
        raw_signal = proba >= threshold
        crossed_threshold = prev_proba is None or prev_proba < threshold
        market_key = row.get("market_id") or row.get("slug") or asset_id
        signal_key = f"{market_key}:{row.get('outcome') or asset_id}"
        cooldown_ok = recv_ns - last_signal_ns_by_key.get(signal_key, 0) >= cooldown_ns
        max_entries_ok = max_entries == 0 or signal_count_by_key.get(signal_key, 0) < max_entries
        has_opposite = row.get("opposite_asset_id") is not None
        entry_filter_ok = (
            row["best_bid"] is not None
            and row["best_ask"] is not None
            and row["current_spread"] is not None
            and row["time_to_expiry_seconds"] is not None
            and row["best_bid"] > 0
            and args.min_entry_ask <= row["best_ask"] <= args.max_entry_ask
            and row["current_spread"] <= args.max_spread
            and row["time_to_expiry_seconds"] >= args.min_time_to_expiry
        )
        if raw_signal and crossed_threshold and cooldown_ok and max_entries_ok and has_opposite and entry_filter_ok:
            selected_indices.append(int(row["_row_idx"]))
            last_signal_ns_by_key[signal_key] = recv_ns
            signal_count_by_key[signal_key] = signal_count_by_key.get(signal_key, 0) + 1
        last_proba_by_asset[asset_id] = float(proba)

    if not selected_indices:
        return df.head(0)
    return rows.filter(pl.col("_row_idx").is_in(selected_indices)).drop("_row_idx")


if __name__ == "__main__":
    main()
