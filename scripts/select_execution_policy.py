#!/usr/bin/env python3
"""Select two-stage execution policies from fill and unwind models.

The policy scores each candidate entry with:

    expected_profit = p_fill * success_profit + (1 - p_fill) * pred_unwind_profit

Then validation data chooses thresholds for expected profit, fill probability,
and predicted unwind profit. Test data is only used after those thresholds are
selected.
"""

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


NS_PER_SECOND = 1_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        nargs=7,
        action="append",
        metavar=(
            "NAME",
            "VALIDATION_DATASET",
            "TEST_DATASET",
            "FILL_MODEL_DIR",
            "FILL_MODEL",
            "UNWIND_MODEL_DIR",
            "UNWIND_MODEL",
        ),
        required=True,
        help="Run definition with explicit fill classifier and unwind regressor model names.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-profit", default="final_profit_10s")
    parser.add_argument("--target-fill", default="y_two_leg_entry_10s")
    parser.add_argument("--positive-class", default="enter")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--split-purge-ms", type=int, default=0)
    parser.add_argument("--split-embargo-ms", type=int, default=0)
    parser.add_argument("--fee-rate", type=float, default=0.072)
    parser.add_argument("--price-buffer", type=float, default=0.01)
    parser.add_argument("--max-total-price", type=float, default=0.96)
    parser.add_argument("--min-validation-entries", type=int, default=25)
    parser.add_argument("--expected-profit-thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--min-p-fill-grid", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.65, 0.75, 0.85])
    parser.add_argument(
        "--min-pred-unwind-profit-grid",
        nargs="*",
        type=float,
        default=[-1.0, -0.08, -0.05, -0.03, -0.02, -0.01, 0.0],
    )
    parser.add_argument(
        "--selection-mode",
        choices=["row", "live"],
        default="live",
        help="row selects every row above thresholds; live applies threshold-crossing, cooldown, and entry filters.",
    )
    parser.add_argument("--signal-cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--max-entries-per-signal-key", type=int, default=3, help="0 means unlimited.")
    parser.add_argument("--min-entry-ask", type=float, default=0.05)
    parser.add_argument("--max-entry-ask", type=float, default=0.95)
    parser.add_argument("--min-time-to-expiry", type=float, default=20.0)
    parser.add_argument("--max-spread", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "target_profit": args.target_profit,
        "target_fill": args.target_fill,
        "selection_mode": args.selection_mode,
        "split_purge_ms": args.split_purge_ms,
        "split_embargo_ms": args.split_embargo_ms,
        "fee_rate": args.fee_rate,
        "price_buffer": args.price_buffer,
        "max_total_price": args.max_total_price,
        "runs": [],
    }

    for run_name, val_dataset, test_dataset, fill_dir, fill_model_name, unwind_dir, unwind_model_name in args.run:
        val_path = Path(val_dataset)
        test_path = Path(test_dataset)
        fill_path = Path(fill_dir) / f"{fill_model_name}.joblib"
        unwind_path = Path(unwind_dir) / f"{unwind_model_name}.joblib"
        if not fill_path.exists():
            raise FileNotFoundError(fill_path)
        if not unwind_path.exists():
            raise FileNotFoundError(unwind_path)

        val_split, val_ranges = load_named_split(val_path, "validation", args)
        test_split, test_ranges = load_named_split(test_path, "test", args)
        fill_payload = joblib.load(fill_path)
        unwind_payload = joblib.load(unwind_path)
        val_scored = add_policy_scores(val_split, fill_payload, unwind_payload, args)
        test_scored = add_policy_scores(test_split, fill_payload, unwind_payload, args)

        thresholds = candidate_expected_profit_thresholds(val_scored, args)
        run_rows: list[dict[str, Any]] = []
        for exp_threshold in thresholds:
            for min_p_fill in args.min_p_fill_grid:
                for min_unwind in args.min_pred_unwind_profit_grid:
                    policy = {
                        "expected_profit_threshold": float(exp_threshold),
                        "min_p_fill": float(min_p_fill),
                        "min_pred_unwind_profit": float(min_unwind),
                    }
                    stats = selection_stats(val_scored, policy, "val", args)
                    row = {
                        "run": run_name,
                        "fill_model": fill_model_name,
                        "unwind_model": unwind_model_name,
                        **policy,
                        **stats,
                    }
                    grid_rows.append(row)
                    run_rows.append(row)

        best = select_best_policy(run_rows, args.min_validation_entries)
        if best is None:
            selected_rows.append(
                    {
                        "run": run_name,
                        "fill_model": fill_model_name,
                        "unwind_model": unwind_model_name,
                        "reason": "no validation policy met min entry constraint",
                    }
                )
            continue
        policy = {
            "expected_profit_threshold": best["expected_profit_threshold"],
            "min_p_fill": best["min_p_fill"],
            "min_pred_unwind_profit": best["min_pred_unwind_profit"],
        }
        test_stats = selection_stats(test_scored, policy, "test", args)
        selected_rows.append(
                {
                    "run": run_name,
                    "fill_model": fill_model_name,
                    "unwind_model": unwind_model_name,
                    "reason": "selected_on_validation_avg_profit",
                **policy,
                **{k: v for k, v in best.items() if k.startswith("val_")},
                **test_stats,
            }
        )
        metadata["runs"].append(
            {
                "name": run_name,
                "validation_dataset": str(val_path),
                "test_dataset": str(test_path),
                "fill_model_path": str(fill_path),
                "unwind_model_path": str(unwind_path),
                "fill_model": fill_model_name,
                "unwind_model": unwind_model_name,
                "validation_split_ranges": val_ranges,
                "test_split_ranges": test_ranges,
            }
        )

    if grid_rows:
        pl.DataFrame(grid_rows).write_csv(str(args.output_dir / "execution_policy_validation_grid.csv"))
    if selected_rows:
        pl.DataFrame(selected_rows).sort(
            ["test_avg_profit", "test_total_profit", "test_entries"],
            descending=[True, True, True],
            nulls_last=True,
        ).write_csv(str(args.output_dir / "execution_policy_selected_test_results.csv"))
    save_json(metadata, args.output_dir / "execution_policy_metadata.json")
    print(f"runs={len(args.run)}")
    print(f"grid={args.output_dir / 'execution_policy_validation_grid.csv'}")
    print(f"selected={args.output_dir / 'execution_policy_selected_test_results.csv'}")


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


def add_policy_scores(
    df: pl.DataFrame,
    fill_payload: dict[str, Any],
    unwind_payload: dict[str, Any],
    args: argparse.Namespace,
) -> pl.DataFrame:
    fill = score_classifier(df, fill_payload, args.positive_class)
    unwind = score_regressor(df, unwind_payload)
    scored = df.with_columns(
        [
            pl.Series("p_fill", fill),
            pl.Series("pred_unwind_profit", unwind),
        ]
    )
    success_profit = compute_success_profit(scored, args)
    return scored.with_columns(
        [
            success_profit.alias("success_profit_estimate"),
        ]
    ).with_columns(
        (
            pl.col("p_fill") * pl.col("success_profit_estimate")
            + (1.0 - pl.col("p_fill")) * pl.col("pred_unwind_profit")
        ).alias("pred_expected_profit")
    )


def score_classifier(df: pl.DataFrame, payload: dict[str, Any], preferred_positive_class: object) -> np.ndarray:
    if payload.get("task") != "classification":
        raise ValueError(f"fill model must be classification, got {payload.get('task')}")
    model = payload["model"]
    feature_columns = payload["feature_columns"]
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing fill feature columns: {missing[:10]}")
    x = df.select(feature_columns).to_pandas()
    proba = model.predict_proba(x)
    classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["model"].classes_)
    positive_class = preferred_positive_class if preferred_positive_class in classes else choose_positive_class(classes)
    positive_idx = classes.index(positive_class)
    return np.asarray(proba[:, positive_idx], dtype=float)


def score_regressor(df: pl.DataFrame, payload: dict[str, Any]) -> np.ndarray:
    if payload.get("task") != "regression":
        raise ValueError(f"unwind model must be regression, got {payload.get('task')}")
    model = payload["model"]
    feature_columns = payload["feature_columns"]
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing unwind feature columns: {missing[:10]}")
    x = df.select(feature_columns).to_pandas()
    return np.asarray(model.predict(x), dtype=float)


def compute_success_profit(df: pl.DataFrame, args: argparse.Namespace) -> pl.Expr:
    entry_price = pl.col("best_ask")
    first_leg_price = entry_price + pl.lit(args.price_buffer)
    fee_per_share = pl.lit(args.fee_rate) * first_leg_price * (1.0 - first_leg_price)
    second_leg_size = 1.0 - fee_per_share
    second_leg_price = pl.lit(args.max_total_price) - entry_price
    return second_leg_size - (first_leg_price + second_leg_size * second_leg_price)


def candidate_expected_profit_thresholds(df: pl.DataFrame, args: argparse.Namespace) -> list[float]:
    if args.expected_profit_thresholds is not None:
        return sorted(set(float(x) for x in args.expected_profit_thresholds))
    grid = list(np.round(np.arange(-0.05, 0.051, 0.005), 6))
    values = df["pred_expected_profit"].drop_nulls().to_numpy()
    derived = []
    if values.size:
        descending = np.sort(values[np.isfinite(values)])[::-1]
        for k in [10, 20, 25, 50, 75, 100, 150, 200, 300, 500, 1000]:
            if 0 < k <= descending.size:
                derived.append(float(np.round(descending[k - 1], 6)))
    return sorted(set(float(x) for x in grid + derived))


def select_best_policy(rows: list[dict[str, Any]], min_entries: int) -> dict[str, Any] | None:
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
            row["expected_profit_threshold"],
            row["min_p_fill"],
            row["min_pred_unwind_profit"],
        ),
    )


def selection_stats(
    df: pl.DataFrame,
    policy: dict[str, float],
    prefix: str,
    args: argparse.Namespace,
) -> dict[str, float | int | None]:
    selected = select_rows(df, policy, args).drop_nulls([args.target_profit])
    if selected.is_empty():
        return empty_stats(prefix)
    profit = selected[args.target_profit]
    fail = selected.filter(pl.col(args.target_profit) < 0)
    fill_success = None
    if args.target_fill in selected.columns:
        fill_success = float((selected[args.target_fill] == args.positive_class).mean())
    avg_fail_loss = None
    if not fail.is_empty():
        avg_fail_loss = float((-fail[args.target_profit]).mean())
    return {
        f"{prefix}_entries": selected.height,
        f"{prefix}_avg_profit": float(profit.mean()),
        f"{prefix}_median_profit": float(profit.median()),
        f"{prefix}_total_profit": float(profit.sum()),
        f"{prefix}_positive_rate": float((profit > 0).mean()),
        f"{prefix}_fill_success_rate": fill_success,
        f"{prefix}_avg_fail_loss": avg_fail_loss,
        f"{prefix}_avg_p_fill": float(selected["p_fill"].mean()),
        f"{prefix}_avg_pred_unwind_profit": float(selected["pred_unwind_profit"].mean()),
        f"{prefix}_avg_pred_expected_profit": float(selected["pred_expected_profit"].mean()),
    }


def empty_stats(prefix: str) -> dict[str, float | int | None]:
    return {
        f"{prefix}_entries": 0,
        f"{prefix}_avg_profit": None,
        f"{prefix}_median_profit": None,
        f"{prefix}_total_profit": 0.0,
        f"{prefix}_positive_rate": None,
        f"{prefix}_fill_success_rate": None,
        f"{prefix}_avg_fail_loss": None,
        f"{prefix}_avg_p_fill": None,
        f"{prefix}_avg_pred_unwind_profit": None,
        f"{prefix}_avg_pred_expected_profit": None,
    }


def select_rows(df: pl.DataFrame, policy: dict[str, float], args: argparse.Namespace) -> pl.DataFrame:
    base = df.filter(
        (pl.col("pred_expected_profit") >= policy["expected_profit_threshold"])
        & (pl.col("p_fill") >= policy["min_p_fill"])
        & (pl.col("pred_unwind_profit") >= policy["min_pred_unwind_profit"])
    )
    if args.selection_mode == "row":
        return base
    return live_like_select_rows(base, args)


def live_like_select_rows(df: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    if df.is_empty():
        return df
    rows = df.sort("recv_ns").with_row_index("_row_idx")
    cooldown_ns = int(args.signal_cooldown_seconds * NS_PER_SECOND)
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
    }
    missing = needed - set(rows.columns)
    if missing:
        raise ValueError(f"live selection missing required columns: {sorted(missing)}")

    for row in rows.select(sorted(needed)).iter_rows(named=True):
        recv_ns = int(row["recv_ns"])
        market_key = row.get("market_id") or row.get("slug") or row.get("asset_id")
        signal_key = f"{market_key}:{row.get('outcome') or row.get('asset_id')}"
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
        if cooldown_ok and max_entries_ok and has_opposite and entry_filter_ok:
            selected_indices.append(int(row["_row_idx"]))
            last_signal_ns_by_key[signal_key] = recv_ns
            signal_count_by_key[signal_key] = signal_count_by_key.get(signal_key, 0) + 1

    if not selected_indices:
        return df.head(0)
    return rows.filter(pl.col("_row_idx").is_in(selected_indices)).drop("_row_idx")


if __name__ == "__main__":
    main()
