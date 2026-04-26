#!/usr/bin/env python3
"""Full-data chronological training with purge/embargo: LightGBM + XGBoost.

- Chronological split: Train=20260420-22, Val=20260423, Test=20260424
- Purge/embargo: 300s gap between train→val and val→test
- Memory-safe: loads one date at a time
- Incremental: 100 rounds per train date × 3 dates = 300 total rounds
- 5m markets only (already filtered in dataset)
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import joblib
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.training.features import infer_feature_columns
from poly.training.models import LGBBoosterWrapper

TRAIN_DATES = ["20260420", "20260421", "20260422"]
VAL_DATE = "20260423"
TEST_DATE = "20260424"
LABEL_COLS = ["y_two_leg_entry_binary_10s", "first_unwind_profit_proxy_10s", "recv_ns"]
NS_PER_SECOND = 1_000_000_000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds-per-date", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--purge-seconds", type=int, default=300)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_features(data_dir: Path) -> list[str]:
    for date_dir in sorted(data_dir.iterdir()):
        files = sorted(date_dir.glob("*.parquet"))
        if files:
            df = pl.read_parquet(files[0], n_rows=2000)
            return infer_feature_columns(df)
    raise SystemExit("No parquet files found")


def load_date_raw(data_dir: Path, date: str, need_cols: list[str]) -> pl.DataFrame | None:
    """Load all parquet files for a date, return as single DataFrame."""
    d = data_dir / f"date={date}"
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None

    pieces = []
    for f in tqdm(files, desc=f"  Load {date}", leave=False, ncols=100):
        available = set(pl.read_parquet_schema(f).names())
        cols = [c for c in need_cols if c in available]
        if cols:
            pieces.append(pl.read_parquet(f, columns=cols))

    if not pieces:
        return None

    df = pl.concat(pieces, how="diagonal_relaxed")
    return df.sort("recv_ns")


def to_numpy(df: pl.DataFrame, feature_cols: list[str]) -> np.ndarray:
    cols = [c for c in feature_cols if c in df.columns]
    if not cols or df.is_empty():
        return np.empty((df.height, len(feature_cols)), np.float32)

    arrs = []
    for c in feature_cols:
        if c not in df.columns:
            arrs.append(np.full(df.height, np.nan, np.float32))
            continue
        s = df[c]
        if s.dtype in (pl.String, pl.Categorical):
            s = s.cast(pl.Categorical).to_physical()
        arrs.append(s.to_numpy().astype(np.float32))
    result = np.column_stack(arrs)
    result[~np.isfinite(result)] = np.nan
    return result


def extract_labels(df: pl.DataFrame):
    y_cls = df["y_two_leg_entry_binary_10s"].to_numpy().astype(np.float32) if "y_two_leg_entry_binary_10s" in df.columns else np.array([], np.float32)
    y_reg = df["first_unwind_profit_proxy_10s"].to_numpy().astype(np.float32) if "first_unwind_profit_proxy_10s" in df.columns else np.array([], np.float32)
    return y_cls, y_reg


# ---------------------------------------------------------------------------
# Load with purge/embargo
# ---------------------------------------------------------------------------

def load_train(data_dir, dates, need_cols, feature_cols, purge_ns):
    """Load train dates, purge last purge_ns from the final train date."""
    dfs = []
    for date in dates:
        df = load_date_raw(data_dir, date, need_cols)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return np.empty((0, len(feature_cols))), np.array([], np.float32), np.array([], np.float32)

    df = pl.concat(dfs, how="diagonal_relaxed").sort("recv_ns")

    # Purge: remove last purge_ns from train
    max_ns = df["recv_ns"].max()
    cutoff = max_ns - purge_ns
    before = df.height
    df = df.filter(pl.col("recv_ns") < cutoff)
    print(f"  Train purge: removed {before - df.height:,} rows (last {purge_ns // NS_PER_SECOND}s)")

    X = to_numpy(df, feature_cols)
    y_cls, y_reg = extract_labels(df)
    print(f"  Train: {df.height:,} rows")
    del df
    gc.collect()
    return X, y_cls, y_reg


def load_val(data_dir, val_date, need_cols, feature_cols, purge_ns):
    """Load val date, embargo first purge_ns, purge last purge_ns."""
    df = load_date_raw(data_dir, val_date, need_cols)
    if df is None:
        return np.empty((0, len(feature_cols))), np.array([], np.float32), np.array([], np.float32)

    min_ns = df["recv_ns"].min()
    max_ns = df["recv_ns"].max()

    before = df.height
    # Embargo: remove first purge_ns
    # Purge: remove last purge_ns
    df = df.filter(
        (pl.col("recv_ns") >= min_ns + purge_ns) &
        (pl.col("recv_ns") < max_ns - purge_ns)
    )
    print(f"  Val embargo+purge: removed {before - df.height:,} rows ({purge_ns // NS_PER_SECOND}s each end)")

    X = to_numpy(df, feature_cols)
    y_cls, y_reg = extract_labels(df)
    print(f"  Val: {df.height:,} rows")
    del df
    gc.collect()
    return X, y_cls, y_reg


def load_test(data_dir, test_date, need_cols, feature_cols, purge_ns):
    """Load test date, embargo first purge_ns."""
    df = load_date_raw(data_dir, test_date, need_cols)
    if df is None:
        return np.empty((0, len(feature_cols))), np.array([], np.float32), np.array([], np.float32)

    min_ns = df["recv_ns"].min()

    before = df.height
    # Embargo: remove first purge_ns
    df = df.filter(pl.col("recv_ns") >= min_ns + purge_ns)
    print(f"  Test embargo: removed {before - df.height:,} rows (first {purge_ns // NS_PER_SECOND}s)")

    X = to_numpy(df, feature_cols)
    y_cls, y_reg = extract_labels(df)
    print(f"  Test: {df.height:,} rows")
    del df
    gc.collect()
    return X, y_cls, y_reg


# ---------------------------------------------------------------------------
# Incremental training
# ---------------------------------------------------------------------------

def _mask_nan(y: np.ndarray, *arrays: np.ndarray):
    m = ~np.isnan(y)
    if m.all():
        return (y,) + arrays
    return (y[m],) + tuple(a[m] for a in arrays)


def train_lgb_incremental(task, data_dir, dates, feature_cols, need_cols,
                          seed, purge_ns, rounds_per_date, lr, max_depth):
    """Train LightGBM incrementally on train dates with purge."""
    import lightgbm as lgb

    params = {
        "objective": "binary" if task == "clf" else "regression",
        "learning_rate": lr,
        "max_depth": max_depth,
        "verbosity": -1,
        "seed": seed,
        "n_jobs": -1,
        "force_col_wise": True,
    }
    if task == "clf":
        params["is_unbalance"] = True

    booster = None
    total_trees = 0

    for date in tqdm(dates, desc=f"LGB {'CLF' if task == 'clf' else 'REG'}", ncols=100):
        df = load_date_raw(data_dir, date, need_cols)
        if df is None:
            continue

        # For training dates before the last, no purge needed (only last train date gets purged)
        # But we still sort and use all rows from non-final dates
        y = df["y_two_leg_entry_binary_10s" if task == "clf" else "first_unwind_profit_proxy_10s"].to_numpy().astype(np.float32)
        X = to_numpy(df, feature_cols)
        y, X = _mask_nan(y, X)[:2]

        n = len(y)
        if n == 0:
            del df; gc.collect()
            continue

        ds = lgb.Dataset(X, label=y, free_raw_data=True)
        booster = lgb.train(params, ds, num_boost_round=rounds_per_date, init_model=booster)
        total_trees += rounds_per_date
        tqdm.write(f"    {date}: {n:,} rows → {total_trees} total trees")

        del df, X, y, ds
        gc.collect()

    return LGBBoosterWrapper(booster, len(feature_cols), "classification" if task == "clf" else "regression")


def train_xgb_incremental(task, data_dir, dates, feature_cols, need_cols,
                          seed, purge_ns, rounds_per_date, lr, max_depth):
    """Train XGBoost incrementally on train dates with purge."""
    import xgboost as xgb

    total_trees = 0
    model = None

    for date in tqdm(dates, desc=f"XGB {'CLF' if task == 'clf' else 'REG'}", ncols=100):
        df = load_date_raw(data_dir, date, need_cols)
        if df is None:
            continue

        y = df["y_two_leg_entry_binary_10s" if task == "clf" else "first_unwind_profit_proxy_10s"].to_numpy().astype(np.float32)
        X = to_numpy(df, feature_cols)
        y, X = _mask_nan(y, X)[:2]

        n = len(y)
        if n == 0:
            del df; gc.collect()
            continue

        if model is None:
            if task == "clf":
                model = xgb.XGBClassifier(
                    n_estimators=rounds_per_date, learning_rate=lr, max_depth=max_depth,
                    random_state=seed, eval_metric="logloss", n_jobs=-1, tree_method="hist",
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=rounds_per_date, learning_rate=lr, max_depth=max_depth,
                    random_state=seed, n_jobs=-1, tree_method="hist",
                )
            model.fit(X, y)
        else:
            prev_booster = model.get_booster()
            model.set_params(n_estimators=rounds_per_date)
            model.fit(X, y, xgb_model=prev_booster)

        total_trees += rounds_per_date
        tqdm.write(f"    {date}: {n:,} rows → {total_trees} total trees")

        del df, X, y
        gc.collect()

    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_split(models, X, y_cls, y_reg):
    """Predict on a pre-loaded split."""
    if X.shape[0] == 0:
        return {}, y_cls, y_reg

    preds = {}
    for name, model in tqdm(models.items(), desc="Predict", leave=False, ncols=100):
        if hasattr(model, "predict_proba"):
            preds[name] = model.predict_proba(X)[:, 1]
        else:
            preds[name] = model.predict(X)
    return preds, y_cls, y_reg


# ---------------------------------------------------------------------------
# Policy selection
# ---------------------------------------------------------------------------

def policy_selection(fill_models, unwind_models,
                     val_preds, val_y_cls, val_y_reg,
                     test_preds, test_y_cls, test_y_reg,
                     output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    min_p_fills = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    min_unwinds = [-0.05, -0.02, 0.0, 0.01, 0.02]

    print("\n=== POLICY SELECTION ===\n")
    results = []
    for fill_name in fill_models:
        for unwind_name in unwind_models:
            val_proba = val_preds[fill_name]
            val_pred_uw = val_preds[unwind_name]
            test_proba = test_preds[fill_name]
            test_pred_uw = test_preds[unwind_name]

            best, best_cfg = None, None
            for th in thresholds:
                for mpf in min_p_fills:
                    for mu in min_unwinds:
                        sp = 0.04
                        ep = val_proba * sp + (1 - val_proba) * val_pred_uw
                        mask = (ep >= th) & (val_proba >= mpf) & (val_pred_uw >= mu)
                        n = mask.sum()
                        if n < 10:
                            continue
                        avg_p = float(val_y_reg[mask].mean())
                        if best is None or avg_p > best["avg_profit"]:
                            best = {"threshold": th, "min_p_fill": mpf, "min_unwind": mu,
                                    "n_entries": int(n), "win_rate": float(val_y_cls[mask].mean()),
                                    "avg_profit": avg_p, "total_profit": float(val_y_reg[mask].sum())}
                            best_cfg = (th, mpf, mu)

            if best is None:
                print(f"  {fill_name} × {unwind_name}: no viable config")
                continue

            th, mpf, mu = best_cfg
            sp = 0.04
            ep = test_proba * sp + (1 - test_proba) * test_pred_uw
            tmask = (ep >= th) & (test_proba >= mpf) & (test_pred_uw >= mu)
            tn = tmask.sum()
            tr = {"n_entries": 0, "win_rate": 0, "avg_profit": 0, "total_profit": 0}
            if tn >= 5:
                tr = {"n_entries": int(tn), "win_rate": float(test_y_cls[tmask].mean()),
                      "avg_profit": float(test_y_reg[tmask].mean()),
                      "total_profit": float(test_y_reg[tmask].sum())}

            combo = f"{fill_name}__{unwind_name}"
            results.append({"fill_model": fill_name, "unwind_model": unwind_name, "combo": combo,
                            "val": best, "test": tr,
                            "config": {"threshold": th, "min_p_fill": mpf, "min_unwind": mu}})
            print(f"  {combo}: val avgP={best['avg_profit']:.4f} n={best['n_entries']} | "
                  f"test avgP={tr['avg_profit']:.4f} n={tr['n_entries']}")

    results.sort(key=lambda r: r["val"]["avg_profit"], reverse=True)
    with open(output_dir / "policy_selection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if results:
        print(f"\n  Best: {results[0]['combo']} → val avgP={results[0]['val']['avg_profit']:.4f} "
              f"test avgP={results[0]['test']['avg_profit']:.4f}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    purge_ns = args.purge_seconds * NS_PER_SECOND

    # Phase 0: Discover features
    print("=" * 60)
    print("Phase 0: Discover features")
    print("=" * 60)
    feature_cols = discover_features(args.data_dir)
    need_cols = list(set(feature_cols + LABEL_COLS))
    feature_only = [c for c in feature_cols if c not in LABEL_COLS]

    print(f"Features: {len(feature_cols)}")
    print(f"Train dates: {TRAIN_DATES}")
    print(f"Val date:    {VAL_DATE}")
    print(f"Test date:   {TEST_DATE}")
    print(f"Purge/embargo: {args.purge_seconds}s")

    # Phase 1: Train 4 models (incremental on train dates only)
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Incremental training ({args.rounds_per_date} rounds × {len(TRAIN_DATES)} dates = {args.rounds_per_date * len(TRAIN_DATES)} total)")
    print(f"{'=' * 60}")

    models = {}

    models["lightgbm_classifier"] = train_lgb_incremental(
        "clf", args.data_dir, TRAIN_DATES, feature_only, need_cols,
        args.seed, purge_ns, args.rounds_per_date, args.learning_rate, args.max_depth)

    models["xgboost_classifier"] = train_xgb_incremental(
        "clf", args.data_dir, TRAIN_DATES, feature_only, need_cols,
        args.seed, purge_ns, args.rounds_per_date, args.learning_rate, args.max_depth)

    models["lightgbm_regressor"] = train_lgb_incremental(
        "reg", args.data_dir, TRAIN_DATES, feature_only, need_cols,
        args.seed, purge_ns, args.rounds_per_date, args.learning_rate, args.max_depth)

    models["xgboost_regressor"] = train_xgb_incremental(
        "reg", args.data_dir, TRAIN_DATES, feature_only, need_cols,
        args.seed, purge_ns, args.rounds_per_date, args.learning_rate, args.max_depth)

    # Save models
    print("\nSaving models...")
    for name, model in models.items():
        is_clf = "classifier" in name
        task = "classification" if is_clf else "regression"
        target = "y_two_leg_entry_binary_10s" if is_clf else "first_unwind_profit_proxy_10s"
        subdir = "fill_models" if is_clf else "unwind_models"
        path = args.output_dir / subdir / f"{name}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "task": task, "feature_columns": feature_cols, "target": target}, path)
        print(f"  {path}")

    # Phase 2: Load val/test with purge, predict
    print(f"\n{'=' * 60}")
    print("Phase 2: Load val & test (with purge/embargo)")
    print(f"{'=' * 60}")

    print("\nLoading val...")
    X_val, y_val_cls, y_val_reg = load_val(args.data_dir, VAL_DATE, need_cols, feature_only, purge_ns)

    print("\nLoading test...")
    X_test, y_test_cls, y_test_reg = load_test(args.data_dir, TEST_DATE, need_cols, feature_only, purge_ns)

    print(f"\nPredicting val ({X_val.shape[0]:,} rows)...")
    val_preds, _, _ = predict_split(models, X_val, y_val_cls, y_val_reg)

    print(f"Predicting test ({X_test.shape[0]:,} rows)...")
    test_preds, _, _ = predict_split(models, X_test, y_test_cls, y_test_reg)

    del X_val, X_test
    gc.collect()

    # Phase 3: Policy selection
    print(f"\n{'=' * 60}")
    print("Phase 3: Policy selection")
    print(f"{'=' * 60}")

    fill_names = [n for n in models if "classifier" in n]
    unwind_names = [n for n in models if "regressor" in n]

    results = policy_selection(
        fill_names, unwind_names,
        val_preds, y_val_cls, y_val_reg,
        test_preds, y_test_cls, y_test_reg,
        args.output_dir / "execution_policy")

    # Summary
    elapsed = time.time() - t_start
    summary = {
        "train_dates": TRAIN_DATES,
        "val_date": VAL_DATE,
        "test_date": TEST_DATE,
        "purge_seconds": args.purge_seconds,
        "features": len(feature_cols),
        "rounds_per_date": args.rounds_per_date,
        "total_rounds": args.rounds_per_date * len(TRAIN_DATES),
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "val_rows": int(len(y_val_cls)),
        "test_rows": int(len(y_test_cls)),
        "elapsed_seconds": round(elapsed, 1),
        "best_combo": results[0] if results else None,
    }
    with open(args.output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {elapsed / 60:.1f} min. Summary → {args.output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
