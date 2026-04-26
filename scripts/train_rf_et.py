#!/usr/bin/env python3
"""Train RF + ET models with importance-weighted sampling on chronological split.

- Chronological split: Train=20260420-22, Val=20260423, Test=20260424
- Purge/embargo: 300s gap between splits
- Importance-weighted sampling to reduce ~57M train rows to ~5-8M
- 4 models: rf_classifier, rf_regressor, et_classifier, et_regressor
- Policy selection on val, evaluation on test
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.training.features import infer_feature_columns

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
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--purge-seconds", type=int, default=300)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def discover_features(data_dir: Path) -> list[str]:
    for date_dir in sorted(data_dir.iterdir()):
        files = sorted(date_dir.glob("*.parquet"))
        if files:
            df = pl.read_parquet(files[0], n_rows=2000)
            return infer_feature_columns(df)
    raise SystemExit("No parquet files found")


def load_date_raw(data_dir: Path, date: str, need_cols: list[str]) -> pl.DataFrame | None:
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

    return pl.concat(pieces, how="diagonal_relaxed").sort("recv_ns")


BALANCED_LABEL_COLS = ["y_two_leg_entry_binary_10s", "first_unwind_profit_proxy_10s", "recv_ns"]


def load_date_balanced(data_dir: Path, date: str, feature_cols: list[str], seed: int) -> pl.DataFrame | None:
    """Load a date in two passes: first get labels for sampling, then load sampled features."""
    d = data_dir / f"date={date}"
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None

    # Pass 1: load only label columns to determine which rows to keep
    label_pieces = []
    file_row_counts = []
    for f in tqdm(files, desc=f"  Labels {date}", leave=False, ncols=100):
        available = set(pl.read_parquet_schema(f).names())
        label_cols = [c for c in BALANCED_LABEL_COLS if c in available]
        if label_cols:
            piece = pl.read_parquet(f, columns=label_cols)
            file_row_counts.append(piece.height)
            label_pieces.append(piece)
        else:
            file_row_counts.append(0)

    if not label_pieces:
        return None

    labels_df = pl.concat(label_pieces, how="diagonal_relaxed").sort("recv_ns")
    del label_pieces
    gc.collect()

    # Determine sample indices
    rng = np.random.default_rng(seed)
    y = labels_df["y_two_leg_entry_binary_10s"].to_numpy()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y != 1)[0]
    n_pos = len(pos_idx)
    n_neg_sample = min(n_pos, len(neg_idx))
    sampled_neg = rng.choice(neg_idx, size=n_neg_sample, replace=False)
    all_idx = np.sort(np.concatenate([pos_idx, sampled_neg]))

    print(f"    {date}: {labels_df.height:,} → {len(all_idx):,} rows (pos={n_pos:,}, neg={n_neg_sample:,})")

    # Extract recv_ns boundaries for sampled rows (to filter in pass 2)
    recv_all = labels_df["recv_ns"].to_numpy()
    sampled_recv_ns = recv_all[all_idx]
    min_recv = int(sampled_recv_ns.min())
    max_recv = int(sampled_recv_ns.max())
    del labels_df, y, pos_idx, neg_idx, sampled_neg, recv_all, sampled_recv_ns
    gc.collect()

    # Pass 2: load only feature + label columns, filtered by recv_ns range
    need_cols = list(set(feature_cols + BALANCED_LABEL_COLS))
    pieces = []
    for f in tqdm(files, desc=f"  Features {date}", leave=False, ncols=100):
        available = set(pl.read_parquet_schema(f).names())
        cols = [c for c in need_cols if c in available]
        if not cols:
            continue
        piece = pl.read_parquet(f, columns=cols)
        if "recv_ns" in piece.columns:
            piece = piece.filter(
                (pl.col("recv_ns") >= min_recv) & (pl.col("recv_ns") <= max_recv)
            )
        if piece.height > 0:
            pieces.append(piece)

    if not pieces:
        return None

    df = pl.concat(pieces, how="diagonal_relaxed").sort("recv_ns")
    del pieces
    gc.collect()
    return df


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
# Importance-weighted sampling
# ---------------------------------------------------------------------------

def balanced_sample_single(df: pl.DataFrame, seed: int, date: str = "") -> pl.DataFrame:
    """Keep all positive labels + equal number of randomly sampled negatives."""
    rng = np.random.default_rng(seed)
    n = df.height
    if n == 0:
        return df

    y = df["y_two_leg_entry_binary_10s"].to_numpy()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y != 1)[0]

    n_pos = len(pos_idx)
    n_neg_sample = min(n_pos, len(neg_idx))  # equal number of negatives
    sampled_neg = rng.choice(neg_idx, size=n_neg_sample, replace=False)

    all_idx = np.concatenate([pos_idx, sampled_neg])
    rng.shuffle(all_idx)
    df_sampled = df[all_idx.tolist()]

    print(f"    {date}: {n:,} → {len(all_idx):,} rows (pos={n_pos:,}, neg={n_neg_sample:,})")
    return df_sampled


# ---------------------------------------------------------------------------
# Load splits with purge/embargo
# ---------------------------------------------------------------------------

def load_train_balanced(data_dir, dates, need_cols, feature_cols, purge_ns, seed):
    """Load train dates with balanced sampling (pos + equal neg), concat, purge."""
    print(f"  Balanced sampling (all positives + equal negatives per date):")
    sampled_parts = []
    for i, date in enumerate(dates):
        df = load_date_balanced(data_dir, date, feature_cols, seed + i)
        if df is None:
            continue
        sampled_parts.append(df)
        gc.collect()

    if not sampled_parts:
        return np.empty((0, len(feature_cols))), np.array([], np.float32), np.array([], np.float32)

    df = pl.concat(sampled_parts, how="diagonal_relaxed").sort("recv_ns")
    del sampled_parts
    gc.collect()
    print(f"  Balanced train (before purge): {df.height:,} rows")

    # Purge: remove last purge_ns from train
    max_ns = df["recv_ns"].max()
    cutoff = max_ns - purge_ns
    before = df.height
    df = df.filter(pl.col("recv_ns") < cutoff)
    print(f"  Train purge: removed {before - df.height:,} rows (last {purge_ns // NS_PER_SECOND}s)")

    X = to_numpy(df, feature_cols)
    y_cls, y_reg = extract_labels(df)
    print(f"  Train (sampled): {df.height:,} rows")
    del df
    gc.collect()
    return X, y_cls, y_reg


def load_val(data_dir, val_date, need_cols, feature_cols, purge_ns):
    df = load_date_raw(data_dir, val_date, need_cols)
    if df is None:
        return np.empty((0, len(feature_cols))), np.array([], np.float32), np.array([], np.float32)

    min_ns = df["recv_ns"].min()
    max_ns = df["recv_ns"].max()
    before = df.height
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
    df = load_date_raw(data_dir, test_date, need_cols)
    if df is None:
        return np.empty((0, len(feature_cols))), np.array([], np.float32), np.array([], np.float32)

    min_ns = df["recv_ns"].min()
    before = df.height
    df = df.filter(pl.col("recv_ns") >= min_ns + purge_ns)
    print(f"  Test embargo: removed {before - df.height:,} rows (first {purge_ns // NS_PER_SECOND}s)")

    X = to_numpy(df, feature_cols)
    y_cls, y_reg = extract_labels(df)
    print(f"  Test: {df.height:,} rows")
    del df
    gc.collect()
    return X, y_cls, y_reg


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _mask_nan(y, *arrays):
    m = ~np.isnan(y)
    if m.all():
        return (y,) + arrays
    return (y[m],) + tuple(a[m] for a in arrays)


def train_sklearn_model(model_cls, is_classifier, X, y, seed):
    """Train a single sklearn model on already-loaded data."""
    task = "CLF" if is_classifier else "REG"
    print(f"  Training {model_cls.__name__} ({task}) on {X.shape[0]:,} rows...")

    y_clean, X_clean = _mask_nan(y, X)[:2]
    model = model_cls(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    if is_classifier:
        model.set_params(class_weight="balanced")

    t0 = time.time()
    model.fit(X_clean, y_clean)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    del X_clean, y_clean
    gc.collect()
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_split(models, X, y_cls, y_reg):
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
    global args
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

    # Phase 1: Load train (balanced) + train RF/ET
    print(f"\n{'=' * 60}")
    print("Phase 1: Load train data (balanced: pos + equal neg)")
    print(f"{'=' * 60}")

    X_train, y_train_cls, y_train_reg = load_train_balanced(
        args.data_dir, TRAIN_DATES, need_cols, feature_only, purge_ns,
        args.seed)

    models = {}

    print(f"\n--- Training RF Classifier ---")
    models["rf_classifier"] = train_sklearn_model(
        RandomForestClassifier, True, X_train, y_train_cls, args.seed)

    print(f"\n--- Training RF Regressor ---")
    models["rf_regressor"] = train_sklearn_model(
        RandomForestRegressor, False, X_train, y_train_reg, args.seed)

    print(f"\n--- Training ET Classifier ---")
    models["et_classifier"] = train_sklearn_model(
        ExtraTreesClassifier, True, X_train, y_train_cls, args.seed)

    print(f"\n--- Training ET Regressor ---")
    models["et_regressor"] = train_sklearn_model(
        ExtraTreesRegressor, False, X_train, y_train_reg, args.seed)

    # Free training data
    del X_train, y_train_cls, y_train_reg
    gc.collect()

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
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "sampling": "balanced (pos + equal neg)",
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
