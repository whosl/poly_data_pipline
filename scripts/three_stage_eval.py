#!/usr/bin/env python3
"""3-stage policy evaluation: first-leg fill → second-leg fill → unwind profit.

Stage 1: P(first_leg fills) — filters out signals where taker order won't execute
Stage 2: P(second_leg maker fill | first_leg filled) — existing two-leg classifier
Stage 3: predicted unwind profit — existing regressor

Combines all three into an expected-profit decision with configurable gates.
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
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.training.features import infer_feature_columns
from poly.training.models import LGBBoosterWrapper

TRAIN_DATES = ["20260420", "20260421", "20260422"]
VAL_DATE = "20260423"
TEST_DATE = "20260424"
NS_PER_SECOND = 1_000_000_000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parts-dir", type=Path, default=Path("artifacts/training_rf_et_win/alpha_dataset_parts"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/three_stage_eval"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-rows", type=int, default=900_000)
    p.add_argument("--purge-seconds", type=int, default=300)
    p.add_argument("--lgb-rounds", type=int, default=100)
    p.add_argument("--xgb-rounds", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--max-val-rows", type=int, default=2_000_000,
                   help="Max rows to load for val/test (avoids OOM)")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--rf-max-depth", type=int, default=12)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_features(parts_dir: Path) -> list[str]:
    for date_dir in sorted(parts_dir.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith("date="):
            continue
        files = sorted(date_dir.glob("*.parquet"))
        if files:
            df = pl.read_parquet(str(files[0]), n_rows=2000)
            return infer_feature_columns(df)
    raise SystemExit(f"No parquet files found in {parts_dir}")


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


def load_date_balanced_numpy(parts_dir, date, feature_cols, seed, max_rows,
                             target_col="y_two_leg_entry_binary_10s", filter_col=None):
    """Balanced sampling → numpy for a given target column.

    If filter_col is set, only rows where filter_col == 1 are kept before
    balanced sampling.  Used so Stage 2/3 train only on first-leg-fill rows.
    """
    d = parts_dir / f"date={date}"
    if not d.exists():
        return None, None
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None, None

    label_cols = [target_col]
    if filter_col:
        label_cols.append(filter_col)
    need_cols = list(set(feature_cols + label_cols))

    # Collect all chunks in a list — no incremental concatenation
    chunks_X, chunks_y = [], []
    rng = np.random.default_rng(seed)
    total_raw, total_sampled = 0, 0
    total_loaded = 0

    for fi, f in enumerate(tqdm(files, desc=f"  Load+sample {date}", leave=False, ncols=100)):
        if total_loaded >= max_rows * 2:
            break

        available = set(pl.read_parquet_schema(str(f)).names())
        cols = [c for c in need_cols if c in available]
        if target_col not in cols:
            continue

        piece = pl.read_parquet(str(f), columns=cols)
        if piece.is_empty():
            continue

        # Pre-filter: keep only rows where filter_col == 1
        if filter_col and filter_col in piece.columns:
            piece = piece.filter(pl.col(filter_col) == 1)
            if piece.is_empty():
                continue

        total_raw += piece.height

        # Balanced sampling
        y = piece[target_col].to_numpy()
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_pos = len(pos_idx)
        n_neg_sample = min(n_pos, len(neg_idx))
        if n_pos > 0 and n_neg_sample > 0:
            sampled_neg = rng.choice(neg_idx, size=n_neg_sample, replace=False)
            all_idx = np.sort(np.concatenate([pos_idx, sampled_neg]))
            piece = piece[all_idx.tolist()]

        if not piece.is_empty():
            feat_piece = piece.select([c for c in feature_cols if c in piece.columns])
            chunks_X.append(to_numpy(feat_piece, feature_cols))
            chunks_y.append(piece[target_col].to_numpy().astype(np.float32))
            total_sampled += piece.height
            total_loaded += piece.height

        del piece

    if not chunks_X:
        return None, None

    # Single final concatenation
    X_all = np.concatenate(chunks_X, axis=0)
    del chunks_X; gc.collect()
    y_all = np.concatenate(chunks_y)
    del chunks_y; gc.collect()

    print(f"    {date} [{target_col}]: {total_raw:,} raw -> {total_sampled:,} balanced")
    if X_all.shape[0] > max_rows:
        idx = np.sort(rng.choice(X_all.shape[0], size=max_rows, replace=False))
        X_all, y_all = X_all[idx], y_all[idx]
        print(f"    {date}: capped to {max_rows:,}")

    return X_all, y_all


def load_date_to_numpy(parts_dir, date, feature_cols, purge_ns=0,
                       purge_head=False, purge_tail=False, max_rows=0):
    """Load a single date → numpy (all rows, no balancing)."""
    d = parts_dir / f"date={date}"
    if not d.exists():
        empty = np.empty((0, len(feature_cols)))
        return empty, np.array([], np.float32), np.array([], np.float32), np.array([], np.float32)
    files = sorted(d.glob("*.parquet"))
    if not files:
        empty = np.empty((0, len(feature_cols)))
        return empty, np.array([], np.float32), np.array([], np.float32), np.array([], np.float32)

    label_cols = ["y_first_leg_fill", "y_two_leg_entry_binary_10s",
                  "first_unwind_profit_proxy_10s", "recv_ns"]
    need_cols = list(set(feature_cols + label_cols))

    # Collect all chunks in a list — no incremental concatenation
    chunks_X, chunks_yf, chunks_yc, chunks_yr, chunks_recv = [], [], [], [], []
    total_loaded = 0

    for fi, f in enumerate(tqdm(files, desc=f"  Load {date}", leave=False, ncols=100)):
        if max_rows > 0 and total_loaded >= max_rows:
            break
        available = set(pl.read_parquet_schema(str(f)).names())
        cols = [c for c in need_cols if c in available]
        if not cols:
            continue
        piece = pl.read_parquet(str(f), columns=cols)
        if piece.is_empty():
            continue

        chunks_X.append(to_numpy(piece, feature_cols))
        chunks_yf.append(
            piece["y_first_leg_fill"].to_numpy().astype(np.float32)
            if "y_first_leg_fill" in piece.columns
            else np.full(piece.height, np.nan, np.float32)
        )
        chunks_yc.append(
            piece["y_two_leg_entry_binary_10s"].to_numpy().astype(np.float32)
            if "y_two_leg_entry_binary_10s" in piece.columns
            else np.full(piece.height, np.nan, np.float32)
        )
        chunks_yr.append(
            piece["first_unwind_profit_proxy_10s"].to_numpy().astype(np.float32)
            if "first_unwind_profit_proxy_10s" in piece.columns
            else np.full(piece.height, np.nan, np.float32)
        )
        chunks_recv.append(piece["recv_ns"].to_numpy().astype(np.int64))
        total_loaded += piece.height
        del piece

    if not chunks_X:
        empty = np.empty((0, len(feature_cols)))
        return empty, np.array([], np.float32), np.array([], np.float32), np.array([], np.float32)

    # Single final concatenation
    X_all = np.concatenate(chunks_X, axis=0)
    y_first_all = np.concatenate(chunks_yf)
    y_cls_all = np.concatenate(chunks_yc)
    y_reg_all = np.concatenate(chunks_yr)
    recv_all = np.concatenate(chunks_recv)
    del chunks_X, chunks_yf, chunks_yc, chunks_yr, chunks_recv
    gc.collect()

    if max_rows > 0 and X_all.shape[0] > max_rows:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_all.shape[0], max_rows, replace=False)
        idx.sort()
        X_all = X_all[idx]
        y_first_all = y_first_all[idx]
        y_cls_all = y_cls_all[idx]
        y_reg_all = y_reg_all[idx]
        recv_all = recv_all[idx]

    if X_all.shape[0] == 0:
        return X_all, y_first_all, y_cls_all, y_reg_all

    if purge_ns > 0:
        if purge_tail:
            mask = recv_all < recv_all.max() - purge_ns
            X_all = X_all[mask]; y_first_all = y_first_all[mask]
            y_cls_all = y_cls_all[mask]; y_reg_all = y_reg_all[mask]
            recv_all = recv_all[mask]
            print(f"    Purge tail: kept {X_all.shape[0]:,}")
        if purge_head:
            mask = recv_all >= recv_all.min() + purge_ns
            X_all = X_all[mask]; y_first_all = y_first_all[mask]
            y_cls_all = y_cls_all[mask]; y_reg_all = y_reg_all[mask]
            recv_all = recv_all[mask]
            print(f"    Embargo head: kept {X_all.shape[0]:,}")

    del recv_all
    gc.collect()
    return X_all, y_first_all, y_cls_all, y_reg_all


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _mask_nan(y, *arrays):
    m = ~np.isnan(y)
    if m.all():
        return (y,) + arrays
    return (y[m],) + tuple(a[m] for a in arrays)


def train_lgb_incremental(task, X, y, feature_cols, seed, rounds, lr, max_depth):
    import lightgbm as lgb
    y_clean, X_clean = _mask_nan(y, X)[:2]
    n = len(y_clean)
    print(f"  Training LightGBM {'CLF' if task == 'clf' else 'REG'} on {n:,} rows, {rounds} rounds...")
    params = {
        "objective": "binary" if task == "clf" else "regression",
        "learning_rate": lr, "max_depth": max_depth,
        "verbosity": -1, "seed": seed, "n_jobs": -1, "force_col_wise": True,
    }
    if task == "clf":
        params["is_unbalance"] = True
    ds = lgb.Dataset(X_clean, label=y_clean, free_raw_data=True)
    t0 = time.time()
    booster = lgb.train(params, ds, num_boost_round=rounds)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    del X_clean, y_clean, ds; gc.collect()
    return LGBBoosterWrapper(booster, len(feature_cols),
                             "classification" if task == "clf" else "regression")


def train_xgb(task, X, y, seed, rounds, lr, max_depth):
    import xgboost as xgb
    y_clean, X_clean = _mask_nan(y, X)[:2]
    n = len(y_clean)
    print(f"  Training XGBoost {'CLF' if task == 'clf' else 'REG'} on {n:,} rows, {rounds} rounds...")
    if task == "clf":
        model = xgb.XGBClassifier(
            n_estimators=rounds, learning_rate=lr, max_depth=max_depth,
            random_state=seed, eval_metric="logloss", n_jobs=-1, tree_method="hist",
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=rounds, learning_rate=lr, max_depth=max_depth,
            random_state=seed, n_jobs=-1, tree_method="hist",
        )
    t0 = time.time()
    model.fit(X_clean, y_clean)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    del X_clean, y_clean; gc.collect()
    return model


def train_rf_et(task, model_cls, X, y, seed, n_estimators, max_depth):
    y_clean, X_clean = _mask_nan(y, X)[:2]
    n = len(y_clean)
    print(f"  Training {model_cls.__name__} {'CLF' if task == 'clf' else 'REG'} "
          f"on {n:,} rows, {n_estimators} trees, depth {max_depth}...")
    model = model_cls(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1,
    )
    if task == "clf":
        model.set_params(class_weight="balanced")
    t0 = time.time()
    model.fit(X_clean, y_clean)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    del X_clean, y_clean; gc.collect()
    return model


# ---------------------------------------------------------------------------
# 3-Stage policy selection
# ---------------------------------------------------------------------------

def three_stage_policy_selection(
    stage1_names, stage2_names, stage3_names,
    val_preds, val_y_first, val_y_cls, val_y_reg,
    test_preds, test_y_first, test_y_cls, test_y_reg,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    min_p_first_fills = [0.5, 0.6, 0.7, 0.8, 0.9]
    min_p_second_fills = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    min_unwinds = [-0.05, -0.02, 0.0, 0.01, 0.02]

    print(f"\n=== 3-STAGE POLICY SELECTION "
          f"({len(stage1_names)} s1 × {len(stage2_names)} s2 × {len(stage3_names)} s3) ===\n")
    results = []

    for s1_name in stage1_names:
        for s2_name in stage2_names:
            for s3_name in stage3_names:
                val_p1 = val_preds[s1_name]
                val_p2 = val_preds[s2_name]
                val_p3 = val_preds[s3_name]
                test_p1 = test_preds[s1_name]
                test_p2 = test_preds[s2_name]
                test_p3 = test_preds[s3_name]

                best, best_cfg = None, None
                for th in thresholds:
                    for mpf1 in min_p_first_fills:
                        for mpf2 in min_p_second_fills:
                            for mu in min_unwinds:
                                sp = 0.04
                                # 3-stage: p_first is gate, not multiplier
                                ep = val_p2 * sp + (1 - val_p2) * val_p3
                                mask = (val_p1 >= mpf1) & (val_p2 >= mpf2) & (ep >= th) & (val_p3 >= mu)
                                n = mask.sum()
                                if n < 10:
                                    continue
                                avg_p = float(val_y_reg[mask].mean())
                                if best is None or avg_p > best["avg_profit"]:
                                    best = {
                                        "threshold": th, "min_p_first_fill": mpf1,
                                        "min_p_second_fill": mpf2, "min_unwind": mu,
                                        "n_entries": int(n),
                                        "win_rate": float(val_y_cls[mask].mean()),
                                        "avg_profit": avg_p,
                                        "total_profit": float(val_y_reg[mask].sum()),
                                        "first_leg_fill_rate": float(val_y_first[mask].mean()),
                                    }
                                    best_cfg = (th, mpf1, mpf2, mu)

                if best is None:
                    continue

                th, mpf1, mpf2, mu = best_cfg
                sp = 0.04
                ep = test_p2 * sp + (1 - test_p2) * test_p3
                tmask = (test_p1 >= mpf1) & (test_p2 >= mpf2) & (ep >= th) & (test_p3 >= mu)
                tn = tmask.sum()
                tr = {"n_entries": 0, "win_rate": 0, "avg_profit": 0, "total_profit": 0,
                      "first_leg_fill_rate": 0}
                if tn >= 5:
                    tr = {
                        "n_entries": int(tn),
                        "win_rate": float(test_y_cls[tmask].mean()),
                        "avg_profit": float(test_y_reg[tmask].mean()),
                        "total_profit": float(test_y_reg[tmask].sum()),
                        "first_leg_fill_rate": float(test_y_first[tmask].mean()),
                    }

                combo = f"{s1_name}__{s2_name}__{s3_name}"
                results.append({
                    "stage1_model": s1_name, "stage2_model": s2_name, "stage3_model": s3_name,
                    "combo": combo, "val": best, "test": tr,
                    "config": {"threshold": th, "min_p_first_fill": mpf1,
                               "min_p_second_fill": mpf2, "min_unwind": mu},
                })
                print(f"  {combo}: val avgP={best['avg_profit']:.4f} n={best['n_entries']} "
                      f"wr={best['win_rate']:.2f} | test avgP={tr['avg_profit']:.4f} n={tr['n_entries']}")

    results.sort(key=lambda r: r["val"]["avg_profit"], reverse=True)
    with open(output_dir / "three_stage_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if results:
        print(f"\n  Top 3:")
        for r in results[:3]:
            print(f"    {r['combo']}: val={r['val']['avg_profit']:.4f} "
                  f"test={r['test']['avg_profit']:.4f} "
                  f"(n_val={r['val']['n_entries']}, n_test={r['test']['n_entries']})")
    return results


def main():
    t_start = time.time()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    purge_ns = args.purge_seconds * NS_PER_SECOND

    print("=" * 70)
    print("3-Stage Policy Evaluation: First-Leg Fill → Second-Leg Fill → Unwind")
    print("=" * 70)
    print(f"Parts dir:    {args.parts_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Train:        {TRAIN_DATES}")
    print(f"Val:          {VAL_DATE}")
    print(f"Test:         {TEST_DATE}")

    # Discover features
    print("\n  Discovering features...")
    feature_cols = discover_features(args.parts_dir)
    print(f"  Features: {len(feature_cols)}")

    # --- Stage 1: Train first-leg fill classifiers ---
    print("\n" + "=" * 70)
    print("STAGE 1: First-Leg Fill Classifiers")
    print("=" * 70)

    per_date_max = args.max_train_rows // len(TRAIN_DATES)

    s1_chunks_X, s1_chunks_y = [], []
    for i, date in enumerate(TRAIN_DATES):
        X_d, y_d = load_date_balanced_numpy(
            args.parts_dir, date, feature_cols, args.seed + i, per_date_max,
            target_col="y_first_leg_fill")
        if X_d is not None:
            s1_chunks_X.append(X_d)
            s1_chunks_y.append(y_d)
            del X_d, y_d; gc.collect()
    X_train_s1 = np.concatenate(s1_chunks_X, axis=0) if s1_chunks_X else np.empty((0, len(feature_cols)), np.float32)
    y_train_s1 = np.concatenate(s1_chunks_y) if s1_chunks_y else np.empty(0, np.float32)
    del s1_chunks_X, s1_chunks_y; gc.collect()

    print(f"\n  Stage 1 train: {X_train_s1.shape[0]:,} rows")
    print(f"  First-leg fill rate: {np.nanmean(y_train_s1):.4f}")

    models = {}
    models["s1_lgb"] = train_lgb_incremental(
        "clf", X_train_s1, y_train_s1, feature_cols,
        args.seed, args.lgb_rounds, args.lr, args.max_depth)
    models["s1_xgb"] = train_xgb(
        "clf", X_train_s1, y_train_s1,
        args.seed, args.xgb_rounds, args.lr, args.max_depth)
    models["s1_rf"] = train_rf_et(
        "clf", RandomForestClassifier, X_train_s1, y_train_s1,
        args.seed, args.n_estimators, args.rf_max_depth)
    models["s1_et"] = train_rf_et(
        "clf", ExtraTreesClassifier, X_train_s1, y_train_s1,
        args.seed, args.n_estimators, args.rf_max_depth)

    del X_train_s1, y_train_s1; gc.collect()

    # --- Stage 2: Train second-leg fill classifiers ---
    print("\n" + "=" * 70)
    print("STAGE 2: Second-Leg Fill Classifiers (y_two_leg_entry_binary_10s)")
    print("=" * 70)

    s2_chunks_X, s2_chunks_y = [], []
    for i, date in enumerate(TRAIN_DATES):
        X_d, y_d = load_date_balanced_numpy(
            args.parts_dir, date, feature_cols, args.seed + i + 100, per_date_max,
            target_col="y_two_leg_entry_binary_10s")
        if X_d is not None:
            s2_chunks_X.append(X_d)
            s2_chunks_y.append(y_d)
            del X_d, y_d; gc.collect()
    X_train_s2 = np.concatenate(s2_chunks_X, axis=0) if s2_chunks_X else np.empty((0, len(feature_cols)), np.float32)
    y_train_s2 = np.concatenate(s2_chunks_y) if s2_chunks_y else np.empty(0, np.float32)
    del s2_chunks_X, s2_chunks_y; gc.collect()

    print(f"\n  Stage 2 train: {X_train_s2.shape[0]:,} rows")
    print(f"  Two-leg entry rate: {np.nanmean(y_train_s2):.4f}")

    models["s2_lgb"] = train_lgb_incremental(
        "clf", X_train_s2, y_train_s2, feature_cols,
        args.seed, args.lgb_rounds, args.lr, args.max_depth)
    models["s2_xgb"] = train_xgb(
        "clf", X_train_s2, y_train_s2,
        args.seed, args.xgb_rounds, args.lr, args.max_depth)
    models["s2_rf"] = train_rf_et(
        "clf", RandomForestClassifier, X_train_s2, y_train_s2,
        args.seed, args.n_estimators, args.rf_max_depth)
    models["s2_et"] = train_rf_et(
        "clf", ExtraTreesClassifier, X_train_s2, y_train_s2,
        args.seed, args.n_estimators, args.rf_max_depth)

    del X_train_s2, y_train_s2; gc.collect()

    # --- Stage 3: Train unwind regressors ---
    print("\n" + "=" * 70)
    print("STAGE 3: Unwind Profit Regressors (first_unwind_profit_proxy_10s)")
    print("=" * 70)

    s3_chunks_X, s3_chunks_y = [], []
    for i, date in enumerate(TRAIN_DATES):
        X_d, y_d = load_date_balanced_numpy(
            args.parts_dir, date, feature_cols, args.seed + i + 200, per_date_max,
            target_col="first_unwind_profit_proxy_10s")
        if X_d is not None:
            s3_chunks_X.append(X_d)
            s3_chunks_y.append(y_d)
            del X_d, y_d; gc.collect()
    X_train_s3 = np.concatenate(s3_chunks_X, axis=0) if s3_chunks_X else np.empty((0, len(feature_cols)), np.float32)
    y_train_s3 = np.concatenate(s3_chunks_y) if s3_chunks_y else np.empty(0, np.float32)
    del s3_chunks_X, s3_chunks_y; gc.collect()

    print(f"\n  Stage 3 train: {X_train_s3.shape[0]:,} rows")

    models["s3_lgb"] = train_lgb_incremental(
        "reg", X_train_s3, y_train_s3, feature_cols,
        args.seed, args.lgb_rounds, args.lr, args.max_depth)
    models["s3_xgb"] = train_xgb(
        "reg", X_train_s3, y_train_s3,
        args.seed, args.xgb_rounds, args.lr, args.max_depth)
    models["s3_rf"] = train_rf_et(
        "reg", RandomForestRegressor, X_train_s3, y_train_s3,
        args.seed, args.n_estimators, args.rf_max_depth)
    models["s3_et"] = train_rf_et(
        "reg", ExtraTreesRegressor, X_train_s3, y_train_s3,
        args.seed, args.n_estimators, args.rf_max_depth)

    del X_train_s3, y_train_s3; gc.collect()

    # Save models
    print("\nSaving 3-stage models...")
    for name, model in models.items():
        is_clf = "s3_" not in name
        subdir = "fill_models" if is_clf else "unwind_models"
        path = args.output_dir / subdir / f"{name}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        target = "y_first_leg_fill" if "s1_" in name else (
            "y_two_leg_entry_binary_10s" if "s2_" in name else "first_unwind_profit_proxy_10s")
        joblib.dump({
            "model": model,
            "task": "classification" if is_clf else "regression",
            "feature_columns": feature_cols,
            "target": target,
        }, path)
        print(f"  {path}")

    # --- Load val, predict, free ---
    print("\n--- Loading val ---")
    X_val, y_val_first, y_val_cls, y_val_reg = load_date_to_numpy(
        args.parts_dir, VAL_DATE, feature_cols, purge_ns, True, True, args.max_val_rows)
    y_val_first = np.nan_to_num(y_val_first, nan=0.0)
    print(f"  Val: {X_val.shape[0]:,}")

    BATCH_SIZE = 2_000_000
    print("--- Predicting val ---")
    val_preds = {}
    for name, model in tqdm(models.items(), desc="Predict val", ncols=100):
        has_proba = hasattr(model, "predict_proba")
        parts_v = []
        for start in range(0, X_val.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, X_val.shape[0])
            xb = X_val[start:end]
            parts_v.append(model.predict_proba(xb)[:, 1] if has_proba else model.predict(xb))
            gc.collect()
        val_preds[name] = np.concatenate(parts_v)
        del parts_v; gc.collect()
    del X_val; gc.collect()

    # --- Load test, predict, free ---
    print("\n--- Loading test ---")
    X_test, y_test_first, y_test_cls, y_test_reg = load_date_to_numpy(
        args.parts_dir, TEST_DATE, feature_cols, purge_ns, True, False, args.max_val_rows)
    y_test_first = np.nan_to_num(y_test_first, nan=0.0)
    print(f"  Test: {X_test.shape[0]:,}")

    print("--- Predicting test ---")
    test_preds = {}
    for name, model in tqdm(models.items(), desc="Predict test", ncols=100):
        has_proba = hasattr(model, "predict_proba")
        parts_t = []
        for start in range(0, X_test.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, X_test.shape[0])
            xb = X_test[start:end]
            parts_t.append(model.predict_proba(xb)[:, 1] if has_proba else model.predict(xb))
            gc.collect()
        test_preds[name] = np.concatenate(parts_t)
        del parts_t; gc.collect()
    del X_test; gc.collect()

    # --- Stage 1 standalone analysis ---
    print("\n" + "=" * 70)
    print("STAGE 1 ANALYSIS: First-Leg Fill Prediction Quality")
    print("=" * 70)
    s1_names = [k for k in models if k.startswith("s1_")]
    s2_names = [k for k in models if k.startswith("s2_")]
    s3_names = [k for k in models if k.startswith("s3_")]

    for name in s1_names:
        p = val_preds[name]
        for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
            mask = p >= thr
            n = mask.sum()
            if n < 10:
                continue
            actual_fill = float(y_val_first[mask].mean())
            print(f"  {name} thr={thr:.1f}: n={n:,} actual_fill_rate={actual_fill:.4f}")

    # --- 2-stage baseline (for comparison) ---
    print("\n" + "=" * 70)
    print("2-STAGE BASELINE (no first-leg filter)")
    print("=" * 70)

    baseline_results = []
    for s2_name in s2_names:
        for s3_name in s3_names:
            val_p2 = val_preds[s2_name]
            val_p3 = val_preds[s3_name]
            test_p2 = test_preds[s2_name]
            test_p3 = test_preds[s3_name]

            best, best_cfg = None, None
            for th in [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]:
                for mpf in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]:
                    for mu in [-0.05, -0.02, 0.0, 0.01, 0.02]:
                        sp = 0.04
                        ep = val_p2 * sp + (1 - val_p2) * val_p3
                        mask = (ep >= th) & (val_p2 >= mpf) & (val_p3 >= mu)
                        n = mask.sum()
                        if n < 10:
                            continue
                        avg_p = float(y_val_reg[mask].mean())
                        if best is None or avg_p > best["avg_profit"]:
                            best = {"threshold": th, "min_p_fill": mpf, "min_unwind": mu,
                                    "n_entries": int(n), "win_rate": float(y_val_cls[mask].mean()),
                                    "avg_profit": avg_p, "total_profit": float(y_val_reg[mask].sum())}
                            best_cfg = (th, mpf, mu)

            if best is None:
                continue

            th, mpf, mu = best_cfg
            sp = 0.04
            ep = test_p2 * sp + (1 - test_p2) * test_p3
            tmask = (ep >= th) & (test_p2 >= mpf) & (test_p3 >= mu)
            tn = tmask.sum()
            tr = {"n_entries": 0, "win_rate": 0, "avg_profit": 0, "total_profit": 0}
            if tn >= 5:
                tr = {"n_entries": int(tn), "win_rate": float(y_test_cls[tmask].mean()),
                      "avg_profit": float(y_test_reg[tmask].mean()),
                      "total_profit": float(y_test_reg[tmask].sum())}

            combo = f"{s2_name}__{s3_name}"
            baseline_results.append({"combo": combo, "val": best, "test": tr, "config": {
                "threshold": th, "min_p_fill": mpf, "min_unwind": mu}})
            print(f"  {combo}: val avgP={best['avg_profit']:.4f} n={best['n_entries']} "
                  f"| test avgP={tr['avg_profit']:.4f} n={tr['n_entries']}")

    # --- 3-stage policy selection ---
    results = three_stage_policy_selection(
        s1_names, s2_names, s3_names,
        val_preds, y_val_first, y_val_cls, y_val_reg,
        test_preds, y_test_first, y_test_cls, y_test_reg,
        args.output_dir / "execution_policy")

    # Summary
    elapsed = time.time() - t_start
    summary = {
        "train_dates": TRAIN_DATES,
        "val_date": VAL_DATE,
        "test_date": TEST_DATE,
        "purge_seconds": args.purge_seconds,
        "features": len(feature_cols),
        "max_train_rows": args.max_train_rows,
        "models": list(models.keys()),
        "n_3stage_combos": len(results),
        "n_2stage_combos": len(baseline_results),
        "elapsed_seconds": round(elapsed, 1),
        "best_3stage": results[0] if results else None,
        "best_2stage": sorted(baseline_results, key=lambda r: r["val"]["avg_profit"], reverse=True)[0] if baseline_results else None,
        "top3_3stage": results[:3],
    }
    with open(args.output_dir / "three_stage_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("COMPARISON: 2-STAGE vs 3-STAGE")
    print(f"{'=' * 70}")
    if baseline_results:
        best_2 = sorted(baseline_results, key=lambda r: r["val"]["avg_profit"], reverse=True)[0]
        print(f"  Best 2-stage: {best_2['combo']}")
        print(f"    Val:  avgP={best_2['val']['avg_profit']:.4f} n={best_2['val']['n_entries']} wr={best_2['val']['win_rate']:.2f}")
        print(f"    Test: avgP={best_2['test']['avg_profit']:.4f} n={best_2['test']['n_entries']}")
    if results:
        best_3 = results[0]
        print(f"  Best 3-stage: {best_3['combo']}")
        print(f"    Val:  avgP={best_3['val']['avg_profit']:.4f} n={best_3['val']['n_entries']} wr={best_3['val']['win_rate']:.2f}")
        print(f"    Test: avgP={best_3['test']['avg_profit']:.4f} n={best_3['test']['n_entries']}")
        print(f"    Config: p_first>={best_3['config']['min_p_first_fill']}, "
              f"p_second>={best_3['config']['min_p_second_fill']}, "
              f"thr>={best_3['config']['threshold']}, "
              f"unwind>={best_3['config']['min_unwind']}")

    print(f"\nDone in {elapsed / 60:.1f} min.")


if __name__ == "__main__":
    main()
