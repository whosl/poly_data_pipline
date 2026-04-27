#!/usr/bin/env python3
"""Combined policy evaluation: RF/ET + XGBoost + LightGBM.

Loads existing RF/ET models, trains XGBoost/LightGBM on the same data,
then runs policy selection across all fill×unwind combinations.
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
NS_PER_SECOND = 1_000_000_000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--parts-dir", type=Path, default=Path("artifacts/training_rf_et_win/alpha_dataset_parts"))
    p.add_argument("--rf-et-dir", type=Path, default=Path("artifacts/training_rf_et_win"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/combined_eval"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-rows", type=int, default=2_000_000)
    p.add_argument("--purge-seconds", type=int, default=300)
    p.add_argument("--lgb-rounds", type=int, default=100)
    p.add_argument("--xgb-rounds", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=6)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading (memory-safe, numpy-first)
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


def load_date_balanced_numpy(parts_dir, date, feature_cols, seed, max_rows):
    """Balanced sampling → numpy, with early stop and periodic consolidation."""
    d = parts_dir / f"date={date}"
    if not d.exists():
        return None, None, None
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None, None, None

    label_cols = ["y_two_leg_entry_binary_10s", "first_unwind_profit_proxy_10s"]
    need_cols = list(set(feature_cols + label_cols))
    n_feat = len(feature_cols)

    X_all = np.empty((0, n_feat), np.float32)
    ycls_all = np.empty(0, np.float32)
    yreg_all = np.empty(0, np.float32)

    CONSOLIDATE_EVERY = 25
    X_buf, ycls_buf, yreg_buf = [], [], []
    rng = np.random.default_rng(seed)
    total_raw, total_sampled = 0, 0

    def consolidate():
        nonlocal X_all, ycls_all, yreg_all
        if not X_buf:
            return
        X_all = np.concatenate([X_all] + X_buf, axis=0)
        ycls_all = np.concatenate([ycls_all] + ycls_buf)
        yreg_all = np.concatenate([yreg_all] + yreg_buf)
        X_buf.clear(); ycls_buf.clear(); yreg_buf.clear()
        gc.collect()

    for fi, f in enumerate(tqdm(files, desc=f"  Load+sample {date}", leave=False, ncols=100)):
        if X_all.shape[0] >= max_rows * 2:
            break

        available = set(pl.read_parquet_schema(str(f)).names())
        cols = [c for c in need_cols if c in available]
        if not cols:
            continue

        piece = pl.read_parquet(str(f), columns=cols)
        if piece.is_empty():
            continue
        total_raw += piece.height

        if "y_two_leg_entry_binary_10s" in piece.columns:
            y = piece["y_two_leg_entry_binary_10s"].to_numpy()
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y != 1)[0]
            n_pos = len(pos_idx)
            n_neg_sample = min(n_pos, len(neg_idx))
            if n_pos > 0 and n_neg_sample > 0:
                sampled_neg = rng.choice(neg_idx, size=n_neg_sample, replace=False)
                all_idx = np.sort(np.concatenate([pos_idx, sampled_neg]))
                piece = piece[all_idx.tolist()]

        if not piece.is_empty():
            X_buf.append(to_numpy(piece, feature_cols))
            ycls_buf.append(
                piece["y_two_leg_entry_binary_10s"].to_numpy().astype(np.float32)
                if "y_two_leg_entry_binary_10s" in piece.columns
                else np.full(piece.height, np.nan, np.float32)
            )
            yreg_buf.append(
                piece["first_unwind_profit_proxy_10s"].to_numpy().astype(np.float32)
                if "first_unwind_profit_proxy_10s" in piece.columns
                else np.full(piece.height, np.nan, np.float32)
            )
            total_sampled += piece.height

        del piece
        if (fi + 1) % CONSOLIDATE_EVERY == 0:
            consolidate()

    consolidate()

    if X_all.shape[0] == 0:
        return None, None, None

    print(f"    {date}: {total_raw:,} raw -> {total_sampled:,} balanced")
    if X_all.shape[0] > max_rows:
        idx = np.sort(rng.choice(X_all.shape[0], size=max_rows, replace=False))
        X_all, ycls_all, yreg_all = X_all[idx], ycls_all[idx], yreg_all[idx]
        print(f"    {date}: capped to {max_rows:,}")

    return X_all, ycls_all, yreg_all


def load_date_to_numpy(parts_dir, date, feature_cols, purge_ns=0,
                       purge_head=False, purge_tail=False):
    """Load a single date → numpy, file by file with consolidation."""
    d = parts_dir / f"date={date}"
    if not d.exists():
        return np.empty((0, len(feature_cols))), np.array([], np.float32), np.array([], np.float32)
    files = sorted(d.glob("*.parquet"))
    if not files:
        return np.empty((0, len(feature_cols))), np.array([], np.float32), np.array([], np.float32)

    label_cols = ["y_two_leg_entry_binary_10s", "first_unwind_profit_proxy_10s", "recv_ns"]
    need_cols = list(set(feature_cols + label_cols))
    n_feat = len(feature_cols)

    X_all = np.empty((0, n_feat), np.float32)
    ycls_all = np.empty(0, np.float32)
    yreg_all = np.empty(0, np.float32)
    recv_all = np.empty(0, np.int64)

    CONSOLIDATE_EVERY = 25
    X_buf, ycls_buf, yreg_buf, recv_buf = [], [], [], []

    def consolidate():
        nonlocal X_all, ycls_all, yreg_all, recv_all
        if not X_buf:
            return
        X_all = np.concatenate([X_all] + X_buf, axis=0)
        ycls_all = np.concatenate([ycls_all] + ycls_buf)
        yreg_all = np.concatenate([yreg_all] + yreg_buf)
        recv_all = np.concatenate([recv_all] + recv_buf)
        X_buf.clear(); ycls_buf.clear(); yreg_buf.clear(); recv_buf.clear()
        gc.collect()

    for fi, f in enumerate(tqdm(files, desc=f"  Load {date}", leave=False, ncols=100)):
        available = set(pl.read_parquet_schema(str(f)).names())
        cols = [c for c in need_cols if c in available]
        if not cols:
            continue
        piece = pl.read_parquet(str(f), columns=cols)
        if piece.is_empty():
            continue

        X_buf.append(to_numpy(piece, feature_cols))
        ycls_buf.append(
            piece["y_two_leg_entry_binary_10s"].to_numpy().astype(np.float32)
            if "y_two_leg_entry_binary_10s" in piece.columns
            else np.full(piece.height, np.nan, np.float32)
        )
        yreg_buf.append(
            piece["first_unwind_profit_proxy_10s"].to_numpy().astype(np.float32)
            if "first_unwind_profit_proxy_10s" in piece.columns
            else np.full(piece.height, np.nan, np.float32)
        )
        recv_buf.append(piece["recv_ns"].to_numpy().astype(np.int64))
        del piece

        if (fi + 1) % CONSOLIDATE_EVERY == 0:
            consolidate()

    consolidate()

    if X_all.shape[0] == 0:
        return X_all, ycls_all, yreg_all

    if purge_ns > 0:
        if purge_tail:
            mask = recv_all < recv_all.max() - purge_ns
            X_all, ycls_all, yreg_all, recv_all = X_all[mask], ycls_all[mask], yreg_all[mask], recv_all[mask]
            print(f"    Purge tail: kept {X_all.shape[0]:,}")
        if purge_head:
            mask = recv_all >= recv_all.min() + purge_ns
            X_all, ycls_all, yreg_all, recv_all = X_all[mask], ycls_all[mask], yreg_all[mask], recv_all[mask]
            print(f"    Embargo head: kept {X_all.shape[0]:,}")

    del recv_all
    gc.collect()
    return X_all, ycls_all, yreg_all


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _mask_nan(y, *arrays):
    m = ~np.isnan(y)
    if m.all():
        return (y,) + arrays
    return (y[m],) + tuple(a[m] for a in arrays)


def train_lgb_incremental(task, X, y, feature_cols, seed, rounds, lr, max_depth):
    """Train LightGBM on pre-loaded numpy arrays."""
    import lightgbm as lgb

    y_clean, X_clean = _mask_nan(y, X)[:2]
    n = len(y_clean)
    print(f"  Training LightGBM {'CLF' if task == 'clf' else 'REG'} on {n:,} rows, {rounds} rounds...")

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

    ds = lgb.Dataset(X_clean, label=y_clean, free_raw_data=True)
    t0 = time.time()
    booster = lgb.train(params, ds, num_boost_round=rounds)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s")
    del X_clean, y_clean, ds
    gc.collect()

    return LGBBoosterWrapper(booster, len(feature_cols),
                             "classification" if task == "clf" else "regression")


def train_xgb(task, X, y, seed, rounds, lr, max_depth):
    """Train XGBoost on pre-loaded numpy arrays."""
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
    del X_clean, y_clean
    gc.collect()
    return model


# ---------------------------------------------------------------------------
# Policy selection
# ---------------------------------------------------------------------------

def policy_selection(fill_names, unwind_names,
                     val_preds, val_y_cls, val_y_reg,
                     test_preds, test_y_cls, test_y_reg, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    min_p_fills = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
    min_unwinds = [-0.05, -0.02, 0.0, 0.01, 0.02]

    print(f"\n=== POLICY SELECTION ({len(fill_names)} fills × {len(unwind_names)} unwinds) ===\n")
    results = []
    for fill_name in fill_names:
        for unwind_name in unwind_names:
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
                            best = {
                                "threshold": th, "min_p_fill": mpf, "min_unwind": mu,
                                "n_entries": int(n),
                                "win_rate": float(val_y_cls[mask].mean()),
                                "avg_profit": avg_p,
                                "total_profit": float(val_y_reg[mask].sum()),
                            }
                            best_cfg = (th, mpf, mu)

            if best is None:
                print(f"  {fill_name} x {unwind_name}: no viable config")
                continue

            th, mpf, mu = best_cfg
            sp = 0.04
            ep = test_proba * sp + (1 - test_proba) * test_pred_uw
            tmask = (ep >= th) & (test_proba >= mpf) & (test_pred_uw >= mu)
            tn = tmask.sum()
            tr = {"n_entries": 0, "win_rate": 0, "avg_profit": 0, "total_profit": 0}
            if tn >= 5:
                tr = {
                    "n_entries": int(tn),
                    "win_rate": float(test_y_cls[tmask].mean()),
                    "avg_profit": float(test_y_reg[tmask].mean()),
                    "total_profit": float(test_y_reg[tmask].sum()),
                }

            combo = f"{fill_name}__{unwind_name}"
            results.append({
                "fill_model": fill_name, "unwind_model": unwind_name, "combo": combo,
                "val": best, "test": tr,
                "config": {"threshold": th, "min_p_fill": mpf, "min_unwind": mu},
            })
            print(f"  {combo}: val avgP={best['avg_profit']:.4f} n={best['n_entries']} "
                  f"wr={best['win_rate']:.2f} | test avgP={tr['avg_profit']:.4f} n={tr['n_entries']}")

    results.sort(key=lambda r: r["val"]["avg_profit"], reverse=True)
    with open(output_dir / "policy_selection_results.json", "w") as f:
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
    print("Combined Policy Evaluation: RF/ET + XGBoost + LightGBM")
    print("=" * 70)
    print(f"Parts dir:    {args.parts_dir}")
    print(f"RF/ET dir:    {args.rf_et_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Train:        {TRAIN_DATES}")
    print(f"Val:          {VAL_DATE}")
    print(f"Test:         {TEST_DATE}")

    # Discover features
    print("\n  Discovering features...")
    feature_cols = discover_features(args.parts_dir)
    print(f"  Features: {len(feature_cols)}")

    # --- Load training data ---
    print("\n--- Loading training data (balanced -> numpy) ---")
    X_train = np.empty((0, len(feature_cols)), np.float32)
    y_train_cls = np.empty(0, np.float32)
    y_train_reg = np.empty(0, np.float32)
    per_date_max = args.max_train_rows // len(TRAIN_DATES)

    for i, date in enumerate(TRAIN_DATES):
        X_d, ycls_d, yreg_d = load_date_balanced_numpy(
            args.parts_dir, date, feature_cols, args.seed + i, per_date_max)
        if X_d is None:
            continue
        X_train = np.concatenate([X_train, X_d], axis=0)
        y_train_cls = np.concatenate([y_train_cls, ycls_d])
        y_train_reg = np.concatenate([y_train_reg, yreg_d])
        del X_d, ycls_d, yreg_d
        gc.collect()

    if X_train.shape[0] == 0:
        print("ERROR: No training data!")
        sys.exit(1)

    print(f"\n  Train: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")

    # --- Train XGBoost + LightGBM ---
    print("\n--- Training XGBoost + LightGBM ---")
    models = {}

    models["lightgbm_classifier"] = train_lgb_incremental(
        "clf", X_train, y_train_cls, feature_cols,
        args.seed, args.lgb_rounds, args.lr, args.max_depth)

    models["xgboost_classifier"] = train_xgb(
        "clf", X_train, y_train_cls,
        args.seed, args.xgb_rounds, args.lr, args.max_depth)

    models["lightgbm_regressor"] = train_lgb_incremental(
        "reg", X_train, y_train_reg, feature_cols,
        args.seed, args.lgb_rounds, args.lr, args.max_depth)

    models["xgboost_regressor"] = train_xgb(
        "reg", X_train, y_train_reg,
        args.seed, args.xgb_rounds, args.lr, args.max_depth)

    # Save new models
    print("\nSaving XGBoost + LightGBM models...")
    for name, model in models.items():
        is_clf = "classifier" in name
        subdir = "fill_models" if is_clf else "unwind_models"
        path = args.output_dir / subdir / f"{name}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": model,
            "task": "classification" if is_clf else "regression",
            "feature_columns": feature_cols,
            "target": "y_two_leg_entry_binary_10s" if is_clf else "first_unwind_profit_proxy_10s",
        }, path)
        print(f"  {path}")

    # Free training data
    del X_train, y_train_cls, y_train_reg
    gc.collect()

    # --- Load RF/ET models ---
    print("\n--- Loading RF/ET models ---")
    for name in ["rf_classifier", "et_classifier", "rf_regressor", "et_regressor"]:
        is_clf = "classifier" in name
        subdir = "fill_models" if is_clf else "unwind_models"
        path = args.rf_et_dir / subdir / f"{name}.joblib"
        if path.exists():
            d = joblib.load(path)
            models[name] = d["model"]
            print(f"  Loaded {name}")
        else:
            print(f"  WARNING: {path} not found, skipping")

    # --- Load val/test ---
    print("\n--- Loading val & test ---")
    X_val, y_val_cls, y_val_reg = load_date_to_numpy(
        args.parts_dir, VAL_DATE, feature_cols, purge_ns, True, True)
    X_test, y_test_cls, y_test_reg = load_date_to_numpy(
        args.parts_dir, TEST_DATE, feature_cols, purge_ns, True, False)
    print(f"  Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")

    # --- Predict (batched to avoid OOM with sklearn RF/ET on 15M rows) ---
    BATCH_SIZE = 2_000_000

    print("\n--- Predicting ---")
    val_preds = {}
    test_preds = {}
    for name, model in tqdm(models.items(), desc="Predict", ncols=100):
        has_proba = hasattr(model, "predict_proba")
        # Predict val in batches
        parts_v = []
        for start in range(0, X_val.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, X_val.shape[0])
            xb = X_val[start:end]
            if has_proba:
                parts_v.append(model.predict_proba(xb)[:, 1])
            else:
                parts_v.append(model.predict(xb))
            gc.collect()
        val_preds[name] = np.concatenate(parts_v)
        del parts_v
        gc.collect()

        # Predict test in batches
        parts_t = []
        for start in range(0, X_test.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, X_test.shape[0])
            xb = X_test[start:end]
            if has_proba:
                parts_t.append(model.predict_proba(xb)[:, 1])
            else:
                parts_t.append(model.predict(xb))
            gc.collect()
        test_preds[name] = np.concatenate(parts_t)
        del parts_t
        gc.collect()

    del X_val, X_test
    gc.collect()

    # --- Policy selection ---
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
        "max_train_rows": args.max_train_rows,
        "models": list(models.keys()),
        "n_combos_tested": len(fill_names) * len(unwind_names),
        "elapsed_seconds": round(elapsed, 1),
        "best_combo": results[0] if results else None,
        "top3": results[:3] if results else [],
    }
    with open(args.output_dir / "combined_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {elapsed / 60:.1f} min.")


if __name__ == "__main__":
    main()
