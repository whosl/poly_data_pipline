#!/usr/bin/env python3
"""Build partitioned features + train RF/ET on Windows (32GB).

Memory-efficient pipeline:
Phase A: Build partitioned alpha dataset by loading one market at a time
         from poly_sampled_book using pyarrow predicate pushdown.
Phase B: Load partitioned data, balanced sample, train RF/ET, evaluate.
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
import pyarrow.dataset as ds
import joblib
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.training.config import DatasetConfig
from poly.training.features import (
    build_feature_frame,
    canonicalize_book,
    canonicalize_metadata,
    canonicalize_trades,
    enrich_with_metadata,
    filter_book_scope,
    infer_feature_columns,
    safe_part_name,
    canonicalize_binance_book,
    get_frame,
)
from poly.training.io import TableLoadResult, read_parquet_safe, table_path

TRAIN_DATES = ["20260420", "20260421", "20260422"]
VAL_DATE = "20260423"
TEST_DATE = "20260424"
ALL_DATES = TRAIN_DATES + [VAL_DATE, TEST_DATE]
NS_PER_SECOND = 1_000_000_000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/training_rf_et_win"))
    p.add_argument("--parts-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--purge-seconds", type=int, default=300)
    p.add_argument("--period", type=str, default="5m")
    p.add_argument("--max-train-rows", type=int, default=4_000_000)
    p.add_argument("--skip-build", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase A: Memory-efficient partitioned build
# ---------------------------------------------------------------------------


def get_market_ids(sampled_path: Path) -> list[str]:
    """Get unique market_ids from sampled_book using pyarrow."""
    dataset = ds.dataset(str(sampled_path), format="parquet")
    scanner = dataset.scanner(columns=["market_id"])
    table = scanner.to_table()
    return sorted(table.column("market_id").unique().to_pylist())


def load_market_sampled_book(sampled_path: Path, market_id: str) -> pl.DataFrame:
    """Load one market's data using pyarrow predicate pushdown."""
    dataset = ds.dataset(str(sampled_path), format="parquet")
    filter_expr = ds.field("market_id") == market_id
    table = dataset.to_table(filter=filter_expr)
    return pl.from_arrow(table)


def build_partitioned_dataset(data_dir: Path, parts_dir: Path, dates: list[str],
                               config: DatasetConfig) -> None:
    """Build one parquet per date/market_id."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    example_frame = None

    for date in dates:
        print(f"\n  Processing {date}...")
        t0 = time.time()

        sampled_path = data_dir / "normalized" / date / "poly_sampled_book.parquet"
        if not sampled_path.exists():
            print(f"    No sampled_book for {date}, skipping")
            continue

        # Load small tables once - use proper canonicalization
        tables = {}
        for name in ["poly_market_metadata", "poly_trades", "binance_l2_book",
                      "binance_best_bid_ask", "binance_trades"]:
            path = table_path(data_dir, "normalized", date, name)
            tables[name] = read_parquet_safe(path, name)

        metadata_df = tables["poly_market_metadata"].frame if tables["poly_market_metadata"].ok else None
        metadata = canonicalize_metadata(metadata_df)

        poly_trades_raw = tables["poly_trades"].frame if tables["poly_trades"].ok else None
        poly_trades_all = canonicalize_trades(
            enrich_with_metadata(poly_trades_raw, metadata)
        )

        binance_book = canonicalize_binance_book(tables)
        binance_trades_raw = tables["binance_trades"].frame if tables["binance_trades"].ok else None
        binance_trades = canonicalize_trades(binance_trades_raw)

        # Get market IDs
        print(f"    Discovering markets...")
        market_ids = get_market_ids(sampled_path)
        print(f"    {len(market_ids)} markets found")

        # Free raw tables dict (canonicalized versions already extracted)
        del tables
        gc.collect()

        date_dir = parts_dir / f"date={date}"
        date_dir.mkdir(parents=True, exist_ok=True)
        date_rows = 0

        for market_id in tqdm(market_ids, desc=f"  Build {date}", leave=False, ncols=100):
            # Load only this market's data
            market_book = load_market_sampled_book(sampled_path, market_id)
            if market_book.is_empty():
                continue

            # Enrich with metadata
            market_book = enrich_with_metadata(market_book, metadata)
            book = canonicalize_book(market_book)
            book = filter_book_scope(book, config)
            if book.is_empty():
                del market_book, book
                continue

            # Filter trades for this market
            if not poly_trades_all.is_empty() and "market_id" in poly_trades_all.columns:
                market_trades = poly_trades_all.filter(pl.col("market_id") == market_id)
            else:
                market_trades = poly_trades_all

            try:
                frame = build_feature_frame(
                    book, market_trades, binance_book, binance_trades,
                    None, config,
                )
            except Exception as exc:
                print(f"      market {market_id[:16]}...: FAILED ({exc})")
                del market_book, book, market_trades
                continue

            if frame.is_empty():
                del market_book, book, market_trades, frame
                continue

            frame = frame.with_columns(pl.lit(date).alias("date"))
            part_path = date_dir / f"market_id={safe_part_name(market_id)}.parquet"
            frame.write_parquet(str(part_path))
            date_rows += frame.height

            if example_frame is None:
                example_frame = frame.head(500)

            del market_book, book, market_trades, frame

        # Free date-level data
        del metadata, poly_trades_all, binance_book, binance_trades
        gc.collect()

        elapsed = time.time() - t0
        print(f"    {date}: {date_rows:,} rows in {elapsed:.0f}s")
        total_rows += date_rows

    print(f"\n  Total partitioned: {total_rows:,} rows")
    if example_frame is not None:
        feature_cols = infer_feature_columns(example_frame)
        print(f"  Features: {len(feature_cols)}")

    meta = {"total_rows": total_rows, "dates": dates}
    with open(parts_dir / "_build_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Phase B: Load partitioned data and train
# ---------------------------------------------------------------------------

def discover_features_from_parts(parts_dir: Path) -> list[str]:
    for date_dir in sorted(parts_dir.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith("date="):
            continue
        files = sorted(date_dir.glob("*.parquet"))
        if files:
            df = pl.read_parquet(str(files[0]), n_rows=2000)
            return infer_feature_columns(df)
    raise SystemExit(f"No parquet files found in {parts_dir}")


def load_date_raw(parts_dir: Path, date: str, need_cols: list[str]) -> pl.DataFrame | None:
    d = parts_dir / f"date={date}"
    if not d.exists():
        return None
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None

    pieces = []
    for f in tqdm(files, desc=f"  Load {date}", leave=False, ncols=100):
        available = set(pl.read_parquet_schema(str(f)).names())
        cols = [c for c in need_cols if c in available]
        if cols:
            pieces.append(pl.read_parquet(str(f), columns=cols))
    if not pieces:
        return None
    return pl.concat(pieces, how="diagonal_relaxed").sort("recv_ns")


def load_date_to_numpy(parts_dir: Path, date: str, feature_cols: list[str],
                       purge_ns: int = 0, purge_head: bool = False,
                       purge_tail: bool = False):
    """Load a single date directly to numpy arrays, file by file.

    Converts each file to numpy immediately and consolidates periodically
    to bound memory usage. Returns (X, y_cls, y_reg).
    """
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

    # Apply purge on recv_ns timestamps
    if purge_ns > 0:
        if purge_tail:
            max_ns = recv_all.max()
            mask = recv_all < max_ns - purge_ns
            X_all, ycls_all, yreg_all = X_all[mask], ycls_all[mask], yreg_all[mask]
            recv_all = recv_all[mask]
            print(f"    Purge tail: kept {X_all.shape[0]:,}")
        if purge_head:
            min_ns = recv_all.min()
            mask = recv_all >= min_ns + purge_ns
            X_all, ycls_all, yreg_all = X_all[mask], ycls_all[mask], yreg_all[mask]
            print(f"    Embargo head: kept {X_all.shape[0]:,}")

    del recv_all
    gc.collect()
    return X_all, ycls_all, yreg_all


def load_date_balanced_numpy(parts_dir: Path, date: str, feature_cols: list[str],
                              seed: int, max_rows: int):
    """Load a date with balanced sampling, returning numpy arrays directly.

    Converts each file's balanced sample to numpy immediately, consolidating
    periodically to bound memory usage.
    Returns (X, y_cls, y_reg) numpy arrays or (None, None, None).
    """
    d = parts_dir / f"date={date}"
    if not d.exists():
        return None, None, None
    files = sorted(d.glob("*.parquet"))
    if not files:
        return None, None, None

    label_cols = ["y_two_leg_entry_binary_10s", "first_unwind_profit_proxy_10s"]
    need_cols = list(set(feature_cols + label_cols))

    n_feat = len(feature_cols)
    # Pre-allocate with generous initial capacity, grow as needed
    X_all = np.empty((0, n_feat), np.float32)
    ycls_all = np.empty(0, np.float32)
    yreg_all = np.empty(0, np.float32)

    # Temporary buffers consolidated every CONSOLIDATE_EVERY files
    CONSOLIDATE_EVERY = 25
    X_buf, ycls_buf, yreg_buf = [], [], []

    total_raw = 0
    total_sampled = 0
    rng = np.random.default_rng(seed)

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
        # Early exit: stop loading once we have enough rows
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

        # Balanced sample within this piece
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

        # Periodic consolidation to bound memory
        if (fi + 1) % CONSOLIDATE_EVERY == 0:
            consolidate()

    # Final consolidation
    consolidate()

    if X_all.shape[0] == 0:
        return None, None, None

    print(f"    {date}: {total_raw:,} raw -> {total_sampled:,} balanced")

    # If still too many, subsample
    if X_all.shape[0] > max_rows:
        idx = np.sort(rng.choice(X_all.shape[0], size=max_rows, replace=False))
        X_all, ycls_all, yreg_all = X_all[idx], ycls_all[idx], yreg_all[idx]
        print(f"    {date}: capped to {max_rows:,}")

    return X_all, ycls_all, yreg_all


def balanced_sample(df: pl.DataFrame, seed: int, max_rows: int, label: str = "") -> pl.DataFrame:
    if df.is_empty() or "y_two_leg_entry_binary_10s" not in df.columns:
        return df

    y = df["y_two_leg_entry_binary_10s"].to_numpy()
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y != 1)[0]
    n_pos = len(pos_idx)
    n_neg_sample = min(n_pos, len(neg_idx))
    rng = np.random.default_rng(seed)
    sampled_neg = rng.choice(neg_idx, size=n_neg_sample, replace=False)
    all_idx = np.sort(np.concatenate([pos_idx, sampled_neg]))
    if len(all_idx) > max_rows:
        all_idx = np.sort(rng.choice(all_idx, size=max_rows, replace=False))

    df_sampled = df[all_idx.tolist()]
    print(f"    {label}: {df.height:,} -> {len(all_idx):,} rows (pos={n_pos:,}, neg={n_neg_sample:,})")
    return df_sampled


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


def apply_purge(df: pl.DataFrame, purge_ns: int, side: str = "tail") -> pl.DataFrame:
    if df.is_empty():
        return df
    if side == "tail":
        max_ns = df["recv_ns"].max()
        before = df.height
        df = df.filter(pl.col("recv_ns") < max_ns - purge_ns)
        print(f"    Purge tail: -{before - df.height:,}")
    else:
        min_ns = df["recv_ns"].min()
        before = df.height
        df = df.filter(pl.col("recv_ns") >= min_ns + purge_ns)
        print(f"    Embargo head: -{before - df.height:,}")
    return df


def _mask_nan(y, *arrays):
    m = ~np.isnan(y)
    if m.all():
        return (y,) + arrays
    return (y[m],) + tuple(a[m] for a in arrays)


def train_model(model_cls, is_classifier, X, y, seed, n_estimators, max_depth):
    task = "CLF" if is_classifier else "REG"
    print(f"  Training {model_cls.__name__} ({task}) on {X.shape[0]:,} rows...")
    y_clean, X_clean = _mask_nan(y, X)[:2]
    model = model_cls(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, n_jobs=-1, verbose=1,
    )
    if is_classifier:
        model.set_params(class_weight="balanced")
    t0 = time.time()
    model.fit(X_clean, y_clean)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    del X_clean, y_clean
    gc.collect()
    return model


def predict_all(models, X):
    preds = {}
    for name, model in tqdm(models.items(), desc="Predict", leave=False, ncols=100):
        preds[name] = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)
    return preds


def policy_selection(fill_models, unwind_models,
                     val_preds, val_y_cls, val_y_reg,
                     test_preds, test_y_cls, test_y_reg, output_dir):
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
                print(f"  {fill_name} x {unwind_name}: no viable config")
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
        print(f"\n  Best: {results[0]['combo']} -> val avgP={results[0]['val']['avg_profit']:.4f} "
              f"test avgP={results[0]['test']['avg_profit']:.4f}")
    return results


def main():
    t_start = time.time()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    purge_ns = args.purge_seconds * NS_PER_SECOND

    config = DatasetConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        periods=[args.period],
        sample_mode="event-driven",
        book_event_source="book",
    )

    parts_dir = args.parts_dir or (args.output_dir / "alpha_dataset_parts")

    print("=" * 60)
    print("RF/ET Training Pipeline (Windows)")
    print("=" * 60)
    print(f"Parts dir:  {parts_dir}")
    print(f"Train:      {TRAIN_DATES}")
    print(f"Val:        {VAL_DATE}")
    print(f"Test:       {TEST_DATE}")

    # Phase A
    if not args.skip_build and not (parts_dir / "_build_metadata.json").exists():
        print(f"\n{'=' * 60}")
        print("Phase A: Build partitioned dataset (per-market)")
        print(f"{'=' * 60}")
        build_partitioned_dataset(args.data_dir, parts_dir, ALL_DATES, config)
    else:
        print(f"\n  Skipping Phase A (using existing parts)")

    # Phase B
    print(f"\n{'=' * 60}")
    print("Phase B: Load & train RF/ET")
    print(f"{'=' * 60}")

    print("  Discovering features...")
    feature_cols = discover_features_from_parts(parts_dir)
    print(f"  Features: {len(feature_cols)}")

    # Load train (balanced incrementally per file -> numpy arrays)
    print("\n  Loading train (balanced per file -> numpy)...")
    X_train = np.empty((0, len(feature_cols)), np.float32)
    y_train_cls = np.empty(0, np.float32)
    y_train_reg = np.empty(0, np.float32)
    per_date_max = args.max_train_rows // len(TRAIN_DATES)
    for i, date in enumerate(TRAIN_DATES):
        X_d, ycls_d, yreg_d = load_date_balanced_numpy(
            parts_dir, date, feature_cols, args.seed + i, per_date_max)
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

    print(f"  Train: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")

    # Train
    print(f"\n--- Training models ---")
    models = {}
    for name, cls, is_clf in [
        ("rf_classifier", RandomForestClassifier, True),
        ("et_classifier", ExtraTreesClassifier, True),
        ("rf_regressor", RandomForestRegressor, False),
        ("et_regressor", ExtraTreesRegressor, False),
    ]:
        y = y_train_cls if is_clf else y_train_reg
        models[name] = train_model(cls, is_clf, X_train, y, args.seed,
                                    args.n_estimators, args.max_depth)

    del X_train, y_train_cls, y_train_reg
    gc.collect()

    # Save models
    print("\nSaving models...")
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

    # Val/test - load directly to numpy (no polars concat needed for single dates)
    print(f"\n  Loading val ({VAL_DATE})...")
    X_val, y_val_cls, y_val_reg = load_date_to_numpy(
        parts_dir, VAL_DATE, feature_cols, purge_ns, purge_head=True, purge_tail=True)

    print(f"  Loading test ({TEST_DATE})...")
    X_test, y_test_cls, y_test_reg = load_date_to_numpy(
        parts_dir, TEST_DATE, feature_cols, purge_ns, purge_head=True, purge_tail=False)

    print(f"  Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")

    # Predict
    print(f"\nPredicting val...")
    val_preds = predict_all(models, X_val)
    print(f"Predicting test...")
    test_preds = predict_all(models, X_test)
    del X_val, X_test
    gc.collect()

    # Policy selection
    results = policy_selection(
        [n for n in models if "classifier" in n],
        [n for n in models if "regressor" in n],
        val_preds, y_val_cls, y_val_reg,
        test_preds, y_test_cls, y_test_reg,
        args.output_dir / "execution_policy")

    elapsed = time.time() - t_start
    summary = {
        "train_dates": TRAIN_DATES, "val_date": VAL_DATE, "test_date": TEST_DATE,
        "purge_seconds": args.purge_seconds, "features": len(feature_cols),
        "n_estimators": args.n_estimators, "max_depth": args.max_depth,
        "period": args.period, "sampling": "balanced (pos + equal neg)",
        "elapsed_seconds": round(elapsed, 1),
        "best_combo": results[0] if results else None,
    }
    with open(args.output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {elapsed / 60:.1f} min.")


if __name__ == "__main__":
    main()
