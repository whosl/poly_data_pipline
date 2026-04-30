#!/usr/bin/env python3
"""Add y_first_leg_fill column to existing partitioned parquet files.

Forward-looking label: scans ALL future snapshots within the validation
window. A fill fails if ANY future best_ask exceeds entry + buffer within
the window. Models end-to-end bot latency (signal + network + chain order).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

PARTS_DIR = Path("artifacts/training_rf_et_win/alpha_dataset_parts")
VALIDATION_MS = 350
PRICE_BUFFER = 0.01


def add_label(df: pl.DataFrame) -> pl.DataFrame:
    if "y_first_leg_fill" in df.columns:
        # Remove old column to recompute
        df = df.drop("y_first_leg_fill")
    if df.is_empty() or "best_ask" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Int8).alias("y_first_leg_fill"))

    validation_ns = VALIDATION_MS * 1_000_000
    df = df.sort(["asset_id", "recv_ns"])

    asset_ids = df["asset_id"].to_numpy()
    recv_ns = df["recv_ns"].to_numpy().astype(np.int64)
    best_ask = df["best_ask"].to_numpy().astype(np.float64)

    n = df.height
    labels = np.ones(n, dtype=np.int8)

    unique_assets = np.unique(asset_ids)
    for asset in unique_assets:
        mask = asset_ids == asset
        idx = np.where(mask)[0]
        a_ns = recv_ns[idx]
        a_ask = best_ask[idx]
        m = len(idx)

        j = 0
        for i in range(m):
            entry_ask = a_ask[i]
            threshold = entry_ask + PRICE_BUFFER
            end_time = a_ns[i] + validation_ns

            if j < i + 1:
                j = i + 1
            while j < m and a_ns[j] <= end_time:
                if a_ask[j] > threshold:
                    labels[idx[i]] = 0
                    break
                j += 1
            if j > i + 1:
                j = i + 1

    ask_null = np.isnan(best_ask)
    labels[ask_null] = -1

    return df.with_columns(
        pl.when(pl.Series(labels) == -1)
        .then(None)
        .otherwise(pl.Series(labels))
        .cast(pl.Int8)
        .alias("y_first_leg_fill")
    )


def main():
    total_files = 0
    total_rows = 0
    for date_dir in sorted(PARTS_DIR.iterdir()):
        if not date_dir.is_dir() or not date_dir.name.startswith("date="):
            continue
        files = sorted(date_dir.glob("*.parquet"))
        for f in tqdm(files, desc=date_dir.name, leave=False, ncols=80):
            df = pl.read_parquet(str(f))
            df = add_label(df)
            df.write_parquet(str(f))
            total_files += 1
            total_rows += df.height
        date_files = len(files)
        print(f"  {date_dir.name}: {date_files} files processed")

    print(f"\nDone: {total_files} files updated, {total_rows:,} total rows")


if __name__ == "__main__":
    main()
