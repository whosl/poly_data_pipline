"""Chronological split helpers."""

from __future__ import annotations

import polars as pl


def chronological_split(
    df: pl.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    time_col: str = "recv_ns",
) -> dict[str, pl.DataFrame]:
    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/validation/test fractions must sum to 1")
    if df.is_empty():
        return {"train": df, "validation": df, "test": df}

    sorted_df = df.sort(time_col)
    n = sorted_df.height
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * validation_fraction)
    return {
        "train": sorted_df.slice(0, train_end),
        "validation": sorted_df.slice(train_end, val_end - train_end),
        "test": sorted_df.slice(val_end, n - val_end),
    }


def split_ranges(splits: dict[str, pl.DataFrame], time_col: str = "recv_ns") -> dict[str, dict[str, int | None]]:
    ranges: dict[str, dict[str, int | None]] = {}
    for name, frame in splits.items():
        if frame.is_empty() or time_col not in frame.columns:
            ranges[name] = {"rows": 0, "start_ns": None, "end_ns": None}
        else:
            ranges[name] = {
                "rows": frame.height,
                "start_ns": int(frame[time_col].min()),
                "end_ns": int(frame[time_col].max()),
            }
    return ranges


def walk_forward_windows(
    df: pl.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    step_fraction: float = 0.10,
    time_col: str = "recv_ns",
) -> list[dict[str, pl.DataFrame]]:
    if df.is_empty():
        return []
    sorted_df = df.sort(time_col)
    n = sorted_df.height
    train_n = max(1, int(n * train_fraction))
    val_n = max(1, int(n * validation_fraction))
    step_n = max(1, int(n * step_fraction))
    windows: list[dict[str, pl.DataFrame]] = []
    start = 0
    while start + train_n + val_n <= n:
        windows.append(
            {
                "train": sorted_df.slice(start, train_n),
                "validation": sorted_df.slice(start + train_n, val_n),
            }
        )
        start += step_n
    return windows

