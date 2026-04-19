"""Chronological split helpers."""

from __future__ import annotations

import polars as pl


def chronological_split(
    df: pl.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    time_col: str = "recv_ns",
    purge_ms: int = 0,
    embargo_ms: int = 0,
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
    splits = {
        "train": sorted_df.slice(0, train_end),
        "validation": sorted_df.slice(train_end, val_end - train_end),
        "test": sorted_df.slice(val_end, n - val_end),
    }
    if purge_ms <= 0 and embargo_ms <= 0:
        return splits
    return apply_boundary_purge_embargo(
        splits,
        time_col=time_col,
        purge_ms=purge_ms,
        embargo_ms=embargo_ms,
    )


def apply_boundary_purge_embargo(
    splits: dict[str, pl.DataFrame],
    time_col: str = "recv_ns",
    purge_ms: int = 0,
    embargo_ms: int = 0,
) -> dict[str, pl.DataFrame]:
    """Drop rows near chronological split boundaries.

    Purge removes rows from the earlier split whose forward-looking labels can
    overlap the next split. Embargo removes rows from the later split just after
    the boundary so adjacent snapshots do not act as near-duplicates.
    """

    if purge_ms <= 0 and embargo_ms <= 0:
        return splits
    if any(time_col not in frame.columns for frame in splits.values() if not frame.is_empty()):
        return splits

    purge_ns = int(purge_ms * 1_000_000)
    embargo_ns = int(embargo_ms * 1_000_000)
    val_start = split_start_ns(splits.get("validation"), time_col)
    test_start = split_start_ns(splits.get("test"), time_col)

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    if val_start is not None:
        if purge_ns > 0:
            train = train.filter(pl.col(time_col) < val_start - purge_ns)
        if embargo_ns > 0:
            validation = validation.filter(pl.col(time_col) >= val_start + embargo_ns)

    if test_start is not None:
        if purge_ns > 0:
            validation = validation.filter(pl.col(time_col) < test_start - purge_ns)
        if embargo_ns > 0:
            test = test.filter(pl.col(time_col) >= test_start + embargo_ns)

    return {"train": train, "validation": validation, "test": test}


def split_start_ns(df: pl.DataFrame | None, time_col: str) -> int | None:
    if df is None or df.is_empty() or time_col not in df.columns:
        return None
    return int(df[time_col].min())


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
