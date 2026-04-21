"""Data discovery and robust parquet loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl
import structlog

logger = structlog.get_logger()


@dataclass
class TableLoadResult:
    name: str
    frame: pl.DataFrame | None
    path: Path | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.frame is not None and self.error is None


def discover_dates(data_dir: Path, requested: Iterable[str] | None = None) -> list[str]:
    if requested:
        return sorted({d for d in requested if d})
    roots = [data_dir / "normalized", data_dir / "research"]
    dates: set[str] = set()
    for root in roots:
        if root.exists():
            dates.update(p.name for p in root.iterdir() if p.is_dir())
    return sorted(dates)


def table_path(data_dir: Path, layer: str, date: str, table_name: str) -> Path:
    return data_dir / layer / date / f"{table_name}.parquet"


def read_parquet_safe(path: Path, table_name: str) -> TableLoadResult:
    if not path.exists():
        return TableLoadResult(table_name, None, path, "missing")
    try:
        frame = pl.read_parquet(str(path))
    except Exception as exc:  # pragma: no cover - exact parquet failures vary
        logger.warning("parquet_read_failed", table=table_name, path=str(path), error=str(exc))
        return TableLoadResult(table_name, None, path, f"{type(exc).__name__}: {exc}")
    return TableLoadResult(table_name, frame, path)


def load_date_tables(data_dir: Path, date: str) -> dict[str, TableLoadResult]:
    specs = {
        "poly_best_bid_ask": ("normalized", "poly_best_bid_ask"),
        "poly_trades": ("normalized", "poly_trades"),
        "binance_l2_book": ("normalized", "binance_l2_book"),
        "binance_best_bid_ask": ("normalized", "binance_best_bid_ask"),
        "binance_trades": ("normalized", "binance_trades"),
        "poly_market_metadata": ("normalized", "poly_market_metadata"),
        "poly_enriched_book": ("research", "poly_enriched_book"),
        "poly_markout_labels": ("research", "poly_markout_labels"),
        "binance_markout_labels": ("research", "binance_markout_labels"),
    }
    sampled_path = table_path(data_dir, "normalized", date, "poly_sampled_book")
    if sampled_path.exists():
        specs["poly_sampled_book"] = ("normalized", "poly_sampled_book")
    else:
        specs["poly_l2_book"] = ("normalized", "poly_l2_book")
    return {
        name: read_parquet_safe(table_path(data_dir, layer, date, table), name)
        for name, (layer, table) in specs.items()
    }


def schema_report(tables_by_date: dict[str, dict[str, TableLoadResult]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for date, tables in tables_by_date.items():
        for name, result in tables.items():
            if result.ok and result.frame is not None:
                rows.append(
                    {
                        "date": date,
                        "table": name,
                        "path": str(result.path),
                        "rows": result.frame.height,
                        "columns": {k: str(v) for k, v in result.frame.schema.items()},
                        "error": None,
                    }
                )
            else:
                rows.append(
                    {
                        "date": date,
                        "table": name,
                        "path": str(result.path) if result.path else None,
                        "rows": 0,
                        "columns": {},
                        "error": result.error,
                    }
                )
    return rows


def concat_non_empty(frames: list[pl.DataFrame]) -> pl.DataFrame:
    frames = [df for df in frames if df is not None and not df.is_empty()]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")
