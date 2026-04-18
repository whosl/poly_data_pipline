#!/usr/bin/env python3
"""Fetch Polymarket Up/Down market metadata into normalized parquet artifacts."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.metadata.polymarket import MetadataFetchConfig, fetch_metadata, write_metadata_by_date
from poly.training.io import discover_dates

structlog.configure(processors=[structlog.processors.add_log_level, structlog.dev.ConsoleRenderer()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dates", nargs="*", default=None, help="YYYYMMDD dates. Defaults to discovered normalized/research dates.")
    parser.add_argument("--gamma-url", default="https://gamma-api.polymarket.com")
    parser.add_argument("--clob-url", default="https://clob.polymarket.com")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=12.0)
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    dates = discover_dates(args.data_dir, args.dates)
    if not dates:
        raise SystemExit(f"No dates found under {args.data_dir}/normalized or {args.data_dir}/research")

    config = MetadataFetchConfig(
        data_dir=args.data_dir,
        gamma_url=args.gamma_url,
        clob_url=args.clob_url,
        dates=tuple(dates),
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
    )
    frame = await fetch_metadata(config)
    paths = write_metadata_by_date(frame, args.data_dir, dates)
    print(f"rows={frame.height}")
    for path in paths:
        print(f"metadata={path}")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
