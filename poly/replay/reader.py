"""Data replay utilities."""

from __future__ import annotations

import asyncio
import gzip
import structlog
import orjson
import polars as pl
from pathlib import Path
from typing import Callable

logger = structlog.get_logger()


class DataReader:
    """Read raw and normalized data for replay."""

    def read_raw(self, data_dir: Path, source: str, date: str):
        """Stream raw messages from JSONL, ordered by recv_ns."""
        import zlib
        pattern = f"{source}_*.jsonl.gz"
        raw_dir = data_dir / "raw_feed" / date
        for path in sorted(raw_dir.glob(pattern)):
            try:
                with gzip.open(path, "rb") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            outer = orjson.loads(line)
                        except Exception:
                            continue
                        recv_ns = outer.get("recv_ns", 0)
                        raw = outer.get("raw", {})
                        if isinstance(raw, str):
                            try:
                                raw = orjson.loads(raw)
                            except Exception:
                                continue
                        yield recv_ns, raw
            except (EOFError, zlib.error) as e:
                logger.warning("replay_truncated_gzip", path=str(path), error=str(e))

    def read_normalized(self, data_dir: Path, data_type: str, date: str) -> pl.DataFrame:
        """Read normalized Parquet file."""
        path = data_dir / "normalized" / date / f"{data_type}.parquet"
        if path.exists():
            return pl.read_parquet(str(path))
        return pl.DataFrame()

    def read_date_range(self, data_dir: Path, data_type: str,
                        start_date: str, end_date: str) -> pl.DataFrame:
        """Read and concat multiple date parquets."""
        frames = []
        norm_dir = data_dir / "normalized"
        for d in sorted(norm_dir.iterdir()):
            if not d.is_dir():
                continue
            date_str = d.name
            if start_date <= date_str <= end_date:
                path = d / f"{data_type}.parquet"
                if path.exists():
                    frames.append(pl.read_parquet(str(path)))
        if frames:
            return pl.concat(frames)
        return pl.DataFrame()


class ReplayPlayer:
    """Replay raw data at original or adjusted speed."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.reader = DataReader()

    async def replay(
        self,
        source: str,
        date: str,
        speed: float = 1.0,
        callback: Callable[[int, dict], None] | None = None,
    ) -> None:
        """Replay messages at scaled original speed."""
        messages = list(self.reader.read_raw(self.data_dir, source, date))
        if not messages:
            logger.warning("replay_no_data", source=source, date=date)
            return

        logger.info("replay_start", source=source, date=date,
                    messages=len(messages), speed=speed)

        prev_ns = messages[0][0]
        for recv_ns, msg in messages:
            if speed > 0:
                delay_ns = recv_ns - prev_ns
                delay_s = (delay_ns / 1e9) / speed
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
            prev_ns = recv_ns

            if callback:
                callback(recv_ns, msg)
            else:
                print(f"[{recv_ns}] {orjson.dumps(msg).decode()}")

        logger.info("replay_done", source=source, date=date)
