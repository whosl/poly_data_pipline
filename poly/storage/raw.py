"""Raw JSONL writer with date-partitioned gzip files."""

from __future__ import annotations

import asyncio
import gzip
import os
import structlog
import orjson
from datetime import datetime, timezone
from pathlib import Path

logger = structlog.get_logger()


class RawWriter:
    """Async raw message writer — never drops a message."""

    def __init__(self, data_dir: Path, source: str, channel: str,
                 flush_interval: float = 5.0) -> None:
        self.data_dir = data_dir
        self.source = source
        self.channel = channel
        self.flush_interval = flush_interval
        self._buffer: list[bytes] = []
        self._current_date: str = ""
        self._file_handle: gzip.GzipFile | None = None
        self._lock = asyncio.Lock()
        self._total_written: int = 0

    def _date_from_ns(self, recv_ns: int) -> str:
        dt = datetime.fromtimestamp(recv_ns / 1e9, tz=timezone.utc)
        return dt.strftime("%Y%m%d")

    def _file_path(self, date_str: str) -> Path:
        d = self.data_dir / "raw_feed" / date_str
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.source}_{self.channel}.jsonl.gz"

    async def write(self, raw_json: bytes, recv_ns: int) -> None:
        date_str = self._date_from_ns(recv_ns)
        line = orjson.dumps({"recv_ns": recv_ns, "raw": orjson.loads(raw_json)})

        async with self._lock:
            if date_str != self._current_date:
                await self._rotate(date_str)
            self._buffer.append(line)
            if len(self._buffer) >= 10000:
                await self._flush_locked()

    async def _rotate(self, new_date: str) -> None:
        if self._file_handle is not None:
            await self._flush_locked()
            self._file_handle.close()
            self._file_handle = None
        self._current_date = new_date

    async def _flush_locked(self) -> None:
        if not self._buffer:
            return
        if self._file_handle is None:
            path = self._file_path(self._current_date)
            self._file_handle = gzip.open(path, "ab")
        self._file_handle.writelines(b + b"\n" for b in self._buffer)
        self._file_handle.flush()
        os.fsync(self._file_handle.fileno())
        self._total_written += len(self._buffer)
        self._buffer.clear()
        logger.debug("raw_flush", source=self.source, channel=self.channel,
                     total=self._total_written)

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def flush_loop(self) -> None:
        """Background task: periodic flush."""
        while True:
            await asyncio.sleep(self.flush_interval)
            await self.flush()

    async def close(self) -> None:
        async with self._lock:
            await self._flush_locked()
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None
        logger.info("raw_writer_closed", source=self.source, channel=self.channel,
                    total=self._total_written)

    @property
    def total_written(self) -> int:
        return self._total_written
