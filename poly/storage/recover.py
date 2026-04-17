"""Recovery helpers for concatenated or truncated gzip JSONL files."""

from __future__ import annotations

import mmap
import zlib
from pathlib import Path
from typing import Iterator

import structlog

logger = structlog.get_logger()

GZIP_MAGIC = b"\x1f\x8b\x08"
CHUNK_SIZE = 1024 * 1024


class _MemberIterator:
    def __init__(self, mm: mmap.mmap, offset: int, file_size: int) -> None:
        self.mm = mm
        self.offset = offset
        self.file_size = file_size
        self.next_offset = offset + 1
        self.complete = False

    def lines(self) -> Iterator[bytes]:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        pos = self.offset
        pending = b""

        try:
            while pos < self.file_size:
                end = min(pos + CHUNK_SIZE, self.file_size)
                chunk = self.mm[pos:end]
                pos = end
                data = decompressor.decompress(chunk)
                if data:
                    parts = data.split(b"\n")
                    parts[0] = pending + parts[0]
                    for line in parts[:-1]:
                        if line.strip():
                            yield line.strip()
                    pending = parts[-1]

                if decompressor.eof:
                    consumed = pos - len(decompressor.unused_data)
                    self.next_offset = consumed
                    if pending.strip():
                        yield pending.strip()
                    self.complete = True
                    return
        except zlib.error:
            self.next_offset = pos
            self.complete = False
            return

        self.next_offset = pos
        self.complete = False


def iter_recovered_gzip_jsonl_lines(path: Path) -> Iterator[bytes]:
    """Yield lines and continue after corrupt gzip members."""
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            file_size = len(mm)
            offset = mm.find(GZIP_MAGIC, 0)
            member_index = 0
            recovered_members = 0
            skipped_members = 0

            while 0 <= offset < file_size:
                member_index += 1
                member = _MemberIterator(mm, offset, file_size)
                yielded = 0
                for line in member.lines():
                    yielded += 1
                    yield line
                if member.complete:
                    recovered_members += 1
                    next_search = max(member.next_offset, offset + 1)
                else:
                    skipped_members += 1
                    next_search = offset + 1
                offset = mm.find(GZIP_MAGIC, next_search)

            logger.info(
                "gzip_jsonl_recovery_done",
                path=str(path),
                recovered_members=recovered_members,
                skipped_members=skipped_members,
            )
