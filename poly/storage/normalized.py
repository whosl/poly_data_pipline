"""Buffered Parquet writer and schema definitions."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import structlog
from pathlib import Path
from datetime import datetime, timezone

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Arrow schemas
# ---------------------------------------------------------------------------

L2_BOOK_SCHEMA = pa.schema([
    ("source", pa.string()),
    ("asset_id", pa.string()),
    ("market", pa.string()),
    ("recv_ns", pa.int64()),
    ("exchange_ts", pa.int64()),
    ("best_bid", pa.float64()),
    ("best_ask", pa.float64()),
    ("spread", pa.float64()),
    ("midpoint", pa.float64()),
    ("microprice", pa.float64()),
    ("imbalance", pa.float64()),
    ("total_bid_levels", pa.int32()),
    ("total_ask_levels", pa.int32()),
])

TRADE_SCHEMA = pa.schema([
    ("source", pa.string()),
    ("asset_id", pa.string()),
    ("market", pa.string()),
    ("recv_ns", pa.int64()),
    ("exchange_ts", pa.int64()),
    ("price", pa.float64()),
    ("size", pa.float64()),
    ("side", pa.string()),
    ("fee_rate_bps", pa.float64()),
])

BEST_BID_ASK_SCHEMA = pa.schema([
    ("source", pa.string()),
    ("asset_id", pa.string()),
    ("recv_ns", pa.int64()),
    ("exchange_ts", pa.int64()),
    ("best_bid", pa.float64()),
    ("best_ask", pa.float64()),
    ("spread", pa.float64()),
])

ORDER_SCHEMA = pa.schema([
    ("order_id", pa.string()),
    ("market", pa.string()),
    ("asset_id", pa.string()),
    ("side", pa.string()),
    ("price", pa.float64()),
    ("original_size", pa.float64()),
    ("size_matched", pa.float64()),
    ("order_type", pa.string()),
    ("recv_ns", pa.int64()),
    ("exchange_ts", pa.int64()),
])

USER_TRADE_SCHEMA = pa.schema([
    ("trade_id", pa.string()),
    ("market", pa.string()),
    ("asset_id", pa.string()),
    ("side", pa.string()),
    ("price", pa.float64()),
    ("size", pa.float64()),
    ("status", pa.string()),
    ("recv_ns", pa.int64()),
    ("exchange_ts", pa.int64()),
])


class ParquetWriter:
    """Buffered Parquet writer with date partitioning."""

    def __init__(self, schema: pa.Schema, output_dir: Path, type_name: str,
                 buffer_size: int = 1000) -> None:
        self.schema = schema
        self.output_dir = output_dir
        self.type_name = type_name
        self.buffer_size = buffer_size
        self._buffers: dict[str, list] = {f.name: [] for f in schema}
        self._writer: pq.ParquetWriter | None = None
        self._current_date: str = ""
        self._total_rows: int = 0

    def _date_from_ns(self, recv_ns: int) -> str:
        dt = datetime.fromtimestamp(recv_ns / 1e9, tz=timezone.utc)
        return dt.strftime("%Y%m%d")

    def _output_path(self, date_str: str) -> Path:
        d = self.output_dir / "normalized" / date_str
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.type_name}.parquet"

    def append(self, row: dict) -> None:
        recv_ns = row.get("recv_ns", 0)
        date_str = self._date_from_ns(recv_ns)

        if date_str != self._current_date:
            if self._current_date and any(self._buffers[f] for f in self._buffers):
                self.flush()
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self._current_date = date_str

        for field in self.schema:
            val = row.get(field.name)
            if val is None:
                val = None
            self._buffers[field.name].append(val)

        if sum(1 for v in self._buffers[self.schema.field(0).name]) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not any(self._buffers[f] for f in self._buffers):
            return
        if not self._current_date:
            return

        arrays = []
        for field in self.schema:
            col = self._buffers[field.name]
            arrays.append(pa.array(col, type=field.type))

        table = pa.Table.from_arrays(arrays, schema=self.schema)

        if self._writer is None:
            path = self._output_path(self._current_date)
            self._writer = pq.ParquetWriter(str(path), self.schema)

        self._writer.write_table(table)
        self._total_rows += table.num_rows

        for f in self._buffers:
            self._buffers[f] = []

        logger.debug("parquet_flush", type=self.type_name,
                     date=self._current_date, total=self._total_rows)

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        logger.info("parquet_writer_closed", type=self.type_name,
                    total=self._total_rows)

    @property
    def total_rows(self) -> int:
        return self._total_rows
