"""Research label generator — markout, imbalance, volatility buckets."""

from __future__ import annotations

import structlog
import polars as pl
from pathlib import Path

logger = structlog.get_logger()

DEFAULT_HORIZONS_NS = [
    1_000_000_000,   # 1s
    3_000_000_000,   # 3s
    5_000_000_000,   # 5s
    10_000_000_000,  # 10s
]


class LabelGenerator:
    """Generate markout and feature bucket labels for research."""

    def generate_markout_labels(
        self,
        l2_book: pl.DataFrame,
        trades: pl.DataFrame,
        horizons_ns: list[int] | None = None,
    ) -> pl.DataFrame:
        """For each trade, compute future midprice at given horizons."""
        if horizons_ns is None:
            horizons_ns = DEFAULT_HORIZONS_NS

        if l2_book.is_empty() or trades.is_empty():
            return trades

        book_sorted = l2_book.sort("recv_ns")
        trades_sorted = trades.sort("recv_ns")

        result = trades_sorted.clone()
        for h_ns in horizons_ns:
            h_sec = h_ns / 1e9
            col_mid = f"future_mid_{h_sec:.0f}s"
            col_markout = f"markout_{h_sec:.0f}s"
            col_markout_bps = f"markout_{h_sec:.0f}s_bps"

            # Create a shifted book with recv_ns + horizon
            future_book = book_sorted.with_columns(
                (pl.col("recv_ns") + pl.lit(h_ns)).alias("future_recv_ns")
            ).select(["future_recv_ns", "midpoint"]).rename(
                {"future_recv_ns": "recv_ns", "midpoint": col_mid}
            )

            # Asof join: for each trade at T, find nearest book at T+H
            result = result.join_asof(
                future_book,
                on="recv_ns",
                strategy="forward",
                tolerance=h_ns,
            )

            # Compute markout
            result = result.with_columns([
                ((pl.col(col_mid) - pl.col("price")) / pl.col("price")).alias(col_markout),
                (((pl.col(col_mid) - pl.col("price")) / pl.col("price")) * 10000).alias(col_markout_bps),
            ])

        return result

    def generate_imbalance_buckets(
        self,
        l2_book: pl.DataFrame,
        n_buckets: int = 10,
    ) -> pl.DataFrame:
        """Add imbalance/spread/price decile buckets."""
        result = l2_book.clone()
        labels = [f"q{i}" for i in range(n_buckets)]

        def safe_qcut(df: pl.DataFrame, col: str, alias: str) -> pl.DataFrame:
            if col not in df.columns:
                return df
            n_unique = df[col].n_unique()
            if n_unique <= 1:
                return df.with_columns(pl.lit("q0").alias(alias))
            return df.with_columns(
                pl.col(col).qcut(n_buckets, labels=labels, allow_duplicates=True).alias(alias)
            )

        result = safe_qcut(result, "imbalance", "imbalance_bucket")
        result = safe_qcut(result, "spread", "spread_bucket")
        result = safe_qcut(result, "midpoint", "price_bucket")
        return result

    def generate_volatility_bucket(
        self,
        l2_book: pl.DataFrame,
        window_ns: int = 60_000_000_000,
    ) -> pl.DataFrame:
        """Add rolling midprice volatility bucket."""
        if l2_book.is_empty() or "midpoint" not in l2_book.columns:
            return l2_book

        result = l2_book.sort("recv_ns")
        # Rolling std of midpoint returns over a window
        result = result.with_columns(
            pl.col("midpoint").pct_change().rolling_std(
                window_size=100,  # approximate 60s in ticks
            ).alias("vol_60s")
        ).with_columns(
            pl.col("vol_60s").qcut(10, labels=[f"v{i}" for i in range(10)], allow_duplicates=True).alias("vol_bucket")
        )
        return result

    def run(self, data_dir: Path, date: str) -> None:
        """Generate all labels for a given date."""
        norm_dir = data_dir / "normalized" / date
        research_dir = data_dir / "research" / date
        research_dir.mkdir(parents=True, exist_ok=True)

        # Polymarket labels
        poly_book_path = norm_dir / "poly_l2_book.parquet"
        poly_trades_path = norm_dir / "poly_trades.parquet"

        if poly_book_path.exists() and poly_trades_path.exists():
            l2_book = pl.read_parquet(str(poly_book_path))
            trades = pl.read_parquet(str(poly_trades_path))

            # Markout labels
            labeled_trades = self.generate_markout_labels(l2_book, trades)
            labeled_trades.write_parquet(str(research_dir / "poly_markout_labels.parquet"))
            logger.info("poly_markout_labels", rows=labeled_trades.height)

            # Enriched book
            enriched = self.generate_imbalance_buckets(l2_book)
            enriched = self.generate_volatility_bucket(enriched)
            enriched.write_parquet(str(research_dir / "poly_enriched_book.parquet"))
            logger.info("poly_enriched_book", rows=enriched.height)

        # Binance labels
        binance_book_path = norm_dir / "binance_l2_book.parquet"
        binance_trades_path = norm_dir / "binance_trades.parquet"

        if binance_book_path.exists() and binance_trades_path.exists():
            l2_book = pl.read_parquet(str(binance_book_path))
            trades = pl.read_parquet(str(binance_trades_path))

            labeled_trades = self.generate_markout_labels(l2_book, trades)
            labeled_trades.write_parquet(str(research_dir / "binance_markout_labels.parquet"))
            logger.info("binance_markout_labels", rows=labeled_trades.height)

        logger.info("labels_done", date=date)
