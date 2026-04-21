"""Reusable feature generation for offline training and replay/backtests."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import polars as pl
import structlog

from poly.metadata.polymarket import enrich_with_metadata
from poly.training.config import DatasetConfig, dataclass_to_json_dict, save_json
from poly.training.io import TableLoadResult, concat_non_empty, load_date_tables, schema_report
from poly.training.labels import (
    add_alpha_labels,
    add_entry_net_edge,
    add_final_profit_labels,
    add_strategy_entry_labels,
    add_two_leg_maker_fill_labels,
    maker_fill_label_design,
)

logger = structlog.get_logger()

NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000
EXPIRY_RE = re.compile(r"(?P<symbol>btc|eth)-updown-(?P<period>5m|15m)-(?P<expiry>\d+)", re.I)
DEPTH_FEATURE_COLUMNS = [
    "depth_top1_imbalance",
    "depth_top3_imbalance",
    "depth_top5_imbalance",
    "depth_top10_imbalance",
    "depth_top20_imbalance",
    "cum_bid_depth_top1",
    "cum_ask_depth_top1",
    "cum_bid_depth_top3",
    "cum_ask_depth_top3",
    "cum_bid_depth_top5",
    "cum_ask_depth_top5",
    "cum_bid_depth_top10",
    "cum_ask_depth_top10",
    "cum_bid_depth_top20",
    "cum_ask_depth_top20",
    "bid_depth_slope_top10",
    "ask_depth_slope_top10",
    "bid_depth_slope_top20",
    "ask_depth_slope_top20",
    "near_touch_bid_notional_5",
    "near_touch_ask_notional_5",
    "near_touch_bid_notional_10",
    "near_touch_ask_notional_10",
    "near_touch_bid_notional_20",
    "near_touch_ask_notional_20",
]
BINANCE_DEPTH_FEATURE_COLUMNS = [f"binance_{col}" for col in DEPTH_FEATURE_COLUMNS]


@dataclass
class BuildResult:
    dataset: pl.DataFrame
    metadata: dict[str, object]


def build_training_dataset(config: DatasetConfig) -> BuildResult:
    tables_by_date = {date: load_date_tables(config.data_dir, date) for date in config.dates}
    table_report = schema_report(tables_by_date)

    date_frames: list[pl.DataFrame] = []
    quality: list[dict[str, object]] = []
    for date, tables in tables_by_date.items():
        try:
            frame = build_date_dataset(date, tables, config)
        except Exception as exc:
            logger.warning("date_dataset_failed", date=date, error=str(exc))
            quality.append({"date": date, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            continue
        if frame.is_empty():
            quality.append({"date": date, "status": "empty", "rows": 0})
        else:
            date_frames.append(frame.with_columns(pl.lit(date).alias("date")))
            quality.append({"date": date, "status": "ok", "rows": frame.height})

    dataset = concat_non_empty(date_frames)
    metadata = make_metadata(dataset, config, table_report, quality)
    return BuildResult(dataset=dataset, metadata=metadata)


def build_date_dataset(
    date: str,
    tables: dict[str, TableLoadResult],
    config: DatasetConfig,
) -> pl.DataFrame:
    metadata = canonicalize_metadata(get_frame(tables, "poly_market_metadata"))
    poly_book = enrich_with_metadata(choose_poly_book(tables), metadata)
    if poly_book is None or poly_book.is_empty():
        logger.warning("missing_poly_book", date=date)
        return pl.DataFrame()

    book = canonicalize_book(poly_book)
    if book.is_empty():
        logger.warning("empty_canonical_book", date=date)
        return pl.DataFrame()

    samples = sample_book(book, config.sample_interval_ms)
    samples = add_lob_features(samples)
    samples = add_book_event_features(samples, book)
    poly_trades = canonicalize_trades(enrich_with_metadata(get_frame(tables, "poly_trades"), metadata))
    samples = add_trade_features(samples, poly_trades)
    samples = add_binance_features(
        samples,
        canonicalize_binance_book(tables),
        canonicalize_trades(get_frame(tables, "binance_trades")),
        config.join_tolerance_ms,
    )
    samples = add_research_buckets(samples, get_frame(tables, "poly_enriched_book"), config.join_tolerance_ms)
    samples = add_future_book_labels(samples, book, [1, 3, 5, 10], config.join_tolerance_ms)
    samples = add_alpha_labels(
        samples,
        horizon_seconds=config.horizon_seconds,
        theta_bps=config.classification_theta_bps,
        entry_threshold_bps=config.entry_threshold_bps,
    )
    samples = add_entry_net_edge(
        samples,
        horizon_seconds=config.horizon_seconds,
        taker_cost_bps=config.taker_cost_bps,
        slippage_buffer_bps=config.slippage_buffer_bps,
        safety_margin_bps=config.safety_margin_bps,
    )
    samples = add_strategy_entry_labels(
        samples,
        horizon_seconds=config.horizon_seconds,
        entry_threshold_price=config.strategy_entry_threshold_price,
    )
    samples = add_two_leg_maker_fill_labels(
        samples,
        trades=poly_trades,
        horizon_seconds=config.horizon_seconds,
        max_total_price=config.two_leg_max_total_price,
        no_fill_edge=config.two_leg_no_fill_edge,
        maker_fill_trade_side=config.two_leg_maker_fill_trade_side,
    )
    samples = add_final_profit_labels(
        samples,
        horizon_seconds=config.horizon_seconds,
        fee_rate=config.fee_rate,
        price_buffer=config.price_buffer,
        max_total_price=config.two_leg_max_total_price,
    )
    return filter_training_rows(samples, config)


def choose_poly_book(tables: dict[str, TableLoadResult]) -> pl.DataFrame | None:
    sampled = get_frame(tables, "poly_sampled_book")
    if sampled is not None and not sampled.is_empty():
        return sampled
    enriched = get_frame(tables, "poly_enriched_book")
    if enriched is not None and not enriched.is_empty():
        return enriched
    return get_frame(tables, "poly_l2_book")


def get_frame(tables: dict[str, TableLoadResult], name: str) -> pl.DataFrame | None:
    result = tables.get(name)
    if result is None or not result.ok:
        return None
    return result.frame


def canonicalize_metadata(metadata: pl.DataFrame | None) -> pl.DataFrame:
    if metadata is None or metadata.is_empty() or "asset_id" not in metadata.columns:
        return pl.DataFrame()
    df = metadata
    for col in ["asset_id", "market_id", "condition_id", "slug", "outcome", "symbol", "period"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.String))
    return df


def canonicalize_book(book: pl.DataFrame) -> pl.DataFrame:
    if book is None or book.is_empty() or "recv_ns" not in book.columns:
        return pl.DataFrame()

    df = book
    if "asset_id" not in df.columns:
        df = df.with_columns(pl.lit("").alias("asset_id"))
    if "market" not in df.columns:
        df = df.with_columns(pl.lit("").alias("market"))
    if "source" not in df.columns:
        df = df.with_columns(pl.lit("").alias("source"))
    for col in ["market_id", "condition_id", "slug", "outcome", "symbol", "period"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))
    if "start_ns" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("start_ns"))
    if "expiry_ns" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Int64).alias("expiry_ns"))
    if "best_bid" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("best_bid"))
    if "best_ask" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("best_ask"))
    if "midpoint" not in df.columns:
        df = df.with_columns(((pl.col("best_bid") + pl.col("best_ask")) / 2).alias("midpoint"))
    if "spread" not in df.columns:
        df = df.with_columns((pl.col("best_ask") - pl.col("best_bid")).alias("spread"))
    if "microprice" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("microprice"))
    if "imbalance" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("imbalance"))
    for col in ["total_bid_levels", "total_ask_levels"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Int32).alias(col))
    for col in DEPTH_FEATURE_COLUMNS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    df = df.with_columns(
        [
            pl.when(pl.col("midpoint").is_null())
            .then((pl.col("best_bid") + pl.col("best_ask")) / 2)
            .otherwise(pl.col("midpoint"))
            .alias("midpoint"),
            pl.when(pl.col("spread").is_null())
            .then(pl.col("best_ask") - pl.col("best_bid"))
            .otherwise(pl.col("spread"))
            .alias("spread"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("recv_ns").cast(pl.Int64),
            pl.col("asset_id").cast(pl.String),
            pl.col("market").cast(pl.String),
            pl.col("source").cast(pl.String),
            pl.col("market_id").cast(pl.String),
            pl.col("condition_id").cast(pl.String),
            pl.col("slug").cast(pl.String),
            pl.col("outcome").cast(pl.String),
            pl.col("symbol").cast(pl.String),
            pl.col("period").cast(pl.String),
            pl.col("start_ns").cast(pl.Int64),
            pl.col("expiry_ns").cast(pl.Int64),
            pl.col("best_bid").cast(pl.Float64),
            pl.col("best_ask").cast(pl.Float64),
            pl.col("midpoint").cast(pl.Float64).alias("current_mid"),
            pl.col("spread").cast(pl.Float64).alias("current_spread"),
            pl.col("microprice").cast(pl.Float64).alias("current_microprice"),
            pl.col("imbalance").cast(pl.Float64).alias("top1_imbalance"),
            *[pl.col(col).cast(pl.Float64) for col in DEPTH_FEATURE_COLUMNS],
        ]
    )
    df = df.with_columns(
        pl.concat_str([pl.col("source"), pl.lit(" "), pl.col("market"), pl.lit(" "), pl.col("asset_id")])
        .alias("_market_hint")
    )
    df = df.with_columns(
        [
            pl.col("_market_hint").map_elements(extract_market_id, return_dtype=pl.String).alias("_hint_market_id"),
            pl.col("_market_hint").map_elements(extract_symbol, return_dtype=pl.String).alias("_hint_symbol"),
            pl.col("_market_hint").map_elements(extract_start_ns, return_dtype=pl.Int64).alias("_hint_start_ns"),
            pl.col("_market_hint").map_elements(extract_expiry_ns, return_dtype=pl.Int64).alias("_hint_expiry_ns"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(pl.col("market_id").is_null() | (pl.col("market_id") == ""))
            .then(
                pl.when(pl.col("_hint_market_id").is_not_null() & (pl.col("_hint_market_id") != ""))
                .then(pl.col("_hint_market_id"))
                .when(pl.col("condition_id").is_not_null() & (pl.col("condition_id") != ""))
                .then(pl.col("condition_id"))
                .otherwise(pl.concat_str([pl.lit("asset:"), pl.col("asset_id")]))
            )
            .otherwise(pl.col("market_id"))
            .alias("market_id"),
            pl.when(pl.col("symbol").is_null() | (pl.col("symbol") == ""))
            .then(
                pl.when(pl.col("_hint_symbol").is_not_null() & (pl.col("_hint_symbol") != ""))
                .then(pl.col("_hint_symbol"))
                .otherwise(infer_symbol_from_asset(pl.col("asset_id")))
            )
            .otherwise(pl.col("symbol"))
            .alias("symbol"),
            pl.coalesce([pl.col("_hint_start_ns"), pl.col("start_ns")]).alias("start_ns"),
            pl.coalesce([pl.col("_hint_expiry_ns"), pl.col("expiry_ns")]).alias("expiry_ns"),
        ]
    )
    df = df.with_columns(
        [
            ((pl.col("expiry_ns") - pl.col("recv_ns")) / NS_PER_SECOND).alias("time_to_expiry_seconds"),
        ]
    )
    df = df.with_columns(
        [
            pl.when(pl.col("current_mid") > 0)
            .then(pl.col("current_spread") / pl.col("current_mid"))
            .otherwise(None)
            .alias("relative_spread"),
            expiry_bucket_expr(pl.col("time_to_expiry_seconds")).alias("expiry_bucket"),
            market_phase_expr(pl.col("time_to_expiry_seconds")).alias("market_phase"),
        ]
    )
    keep_cols = [
        "recv_ns",
        "exchange_ts",
        "market_id",
        "asset_id",
        "symbol",
        "market",
        "source",
        "condition_id",
        "slug",
        "outcome",
        "period",
        "start_ns",
        "expiry_ns",
        "time_to_expiry_seconds",
        "expiry_bucket",
        "market_phase",
        "best_bid",
        "best_ask",
        "current_mid",
        "current_spread",
        "relative_spread",
        "current_microprice",
        "top1_imbalance",
        "total_bid_levels",
        "total_ask_levels",
        *DEPTH_FEATURE_COLUMNS,
        "imbalance_bucket",
        "spread_bucket",
        "price_bucket",
        "vol_bucket",
        "vol_60s",
        "tick_size",
        "min_order_size",
        "maker_base_fee",
        "taker_base_fee",
        "volume_24h",
        "liquidity",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return (
        df.select(keep_cols)
        .filter(
            (pl.col("recv_ns").is_not_null())
            & (pl.col("current_mid").is_not_null())
            & (pl.col("current_mid") > 0)
            & (pl.col("best_bid").is_not_null())
            & (pl.col("best_ask").is_not_null())
        )
        .sort(["market_id", "asset_id", "recv_ns"])
    )


def sample_book(book: pl.DataFrame, sample_interval_ms: int) -> pl.DataFrame:
    interval_ns = sample_interval_ms * NS_PER_MS
    return (
        book.with_columns(((pl.col("recv_ns") // interval_ns) * interval_ns).alias("sample_bucket_ns"))
        .sort(["market_id", "asset_id", "recv_ns"])
        .unique(subset=["market_id", "asset_id", "sample_bucket_ns"], keep="last", maintain_order=True)
        .drop("sample_bucket_ns")
        .sort(["market_id", "asset_id", "recv_ns"])
    )


def add_lob_features(samples: pl.DataFrame) -> pl.DataFrame:
    df = samples
    for col in ["top1_imbalance", "total_bid_levels", "total_ask_levels"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    df = df.with_columns(
        [
            # Use real top-N imbalance from depth features when available;
            # fall back to top1 proxy for old data that lacks depth columns.
            pl.coalesce([pl.col("depth_top3_imbalance"), pl.col("top1_imbalance")]).alias("top3_imbalance"),
            pl.coalesce([pl.col("depth_top5_imbalance"), pl.col("top1_imbalance")]).alias("top5_imbalance"),
            pl.coalesce([pl.col("depth_top10_imbalance"), pl.col("top1_imbalance")]).alias("top10_imbalance"),
            # Use real cumulative depth when available; fall back to level-count proxy.
            pl.coalesce([pl.col("cum_bid_depth_top10"), pl.col("total_bid_levels").cast(pl.Float64)]).alias("cum_bid_depth_topN_proxy"),
            pl.coalesce([pl.col("cum_ask_depth_top10"), pl.col("total_ask_levels").cast(pl.Float64)]).alias("cum_ask_depth_topN_proxy"),
            pl.coalesce(
                [pl.col("depth_top10_imbalance"),
                 (pl.col("total_bid_levels").cast(pl.Float64) - pl.col("total_ask_levels").cast(pl.Float64))]
            ).alias("depth_level_imbalance_proxy"),
            # Use real depth slope when available; fall back to None.
            pl.coalesce([pl.col("bid_depth_slope_top10"), pl.lit(None, dtype=pl.Float64)]).alias("bid_depth_slope"),
            pl.coalesce([pl.col("ask_depth_slope_top10"), pl.lit(None, dtype=pl.Float64)]).alias("ask_depth_slope"),
            pl.lit(None, dtype=pl.Float64).alias("queue_depletion_proxy"),
        ]
    )
    return df


def add_book_event_features(samples: pl.DataFrame, book: pl.DataFrame) -> pl.DataFrame:
    events = book.sort(["market_id", "asset_id", "recv_ns"]).with_columns(
        [
            pl.from_epoch("recv_ns", time_unit="ns").alias("_recv_dt"),
            pl.lit(1).alias("_one"),
            (pl.col("recv_ns").diff().over(["market_id", "asset_id"]) <= 100 * NS_PER_MS)
            .fill_null(False)
            .cast(pl.Int8)
            .alias("_fast_update"),
            pl.col("current_spread").diff().over(["market_id", "asset_id"]).alias("_spread_delta"),
            pl.col("current_mid").pct_change().over(["market_id", "asset_id"]).alias("_mid_ret"),
        ]
    )
    event_features = events.with_columns(
        [
            pl.col("_one").rolling_sum_by("_recv_dt", window_size="100ms").over(["market_id", "asset_id"]).alias("book_update_count_100ms"),
            pl.col("_one").rolling_sum_by("_recv_dt", window_size="500ms").over(["market_id", "asset_id"]).alias("book_update_count_500ms"),
            pl.col("_one").rolling_sum_by("_recv_dt", window_size="1s").over(["market_id", "asset_id"]).alias("book_update_count_1s"),
            (pl.col("_spread_delta") > 0)
            .cast(pl.Int8)
            .rolling_sum_by("_recv_dt", window_size="1s")
            .over(["market_id", "asset_id"])
            .alias("spread_widen_count_recent"),
            (pl.col("_spread_delta") < 0)
            .cast(pl.Int8)
            .rolling_sum_by("_recv_dt", window_size="1s")
            .over(["market_id", "asset_id"])
            .alias("spread_narrow_count_recent"),
            pl.col("_mid_ret").rolling_std_by("_recv_dt", window_size="3s").over(["market_id", "asset_id"]).alias("realized_vol_short"),
        ]
    ).select(
        [
            "market_id",
            "asset_id",
            "recv_ns",
            "book_update_count_100ms",
            "book_update_count_500ms",
            "book_update_count_1s",
            "spread_widen_count_recent",
            "spread_narrow_count_recent",
            "realized_vol_short",
        ]
    )
    return join_asof_by_group(samples, event_features, by=["market_id", "asset_id"], tolerance_ms=1000)


def canonicalize_trades(trades: pl.DataFrame | None) -> pl.DataFrame:
    if trades is None or trades.is_empty() or "recv_ns" not in trades.columns:
        return pl.DataFrame()
    df = trades
    for col in ["asset_id", "market", "source", "side"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))
    for col in ["market_id", "condition_id", "slug", "outcome", "symbol", "period"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit("").alias(col))
    for col in ["price", "size"]:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    df = df.with_columns(
        [
            pl.col("recv_ns").cast(pl.Int64),
            pl.col("asset_id").cast(pl.String),
            pl.col("market").cast(pl.String),
            pl.col("source").cast(pl.String),
            pl.col("market_id").cast(pl.String),
            pl.col("condition_id").cast(pl.String),
            pl.col("slug").cast(pl.String),
            pl.col("outcome").cast(pl.String),
            pl.col("symbol").cast(pl.String),
            pl.col("period").cast(pl.String),
            pl.col("side").cast(pl.String).str.to_uppercase(),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64),
            pl.concat_str([pl.col("source"), pl.lit(" "), pl.col("market"), pl.lit(" "), pl.col("asset_id")])
            .alias("_market_hint"),
        ]
    )
    return df.with_columns(
        [
            pl.col("_market_hint").map_elements(extract_market_id, return_dtype=pl.String).alias("_hint_market_id"),
            pl.col("_market_hint").map_elements(extract_symbol, return_dtype=pl.String).alias("_hint_symbol"),
            pl.when(pl.col("side").is_in(["BUY", "BID"]))
            .then(pl.lit(1.0))
            .when(pl.col("side").is_in(["SELL", "ASK"]))
            .then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
            .alias("signed_side"),
        ]
    ).with_columns(
        pl.when(pl.col("market_id").is_null() | (pl.col("market_id") == ""))
        .then(
            pl.when(pl.col("_hint_market_id").is_not_null() & (pl.col("_hint_market_id") != ""))
            .then(pl.col("_hint_market_id"))
            .when(pl.col("condition_id").is_not_null() & (pl.col("condition_id") != ""))
            .then(pl.col("condition_id"))
            .otherwise(pl.concat_str([pl.lit("asset:"), pl.col("asset_id")]))
        )
        .otherwise(pl.col("market_id"))
        .alias("market_id")
    ).with_columns(
        pl.when(pl.col("symbol").is_null() | (pl.col("symbol") == ""))
        .then(pl.col("_hint_symbol"))
        .otherwise(pl.col("symbol"))
        .alias("symbol")
    )


def add_trade_features(samples: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty():
        return samples.with_columns(empty_trade_feature_exprs())
    enriched = trades.sort(["market_id", "asset_id", "recv_ns"]).with_columns(
        [
            pl.from_epoch("recv_ns", time_unit="ns").alias("_recv_dt"),
            pl.lit(1.0).alias("_one"),
            (pl.col("size") * (pl.col("signed_side") > 0).cast(pl.Float64)).alias("_buy_volume"),
            (pl.col("size") * (pl.col("signed_side") < 0).cast(pl.Float64)).alias("_sell_volume"),
            (pl.col("size") * pl.col("signed_side")).alias("_signed_volume"),
            (pl.col("price") * pl.col("size")).alias("_notional"),
        ]
    )
    features = enriched.with_columns(
        [
            pl.col("_one").rolling_sum_by("_recv_dt", window_size="100ms").over(["market_id", "asset_id"]).alias("poly_trade_count_100ms"),
            pl.col("_one").rolling_sum_by("_recv_dt", window_size="500ms").over(["market_id", "asset_id"]).alias("poly_trade_count_500ms"),
            pl.col("_one").rolling_sum_by("_recv_dt", window_size="1s").over(["market_id", "asset_id"]).alias("poly_trade_count_1s"),
            pl.col("_buy_volume").rolling_sum_by("_recv_dt", window_size="1s").over(["market_id", "asset_id"]).alias("poly_aggressive_buy_volume_1s"),
            pl.col("_sell_volume").rolling_sum_by("_recv_dt", window_size="1s").over(["market_id", "asset_id"]).alias("poly_aggressive_sell_volume_1s"),
            pl.col("_signed_volume").rolling_sum_by("_recv_dt", window_size="1s").over(["market_id", "asset_id"]).alias("poly_signed_volume_1s"),
            pl.col("_notional").rolling_sum_by("_recv_dt", window_size="1s").over(["market_id", "asset_id"]).alias("_rolling_notional_1s"),
            pl.col("size").rolling_sum_by("_recv_dt", window_size="1s").over(["market_id", "asset_id"]).alias("_rolling_size_1s"),
            (pl.col("signed_side") > 0).cast(pl.Int8).rolling_sum(window_size=10).over(["market_id", "asset_id"]).alias("consecutive_buy_trade_run_proxy"),
            (pl.col("signed_side") < 0).cast(pl.Int8).rolling_sum(window_size=10).over(["market_id", "asset_id"]).alias("consecutive_sell_trade_run_proxy"),
        ]
    ).with_columns(
        [
            pl.col("poly_trade_count_1s").alias("poly_trade_count_recent"),
            pl.col("poly_aggressive_buy_volume_1s").alias("poly_aggressive_buy_volume_recent"),
            pl.col("poly_aggressive_sell_volume_1s").alias("poly_aggressive_sell_volume_recent"),
            pl.col("poly_signed_volume_1s").alias("poly_signed_volume_recent"),
            (pl.col("poly_signed_volume_1s") / (pl.col("poly_aggressive_buy_volume_1s") + pl.col("poly_aggressive_sell_volume_1s") + 1e-12))
            .alias("poly_signed_volume_imbalance_recent"),
            (pl.col("_rolling_notional_1s") / (pl.col("_rolling_size_1s") + 1e-12)).alias("poly_recent_vwap"),
            (pl.col("size") > pl.col("size").rolling_quantile(0.95, window_size=100).over(["market_id", "asset_id"]))
            .cast(pl.Int8)
            .alias("poly_sweep_indicator_proxy"),
        ]
    ).select(
        [
            "market_id",
            "asset_id",
            "recv_ns",
            "poly_trade_count_100ms",
            "poly_trade_count_500ms",
            "poly_trade_count_1s",
            "poly_trade_count_recent",
            "poly_aggressive_buy_volume_1s",
            "poly_aggressive_sell_volume_1s",
            "poly_signed_volume_1s",
            "poly_aggressive_buy_volume_recent",
            "poly_aggressive_sell_volume_recent",
            "poly_signed_volume_recent",
            "poly_signed_volume_imbalance_recent",
            "poly_recent_vwap",
            "poly_sweep_indicator_proxy",
            "consecutive_buy_trade_run_proxy",
            "consecutive_sell_trade_run_proxy",
        ]
    )
    joined = join_asof_by_group(samples, features, by=["market_id", "asset_id"], tolerance_ms=1000)
    return joined.with_columns(
        (pl.col("poly_recent_vwap") - pl.col("current_mid")).alias("poly_recent_vwap_deviation")
    )


def empty_trade_feature_exprs() -> list[pl.Expr]:
    return [
        pl.lit(0.0).alias("poly_trade_count_100ms"),
        pl.lit(0.0).alias("poly_trade_count_500ms"),
        pl.lit(0.0).alias("poly_trade_count_1s"),
        pl.lit(0.0).alias("poly_trade_count_recent"),
        pl.lit(0.0).alias("poly_aggressive_buy_volume_1s"),
        pl.lit(0.0).alias("poly_aggressive_sell_volume_1s"),
        pl.lit(0.0).alias("poly_signed_volume_1s"),
        pl.lit(0.0).alias("poly_aggressive_buy_volume_recent"),
        pl.lit(0.0).alias("poly_aggressive_sell_volume_recent"),
        pl.lit(0.0).alias("poly_signed_volume_recent"),
        pl.lit(0.0).alias("poly_signed_volume_imbalance_recent"),
        pl.lit(None, dtype=pl.Float64).alias("poly_recent_vwap"),
        pl.lit(None, dtype=pl.Float64).alias("poly_recent_vwap_deviation"),
        pl.lit(0).alias("poly_sweep_indicator_proxy"),
        pl.lit(0).alias("consecutive_buy_trade_run_proxy"),
        pl.lit(0).alias("consecutive_sell_trade_run_proxy"),
    ]


def canonicalize_binance_book(tables: dict[str, TableLoadResult]) -> pl.DataFrame:
    frames = []
    for name in ["binance_l2_book", "binance_best_bid_ask"]:
        frame = get_frame(tables, name)
        if frame is not None and not frame.is_empty():
            frames.append(frame)
    raw = concat_non_empty(frames)
    if raw.is_empty() or "recv_ns" not in raw.columns:
        return pl.DataFrame()
    df = canonicalize_book(raw)
    if df.is_empty():
        return df
    base = df.select(
        [
            "recv_ns",
            "asset_id",
            "symbol",
            "current_mid",
            "current_spread",
            "top1_imbalance",
        ]
    ).rename(
        {
            "asset_id": "binance_asset_id",
            "symbol": "binance_symbol",
            "current_mid": "binance_mid",
            "current_spread": "binance_spread",
            "top1_imbalance": "binance_book_imbalance",
        }
    ).sort("recv_ns")

    depth_raw = get_frame(tables, "binance_l2_book")
    if depth_raw is None or depth_raw.is_empty():
        return base.with_columns(empty_binance_depth_feature_exprs())
    depth = canonicalize_book(depth_raw)
    if depth.is_empty():
        return base.with_columns(empty_binance_depth_feature_exprs())

    depth_state = depth.select(["recv_ns", "symbol", *DEPTH_FEATURE_COLUMNS]).rename(
        {
            "symbol": "binance_symbol",
            **{col: f"binance_{col}" for col in DEPTH_FEATURE_COLUMNS},
        }
    )
    join_globally = depth_state["binance_symbol"].drop_nulls().n_unique() <= 1
    depth_tolerance_ns = 2_000 * NS_PER_MS
    if join_globally:
        joined = base.sort("recv_ns").join_asof(
            depth_state.drop(["binance_symbol"], strict=False).sort("recv_ns"),
            on="recv_ns",
            strategy="backward",
            tolerance=depth_tolerance_ns,
        )
    else:
        joined = join_asof_by_group(
            base,
            depth_state.rename({"binance_symbol": "symbol"}),
            by=["symbol"],
            tolerance_ms=2_000,
        )
    for expr in empty_binance_depth_feature_exprs():
        name = expr.meta.output_name()
        if name not in joined.columns:
            joined = joined.with_columns(expr)
    return joined.sort("recv_ns")


def add_binance_features(
    samples: pl.DataFrame,
    binance_book: pl.DataFrame,
    binance_trades: pl.DataFrame,
    tolerance_ms: int,
) -> pl.DataFrame:
    df = samples
    if binance_book.is_empty():
        df = df.with_columns(empty_binance_feature_exprs())
    else:
        join_globally = binance_book["binance_symbol"].drop_nulls().n_unique() <= 1
        b = binance_book.sort("recv_ns").with_columns(
            [
                pl.col("binance_mid").pct_change().alias("binance_return_tick"),
                pl.col("binance_mid").pct_change(n=1).alias("binance_return_100ms"),
                pl.col("binance_mid").pct_change(n=5).alias("binance_return_500ms"),
                pl.col("binance_mid").pct_change(n=10).alias("binance_return_1s"),
                pl.col("binance_mid").pct_change(n=30).alias("binance_return_3s"),
            ]
        )
        if join_globally:
            df = df.sort("recv_ns").join_asof(
                b.drop(["binance_symbol"], strict=False),
                on="recv_ns",
                strategy="backward",
                tolerance=tolerance_ms * NS_PER_MS,
            )
        else:
            df = join_asof_by_group(
                df,
                b.rename({"binance_symbol": "symbol"}),
                by=["symbol"],
                tolerance_ms=tolerance_ms,
            )
        for expr in empty_binance_feature_exprs():
            name = expr.meta.output_name()
            if name not in df.columns:
                df = df.with_columns(expr)

    if not binance_trades.is_empty():
        join_globally = binance_trades["symbol"].drop_nulls().n_unique() <= 1
        bt = binance_trades.sort("recv_ns").with_columns(
            [
                (pl.col("size") * pl.col("signed_side")).rolling_sum(window_size=20).alias("binance_recent_trade_signed_volume"),
                pl.col("size").rolling_sum(window_size=20).alias("binance_recent_trade_volume"),
            ]
        ).with_columns(
            (pl.col("binance_recent_trade_signed_volume") / (pl.col("binance_recent_trade_volume") + 1e-12))
            .alias("binance_recent_trade_imbalance")
        ).select(["recv_ns", "symbol", "binance_recent_trade_imbalance"])
        if join_globally:
            df = df.sort("recv_ns").join_asof(
                bt.drop(["symbol"], strict=False),
                on="recv_ns",
                strategy="backward",
                tolerance=tolerance_ms * NS_PER_MS,
            )
        else:
            df = join_asof_by_group(
                df,
                bt,
                by=["symbol"],
                tolerance_ms=tolerance_ms,
            )
    elif "binance_recent_trade_imbalance" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("binance_recent_trade_imbalance"))

    df = df.with_columns(
        [
            pl.col("current_mid").pct_change(n=10).over(["market_id", "asset_id"]).alias("poly_return_1s"),
            (pl.col("binance_return_1s") - pl.col("current_mid").pct_change(n=10).over(["market_id", "asset_id"]))
            .alias("lead_lag_binance_minus_poly_1s"),
            (pl.col("binance_return_500ms") - pl.col("current_mid").pct_change(n=5).over(["market_id", "asset_id"]))
            .alias("lead_lag_binance_minus_poly_500ms"),
        ]
    )
    return df


def empty_binance_feature_exprs() -> list[pl.Expr]:
    return [
        pl.lit(None, dtype=pl.Float64).alias("binance_mid"),
        pl.lit(None, dtype=pl.Float64).alias("binance_spread"),
        pl.lit(None, dtype=pl.Float64).alias("binance_book_imbalance"),
        pl.lit(None, dtype=pl.Float64).alias("binance_return_tick"),
        pl.lit(None, dtype=pl.Float64).alias("binance_return_100ms"),
        pl.lit(None, dtype=pl.Float64).alias("binance_return_500ms"),
        pl.lit(None, dtype=pl.Float64).alias("binance_return_1s"),
        pl.lit(None, dtype=pl.Float64).alias("binance_return_3s"),
        *empty_binance_depth_feature_exprs(),
    ]


def empty_binance_depth_feature_exprs() -> list[pl.Expr]:
    return [pl.lit(None, dtype=pl.Float64).alias(col) for col in BINANCE_DEPTH_FEATURE_COLUMNS]


def add_research_buckets(samples: pl.DataFrame, enriched: pl.DataFrame | None, tolerance_ms: int) -> pl.DataFrame:
    if enriched is None or enriched.is_empty():
        df = samples
    else:
        buckets = canonicalize_book(enriched)
        bucket_cols = [c for c in ["imbalance_bucket", "spread_bucket", "price_bucket", "vol_bucket", "vol_60s"] if c in buckets.columns]
        if not bucket_cols:
            df = samples
        else:
            right = buckets.select(["market_id", "asset_id", "recv_ns", *bucket_cols])
            df = join_asof_by_group(samples.drop([c for c in bucket_cols if c in samples.columns]), right, by=["market_id", "asset_id"], tolerance_ms=tolerance_ms)
    for col in ["imbalance_bucket", "spread_bucket", "price_bucket", "vol_bucket"]:
        if col not in df.columns:
            base = col.replace("_bucket", "")
            if base == "price":
                df = add_quantile_bucket(df, "current_mid", col)
            elif base == "spread":
                df = add_quantile_bucket(df, "current_spread", col)
            elif base == "imbalance":
                df = add_quantile_bucket(df, "top1_imbalance", col)
            else:
                df = df.with_columns(pl.lit("unknown").alias(col))
    return df


def add_future_mid_labels(
    samples: pl.DataFrame,
    book: pl.DataFrame,
    horizons_seconds: list[int],
    tolerance_ms: int,
) -> pl.DataFrame:
    return add_future_book_labels(samples, book, horizons_seconds, tolerance_ms)


def add_future_book_labels(
    samples: pl.DataFrame,
    book: pl.DataFrame,
    horizons_seconds: list[int],
    tolerance_ms: int,
) -> pl.DataFrame:
    df = samples
    future_book = book.select(["market_id", "asset_id", "recv_ns", "current_mid", "best_bid", "best_ask"]).sort(
        ["market_id", "asset_id", "recv_ns"]
    )
    for horizon in horizons_seconds:
        target_ns_col = f"_target_ns_{horizon}s"
        future_col = f"future_mid_{horizon}s"
        future_bid_col = f"future_best_bid_{horizon}s"
        future_ask_col = f"future_best_ask_{horizon}s"
        future = future_book.rename(
            {
                "current_mid": future_col,
                "best_bid": future_bid_col,
                "best_ask": future_ask_col,
                "recv_ns": "_future_recv_ns",
            }
        )
        df = df.with_columns((pl.col("recv_ns") + horizon * NS_PER_SECOND).alias(target_ns_col))
        df = join_asof_by_group(
            df,
            future,
            by=["market_id", "asset_id"],
            left_on=target_ns_col,
            right_on="_future_recv_ns",
            tolerance_ms=tolerance_ms,
            strategy="forward",
        )
        df = df.drop([target_ns_col, "_future_recv_ns"], strict=False)
        df = df.with_columns(
            [
                ((pl.col(future_col) - pl.col("current_mid")) / pl.col("current_mid")).alias(f"markout_{horizon}s"),
                (((pl.col(future_col) - pl.col("current_mid")) / pl.col("current_mid")) * 10000).alias(f"markout_{horizon}s_bps"),
            ]
        )
    return df


def join_asof_by_group(
    left: pl.DataFrame,
    right: pl.DataFrame,
    by: list[str],
    tolerance_ms: int,
    left_on: str = "recv_ns",
    right_on: str = "recv_ns",
    strategy: str = "backward",
) -> pl.DataFrame:
    if left.is_empty() or right.is_empty():
        return left
    pieces: list[pl.DataFrame] = []
    group_keys = left.select(by).unique().iter_rows(named=True)
    for key in group_keys:
        left_filter = pl.all_horizontal([pl.col(k) == v for k, v in key.items()])
        right_filter = pl.all_horizontal([pl.col(k) == v for k, v in key.items()])
        l = left.filter(left_filter).sort(left_on)
        r = right.filter(right_filter).sort(right_on)
        if r.is_empty():
            pieces.append(l)
            continue
        r_join = r.drop(by, strict=False)
        pieces.append(
            l.join_asof(
                r_join,
                left_on=left_on,
                right_on=right_on,
                strategy=strategy,
                tolerance=tolerance_ms * NS_PER_MS,
            )
        )
    return concat_non_empty(pieces).sort(["market_id", "asset_id", "recv_ns"])


def filter_training_rows(samples: pl.DataFrame, config: DatasetConfig) -> pl.DataFrame:
    if samples.is_empty():
        return samples
    required = [
        "current_mid",
        f"future_mid_{config.horizon_seconds}s",
        f"markout_{config.horizon_seconds}s_bps",
        f"y_reg_{config.horizon_seconds}s",
        f"y_cls_{config.horizon_seconds}s",
    ]
    present = [c for c in required if c in samples.columns]
    filtered = samples.drop_nulls(present)
    if {"recv_ns", "expiry_ns"}.issubset(filtered.columns):
        horizon_ns = config.horizon_seconds * NS_PER_SECOND
        filtered = filtered.filter(
            pl.col("expiry_ns").is_null()
            | ((pl.col("recv_ns") + pl.lit(horizon_ns)) <= pl.col("expiry_ns"))
        )
    if filtered.is_empty():
        return filtered
    feature_cols = infer_feature_columns(filtered)
    keep_features = []
    for col in feature_cols:
        null_frac = filtered[col].null_count() / max(filtered.height, 1)
        if null_frac <= config.max_null_fraction:
            keep_features.append(col)
    id_label_cols = [c for c in filtered.columns if c not in feature_cols]
    return filtered.select(id_label_cols + keep_features)


def infer_feature_columns(df: pl.DataFrame) -> list[str]:
    excluded_prefixes = (
        "future_mid_",
        "future_best_bid_",
        "future_best_ask_",
        "markout_",
        "y_",
        "date",
        "strategy_entry_edge_",
        "future_opposite_maker_fill_",
        "two_leg_",
        "first_unwind_loss_proxy_",
        "final_profit_",
        "final_profit_weight_",
    )
    excluded = {
        "recv_ns",
        "exchange_ts",
        "market_id",
        "asset_id",
        "symbol",
        "market",
        "source",
        "condition_id",
        "slug",
        "outcome",
        "period",
        "start_ns",
        "question",
        "expiry_ns",
        "opposite_asset_id",
        "realized_edge_after_entry_cost_bps",
    }
    numeric = {pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
    feature_cols = [
        col
        for col, dtype in df.schema.items()
        if dtype in numeric
        and col not in excluded
        and not col.startswith(excluded_prefixes)
    ]
    categorical_features = [
        "imbalance_bucket",
        "spread_bucket",
        "price_bucket",
        "vol_bucket",
    ]
    feature_cols.extend([col for col in categorical_features if col in df.columns and col not in feature_cols])
    return feature_cols


def make_metadata(
    dataset: pl.DataFrame,
    config: DatasetConfig,
    table_report: list[dict[str, object]],
    quality: list[dict[str, object]],
) -> dict[str, object]:
    label_columns = [
        c
        for c in dataset.columns
        if c.startswith("y_")
        or c.startswith("future_mid_")
        or c.startswith("markout_")
        or c.startswith("strategy_entry_edge_")
        or c.startswith("future_opposite_maker_fill_")
        or c.startswith("two_leg_")
        or c.startswith("first_unwind_loss_proxy_")
        or c.startswith("final_profit_")
        or c.startswith("final_profit_weight_")
    ]
    feature_columns = infer_feature_columns(dataset) if not dataset.is_empty() else []
    if dataset.is_empty() or "recv_ns" not in dataset.columns:
        date_range = {"start_ns": None, "end_ns": None}
    else:
        date_range = {"start_ns": int(dataset["recv_ns"].min()), "end_ns": int(dataset["recv_ns"].max())}
    return {
        "config": dataclass_to_json_dict(config),
        "rows": dataset.height,
        "feature_columns": feature_columns,
        "label_columns": label_columns,
        "sampling_interval_ms": config.sample_interval_ms,
        "date_ranges": date_range,
        "input_tables": table_report,
        "data_quality": quality,
        "second_leg_maker_fill_label_design": maker_fill_label_design(),
    }


def write_dataset_artifacts(
    result: BuildResult,
    output_dir: Path,
    basename: str = "alpha_dataset",
    write_csv: bool = True,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{basename}.parquet"
    csv_path = output_dir / f"{basename}.csv"
    metadata_path = output_dir / f"{basename}_metadata.json"
    result.dataset.write_parquet(str(parquet_path))
    save_json(result.metadata, metadata_path)
    paths = {"parquet": str(parquet_path), "metadata": str(metadata_path)}
    if write_csv:
        result.dataset.write_csv(str(csv_path))
        paths["csv"] = str(csv_path)
    return paths


def add_quantile_bucket(df: pl.DataFrame, source_col: str, bucket_col: str, n_buckets: int = 10) -> pl.DataFrame:
    if source_col not in df.columns or df[source_col].n_unique() <= 1:
        return df.with_columns(pl.lit("q0").alias(bucket_col))
    labels = [f"q{i}" for i in range(n_buckets)]
    try:
        return df.with_columns(pl.col(source_col).qcut(n_buckets, labels=labels, allow_duplicates=True).alias(bucket_col))
    except Exception:
        return df.with_columns(pl.lit("q0").alias(bucket_col))


def extract_market_id(text: str | None) -> str:
    if not text:
        return ""
    match = EXPIRY_RE.search(str(text))
    return match.group(0).lower() if match else ""


def extract_symbol(text: str | None) -> str:
    if not text:
        return ""
    match = EXPIRY_RE.search(str(text))
    if match:
        return match.group("symbol").upper()
    lower = str(text).lower()
    if "eth" in lower:
        return "ETH"
    if "btc" in lower or "bitcoin" in lower:
        return "BTC"
    return ""


def extract_expiry_ns(text: str | None) -> int | None:
    if not text:
        return None
    match = EXPIRY_RE.search(str(text))
    if not match:
        return None
    start = int(match.group("expiry"))
    period = match.group("period").lower()
    seconds = 300 if period == "5m" else 900 if period == "15m" else 0
    expiry = start + seconds
    if expiry < 10_000_000_000:
        expiry *= NS_PER_SECOND
    return expiry


def extract_start_ns(text: str | None) -> int | None:
    if not text:
        return None
    match = EXPIRY_RE.search(str(text))
    if not match:
        return None
    start = int(match.group("expiry"))
    if start < 10_000_000_000:
        start *= NS_PER_SECOND
    return start


def infer_symbol_from_asset(asset_expr: pl.Expr) -> pl.Expr:
    lower = asset_expr.str.to_lowercase()
    return (
        pl.when(lower.str.contains("eth"))
        .then(pl.lit("ETH"))
        .when(lower.str.contains("btc") | lower.str.contains("bitcoin"))
        .then(pl.lit("BTC"))
        .otherwise(pl.lit("UNKNOWN"))
    )


def expiry_bucket_expr(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr.is_null())
        .then(pl.lit("unknown"))
        .when(expr > 180)
        .then(pl.lit(">180s"))
        .when(expr > 60)
        .then(pl.lit("60-180s"))
        .when(expr > 30)
        .then(pl.lit("30-60s"))
        .when(expr > 10)
        .then(pl.lit("10-30s"))
        .otherwise(pl.lit("<10s"))
    )


def market_phase_expr(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr.is_null())
        .then(pl.lit("unknown"))
        .when(expr > 180)
        .then(pl.lit("early"))
        .when(expr > 30)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("tail"))
    )
