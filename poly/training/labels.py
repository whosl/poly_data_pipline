"""Label construction for alpha and entry-worthiness models."""

from __future__ import annotations

from collections import deque

import numpy as np
import polars as pl


def add_alpha_labels(
    samples: pl.DataFrame,
    horizon_seconds: int,
    theta_bps: float,
    entry_threshold_bps: float,
) -> pl.DataFrame:
    markout_col = f"markout_{horizon_seconds}s_bps"
    future_mid_col = f"future_mid_{horizon_seconds}s"
    if markout_col not in samples.columns:
        raise ValueError(f"missing required markout column: {markout_col}")

    cls_col = f"y_cls_{horizon_seconds}s"
    reg_col = f"y_reg_{horizon_seconds}s"
    entry_col = "y_entry"
    return samples.with_columns(
        [
            pl.col(markout_col).alias(reg_col),
            pl.when(pl.col(markout_col) > theta_bps)
            .then(pl.lit("up"))
            .when(pl.col(markout_col) < -theta_bps)
            .then(pl.lit("down"))
            .otherwise(pl.lit("neutral"))
            .alias(cls_col),
            (pl.col(markout_col) > entry_threshold_bps).cast(pl.Int8).alias(entry_col),
            (
                pl.when(pl.col(future_mid_col).is_not_null())
                .then(pl.col(future_mid_col) - pl.col("current_mid"))
                .otherwise(None)
            ).alias(f"future_mid_delta_{horizon_seconds}s"),
        ]
    )


def add_entry_net_edge(
    samples: pl.DataFrame,
    horizon_seconds: int,
    taker_cost_bps: float,
    slippage_buffer_bps: float,
    safety_margin_bps: float,
) -> pl.DataFrame:
    reg_col = f"y_reg_{horizon_seconds}s"
    threshold = taker_cost_bps + slippage_buffer_bps + safety_margin_bps
    return samples.with_columns(
        [
            (pl.col(reg_col) - pl.lit(threshold)).alias("realized_edge_after_entry_cost_bps"),
            (pl.col(reg_col) > pl.lit(threshold)).cast(pl.Int8).alias("y_entry_after_cost"),
        ]
    )


def add_strategy_entry_labels(
    samples: pl.DataFrame,
    horizon_seconds: int,
    entry_threshold_price: float,
) -> pl.DataFrame:
    """Label first-leg taker entries by absolute price edge, not bps."""
    future_mid_col = f"future_mid_{horizon_seconds}s"
    edge_col = f"strategy_entry_edge_{horizon_seconds}s"
    edge_after_threshold_col = f"strategy_entry_edge_after_threshold_{horizon_seconds}s"
    cls_col = f"y_strategy_entry_{horizon_seconds}s"
    binary_col = f"y_strategy_entry_binary_{horizon_seconds}s"
    edge = pl.col(future_mid_col) - pl.col("best_ask")
    return samples.with_columns(
        [
            edge.alias(edge_col),
            (edge - pl.lit(entry_threshold_price)).alias(edge_after_threshold_col),
            pl.when(pl.col(future_mid_col).is_null() | pl.col("best_ask").is_null())
            .then(None)
            .when(edge >= pl.lit(entry_threshold_price))
            .then(pl.lit("enter"))
            .otherwise(pl.lit("skip"))
            .alias(cls_col),
            pl.when(pl.col(future_mid_col).is_null() | pl.col("best_ask").is_null())
            .then(None)
            .when(edge >= pl.lit(entry_threshold_price))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias(binary_col),
        ]
    )


def add_two_leg_maker_fill_labels(
    samples: pl.DataFrame,
    trades: pl.DataFrame,
    horizon_seconds: int,
    max_total_price: float,
    no_fill_edge: float = -1.0,
    maker_fill_trade_side: str = "SELL",
) -> pl.DataFrame:
    """Label whether a taker first leg plus future opposite maker fill clears max total price.

    The first leg is a taker buy at this row's `best_ask`. The second leg is a
    maker buy on the opposite outcome in the same market. The first implementation
    treats a future opposite-side trade with `side == maker_fill_trade_side` as
    fill evidence and uses the lowest such trade price within the horizon.
    """
    fill_price_col = f"future_opposite_maker_fill_price_{horizon_seconds}s"
    fill_ts_col = f"future_opposite_maker_fill_recv_ns_{horizon_seconds}s"
    total_col = f"two_leg_total_price_{horizon_seconds}s"
    edge_col = f"two_leg_edge_{horizon_seconds}s"
    realized_edge_col = f"two_leg_realized_edge_{horizon_seconds}s"
    cls_col = f"y_two_leg_entry_{horizon_seconds}s"
    binary_col = f"y_two_leg_entry_binary_{horizon_seconds}s"

    if samples.is_empty():
        return samples

    working = samples.with_row_index("_row_nr")
    opposite_by_row = opposite_asset_ids(working)
    n = working.height
    fill_prices = np.full(n, np.nan, dtype=float)
    fill_times = np.full(n, -1, dtype=np.int64)

    if trades is not None and not trades.is_empty():
        side = maker_fill_trade_side.upper()
        trade_events = (
            trades.filter((pl.col("side").str.to_uppercase() == side) & pl.col("price").is_not_null())
            .select(["asset_id", "recv_ns", "price"])
            .sort(["asset_id", "recv_ns"])
        )
        events_by_asset = {
            str(partition_key_value(asset_id)): group.select(["recv_ns", "price"]).to_numpy()
            for asset_id, group in trade_events.partition_by("asset_id", as_dict=True).items()
        }
        horizon_ns = horizon_seconds * 1_000_000_000
        for opposite_asset_id, group in working.with_columns(pl.Series("opposite_asset_id", opposite_by_row)).partition_by(
            "opposite_asset_id", as_dict=True
        ).items():
            event_array = events_by_asset.get(str(partition_key_value(opposite_asset_id)))
            if event_array is None or len(event_array) == 0:
                continue
            row_idx = group["_row_nr"].to_numpy()
            query_times = group["recv_ns"].to_numpy()
            prices, times = future_window_min_prices(
                event_array[:, 0].astype(np.int64),
                event_array[:, 1].astype(float),
                query_times.astype(np.int64),
                horizon_ns,
            )
            fill_prices[row_idx] = prices
            fill_times[row_idx] = times

    return (
        working.with_columns(
            [
                pl.Series("opposite_asset_id", opposite_by_row),
                pl.Series(fill_price_col, fill_prices).fill_nan(None),
                pl.Series(fill_ts_col, fill_times).replace(-1, None),
            ]
        )
        .with_columns((pl.col("best_ask") + pl.col(fill_price_col)).alias(total_col))
        .with_columns((pl.lit(max_total_price) - pl.col(total_col)).alias(edge_col))
        .with_columns(pl.coalesce([pl.col(edge_col), pl.lit(no_fill_edge)]).alias(realized_edge_col))
        .with_columns(
            [
                pl.when(pl.col(edge_col).is_not_null() & (pl.col(edge_col) >= 0))
                .then(pl.lit("enter"))
                .otherwise(pl.lit("skip"))
                .alias(cls_col),
                (pl.col(edge_col).is_not_null() & (pl.col(edge_col) >= 0)).cast(pl.Int8).alias(binary_col),
            ]
        )
        .drop("_row_nr")
    )


def add_final_profit_labels(
    samples: pl.DataFrame,
    horizon_seconds: int,
    fee_rate: float = 0.072,
    price_buffer: float = 0.01,
    max_total_price: float = 0.96,
) -> pl.DataFrame:
    """Realized payoff target for two-leg capped-win / first-leg-unwind strategy.

    Success (two-leg fills):
        entry_price     = best_ask
        first_leg_price = entry_price + price_buffer
        fee_per_share   = fee_rate * first_leg_price * (1 - first_leg_price)
        second_leg_size = 1 - fee_per_share              (fee deducted in shares)
        second_leg_price = max_total_price - entry_price  (opposite-leg maker quote)
        success_profit  = second_leg_size - (first_leg_price + second_leg_size * second_leg_price)

    Failure (unwind first leg):
        unwind_profit = second_leg_size * future_best_bid - first_leg_price
        unwind_loss = -unwind_profit
    """
    future_mid_col = f"future_mid_{horizon_seconds}s"
    future_bid_col = f"future_best_bid_{horizon_seconds}s"
    two_leg_cls_col = f"y_two_leg_entry_{horizon_seconds}s"
    unwind_loss_col = f"first_unwind_loss_proxy_{horizon_seconds}s"
    unwind_profit_col = f"first_unwind_profit_proxy_{horizon_seconds}s"
    final_profit_col = f"final_profit_{horizon_seconds}s"
    final_profit_weight_col = f"final_profit_weight_{horizon_seconds}s"
    cls_col = f"y_final_profit_entry_{horizon_seconds}s"
    binary_col = f"y_final_profit_entry_binary_{horizon_seconds}s"

    entry_price = pl.col("best_ask")
    first_leg_price = entry_price + pl.lit(price_buffer)
    fee_per_share = pl.lit(fee_rate) * first_leg_price * (1.0 - first_leg_price)
    second_leg_size = 1.0 - fee_per_share
    second_leg_price = pl.lit(max_total_price) - entry_price

    # Success profit (per share): revenue - cost = second_leg_size - (first_leg_price + second_leg_size * second_leg_price)
    success_profit = second_leg_size - (first_leg_price + second_leg_size * second_leg_price)

    # Unwind PnL for the first-leg failure branch:
    # buy one share at first_leg_price, then sell the remaining post-fee
    # share quantity at the future same-leg best bid. This can be positive
    # when the first leg moves favorably even if the maker exit never fills.
    unwind_reference = pl.col(future_bid_col) if future_bid_col in samples.columns else pl.col(future_mid_col)
    unwind_profit = second_leg_size * unwind_reference - first_leg_price
    unwind_loss = -unwind_profit

    final_profit = (
        pl.when(pl.col(two_leg_cls_col) == "enter")
        .then(success_profit)
        .otherwise(unwind_profit)
    )
    return (
        samples.with_columns(unwind_loss.alias(unwind_loss_col))
        .with_columns(unwind_profit.alias(unwind_profit_col))
        .with_columns(final_profit.alias(final_profit_col))
        .with_columns(
            [
                pl.when(pl.col(final_profit_col) > 0)
                .then(pl.lit("enter"))
                .otherwise(pl.lit("skip"))
                .alias(cls_col),
                (pl.col(final_profit_col) > 0).cast(pl.Int8).alias(binary_col),
                pl.col(final_profit_col).abs().clip(1e-6, None).alias(final_profit_weight_col),
            ]
        )
    )


def opposite_asset_ids(samples: pl.DataFrame) -> list[str | None]:
    market_pairs: dict[str, dict[str, str | None]] = {}
    pairs = samples.select(["market_id", "asset_id", "outcome"]).unique()
    for market_id, group in pairs.partition_by("market_id", as_dict=True).items():
        rows = group.select(["asset_id", "outcome"]).to_dicts()
        if len(rows) != 2:
            continue
        a, b = rows
        market_pairs[str(partition_key_value(market_id))] = {
            str(a["asset_id"]): str(b["asset_id"]),
            str(b["asset_id"]): str(a["asset_id"]),
        }
    return [market_pairs.get(str(market_id), {}).get(str(asset_id)) for market_id, asset_id in samples.select(["market_id", "asset_id"]).iter_rows()]


def partition_key_value(key: object) -> object:
    if isinstance(key, tuple) and len(key) == 1:
        return key[0]
    return key


def future_window_min_prices(
    event_times: np.ndarray,
    event_prices: np.ndarray,
    query_times: np.ndarray,
    horizon_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(query_times, kind="stable")
    sorted_queries = query_times[order]
    out_prices = np.full(len(query_times), np.nan, dtype=float)
    out_times = np.full(len(query_times), -1, dtype=np.int64)
    candidates: deque[int] = deque()
    event_idx = 0
    for query_pos in order:
        query_time = query_times[query_pos]
        end_time = query_time + horizon_ns
        while event_idx < len(event_times) and event_times[event_idx] <= end_time:
            while candidates and event_prices[candidates[-1]] >= event_prices[event_idx]:
                candidates.pop()
            candidates.append(event_idx)
            event_idx += 1
        while candidates and event_times[candidates[0]] < query_time:
            candidates.popleft()
        if candidates:
            best_idx = candidates[0]
            out_prices[query_pos] = event_prices[best_idx]
            out_times[query_pos] = event_times[best_idx]
    return out_prices, out_times


def maker_fill_label_design() -> dict[str, object]:
    """Document the next layer without guessing labels from current snapshots."""
    return {
        "status": "designed_not_implemented",
        "unit": "first_leg_candidate",
        "required_inputs": [
            "future top-of-book updates for the candidate market/asset",
            "future public trades with side, price, and size",
            "candidate maker quote price and side",
            "queue position proxy or conservative no-priority assumption",
        ],
        "labels": [
            "fill_1s",
            "fill_3s",
            "fill_5s",
            "fill_10s",
            "time_to_fill_ms",
            "realized_second_leg_price",
            "best_possible_exit_price_within_horizon",
            "forced_exit_price_if_not_filled",
        ],
        "derivation": (
            "For each taker-entry candidate, place an opposite-side maker quote at "
            "the configured price, replay future book/trade events chronologically, "
            "mark fill only when future trade-through or conservative queue depletion "
            "evidence reaches the quote within the horizon, and use best bid/ask at "
            "horizon as the forced exit fallback."
        ),
    }
