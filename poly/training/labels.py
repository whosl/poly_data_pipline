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
    maker_fill_latency_ms: int = 250,
    maker_fill_trade_through_ticks: float = 1.0,
    tick_size_fallback: float = 0.01,
    first_leg_fill_validation_ms: int = 0,
    price_buffer: float = 0.01,
) -> pl.DataFrame:
    """Label whether a taker first leg plus future opposite maker fill clears max total price.

    The first leg is a taker buy at this row's `best_ask`. The second leg is a
    maker buy on the opposite outcome in the same market. A future opposite-side
    public trade is conservative fill evidence only after the configured latency
    and only if it trades through the maker quote by the configured tick count.
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
    latency_ns = max(0, int(maker_fill_latency_ms)) * 1_000_000
    horizon_ns = horizon_seconds * 1_000_000_000
    effective_horizon_ns = max(0, horizon_ns - latency_ns)

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
            query_times = group["recv_ns"].to_numpy() + latency_ns
            prices, times = future_window_min_prices(
                event_array[:, 0].astype(np.int64),
                event_array[:, 1].astype(float),
                query_times.astype(np.int64),
                effective_horizon_ns,
            )
            fill_prices[row_idx] = prices
            fill_times[row_idx] = times

    candidate_fill_price_col = "_candidate_opposite_maker_fill_price"
    candidate_fill_ts_col = "_candidate_opposite_maker_fill_recv_ns"
    fill_threshold_col = "_opposite_maker_fill_threshold"
    quote_col = "_opposite_maker_quote_price"
    tick_col = "_maker_fill_tick_size"
    fill_epsilon = 1e-12
    tick_expr = (
        pl.when(pl.col("tick_size").is_not_null() & (pl.col("tick_size") > 0))
        .then(pl.col("tick_size"))
        .otherwise(pl.lit(tick_size_fallback))
        if "tick_size" in working.columns
        else pl.lit(tick_size_fallback)
    )

    result = (
        working.with_columns(
            [
                pl.Series("opposite_asset_id", opposite_by_row),
                pl.Series(candidate_fill_price_col, fill_prices).fill_nan(None),
                pl.Series(candidate_fill_ts_col, fill_times).replace(-1, None),
            ]
        )
        .with_columns(
            [
                tick_expr.alias(tick_col),
                (pl.lit(max_total_price) - pl.col("best_ask")).alias(quote_col),
            ]
        )
        .with_columns(
            (pl.col(quote_col) - pl.lit(maker_fill_trade_through_ticks) * pl.col(tick_col)).alias(fill_threshold_col)
        )
        .with_columns(
            [
                pl.when(pl.col(candidate_fill_price_col) <= pl.col(fill_threshold_col) + pl.lit(fill_epsilon))
                .then(pl.col(candidate_fill_price_col))
                .otherwise(None)
                .alias(fill_price_col),
                pl.when(pl.col(candidate_fill_price_col) <= pl.col(fill_threshold_col) + pl.lit(fill_epsilon))
                .then(pl.col(candidate_fill_ts_col))
                .otherwise(None)
                .alias(fill_ts_col),
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
    )

    # First-leg fill validation: check if best_ask stayed at/below fill price
    # within validation window. If price moved away, the first leg couldn't fill
    # so this row should not be labeled as "enter".
    if first_leg_fill_validation_ms > 0:
        validation_ns = first_leg_fill_validation_ms * 1_000_000
        fill_price = pl.col("best_ask") + pl.lit(price_buffer)
        result = result.sort(["asset_id", "recv_ns"]).with_columns(
            [
                pl.col("best_ask").shift(-1).over("asset_id").alias("_next_best_ask"),
                pl.col("recv_ns").shift(-1).over("asset_id").alias("_next_recv_ns"),
            ]
        ).with_columns(
            [
                # Next row is within validation window AND price moved above fill_price
                pl.when(
                    pl.col("_next_recv_ns").is_not_null()
                    & ((pl.col("_next_recv_ns") - pl.col("recv_ns")) <= pl.lit(validation_ns))
                    & (pl.col("_next_best_ask") > fill_price)
                )
                .then(pl.lit("skip"))
                .otherwise(pl.col(cls_col))
                .alias(cls_col),
                pl.when(
                    pl.col("_next_recv_ns").is_not_null()
                    & ((pl.col("_next_recv_ns") - pl.col("recv_ns")) <= pl.lit(validation_ns))
                    & (pl.col("_next_best_ask") > fill_price)
                )
                .then(pl.lit(0))
                .otherwise(pl.col(binary_col))
                .alias(binary_col),
            ]
        ).drop(["_next_best_ask", "_next_recv_ns"])

    return result.drop([candidate_fill_price_col, candidate_fill_ts_col, fill_threshold_col, quote_col, tick_col, "_row_nr"], strict=False)


def add_first_leg_fill_labels(
    samples: pl.DataFrame,
    validation_ms: int = 350,
    price_buffer: float = 0.01,
) -> pl.DataFrame:
    """Label whether the first-leg taker buy at best_ask would fill after latency.

    Scans ALL future snapshots within the validation window (not just the next
    one). A fill fails if ANY future best_ask exceeds entry_ask + price_buffer
    within the window. This models the bot's end-to-end latency: the order
    reaches the exchange validation_ms after the orderbook snapshot.

    Uses a two-pointer sliding-window-max approach per asset for efficiency.
    """
    label_col = "y_first_leg_fill"
    if samples.is_empty():
        return samples.with_columns(pl.lit(None).cast(pl.Int8).alias(label_col))

    validation_ns = validation_ms * 1_000_000
    samples = samples.sort(["asset_id", "recv_ns"])

    asset_ids = samples["asset_id"].to_numpy()
    recv_ns = samples["recv_ns"].to_numpy().astype(np.int64)
    best_ask = samples["best_ask"].to_numpy().astype(np.float64)

    n = len(samples)
    labels = np.ones(n, dtype=np.int8)  # default: fill

    # Per-asset two-pointer sliding window max
    unique_assets = np.unique(asset_ids)
    for asset in unique_assets:
        mask = asset_ids == asset
        idx = np.where(mask)[0]
        a_ns = recv_ns[idx]
        a_ask = best_ask[idx]
        m = len(idx)

        j = 0
        for i in range(m):
            entry_ask = a_ask[i]
            threshold = entry_ask + price_buffer
            end_time = a_ns[i] + validation_ns

            if j < i + 1:
                j = i + 1
            # Scan forward — check if any snapshot in window has ask > threshold
            while j < m and a_ns[j] <= end_time:
                if a_ask[j] > threshold:
                    labels[idx[i]] = 0
                    break
                j += 1
            # Reset j for next row only if it went too far
            if j > i + 1:
                j = i + 1

    # Null out rows where best_ask is null or no future data
    ask_null = np.isnan(best_ask)
    labels[ask_null] = -1  # sentinel for null

    return samples.with_columns(
        pl.when(pl.Series(labels) == -1)
        .then(None)
        .otherwise(pl.Series(labels))
        .cast(pl.Int8)
        .alias(label_col)
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
            "mark fill only after the configured latency when future public trades "
            "trade through the quote by the configured tick threshold, and use best "
            "bid/ask at horizon as the forced exit fallback."
        ),
    }
