"""Label construction for alpha and entry-worthiness models."""

from __future__ import annotations

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

