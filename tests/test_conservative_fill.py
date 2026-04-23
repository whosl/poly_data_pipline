from __future__ import annotations

import math

import polars as pl

from poly.predict.pipeline import NS_PER_MS, NS_PER_SECOND, OutcomeMonitor, PendingPrediction
from poly.training.labels import add_final_profit_labels, add_two_leg_maker_fill_labels


def _samples() -> pl.DataFrame:
    ts = NS_PER_SECOND
    return pl.DataFrame(
        [
            {
                "recv_ns": ts,
                "market_id": "m1",
                "asset_id": "A",
                "outcome": "Up",
                "best_ask": 0.50,
                "future_best_bid_10s": 0.48,
                "future_mid_10s": 0.49,
                "tick_size": 0.01,
            },
            {
                "recv_ns": ts,
                "market_id": "m1",
                "asset_id": "B",
                "outcome": "Down",
                "best_ask": 0.50,
                "future_best_bid_10s": 0.48,
                "future_mid_10s": 0.49,
                "tick_size": 0.01,
            },
        ]
    )


def _label_for_a(trades: pl.DataFrame) -> dict:
    labeled = add_two_leg_maker_fill_labels(
        _samples(),
        trades=trades,
        horizon_seconds=10,
        max_total_price=0.96,
        maker_fill_latency_ms=250,
        maker_fill_trade_through_ticks=1.0,
    )
    return labeled.filter(pl.col("asset_id") == "A").to_dicts()[0]


def test_exact_quote_does_not_fill() -> None:
    trades = pl.DataFrame([{"asset_id": "B", "recv_ns": NS_PER_SECOND + 300 * NS_PER_MS, "side": "SELL", "price": 0.46}])
    row = _label_for_a(trades)
    assert row["future_opposite_maker_fill_price_10s"] is None
    assert row["y_two_leg_entry_10s"] == "skip"


def test_one_tick_trade_through_fills_after_latency() -> None:
    trades = pl.DataFrame([{"asset_id": "B", "recv_ns": NS_PER_SECOND + 300 * NS_PER_MS, "side": "SELL", "price": 0.45}])
    row = _label_for_a(trades)
    assert math.isclose(row["future_opposite_maker_fill_price_10s"], 0.45)
    assert row["y_two_leg_entry_10s"] == "enter"


def test_trade_through_before_latency_does_not_fill() -> None:
    trades = pl.DataFrame([{"asset_id": "B", "recv_ns": NS_PER_SECOND + 100 * NS_PER_MS, "side": "SELL", "price": 0.45}])
    row = _label_for_a(trades)
    assert row["future_opposite_maker_fill_price_10s"] is None
    assert row["y_two_leg_entry_10s"] == "skip"


def test_trade_through_at_latency_boundary_fills() -> None:
    trades = pl.DataFrame([{"asset_id": "B", "recv_ns": NS_PER_SECOND + 250 * NS_PER_MS, "side": "SELL", "price": 0.45}])
    row = _label_for_a(trades)
    assert math.isclose(row["future_opposite_maker_fill_price_10s"], 0.45)
    assert row["y_two_leg_entry_10s"] == "enter"


def test_no_fill_uses_unwind_profit() -> None:
    row = _label_for_a(pl.DataFrame(schema={"asset_id": pl.String, "recv_ns": pl.Int64, "side": pl.String, "price": pl.Float64}))
    final = add_final_profit_labels(pl.DataFrame([row]), horizon_seconds=10).to_dicts()[0]
    expected = (1 - 0.072 * 0.51 * (1 - 0.51)) * 0.48 - 0.51
    assert final["y_two_leg_entry_10s"] == "skip"
    assert math.isclose(final["final_profit_10s"], expected)


def _prediction(track_stats: bool = True) -> PendingPrediction:
    return PendingPrediction(
        signal_id="S000001",
        predict_ns=NS_PER_SECOND,
        asset_id="A",
        market_id="m1",
        outcome="Up",
        signal_key="m1:Up",
        signal_seq_for_key=1,
        best_bid_at_entry=0.49,
        best_ask_at_entry=0.50,
        current_mid_at_entry=0.495,
        spread_at_entry=0.01,
        time_to_expiry_at_entry=60.0,
        proba=0.8,
        opposite_asset_id="B",
        slug="btc-updown-5m-1",
        second_leg_quote_price=0.46,
        tick_size_at_entry=0.01,
        track_stats=track_stats,
    )


def test_live_monitor_uses_same_trade_through_and_latency_rule() -> None:
    monitor = OutcomeMonitor(horizon_seconds=10, maker_fill_latency_ms=250, maker_fill_trade_through_ticks=1.0)
    pred = _prediction()
    monitor.add_signal(pred)
    monitor.observe_trade("B", "SELL", 0.46, NS_PER_SECOND + 300 * NS_PER_MS)
    assert pred.maker_fill_price is None
    monitor.observe_trade("B", "SELL", 0.45, NS_PER_SECOND + 100 * NS_PER_MS)
    assert pred.maker_fill_price is None
    monitor.observe_trade("B", "SELL", 0.45, NS_PER_SECOND + 300 * NS_PER_MS)
    assert pred.maker_fill_price == 0.45

    resolved = monitor.check_outcomes(11 * NS_PER_SECOND, {"A": {"best_bid": 0.48, "best_ask": 0.50, "current_mid": 0.49}})
    assert resolved[0][1]["result"] == "success"


def test_shadow_prediction_does_not_update_signal_stats() -> None:
    monitor = OutcomeMonitor(horizon_seconds=10, maker_fill_latency_ms=250, maker_fill_trade_through_ticks=1.0)
    pred = _prediction(track_stats=False)
    monitor.add_signal(pred)
    assert monitor.stats.total_signals == 0
    resolved = monitor.check_outcomes(11 * NS_PER_SECOND, {"A": {"best_bid": 0.48, "best_ask": 0.50, "current_mid": 0.49}})
    assert resolved[0][1]["result"] == "unwind"
    assert monitor.stats.total_resolved == 0
