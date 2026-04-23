"""Live prediction pipeline: WebSocket data → features → model prediction + outcome monitoring."""

from __future__ import annotations

import asyncio
import collections
from collections import Counter
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

import pandas as pd
import joblib

logger = structlog.get_logger()

NS_PER_MS = 1_000_000
NS_PER_SECOND = 1_000_000_000

# EWMA decay factors: alpha = 2 / (span + 1)
_EWMA_ALPHA_FAST = 2.0 / (5 + 1)   # span=5,  ~500ms half-life at 100ms
_EWMA_ALPHA_SLOW = 2.0 / (20 + 1)  # span=20, ~2s half-life at 100ms


def _ewma_update(prev: float | None, new_val: float | None, alpha: float) -> float | None:
    """Incrementally update an EWMA value. Returns None if new_val is None."""
    if new_val is None:
        return prev
    if prev is None:
        return new_val
    return alpha * new_val + (1.0 - alpha) * prev


def short_id(value: str | None, n: int = 8) -> str:
    return str(value or "")[:n]


def short_slug(value: str | None) -> str:
    slug = str(value or "")
    return slug.split("-updown-")[-1] if "-updown-" in slug else slug[:24]


def fmt_price(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def fmt_float(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def fmt_ms(delta_ns: int | float | None) -> str:
    return "" if delta_ns is None else f"{float(delta_ns) / NS_PER_MS:.0f}"


def quantile_summary(values: list[float]) -> dict[str, float | None]:
    clean = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if clean.size == 0:
        return {"min": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "max": None}
    qs = np.quantile(clean, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return {
        "min": float(qs[0]),
        "p10": float(qs[1]),
        "p25": float(qs[2]),
        "p50": float(qs[3]),
        "p75": float(qs[4]),
        "p90": float(qs[5]),
        "max": float(qs[6]),
    }


# ---------------------------------------------------------------------------
# Rolling window helper
# ---------------------------------------------------------------------------
class RollingWindow:
    """Time-based rolling window for numeric values."""

    def __init__(self, window_ns: int):
        self._window_ns = window_ns
        self._data: deque[tuple[int, float]] = deque()

    def add(self, ts_ns: int, value: float) -> None:
        self._data.append((ts_ns, value))
        self._evict(ts_ns)

    def _evict(self, now_ns: int) -> None:
        cutoff = now_ns - self._window_ns
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

    def sum(self, now_ns: int) -> float:
        self._evict(now_ns)
        return sum(v for _, v in self._data)

    def count(self, now_ns: int) -> int:
        self._evict(now_ns)
        return len(self._data)

    def values(self, now_ns: int) -> list[float]:
        self._evict(now_ns)
        return [v for _, v in self._data]


# ---------------------------------------------------------------------------
# Live feature state per asset
# ---------------------------------------------------------------------------
@dataclass
class AssetState:
    """Maintains rolling feature state for a single asset."""

    asset_id: str
    market_id: str = ""
    outcome: str = ""
    slug: str = ""
    tick_size: float = 0.01
    min_order_size: float = 1.0
    maker_base_fee: float = 0.0
    taker_base_fee: float = 0.0
    expiry_ns: int = 0

    # Current book state (from Rust engine)
    best_bid: float = 0.0
    best_ask: float = 0.0
    current_mid: float = 0.0
    current_spread: float = 0.0
    relative_spread: float = 0.0
    current_microprice: float = 0.0
    top1_imbalance: float = 0.0
    total_bid_levels: int = 0
    total_ask_levels: int = 0
    depth_features: dict[str, float | None] = field(default_factory=dict)

    # Book event rolling windows
    book_updates_100ms: RollingWindow = field(default_factory=lambda: RollingWindow(100 * NS_PER_MS))
    book_updates_500ms: RollingWindow = field(default_factory=lambda: RollingWindow(500 * NS_PER_MS))
    book_updates_1s: RollingWindow = field(default_factory=lambda: RollingWindow(NS_PER_SECOND))
    spread_deltas_1s: RollingWindow = field(default_factory=lambda: RollingWindow(NS_PER_SECOND))
    mid_history: deque = field(default_factory=lambda: deque(maxlen=200))

    # Trade rolling windows
    trades_100ms: RollingWindow = field(default_factory=lambda: RollingWindow(100 * NS_PER_MS))
    trades_500ms: RollingWindow = field(default_factory=lambda: RollingWindow(500 * NS_PER_MS))
    trades_1s: RollingWindow = field(default_factory=lambda: RollingWindow(NS_PER_SECOND))
    buy_vol_1s: RollingWindow = field(default_factory=lambda: RollingWindow(NS_PER_SECOND))
    sell_vol_1s: RollingWindow = field(default_factory=lambda: RollingWindow(NS_PER_SECOND))
    signed_vol_1s: RollingWindow = field(default_factory=lambda: RollingWindow(NS_PER_SECOND))
    trade_notional_1s: RollingWindow = field(default_factory=lambda: RollingWindow(NS_PER_SECOND))
    trade_size_1s: RollingWindow = field(default_factory=lambda: RollingWindow(NS_PER_SECOND))
    recent_trade_sides: deque = field(default_factory=lambda: deque(maxlen=10))

    # Sampling control
    last_sample_ns: int = 0
    sample_interval_ns: int = 100 * NS_PER_MS  # 100ms

    # Opposite asset
    opposite_asset_id: str | None = None

    # EWMA state for smoothed features (fast span=5, slow span=20)
    realized_vol_ewma_fast: float | None = None
    realized_vol_ewma_slow: float | None = None
    depth_top10_imbalance_ewma_fast: float | None = None
    depth_top10_imbalance_ewma_slow: float | None = None
    binance_return_1s_ewma_fast: float | None = None
    binance_return_1s_ewma_slow: float | None = None


# ---------------------------------------------------------------------------
# Binance state
# ---------------------------------------------------------------------------
@dataclass
class BinanceState:
    """Maintains state for Binance reference market."""

    mid: float = 0.0
    spread: float = 0.0
    depth_features: dict[str, float | None] = field(default_factory=dict)
    mid_history: deque = field(default_factory=lambda: deque(maxlen=200))
    recent_trade_imbalance: float = 0.0
    last_recv_ns: int = 0


# ---------------------------------------------------------------------------
# Live feature assembler
# ---------------------------------------------------------------------------
class LiveFeatureAssembler:
    """Assembles 108 features from live market data for model prediction."""

    def __init__(self, feature_columns: list[str], cat_columns: list[str]):
        self.feature_columns = feature_columns
        self.cat_columns = cat_columns
        self.num_columns = [c for c in feature_columns if c not in cat_columns]

    def assemble(self, asset: AssetState, binance: BinanceState) -> dict[str, float | str | None]:
        """Build feature dict for a single asset given current state."""
        now_ns = int(time.time() * NS_PER_SECOND)
        features: dict[str, float | str | None] = {}

        # --- Time / metadata features ---
        if asset.expiry_ns > 0:
            tte = (asset.expiry_ns - now_ns) / NS_PER_SECOND
        else:
            tte = 0.0
        features["time_to_expiry_seconds"] = tte

        # --- Poly book basics ---
        features["best_bid"] = asset.best_bid
        features["best_ask"] = asset.best_ask
        features["current_mid"] = asset.current_mid
        features["current_spread"] = asset.current_spread
        features["relative_spread"] = asset.relative_spread
        features["current_microprice"] = asset.current_microprice
        features["top1_imbalance"] = asset.top1_imbalance
        features["total_bid_levels"] = float(asset.total_bid_levels)
        features["total_ask_levels"] = float(asset.total_ask_levels)

        # --- Depth features (from Rust engine) ---
        for key, val in asset.depth_features.items():
            features[key] = val

        # --- Metadata ---
        features["tick_size"] = asset.tick_size
        features["min_order_size"] = asset.min_order_size
        features["maker_base_fee"] = asset.maker_base_fee
        features["taker_base_fee"] = asset.taker_base_fee

        # --- LOB derived features ---
        features["top3_imbalance"] = asset.depth_features.get("depth_top3_imbalance", asset.top1_imbalance)
        features["top5_imbalance"] = asset.depth_features.get("depth_top5_imbalance", asset.top1_imbalance)
        features["top10_imbalance"] = asset.depth_features.get("depth_top10_imbalance", asset.top1_imbalance)
        features["cum_bid_depth_topN_proxy"] = asset.depth_features.get("cum_bid_depth_top10", float(asset.total_bid_levels))
        features["cum_ask_depth_topN_proxy"] = asset.depth_features.get("cum_ask_depth_top10", float(asset.total_ask_levels))
        depth10_imb = asset.depth_features.get("depth_top10_imbalance")
        features["depth_level_imbalance_proxy"] = depth10_imb if depth10_imb is not None else 0.0
        features["bid_depth_slope"] = asset.depth_features.get("bid_depth_slope_top10")
        features["ask_depth_slope"] = asset.depth_features.get("ask_depth_slope_top10")

        # --- Book event features ---
        features["book_update_count_100ms"] = float(asset.book_updates_100ms.count(now_ns))
        features["book_update_count_500ms"] = float(asset.book_updates_500ms.count(now_ns))
        features["book_update_count_1s"] = float(asset.book_updates_1s.count(now_ns))
        spread_deltas = asset.spread_deltas_1s.values(now_ns)
        features["spread_widen_count_recent"] = float(sum(1 for d in spread_deltas if d > 0))
        features["spread_narrow_count_recent"] = float(sum(1 for d in spread_deltas if d < 0))
        # Realized vol
        mid_vals = list(asset.mid_history)
        if len(mid_vals) >= 3:
            rets = np.diff(np.log(np.array(mid_vals[-30:]))) if len(mid_vals) >= 30 else np.diff(np.log(np.array(mid_vals)))
            features["realized_vol_short"] = float(np.std(rets)) if len(rets) > 1 else 0.0
        else:
            features["realized_vol_short"] = 0.0

        # --- Poly trade features ---
        features["poly_trade_count_100ms"] = float(asset.trades_100ms.count(now_ns))
        features["poly_trade_count_500ms"] = float(asset.trades_500ms.count(now_ns))
        features["poly_trade_count_1s"] = float(asset.trades_1s.count(now_ns))
        features["poly_trade_count_recent"] = float(asset.trades_1s.count(now_ns))
        buy_vol = asset.buy_vol_1s.sum(now_ns)
        sell_vol = asset.sell_vol_1s.sum(now_ns)
        signed_vol = asset.signed_vol_1s.sum(now_ns)
        features["poly_aggressive_buy_volume_1s"] = buy_vol
        features["poly_aggressive_sell_volume_1s"] = sell_vol
        features["poly_signed_volume_1s"] = signed_vol
        features["poly_aggressive_buy_volume_recent"] = buy_vol
        features["poly_aggressive_sell_volume_recent"] = sell_vol
        features["poly_signed_volume_recent"] = signed_vol
        total_vol = buy_vol + sell_vol
        features["poly_signed_volume_imbalance_recent"] = signed_vol / (total_vol + 1e-12)
        notional_1s = asset.trade_notional_1s.sum(now_ns)
        size_1s = asset.trade_size_1s.sum(now_ns)
        if size_1s > 1e-12:
            vwap = notional_1s / size_1s
            features["poly_recent_vwap"] = vwap
            features["poly_recent_vwap_deviation"] = vwap - asset.current_mid if asset.current_mid > 0 else 0.0
        else:
            features["poly_recent_vwap"] = None
            features["poly_recent_vwap_deviation"] = None

        # Consecutive trade run
        sides = list(asset.recent_trade_sides)
        buy_run = sum(1 for s in sides if s > 0)
        sell_run = sum(1 for s in sides if s < 0)
        features["consecutive_buy_trade_run_proxy"] = float(buy_run)
        features["consecutive_sell_trade_run_proxy"] = float(sell_run)

        # --- Binance features ---
        features["binance_mid"] = binance.mid
        features["binance_spread"] = binance.spread
        for key, val in binance.depth_features.items():
            features[f"binance_{key}"] = val

        # Binance returns
        bin_mids = list(binance.mid_history)
        if len(bin_mids) >= 2:
            features["binance_return_tick"] = (bin_mids[-1] - bin_mids[-2]) / bin_mids[-2] if bin_mids[-2] > 0 else 0.0
        else:
            features["binance_return_tick"] = 0.0
        if len(bin_mids) >= 6:
            features["binance_return_100ms"] = (bin_mids[-1] - bin_mids[-2]) / bin_mids[-2] if bin_mids[-2] > 0 else 0.0
            features["binance_return_500ms"] = (bin_mids[-1] - bin_mids[-6]) / bin_mids[-6] if bin_mids[-6] > 0 else 0.0
        else:
            features["binance_return_100ms"] = 0.0
            features["binance_return_500ms"] = 0.0
        if len(bin_mids) >= 11:
            features["binance_return_1s"] = (bin_mids[-1] - bin_mids[-11]) / bin_mids[-11] if bin_mids[-11] > 0 else 0.0
        else:
            features["binance_return_1s"] = 0.0
        if len(bin_mids) >= 31:
            features["binance_return_3s"] = (bin_mids[-1] - bin_mids[-31]) / bin_mids[-31] if bin_mids[-31] > 0 else 0.0
        else:
            features["binance_return_3s"] = 0.0
        features["binance_recent_trade_imbalance"] = binance.recent_trade_imbalance

        # --- Cross features ---
        mid_vals = list(asset.mid_history)
        if len(mid_vals) >= 11:
            features["poly_return_1s"] = (mid_vals[-1] - mid_vals[-11]) / mid_vals[-11] if mid_vals[-11] > 0 else 0.0
        else:
            features["poly_return_1s"] = 0.0
        features["lead_lag_binance_minus_poly_1s"] = features.get("binance_return_1s", 0.0) - features.get("poly_return_1s", 0.0)
        features["lead_lag_binance_minus_poly_500ms"] = features.get("binance_return_500ms", 0.0) - (
            (mid_vals[-1] - mid_vals[-6]) / mid_vals[-6] if len(mid_vals) >= 6 and mid_vals[-6] > 0 else 0.0
        )

        # --- EWMA features (incremental update on asset state) ---
        rv = features.get("realized_vol_short")
        asset.realized_vol_ewma_fast = _ewma_update(asset.realized_vol_ewma_fast, rv, _EWMA_ALPHA_FAST)
        asset.realized_vol_ewma_slow = _ewma_update(asset.realized_vol_ewma_slow, rv, _EWMA_ALPHA_SLOW)
        features["realized_vol_ewma_fast"] = asset.realized_vol_ewma_fast
        features["realized_vol_ewma_slow"] = asset.realized_vol_ewma_slow
        if asset.realized_vol_ewma_fast is not None and asset.realized_vol_ewma_slow is not None:
            features["realized_vol_ewma_diff"] = asset.realized_vol_ewma_fast - asset.realized_vol_ewma_slow
        else:
            features["realized_vol_ewma_diff"] = None

        d10i = asset.depth_features.get("depth_top10_imbalance")
        asset.depth_top10_imbalance_ewma_fast = _ewma_update(asset.depth_top10_imbalance_ewma_fast, d10i, _EWMA_ALPHA_FAST)
        asset.depth_top10_imbalance_ewma_slow = _ewma_update(asset.depth_top10_imbalance_ewma_slow, d10i, _EWMA_ALPHA_SLOW)
        features["depth_top10_imbalance_ewma_fast"] = asset.depth_top10_imbalance_ewma_fast
        features["depth_top10_imbalance_ewma_slow"] = asset.depth_top10_imbalance_ewma_slow
        if asset.depth_top10_imbalance_ewma_fast is not None and asset.depth_top10_imbalance_ewma_slow is not None:
            features["depth_top10_imbalance_ewma_diff"] = asset.depth_top10_imbalance_ewma_fast - asset.depth_top10_imbalance_ewma_slow
        else:
            features["depth_top10_imbalance_ewma_diff"] = None

        br1 = features.get("binance_return_1s")
        asset.binance_return_1s_ewma_fast = _ewma_update(asset.binance_return_1s_ewma_fast, br1, _EWMA_ALPHA_FAST)
        asset.binance_return_1s_ewma_slow = _ewma_update(asset.binance_return_1s_ewma_slow, br1, _EWMA_ALPHA_SLOW)
        features["binance_return_1s_ewma_fast"] = asset.binance_return_1s_ewma_fast
        features["binance_return_1s_ewma_slow"] = asset.binance_return_1s_ewma_slow
        if asset.binance_return_1s_ewma_fast is not None and asset.binance_return_1s_ewma_slow is not None:
            features["binance_return_1s_ewma_diff"] = asset.binance_return_1s_ewma_fast - asset.binance_return_1s_ewma_slow
        else:
            features["binance_return_1s_ewma_diff"] = None

        # --- Categorical buckets (simple thresholds) ---
        imb = features.get("top1_imbalance", 0.0) or 0.0
        features["imbalance_bucket"] = "neg_high" if imb < -0.3 else "neg_low" if imb < 0 else "pos_low" if imb < 0.3 else "pos_high"
        spr = features.get("current_spread", 0.0) or 0.0
        mid = features.get("current_mid", 0.01) or 0.01
        rel_spread = spr / mid
        features["spread_bucket"] = "tight" if rel_spread < 0.005 else "normal" if rel_spread < 0.02 else "wide"
        features["price_bucket"] = "low" if mid < 0.1 else "mid" if mid < 0.5 else "high" if mid < 0.9 else "very_high"
        vol = features.get("realized_vol_short", 0.0) or 0.0
        features["vol_bucket"] = "low" if vol < 0.0001 else "mid" if vol < 0.001 else "high"

        # --- Fill missing features with None/0 ---
        for col in self.feature_columns:
            if col not in features:
                features[col] = 0.0 if col in self.num_columns else "unknown"

        return features


# ---------------------------------------------------------------------------
# Outcome monitor
# ---------------------------------------------------------------------------
@dataclass
class PendingPrediction:
    """A prediction waiting for 10s outcome evaluation."""

    signal_id: str
    predict_ns: int
    asset_id: str
    market_id: str
    outcome: str
    signal_key: str
    signal_seq_for_key: int
    best_bid_at_entry: float
    best_ask_at_entry: float
    current_mid_at_entry: float
    spread_at_entry: float
    time_to_expiry_at_entry: float
    proba: float
    pred_unwind_profit: float | None = None
    pred_expected_profit: float | None = None
    decision_score: float | None = None
    opposite_asset_id: str | None = None
    slug: str = ""
    two_leg_max_total_price: float = 0.96
    fee_rate: float = 0.072
    price_buffer: float = 0.01
    second_leg_quote_price: float = 0.0
    maker_fill_price: float | None = None
    maker_fill_ns: int | None = None
    future_best_bid_at_resolve: float | None = None
    future_best_ask_at_resolve: float | None = None
    future_mid_at_resolve: float | None = None


@dataclass
class MonitorStats:
    """Running statistics for prediction monitoring."""

    total_predictions: int = 0
    total_signals: int = 0  # predictions above threshold
    total_resolved: int = 0
    total_success: int = 0
    total_failure: int = 0
    total_profit: float = 0.0
    recent_profits: deque = field(default_factory=lambda: deque(maxlen=100))

    def accuracy(self) -> float:
        return self.total_success / max(self.total_resolved, 1)

    def avg_profit(self) -> float:
        if not self.recent_profits:
            return 0.0
        return sum(self.recent_profits) / len(self.recent_profits)

    def summary(self) -> str:
        return (
            f"signals={self.total_signals}  "
            f"resolved={self.total_resolved}  "
            f"accuracy={self.accuracy():.1%}  "
            f"total_profit={self.total_profit:+.4f}  "
            f"avg_profit={self.avg_profit():+.4f}  "
            f"recent_n={len(self.recent_profits)}"
        )


class OutcomeMonitor:
    """Monitors 10-second outcomes for each prediction signal."""

    def __init__(
        self,
        horizon_seconds: int = 10,
        fee_rate: float = 0.072,
        price_buffer: float = 0.01,
        maker_fill_trade_side: str = "SELL",
    ):
        self.horizon_seconds = horizon_seconds
        self.fee_rate = fee_rate
        self.price_buffer = price_buffer
        self.maker_fill_trade_side = maker_fill_trade_side.upper()
        self.pending: list[PendingPrediction] = []
        self.stats = MonitorStats()

    def add_signal(self, pred: PendingPrediction) -> None:
        self.pending.append(pred)
        self.stats.total_signals += 1

    def observe_trade(self, asset_id: str, side: str, price: float, recv_ns: int) -> None:
        """Record future opposite trade evidence for pending maker fills.

        This mirrors the offline label: a pending first-leg entry succeeds only
        if the opposite asset has a future trade on the configured side at or
        better than our maker quote within the horizon.
        """
        if side.upper() != self.maker_fill_trade_side:
            return
        for pred in self.pending:
            if pred.maker_fill_price is not None:
                continue
            if pred.opposite_asset_id != asset_id:
                continue
            if recv_ns <= pred.predict_ns:
                continue
            if recv_ns > pred.predict_ns + self.horizon_seconds * NS_PER_SECOND:
                continue
            if price <= pred.second_leg_quote_price:
                pred.maker_fill_price = price
                pred.maker_fill_ns = recv_ns
                logger.info(
                    "maker_fill_observed",
                    signal_id=pred.signal_id,
                    signal_key=pred.signal_key,
                    slug=short_slug(pred.slug),
                    outcome=pred.outcome,
                    asset=short_id(pred.asset_id),
                    opposite_asset=short_id(pred.opposite_asset_id),
                    quote_price=fmt_price(pred.second_leg_quote_price),
                    trade_price=fmt_price(price),
                    fill_lag_ms=fmt_ms(recv_ns - pred.predict_ns),
                )

    def check_outcomes(
        self,
        now_ns: int,
        asset_books: dict[str, dict],
    ) -> list[tuple[PendingPrediction, dict[str, object]]]:
        """Check all pending predictions. Returns list of (prediction, outcome details)."""
        resolved = []
        still_pending = []
        for pred in self.pending:
            elapsed_ns = now_ns - pred.predict_ns
            if elapsed_ns < self.horizon_seconds * NS_PER_SECOND:
                still_pending.append(pred)
                continue

            # Evaluate outcome
            resolved.append((pred, self._evaluate(pred, asset_books)))

        self.pending = still_pending
        return resolved

    def _evaluate(
        self,
        pred: PendingPrediction,
        asset_books: dict[str, dict],
    ) -> dict[str, object]:
        """Evaluate a single prediction's outcome using actual trading mechanics.

        Two-leg strategy:
        - First leg: taker BUY this asset at (best_ask + price_buffer)
        - Fee deducted in shares: fee = fee_rate * first_leg_price * (1 - first_leg_price)
        - Second leg size = 1 - fee (shares remaining after fee)
        - Second leg: maker quote opposite asset at (max_total_price - best_ask)
        - If a future opposite trade fills that quote within the horizon → success
        - Otherwise → failure, unwind the first leg at future same-leg best_bid
        """
        # Compute entry parameters (same as labels.py formula)
        entry_price = pred.best_ask_at_entry
        first_leg_price = entry_price + pred.price_buffer
        fee_per_share = pred.fee_rate * first_leg_price * (1.0 - first_leg_price)
        second_leg_size = 1.0 - fee_per_share
        second_leg_price = pred.second_leg_quote_price or (pred.two_leg_max_total_price - entry_price)
        success_first_leg_cost = first_leg_price
        success_second_leg_cost = second_leg_size * second_leg_price
        success_total_cost = success_first_leg_cost + success_second_leg_cost
        success_revenue = second_leg_size

        # Get current state of this asset (for unwind)
        my_book = asset_books.get(pred.asset_id, {})
        future_best_bid = my_book.get("best_bid", 0.0)
        future_best_ask = my_book.get("best_ask", 0.0)
        future_mid = my_book.get("current_mid", 0.0)
        pred.future_best_bid_at_resolve = future_best_bid
        pred.future_best_ask_at_resolve = future_best_ask
        pred.future_mid_at_resolve = future_mid

        if pred.maker_fill_price is not None:
            # Success profit mirrors add_final_profit_labels: use our posted
            # quote price, while trade evidence only proves the quote was
            # marketable/fillable within the horizon.
            profit = success_revenue - success_total_cost
            self.stats.total_success += 1
            self.stats.recent_profits.append(profit)
            self.stats.total_profit += profit
            self.stats.total_resolved += 1
            return {
                "result": "success",
                "profit": profit,
                "first_leg_price": first_leg_price,
                "fee_per_share": fee_per_share,
                "second_leg_size": second_leg_size,
                "second_leg_quote": second_leg_price,
                "second_leg_fill_price": pred.maker_fill_price,
                "second_leg_fill_lag_ms": (pred.maker_fill_ns - pred.predict_ns) / NS_PER_MS
                if pred.maker_fill_ns is not None
                else None,
                "success_revenue": success_revenue,
                "first_leg_cost": success_first_leg_cost,
                "second_leg_cost": success_second_leg_cost,
                "total_cost": success_total_cost,
                "future_best_bid": future_best_bid,
                "future_best_ask": future_best_ask,
                "future_mid": future_mid,
            }

        # No fill evidence: unwind the first leg at future same-leg best_bid.
        # This is signed PnL, so a favorable first-leg move can still be
        # profitable even when the maker exit never fills.
        unwind_revenue = second_leg_size * future_best_bid
        profit = unwind_revenue - first_leg_price
        result = "unwind" if pred.opposite_asset_id else "no_opposite"
        if pred.opposite_asset_id:
            self.stats.total_failure += 1
            self.stats.recent_profits.append(profit)
            self.stats.total_profit += profit
            self.stats.total_resolved += 1
            return {
                "result": result,
                "profit": profit,
                "first_leg_price": first_leg_price,
                "fee_per_share": fee_per_share,
                "second_leg_size": second_leg_size,
                "second_leg_quote": second_leg_price,
                "second_leg_fill_price": None,
                "second_leg_fill_lag_ms": None,
                "unwind_revenue": unwind_revenue,
                "unwind_price": future_best_bid,
                "unwind_price_move": future_best_bid - first_leg_price,
                "first_leg_cost": first_leg_price,
                "future_best_bid": future_best_bid,
                "future_best_ask": future_best_ask,
                "future_mid": future_mid,
            }

        self.stats.total_failure += 1
        self.stats.recent_profits.append(profit)
        self.stats.total_profit += profit
        self.stats.total_resolved += 1
        return {
            "result": result,
            "profit": profit,
            "first_leg_price": first_leg_price,
            "fee_per_share": fee_per_share,
            "second_leg_size": second_leg_size,
            "second_leg_quote": second_leg_price,
            "second_leg_fill_price": None,
            "second_leg_fill_lag_ms": None,
            "unwind_revenue": unwind_revenue,
            "unwind_price": future_best_bid,
            "unwind_price_move": future_best_bid - first_leg_price,
            "first_leg_cost": first_leg_price,
            "future_best_bid": future_best_bid,
            "future_best_ask": future_best_ask,
            "future_mid": future_mid,
        }


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------
class PredictionPipeline:
    """Live prediction pipeline connecting to WebSocket feeds."""

    def __init__(
        self,
        model_path: str | Path,
        unwind_model_path: str | Path | None = None,
        threshold: float = 0.6,
        min_p_fill: float = 0.0,
        min_pred_unwind_profit: float = -1.0,
        sample_interval_ms: int = 100,
        horizon_seconds: int = 10,
        fee_rate: float = 0.072,
        price_buffer: float = 0.01,
        signal_cooldown_seconds: float | None = None,
        log_near_threshold: bool = False,
        stats_interval: int = 1000,
        min_entry_ask: float = 0.05,
        max_entry_ask: float = 0.95,
        min_time_to_expiry_seconds: float = 20.0,
        max_spread: float = 0.05,
        max_entries_per_signal_key: int = 0,
    ):
        self.threshold = threshold
        self.min_p_fill = min_p_fill
        self.min_pred_unwind_profit = min_pred_unwind_profit
        self.sample_interval_ns = sample_interval_ms * NS_PER_MS
        self.horizon_seconds = horizon_seconds
        self.fee_rate = fee_rate
        self.price_buffer = price_buffer
        self.signal_cooldown_ns = int((signal_cooldown_seconds if signal_cooldown_seconds is not None else horizon_seconds) * NS_PER_SECOND)
        self.log_near_threshold = log_near_threshold
        self.stats_interval = max(1, stats_interval)
        self.min_entry_ask = min_entry_ask
        self.max_entry_ask = max_entry_ask
        self.min_time_to_expiry_seconds = min_time_to_expiry_seconds
        self.max_spread = max_spread
        self.max_entries_per_signal_key = max(0, int(max_entries_per_signal_key))

        # Load model. In two-stage mode, model_path is the fill classifier and
        # unwind_model_path is a signed-PnL regressor for the failure branch.
        blob = joblib.load(str(model_path))
        self.pipeline = blob["model"]
        self.unwind_pipeline = None
        unwind_feature_cols: list[str] = []
        if unwind_model_path is not None:
            unwind_blob = joblib.load(str(unwind_model_path))
            self.unwind_pipeline = unwind_blob["model"]
            unwind_feature_cols = unwind_blob.get("feature_columns", [])
        model_feature_cols = blob.get("feature_columns", [])

        # Load training metadata for feature column order
        metadata_path = Path(model_path).parent / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            self.feature_columns = meta["feature_columns"]
            self.cat_columns = meta.get("categorical_feature_columns", [])
        elif model_feature_cols:
            self.feature_columns = model_feature_cols
            self.cat_columns = [c for c in model_feature_cols if "bucket" in c]
        else:
            raise ValueError("Cannot determine feature columns from model or metadata")
        for col in unwind_feature_cols:
            if col not in self.feature_columns:
                self.feature_columns.append(col)
        self.model_feature_columns = model_feature_cols or self.feature_columns
        self.unwind_feature_columns = unwind_feature_cols or self.feature_columns

        self.assembler = LiveFeatureAssembler(self.feature_columns, self.cat_columns)
        self.monitor = OutcomeMonitor(horizon_seconds, fee_rate, price_buffer)

        # State
        self.asset_states: dict[str, AssetState] = {}
        self.binance_state = BinanceState()
        self.engine = None  # OrderBookEngine, initialized on start

        # Market mapping: market_id -> {asset_id -> outcome}
        self.market_assets: dict[str, dict[str, str]] = {}
        # asset_id -> opposite asset_id
        self.opposite_map: dict[str, str] = {}

        self._running = False
        self._signal_count_since_display = 0
        self._last_proba_by_asset: dict[str, float] = {}
        self._last_signal_ns_by_key: dict[str, int] = {}
        self._signal_count_by_key: dict[str, int] = {}
        self._next_signal_id = 1
        self._score_window: deque[dict[str, float | None]] = deque(maxlen=10_000)
        self._block_counts: Counter[str] = Counter()

    def _get_or_create_asset(self, asset_id: str) -> AssetState:
        if asset_id not in self.asset_states:
            self.asset_states[asset_id] = AssetState(asset_id=asset_id)
        return self.asset_states[asset_id]

    def update_market_mapping(self, market_id: str, asset_id: str, outcome: str) -> None:
        """Update market_id -> {asset_id -> outcome} mapping."""
        if not market_id or not asset_id:
            return
        if market_id not in self.market_assets:
            self.market_assets[market_id] = {}
        self.market_assets[market_id][asset_id] = outcome

        # If we have 2 assets for this market, set opposite mapping
        assets = self.market_assets[market_id]
        if len(assets) == 2:
            ids = list(assets.keys())
            self.opposite_map[ids[0]] = ids[1]
            self.opposite_map[ids[1]] = ids[0]
            self._get_or_create_asset(ids[0]).opposite_asset_id = ids[1]
            self._get_or_create_asset(ids[1]).opposite_asset_id = ids[0]

    def handle_poly_book(self, asset_id: str, features: dict, recv_ns: int, metadata: dict | None = None) -> None:
        """Process a poly orderbook update and potentially run prediction."""
        from poly.engine.orderbook import extract_depth_features

        asset = self._get_or_create_asset(asset_id)
        prev_mid = asset.current_mid
        prev_spread = asset.current_spread

        # Update metadata if provided
        if metadata:
            for key in ["tick_size", "min_order_size", "maker_base_fee", "taker_base_fee"]:
                if key in metadata and metadata[key] is not None:
                    setattr(asset, key, float(metadata[key]))
            for key in ["market_id", "outcome", "slug"]:
                if key in metadata and metadata[key] is not None:
                    setattr(asset, key, str(metadata[key]))
            if "expiry_ns" in metadata and metadata["expiry_ns"] is not None:
                asset.expiry_ns = int(metadata["expiry_ns"])
            if "market_id" in metadata and "outcome" in metadata:
                self.update_market_mapping(asset.market_id, asset_id, asset.outcome)

        # Update book state
        asset.best_bid = float(features.get("best_bid") or 0)
        asset.best_ask = float(features.get("best_ask") or 0)
        asset.current_mid = float(features.get("midpoint") or (asset.best_bid + asset.best_ask) / 2)
        asset.current_spread = float(features.get("spread") or (asset.best_ask - asset.best_bid))
        asset.relative_spread = asset.current_spread / asset.current_mid if asset.current_mid > 0 else 0.0
        asset.current_microprice = float(features.get("microprice") or 0)
        asset.top1_imbalance = float(features.get("imbalance") or 0)
        asset.total_bid_levels = int(features.get("total_bid_levels") or 0)
        asset.total_ask_levels = int(features.get("total_ask_levels") or 0)
        asset.depth_features = extract_depth_features(features)

        # Rolling windows
        asset.book_updates_100ms.add(recv_ns, 1.0)
        asset.book_updates_500ms.add(recv_ns, 1.0)
        asset.book_updates_1s.add(recv_ns, 1.0)
        if prev_spread > 0:
            asset.spread_deltas_1s.add(recv_ns, asset.current_spread - prev_spread)
        asset.mid_history.append(asset.current_mid)

        # Check sampling
        if recv_ns - asset.last_sample_ns < self.sample_interval_ns:
            return
        asset.last_sample_ns = recv_ns

        # Assemble features and predict
        self._predict(asset, recv_ns)

    def handle_poly_trade(self, asset_id: str, side: str, price: float, size: float, recv_ns: int) -> None:
        """Process a poly trade."""
        self.monitor.observe_trade(asset_id, side, price, recv_ns)
        asset = self._get_or_create_asset(asset_id)
        signed_side = 1.0 if side.upper() in ("BUY", "BID") else -1.0
        asset.trades_100ms.add(recv_ns, 1.0)
        asset.trades_500ms.add(recv_ns, 1.0)
        asset.trades_1s.add(recv_ns, 1.0)
        if signed_side > 0:
            asset.buy_vol_1s.add(recv_ns, size)
        else:
            asset.sell_vol_1s.add(recv_ns, size)
        asset.signed_vol_1s.add(recv_ns, size * signed_side)
        asset.trade_notional_1s.add(recv_ns, price * size)
        asset.trade_size_1s.add(recv_ns, size)
        asset.recent_trade_sides.append(signed_side)

    def handle_binance_depth(self, bids: list, asks: list, recv_ns: int) -> None:
        """Process Binance depth update."""
        from poly.collector.binance_depth import depth_features

        if not bids or not asks:
            return
        self.binance_state.mid = (float(bids[0][0]) + float(asks[0][0])) / 2
        self.binance_state.spread = float(asks[0][0]) - float(bids[0][0])
        self.binance_state.depth_features = depth_features(bids, asks, max_depth=20)
        self.binance_state.mid_history.append(self.binance_state.mid)
        self.binance_state.last_recv_ns = recv_ns

    def handle_binance_trade(self, side: str, size: float, recv_ns: int) -> None:
        """Process Binance trade for trade imbalance."""
        signed = 1.0 if side.upper() == "BUY" else -1.0
        self.binance_state.recent_trade_imbalance = signed * size  # simplified

    def _predict(self, asset: AssetState, recv_ns: int) -> None:
        """Assemble features, run prediction, output and track."""
        if asset.current_mid <= 0:
            return

        self.monitor.stats.total_predictions += 1

        # Assemble features
        feat_dict = self.assembler.assemble(asset, self.binance_state)

        # Build DataFrame row
        row = {}
        for col in self.feature_columns:
            val = feat_dict.get(col)
            if val is None:
                row[col] = 0.0 if col not in self.cat_columns else "unknown"
            else:
                row[col] = val
        df = pd.DataFrame([row])

        # Predict
        proba = self.pipeline.predict_proba(df[self.model_feature_columns])[0, 1]
        pred_unwind_profit = None
        pred_expected_profit = None
        decision_score = proba
        if self.unwind_pipeline is not None:
            pred_unwind_profit = float(self.unwind_pipeline.predict(df[self.unwind_feature_columns])[0])

        if asset.opposite_asset_id is None and asset.asset_id in self.opposite_map:
            asset.opposite_asset_id = self.opposite_map[asset.asset_id]

        # Output
        tte = (asset.expiry_ns - recv_ns) / NS_PER_SECOND if asset.expiry_ns > 0 else 0
        slug_short = asset.slug.split("-updown-")[-1] if "-updown-" in asset.slug else asset.slug[:20]
        outcome_char = "U" if asset.outcome == "Up" else "D" if asset.outcome == "Down" else "?"

        prev_score = self._last_proba_by_asset.get(asset.asset_id)
        second_leg_quote_price = 0.96 - asset.best_ask
        first_leg_price = asset.best_ask + self.price_buffer
        fee_per_share = self.fee_rate * first_leg_price * (1.0 - first_leg_price)
        second_leg_size = 1.0 - fee_per_share
        success_profit_estimate = second_leg_size - (
            first_leg_price + second_leg_size * second_leg_quote_price
        )
        if pred_unwind_profit is not None:
            pred_expected_profit = proba * success_profit_estimate + (1.0 - proba) * pred_unwind_profit
            decision_score = pred_expected_profit

        raw_signal = decision_score >= self.threshold
        if self.unwind_pipeline is not None:
            # The offline two-stage execution policy selects rows that satisfy
            # the full policy, then applies cooldown/max-entry gates. It does
            # not require a raw score threshold crossing, because p_fill,
            # expected profit, and unwind risk can become eligible at different
            # times. Keep live semantics aligned with that evaluator.
            crossed_threshold = True
        else:
            crossed_threshold = prev_score is None or prev_score < self.threshold
        market_key = asset.market_id or asset.slug or asset.asset_id
        signal_key = f"{market_key}:{asset.outcome or asset.asset_id}"
        last_signal_ns = self._last_signal_ns_by_key.get(signal_key, 0)
        cooldown_ok = recv_ns - last_signal_ns >= self.signal_cooldown_ns
        max_entries_ok = (
            self.max_entries_per_signal_key == 0
            or self._signal_count_by_key.get(signal_key, 0) < self.max_entries_per_signal_key
        )
        has_opposite = asset.opposite_asset_id is not None
        structural_entry_filter_ok = (
            self.min_entry_ask <= asset.best_ask <= self.max_entry_ask
            and tte >= self.min_time_to_expiry_seconds
            and asset.current_spread <= self.max_spread
            and asset.best_bid > 0
            and asset.best_ask > 0
        )
        p_fill_ok = proba >= self.min_p_fill
        unwind_ok = pred_unwind_profit is None or pred_unwind_profit >= self.min_pred_unwind_profit
        signal = (
            raw_signal
            and crossed_threshold
            and cooldown_ok
            and max_entries_ok
            and has_opposite
            and structural_entry_filter_ok
            and p_fill_ok
            and unwind_ok
        )
        self._score_window.append(
            {
                "p_fill": float(proba),
                "pred_unwind_profit": pred_unwind_profit,
                "pred_expected_profit": pred_expected_profit,
                "decision_score": float(decision_score),
            }
        )
        self._record_gate_counts(
            raw_signal=raw_signal,
            crossed_threshold=crossed_threshold,
            cooldown_ok=cooldown_ok,
            max_entries_ok=max_entries_ok,
            has_opposite=has_opposite,
            entry_filter_ok=structural_entry_filter_ok,
            p_fill_ok=p_fill_ok,
            unwind_ok=unwind_ok,
            signal=signal,
        )
        self._last_proba_by_asset[asset.asset_id] = decision_score
        if (not signal) and self.log_near_threshold and decision_score > self.threshold * 0.8:
            logger.info(
                "prediction_near_threshold",
                asset=short_id(asset.asset_id),
                slug=slug_short,
                outcome=outcome_char,
                mid=f"{asset.current_mid:.3f}",
                ask=f"{asset.best_ask:.3f}",
                proba=f"{proba:.3f}",
                decision_score=f"{decision_score:.6f}",
                tte=f"{tte:.0f}s",
            )

        if signal:
            signal_seq_for_key = self._signal_count_by_key.get(signal_key, 0) + 1
            signal_id = f"S{self._next_signal_id:06d}"
            self._next_signal_id += 1
            pred = PendingPrediction(
                signal_id=signal_id,
                predict_ns=recv_ns,
                asset_id=asset.asset_id,
                market_id=asset.market_id,
                outcome=asset.outcome,
                signal_key=signal_key,
                signal_seq_for_key=signal_seq_for_key,
                best_bid_at_entry=asset.best_bid,
                best_ask_at_entry=asset.best_ask,
                current_mid_at_entry=asset.current_mid,
                spread_at_entry=asset.current_spread,
                time_to_expiry_at_entry=tte,
                proba=proba,
                pred_unwind_profit=pred_unwind_profit,
                pred_expected_profit=pred_expected_profit,
                decision_score=decision_score,
                opposite_asset_id=asset.opposite_asset_id,
                slug=asset.slug,
                fee_rate=self.fee_rate,
                price_buffer=self.price_buffer,
                second_leg_quote_price=second_leg_quote_price,
            )
            self.monitor.add_signal(pred)
            self._last_signal_ns_by_key[signal_key] = recv_ns
            self._signal_count_by_key[signal_key] = signal_seq_for_key
            self._signal_count_since_display += 1
            logger.info(
                "signal_open",
                signal_id=signal_id,
                signal_key=signal_key,
                signal_seq_for_key=signal_seq_for_key,
                max_entries_per_signal_key=self.max_entries_per_signal_key,
                slug=slug_short,
                market_id=short_id(asset.market_id, 12),
                outcome=asset.outcome,
                asset=short_id(asset.asset_id),
                opposite_asset=short_id(asset.opposite_asset_id),
                proba=f"{proba:.6f}",
                threshold=f"{self.threshold:.6f}",
                decision_score=f"{decision_score:.6f}",
                p_fill=f"{proba:.6f}",
                min_p_fill=f"{self.min_p_fill:.6f}",
                pred_unwind_profit=fmt_float(pred_unwind_profit),
                min_pred_unwind_profit=fmt_float(self.min_pred_unwind_profit),
                pred_expected_profit=fmt_float(pred_expected_profit),
                entry_bid=fmt_price(asset.best_bid),
                entry_ask=fmt_price(asset.best_ask),
                entry_mid=fmt_price(asset.current_mid),
                spread=fmt_price(asset.current_spread),
                tte_seconds=f"{tte:.1f}",
                first_leg_price=fmt_price(first_leg_price),
                price_buffer=fmt_price(self.price_buffer),
                fee_per_share=fmt_float(fee_per_share),
                second_leg_size=fmt_float(second_leg_size),
                second_leg_quote=fmt_price(second_leg_quote_price),
                success_profit_estimate=f"{success_profit_estimate:+.6f}",
            )

        # Check outcomes for pending predictions
        asset_books = {}
        for aid, state in self.asset_states.items():
            asset_books[aid] = {
                "best_bid": state.best_bid,
                "best_ask": state.best_ask,
                "current_mid": state.current_mid,
            }
        resolved = self.monitor.check_outcomes(recv_ns, asset_books)
        for pred, details in resolved:
            profit = float(details["profit"])
            result = str(details["result"])
            logger.info(
                "signal_close",
                signal_id=pred.signal_id,
                signal_key=pred.signal_key,
                signal_seq_for_key=pred.signal_seq_for_key,
                slug=short_slug(pred.slug),
                market_id=short_id(pred.market_id, 12),
                outcome_pred=pred.outcome,
                asset=short_id(pred.asset_id),
                opposite_asset=short_id(pred.opposite_asset_id),
                proba=f"{pred.proba:.6f}",
                decision_score=fmt_float(pred.decision_score),
                p_fill=f"{pred.proba:.6f}",
                pred_unwind_profit=fmt_float(pred.pred_unwind_profit),
                pred_expected_profit=fmt_float(pred.pred_expected_profit),
                result=result,
                profit=f"{profit:+.6f}",
                entry_bid=fmt_price(pred.best_bid_at_entry),
                entry_ask=fmt_price(pred.best_ask_at_entry),
                entry_mid=fmt_price(pred.current_mid_at_entry),
                entry_spread=fmt_price(pred.spread_at_entry),
                entry_tte_seconds=f"{pred.time_to_expiry_at_entry:.1f}",
                first_leg_price=fmt_price(float(details["first_leg_price"])),
                fee_per_share=fmt_float(float(details["fee_per_share"])),
                second_leg_size=fmt_float(float(details["second_leg_size"])),
                second_leg_quote=fmt_price(float(details["second_leg_quote"])),
                second_leg_fill_price=fmt_price(details.get("second_leg_fill_price")),
                second_leg_fill_lag_ms=(
                    "" if details.get("second_leg_fill_lag_ms") is None else f"{float(details['second_leg_fill_lag_ms']):.0f}"
                ),
                future_best_bid=fmt_price(float(details["future_best_bid"])),
                future_best_ask=fmt_price(float(details["future_best_ask"])),
                future_mid=fmt_price(float(details["future_mid"])),
                unwind_revenue=fmt_float(details.get("unwind_revenue")),
                unwind_price=fmt_price(details.get("unwind_price")),
                unwind_price_move=fmt_price(details.get("unwind_price_move")),
                success_revenue=fmt_float(details.get("success_revenue")),
                first_leg_cost=fmt_float(details.get("first_leg_cost")),
                second_leg_cost=fmt_float(details.get("second_leg_cost")),
                total_cost=fmt_float(details.get("total_cost")),
                running=self.monitor.stats.summary(),
            )

        # Periodic stats display
        if self.monitor.stats.total_predictions % self.stats_interval == 0:
            score_stats = self._score_stats()
            logger.info(
                "stats",
                predictions=self.monitor.stats.total_predictions,
                signals=self.monitor.stats.total_signals,
                pending=self.monitor.stats.total_signals - self.monitor.stats.total_resolved,
                resolved=self.monitor.stats.total_resolved,
                accuracy=f"{self.monitor.stats.accuracy():.1%}",
                total_profit=f"{self.monitor.stats.total_profit:+.4f}",
                score_stats=score_stats,
                gate_counts=dict(self._block_counts),
            )
            self._block_counts.clear()

    def _record_gate_counts(
        self,
        *,
        raw_signal: bool,
        crossed_threshold: bool,
        cooldown_ok: bool,
        max_entries_ok: bool,
        has_opposite: bool,
        entry_filter_ok: bool,
        p_fill_ok: bool,
        unwind_ok: bool,
        signal: bool,
    ) -> None:
        self._block_counts["samples"] += 1
        if raw_signal:
            self._block_counts["pass_threshold"] += 1
        if p_fill_ok:
            self._block_counts["pass_p_fill"] += 1
        if unwind_ok:
            self._block_counts["pass_unwind"] += 1
        if raw_signal and p_fill_ok and unwind_ok:
            self._block_counts["pass_model_gates"] += 1
        if raw_signal and p_fill_ok and unwind_ok and entry_filter_ok and has_opposite:
            self._block_counts["pass_model_and_entry_gates"] += 1
        if signal:
            self._block_counts["signals"] += 1
            return
        if not raw_signal:
            self._block_counts["blocked_threshold"] += 1
        if not p_fill_ok:
            self._block_counts["blocked_p_fill"] += 1
        if not unwind_ok:
            self._block_counts["blocked_unwind"] += 1
        if not crossed_threshold:
            self._block_counts["blocked_crossing"] += 1
        if not cooldown_ok:
            self._block_counts["blocked_cooldown"] += 1
        if not max_entries_ok:
            self._block_counts["blocked_max_entries"] += 1
        if not has_opposite:
            self._block_counts["blocked_no_opposite"] += 1
        if not entry_filter_ok:
            self._block_counts["blocked_entry_filter"] += 1

    def _score_stats(self) -> dict[str, object]:
        rows = list(self._score_window)
        return {
            "window": len(rows),
            "p_fill": quantile_summary([float(r["p_fill"]) for r in rows if r.get("p_fill") is not None]),
            "pred_unwind_profit": quantile_summary(
                [float(r["pred_unwind_profit"]) for r in rows if r.get("pred_unwind_profit") is not None]
            ),
            "pred_expected_profit": quantile_summary(
                [float(r["pred_expected_profit"]) for r in rows if r.get("pred_expected_profit") is not None]
            ),
            "decision_score": quantile_summary(
                [float(r["decision_score"]) for r in rows if r.get("decision_score") is not None]
            ),
        }
