"""Live prediction pipeline: WebSocket data → features → model prediction + outcome monitoring."""

from __future__ import annotations

import asyncio
import collections
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

    predict_ns: int
    asset_id: str
    market_id: str
    outcome: str
    best_ask_at_entry: float
    current_mid_at_entry: float
    proba: float
    opposite_asset_id: str | None
    slug: str
    two_leg_max_total_price: float = 0.96
    fee_rate: float = 0.072
    price_buffer: float = 0.01


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

    def __init__(self, horizon_seconds: int = 10, fee_rate: float = 0.072, price_buffer: float = 0.01):
        self.horizon_seconds = horizon_seconds
        self.fee_rate = fee_rate
        self.price_buffer = price_buffer
        self.pending: list[PendingPrediction] = []
        self.stats = MonitorStats()

    def add_signal(self, pred: PendingPrediction) -> None:
        self.pending.append(pred)
        self.stats.total_signals += 1

    def check_outcomes(
        self,
        now_ns: int,
        asset_books: dict[str, dict],
    ) -> list[tuple[PendingPrediction, float, str]]:
        """Check all pending predictions for 10s outcome. Returns list of (pred, profit, outcome)."""
        resolved = []
        still_pending = []
        for pred in self.pending:
            elapsed_ns = now_ns - pred.predict_ns
            if elapsed_ns < self.horizon_seconds * NS_PER_SECOND:
                still_pending.append(pred)
                continue

            # Evaluate outcome
            profit, outcome = self._evaluate(pred, asset_books)
            resolved.append((pred, profit, outcome))

        self.pending = still_pending
        return resolved

    def _evaluate(
        self,
        pred: PendingPrediction,
        asset_books: dict[str, dict],
    ) -> tuple[float, str]:
        """Evaluate a single prediction's outcome using actual trading mechanics.

        Two-leg strategy:
        - First leg: taker BUY this asset at (best_ask + price_buffer)
        - Fee deducted in shares: fee = fee_rate * first_leg_price * (1 - first_leg_price)
        - Second leg size = 1 - fee (shares remaining after fee)
        - Second leg: maker SELL opposite asset at (max_total_price - best_ask)
        - If opposite best_bid enables fill → success, profit = second_leg_size - total_cost
        - Otherwise → failure, unwind first leg by selling second_leg_size shares at future best_bid
        """
        # Compute entry parameters (same as labels.py formula)
        entry_price = pred.best_ask_at_entry
        first_leg_price = entry_price + pred.price_buffer
        fee_per_share = pred.fee_rate * first_leg_price * (1.0 - first_leg_price)
        second_leg_size = 1.0 - fee_per_share
        second_leg_price = pred.two_leg_max_total_price - entry_price

        # Get current state of this asset (for unwind)
        my_book = asset_books.get(pred.asset_id, {})
        future_best_bid = my_book.get("best_bid", 0.0)

        # Get opposite asset state
        if pred.opposite_asset_id and pred.opposite_asset_id in asset_books:
            opp_book = asset_books[pred.opposite_asset_id]
            opp_best_bid = opp_book.get("best_bid", 0.0)

            # Check if two-leg would have succeeded
            # If opposite asset's best_bid is high enough that our total <= max_total_price
            total_price = entry_price + opp_best_bid

            if total_price <= pred.two_leg_max_total_price:
                # Success: both legs fill
                # cost = first_leg_price + second_leg_size * second_leg_price
                # revenue = second_leg_size (both legs pay $1/share at expiry)
                # profit = revenue - cost = second_leg_size - (first_leg_price + second_leg_size * second_leg_price)
                profit = second_leg_size - (first_leg_price + second_leg_size * second_leg_price)
                self.stats.total_success += 1
                self.stats.recent_profits.append(profit)
                self.stats.total_profit += profit
                self.stats.total_resolved += 1
                return profit, "success"
            else:
                # Unwind: sell second_leg_size shares at future best_bid
                # loss = first_leg_price - second_leg_size * future_best_bid
                unwind_loss = max(0.0, first_leg_price - second_leg_size * future_best_bid)
                profit = -unwind_loss
                self.stats.total_failure += 1
                self.stats.recent_profits.append(profit)
                self.stats.total_profit += profit
                self.stats.total_resolved += 1
                return profit, "unwind"
        else:
            # No opposite asset tracked → unwind with loss
            unwind_loss = max(0.0, first_leg_price - second_leg_size * future_best_bid)
            profit = -unwind_loss
            self.stats.total_failure += 1
            self.stats.recent_profits.append(profit)
            self.stats.total_profit += profit
            self.stats.total_resolved += 1
            return profit, "no_opposite"


# ---------------------------------------------------------------------------
# Main prediction pipeline
# ---------------------------------------------------------------------------
class PredictionPipeline:
    """Live prediction pipeline connecting to WebSocket feeds."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.6,
        sample_interval_ms: int = 100,
        horizon_seconds: int = 10,
        fee_rate: float = 0.072,
        price_buffer: float = 0.01,
    ):
        self.threshold = threshold
        self.sample_interval_ns = sample_interval_ms * NS_PER_MS
        self.horizon_seconds = horizon_seconds
        self.fee_rate = fee_rate
        self.price_buffer = price_buffer

        # Load model
        blob = joblib.load(str(model_path))
        self.pipeline = blob["model"]
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
            self.asset_states.get(ids[0], AssetState(asset_id=ids[0])).opposite_asset_id = ids[1]
            self.asset_states.get(ids[1], AssetState(asset_id=ids[1])).opposite_asset_id = ids[0]

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
        df = pd.DataFrame([row])[self.feature_columns]

        # Predict
        proba = self.pipeline.predict_proba(df)[0, 1]

        # Output
        tte = (asset.expiry_ns - recv_ns) / NS_PER_SECOND if asset.expiry_ns > 0 else 0
        slug_short = asset.slug.split("-updown-")[-1] if "-updown-" in asset.slug else asset.slug[:20]
        outcome_char = "U" if asset.outcome == "Up" else "D" if asset.outcome == "Down" else "?"

        signal = proba >= self.threshold
        signal_str = "** SIGNAL **" if signal else ""

        if signal or proba > self.threshold * 0.8:
            logger.info(
                "predict",
                asset=asset.asset_id[:8],
                slug=slug_short,
                outcome=outcome_char,
                mid=f"{asset.current_mid:.3f}",
                ask=f"{asset.best_ask:.3f}",
                proba=f"{proba:.3f}",
                tte=f"{tte:.0f}s",
                signal=signal_str,
            )

        if signal:
            pred = PendingPrediction(
                predict_ns=recv_ns,
                asset_id=asset.asset_id,
                market_id=asset.market_id,
                outcome=asset.outcome,
                best_ask_at_entry=asset.best_ask,
                current_mid_at_entry=asset.current_mid,
                proba=proba,
                opposite_asset_id=asset.opposite_asset_id,
                slug=asset.slug,
                fee_rate=self.fee_rate,
                price_buffer=self.price_buffer,
            )
            self.monitor.add_signal(pred)
            self._signal_count_since_display += 1

        # Check outcomes for pending predictions
        asset_books = {}
        for aid, state in self.asset_states.items():
            asset_books[aid] = {
                "best_bid": state.best_bid,
                "best_ask": state.best_ask,
                "current_mid": state.current_mid,
            }
        resolved = self.monitor.check_outcomes(recv_ns, asset_books)
        for pred, profit, outcome in resolved:
            logger.info(
                "outcome",
                asset=pred.asset_id[:8],
                slug=pred.slug.split("-updown-")[-1] if "-updown-" in pred.slug else pred.slug[:20],
                outcome_pred=pred.outcome,
                entry_ask=f"{pred.best_ask_at_entry:.3f}",
                proba=f"{pred.proba:.3f}",
                result=outcome,
                profit=f"{profit:+.4f}",
                running=self.monitor.stats.summary(),
            )

        # Periodic stats display
        if self.monitor.stats.total_predictions % 100 == 0:
            logger.info(
                "stats",
                predictions=self.monitor.stats.total_predictions,
                signals=self.monitor.stats.total_signals,
                pending=self.monitor.stats.total_signals - self.monitor.stats.total_resolved,
                resolved=self.monitor.stats.total_resolved,
                accuracy=f"{self.monitor.stats.accuracy():.1%}",
                total_profit=f"{self.monitor.stats.total_profit:+.4f}",
            )
