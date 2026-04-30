"""Python wrapper for the Rust orderbook engine."""

from __future__ import annotations

import logging
from typing import Any

import poly_core

logger = logging.getLogger(__name__)

# Keys returned by Rust depth_summary() for top-N depth features.
# Must stay in sync with src/orderbook.rs.
DEPTH_FEATURE_KEYS = [
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


def extract_depth_features(features: dict) -> dict[str, float | None]:
    """Extract depth feature values from a Rust depth_summary() dict.

    Rust returns Decimal values as strings; this converts them to float.
    Missing / None keys become None.
    """
    result: dict[str, float | None] = {}
    for key in DEPTH_FEATURE_KEYS:
        val = features.get(key)
        if val is not None:
            try:
                result[key] = float(val)
            except (TypeError, ValueError):
                result[key] = None
        else:
            result[key] = None
    return result


class OrderBookEngine:
    """Manages per-asset OrderBook instances backed by Rust.

    Event-driven depth_summary: only recomputes the expensive top-N depth
    features when midpoint moves beyond ``midpoint_threshold`` (relative).
    Otherwise returns a cheap quick_summary (O(1) fields only).
    """

    def __init__(self, midpoint_threshold: float = 0.005) -> None:
        self._books: dict[str, poly_core.OrderBook] = {}
        # Per-asset tracking for event-driven depth recomputation.
        self._last_mid: dict[str, float] = {}
        self._last_summary: dict[str, dict] = {}
        self.midpoint_threshold = midpoint_threshold

    def get_or_create(self, asset_id: str) -> poly_core.OrderBook:
        if asset_id not in self._books:
            self._books[asset_id] = poly_core.OrderBook(asset_id)
        return self._books[asset_id]

    def handle_book(self, asset_id: str, bids: list, asks: list, exchange_ts: int) -> dict[str, Any] | None:
        """Apply a full snapshot and always return full depth summary."""
        book = self.get_or_create(asset_id)
        bids_data = [(b["price"], b["size"]) for b in bids]
        asks_data = [(a["price"], a["size"]) for a in asks]
        book.apply_snapshot(bids_data, asks_data, exchange_ts)
        summary = book.depth_summary()
        # Cache the full summary and midpoint.
        mid_str = summary.get("midpoint")
        if mid_str is not None:
            self._last_mid[asset_id] = float(mid_str)
        self._last_summary[asset_id] = summary
        return summary

    def handle_price_change(
        self, asset_id: str, side: str, price: str, size: str, exchange_ts: int
    ) -> dict[str, Any] | None:
        """Apply a delta; only recompute full depth_summary when midpoint moved.

        When the midpoint barely moves, returns a quick_summary (no depth
        features) but preserves the cached full depth_summary so that
        get_features() still returns depth features from the last full compute.
        """
        book = self.get_or_create(asset_id)
        book.apply_delta(side, price, size, exchange_ts)

        # Check midpoint delta vs cached value.
        mid_str = book.midpoint()
        if mid_str is None:
            return book.quick_summary()

        new_mid = float(mid_str)
        prev_mid = self._last_mid.get(asset_id)

        if prev_mid is not None and prev_mid > 0:
            rel_change = abs(new_mid - prev_mid) / prev_mid
            if rel_change < self.midpoint_threshold:
                # Midpoint barely moved — return cheap summary but patch the
                # cached full summary with updated top-of-book fields so
                # get_features() returns depth features + current prices.
                qs = book.quick_summary()
                cached = self._last_summary.get(asset_id)
                if cached is not None:
                    for key in ("best_bid", "best_bid_size", "best_ask",
                                "best_ask_size", "spread", "midpoint",
                                "microprice", "imbalance", "total_bid_levels",
                                "total_ask_levels", "last_exchange_ts"):
                        if key in qs:
                            cached[key] = qs[key]
                return qs

        # Midpoint moved beyond threshold (or first update) — full recompute.
        summary = book.depth_summary()
        self._last_mid[asset_id] = new_mid
        self._last_summary[asset_id] = summary
        return summary

    def get_features(self, asset_id: str) -> dict[str, Any] | None:
        """Get current depth summary without updating. Returns cached if available."""
        cached = self._last_summary.get(asset_id)
        if cached is not None:
            return cached
        book = self._books.get(asset_id)
        if book is None:
            return None
        return book.depth_summary()

    def remove(self, asset_id: str) -> None:
        """Drop local state for an asset that is no longer subscribed."""
        self._books.pop(asset_id, None)
        self._last_mid.pop(asset_id, None)
        self._last_summary.pop(asset_id, None)

    def clear(self) -> None:
        """Drop all local order book state after a websocket reconnect."""
        self._books.clear()
        self._last_mid.clear()
        self._last_summary.clear()

    def top_n(self, asset_id: str, n: int = 10) -> tuple[list, list] | None:
        book = self._books.get(asset_id)
        if book is None:
            return None
        return book.top_n_levels(n)
