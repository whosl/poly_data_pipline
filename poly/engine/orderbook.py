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
    """Manages per-asset OrderBook instances backed by Rust."""

    def __init__(self) -> None:
        self._books: dict[str, poly_core.OrderBook] = {}

    def get_or_create(self, asset_id: str) -> poly_core.OrderBook:
        if asset_id not in self._books:
            self._books[asset_id] = poly_core.OrderBook(asset_id)
        return self._books[asset_id]

    def handle_book(self, asset_id: str, bids: list, asks: list, exchange_ts: int) -> dict[str, Any] | None:
        """Apply a full snapshot and return depth summary."""
        book = self.get_or_create(asset_id)
        bids_data = [(b["price"], b["size"]) for b in bids]
        asks_data = [(a["price"], a["size"]) for a in asks]
        book.apply_snapshot(bids_data, asks_data, exchange_ts)
        return book.depth_summary()

    def handle_price_change(
        self, asset_id: str, side: str, price: str, size: str, exchange_ts: int
    ) -> dict[str, Any] | None:
        """Apply a delta and return depth summary."""
        book = self.get_or_create(asset_id)
        book.apply_delta(side, price, size, exchange_ts)
        return book.depth_summary()

    def get_features(self, asset_id: str) -> dict[str, Any] | None:
        """Get current depth summary without updating."""
        book = self._books.get(asset_id)
        if book is None:
            return None
        return book.depth_summary()

    def remove(self, asset_id: str) -> None:
        """Drop local state for an asset that is no longer subscribed."""
        self._books.pop(asset_id, None)

    def top_n(self, asset_id: str, n: int = 10) -> tuple[list, list] | None:
        book = self._books.get(asset_id)
        if book is None:
            return None
        return book.top_n_levels(n)
