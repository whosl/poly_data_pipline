"""Python wrapper for the Rust orderbook engine."""

from __future__ import annotations

import logging
from typing import Any

import poly_core

logger = logging.getLogger(__name__)


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

    def top_n(self, asset_id: str, n: int = 10) -> tuple[list, list] | None:
        book = self._books.get(asset_id)
        if book is None:
            return None
        return book.top_n_levels(n)
