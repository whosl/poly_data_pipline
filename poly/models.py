"""Pydantic v2 models for all WebSocket and API message types."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class _Base(BaseModel):
    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Polymarket Market WebSocket
# ---------------------------------------------------------------------------

class PolyLevel(_Base):
    price: str
    size: str


class BookEvent(_Base):
    event_type: Literal["book"]
    asset_id: str
    market: str
    bids: list[PolyLevel]
    asks: list[PolyLevel]
    timestamp: str
    hash: str | None = None


class PriceChangeEntry(_Base):
    asset_id: str
    price: str
    size: str
    side: Literal["BUY", "SELL"]
    hash: str | None = None
    best_bid: str | None = None
    best_ask: str | None = None


class PriceChangeEvent(_Base):
    event_type: Literal["price_change"]
    market: str
    price_changes: list[PriceChangeEntry]
    timestamp: str


class LastTradePriceEvent(_Base):
    event_type: Literal["last_trade_price"]
    asset_id: str
    market: str
    price: str
    size: str
    fee_rate_bps: str
    side: Literal["BUY", "SELL"]
    timestamp: str


class TickSizeChangeEvent(_Base):
    event_type: Literal["tick_size_change"]
    asset_id: str
    market: str
    old_tick_size: str
    new_tick_size: str
    timestamp: str


class BestBidAskEvent(_Base):
    event_type: Literal["best_bid_ask"]
    market: str
    asset_id: str
    best_bid: str
    best_ask: str
    spread: str
    timestamp: str


# ---------------------------------------------------------------------------
# Polymarket User WebSocket
# ---------------------------------------------------------------------------

class MakerOrder(_Base):
    order_id: str
    matched_amount: str
    price: str
    side: str
    asset_id: str


class OrderEvent(_Base):
    event_type: Literal["order"]
    id: str
    market: str
    asset_id: str
    side: Literal["BUY", "SELL"]
    original_size: str
    size_matched: str
    price: str
    type: Literal["PLACEMENT", "UPDATE", "CANCELLATION"]
    timestamp: str
    outcome: str | None = None
    order_owner: str | None = None
    associate_trades: list | None = None


class TradeEvent(_Base):
    event_type: Literal["trade"]
    id: str
    market: str
    asset_id: str
    side: Literal["BUY", "SELL"]
    size: str
    price: str
    fee_rate_bps: str | None = None
    status: str
    timestamp: str
    matchtime: str | None = None
    taker_order_id: str | None = None
    maker_orders: list[MakerOrder] | None = None
    outcome: str | None = None


# ---------------------------------------------------------------------------
# Binance WebSocket
# ---------------------------------------------------------------------------

class BinanceBookTicker(_Base):
    u: int
    s: str
    b: str
    B: str
    a: str
    A: str


class BinanceAggTrade(_Base):
    e: str
    E: int
    s: str
    a: int
    p: str
    q: str
    f: int
    l: int
    T: int
    m: bool
    M: bool


class BinanceDepth(_Base):
    lastUpdateId: int
    bids: list[list[str]]
    asks: list[list[str]]


# ---------------------------------------------------------------------------
# Gamma API
# ---------------------------------------------------------------------------

class GammaMarket(_Base):
    id: str
    question: str
    slug: str
    conditionId: str
    outcomes: str
    clobTokenIds: str
    active: bool
    closed: bool
    endDate: str | None = None
    startDate: str | None = None
    category: str | None = None
    tags: str | None = None
    minimumOrderSize: str | None = None
    minimumTickSize: str | None = None
    volume: str | None = None
    liquidity: str | None = None


class GammaEvent(_Base):
    id: str
    title: str
    slug: str
    active: bool
    closed: bool
    category: str | None = None
    markets: list[GammaMarket] | None = None


# ---------------------------------------------------------------------------
# Internal models
# ---------------------------------------------------------------------------

class MarketInfo(_Base):
    event_id: str
    market_id: str
    condition_id: str
    question: str
    slug: str
    outcomes: list[str]
    token_ids: list[str]
    active: bool
    closed: bool
    tick_size: str
    min_order_size: str
    category: str | None = None
    tags: list[str] = []
    end_date: str | None = None


class RawMessage(_Base):
    recv_ns: int
    source: str
    channel: str
    payload: bytes
