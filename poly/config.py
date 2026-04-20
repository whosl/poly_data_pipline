"""Environment-based configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    data_dir: Path
    gamma_url: str
    clob_url: str
    poly_market_ws_url: str
    poly_user_ws_url: str
    binance_ws_url: str
    binance_symbols: list[str]
    updown_markets: list[str]
    api_key: str
    api_secret: str
    api_passphrase: str
    filter_tags: list[str]
    ws_reconnect_delay: float
    ws_ping_interval: float
    raw_flush_interval: float
    norm_buffer_size: int
    watchdog_timeout: float


def get_config() -> Config:
    """Load config from environment variables with sensible defaults."""
    data_dir = Path(os.environ.get("POLY_DATA_DIR", str(_PROJECT_ROOT / "data")))
    symbols_str = os.environ.get("POLY_BINANCE_SYMBOLS", "btcusdt")
    updown_markets_str = os.environ.get("POLY_UPDOWN_MARKETS", "btc-updown-5m,btc-updown-15m")
    tags_str = os.environ.get("POLY_FILTER_TAGS", "bitcoin,crypto")

    return Config(
        data_dir=data_dir,
        gamma_url=os.environ.get("POLY_GAMMA_URL", "https://gamma-api.polymarket.com"),
        clob_url=os.environ.get("POLY_CLOB_URL", "https://clob.polymarket.com"),
        poly_market_ws_url=os.environ.get(
            "POLY_MARKET_WS_URL",
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
        ),
        poly_user_ws_url=os.environ.get(
            "POLY_USER_WS_URL",
            "wss://ws-subscriptions-clob.polymarket.com/ws/user",
        ),
        binance_ws_url=os.environ.get(
            "POLY_BINANCE_WS_URL", "wss://stream.binance.com:9443"
        ),
        binance_symbols=[s.strip().lower() for s in symbols_str.split(",")],
        updown_markets=[s.strip().lower() for s in updown_markets_str.split(",") if s.strip()],
        api_key=os.environ.get("POLY_API_KEY", ""),
        api_secret=os.environ.get("POLY_API_SECRET", ""),
        api_passphrase=os.environ.get("POLY_API_PASSPHRASE", ""),
        filter_tags=[t.strip().lower() for t in tags_str.split(",")],
        ws_reconnect_delay=float(os.environ.get("POLY_WS_RECONNECT_DELAY", "5")),
        ws_ping_interval=float(os.environ.get("POLY_WS_PING_INTERVAL", "10")),
        raw_flush_interval=float(os.environ.get("POLY_RAW_FLUSH_INTERVAL", "5")),
        norm_buffer_size=int(os.environ.get("POLY_NORM_BUFFER_SIZE", "1000")),
        watchdog_timeout=float(os.environ.get("POLY_WATCHDOG_TIMEOUT", "120")),
    )
