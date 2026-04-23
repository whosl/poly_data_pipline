#!/usr/bin/env python3
"""Live prediction CLI: connects to WebSocket feeds, runs model predictions, monitors 10s outcomes.

Usage:
    python scripts/live_predict.py --model xgboost_classifier --threshold 0.67
    python scripts/live_predict.py --model random_forest_classifier --threshold 0.751182 --sample-interval 100
    python scripts/live_predict.py --model-path artifacts/training_reprofit_20260420_21_5m/final_profit_models/random_forest_classifier.joblib
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from pathlib import Path

import click
import structlog
import orjson
import aiohttp
import re

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poly.config import get_config
from poly.engine.orderbook import OrderBookEngine
from poly.predict.pipeline import PredictionPipeline

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()


async def run_pipeline(
    model_name: str = "xgboost_classifier",
    model_dir: str | Path = "artifacts/training_reprofit_20260420_21_5m/final_profit_models",
    model_path: str | Path | None = None,
    unwind_model_name: str | None = None,
    unwind_model_dir: str | Path | None = None,
    unwind_model_path: str | Path | None = None,
    threshold: float = 0.6,
    min_p_fill: float = 0.0,
    min_pred_unwind_profit: float = -1.0,
    sample_interval_ms: int = 100,
    horizon_seconds: int = 10,
    fee_rate: float = 0.072,
    price_buffer: float = 0.01,
    symbols: str = "btcusdt",
    signal_cooldown_seconds: float | None = None,
    log_near_threshold: bool = False,
    stats_interval: int = 1000,
    min_entry_ask: float = 0.05,
    max_entry_ask: float = 0.95,
    min_time_to_expiry: float = 20.0,
    max_spread: float = 0.05,
    max_entries_per_signal_key: int = 0,
) -> None:
    config = get_config()

    # Resolve model path
    resolved_model_path = Path(model_path) if model_path else Path(model_dir) / f"{model_name}.joblib"
    if not resolved_model_path.exists():
        logger.error("model_not_found", path=str(resolved_model_path))
        return
    resolved_unwind_model_path = None
    if unwind_model_path or unwind_model_name:
        if unwind_model_path:
            resolved_unwind_model_path = Path(unwind_model_path)
        else:
            base = Path(unwind_model_dir) if unwind_model_dir else Path(model_dir)
            resolved_unwind_model_path = base / f"{unwind_model_name}.joblib"
        if not resolved_unwind_model_path.exists():
            logger.error("unwind_model_not_found", path=str(resolved_unwind_model_path))
            return

    # Init pipeline
    pipeline = PredictionPipeline(
        model_path=resolved_model_path,
        unwind_model_path=resolved_unwind_model_path,
        threshold=threshold,
        min_p_fill=min_p_fill,
        min_pred_unwind_profit=min_pred_unwind_profit,
        sample_interval_ms=sample_interval_ms,
        horizon_seconds=horizon_seconds,
        fee_rate=fee_rate,
        price_buffer=price_buffer,
        signal_cooldown_seconds=signal_cooldown_seconds,
        log_near_threshold=log_near_threshold,
        stats_interval=stats_interval,
        min_entry_ask=min_entry_ask,
        max_entry_ask=max_entry_ask,
        min_time_to_expiry_seconds=min_time_to_expiry,
        max_spread=max_spread,
        max_entries_per_signal_key=max_entries_per_signal_key,
    )
    engine = OrderBookEngine()

    logger.info(
        "pipeline_starting",
        model=model_name,
        model_path=str(resolved_model_path),
        unwind_model=unwind_model_name,
        unwind_model_path=str(resolved_unwind_model_path) if resolved_unwind_model_path else None,
        threshold=threshold,
        min_p_fill=min_p_fill,
        min_pred_unwind_profit=min_pred_unwind_profit,
        sample_interval_ms=sample_interval_ms,
        horizon_seconds=horizon_seconds,
        fee_rate=fee_rate,
        price_buffer=price_buffer,
        signal_cooldown_seconds=signal_cooldown_seconds if signal_cooldown_seconds is not None else horizon_seconds,
        log_near_threshold=log_near_threshold,
        stats_interval=stats_interval,
        min_entry_ask=min_entry_ask,
        max_entry_ask=max_entry_ask,
        min_time_to_expiry=min_time_to_expiry,
        max_spread=max_spread,
        max_entries_per_signal_key=max_entries_per_signal_key,
    )

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    # Start poly + binance tasks
    poly_task = asyncio.create_task(
        _run_poly_ws(config, pipeline, engine, shutdown_event),
        name="poly_ws",
    )
    binance_task = asyncio.create_task(
        _run_binance_ws(config, pipeline, symbols.split(","), shutdown_event),
        name="binance_ws",
    )

    await asyncio.gather(poly_task, binance_task, return_exceptions=True)
    logger.info("pipeline_stopped")


# ---------------------------------------------------------------------------
# Polymarket WS (UpDown markets)
# ---------------------------------------------------------------------------
async def _run_poly_ws(
    config,
    pipeline: PredictionPipeline,
    engine: OrderBookEngine,
    shutdown: asyncio.Event,
) -> None:
    from poly.collector.updown_ws import (
        UpDownCollector, fetch_market_tokens, _market_defs_from_config,
        _upcoming_timestamps, EXPIRED_MARKET_GRACE_SECONDS,
    )

    market_defs = _market_defs_from_config(config)
    subscribed_assets: dict[str, str] = {}  # asset_id -> slug
    asset_metadata: dict[str, dict] = {}
    slug_assets: dict[str, set[str]] = {}
    slug_expiry_ts: dict[str, int] = {}
    known_slugs: set[str] = set()

    backoff = 1.0
    while not shutdown.is_set():
        try:
            url = config.poly_market_ws_url
            logger.info("poly_ws_connecting", url=url)

            async with websockets.connect(url) as ws:
                rotation_task = asyncio.create_task(
                    _rotation_loop(ws, config, market_defs, subscribed_assets, asset_metadata,
                                   slug_assets, slug_expiry_ts, known_slugs, engine, shutdown),
                )
                try:
                    async for message in ws:
                        if shutdown.is_set():
                            break
                        import poly_core
                        recv_ns = poly_core.now_ns()

                        if isinstance(message, bytes):
                            text = message.decode("utf-8", errors="replace")
                        else:
                            text = message
                        text = text.strip()
                        if not text or text in ("PONG", "PING"):
                            continue

                        try:
                            parsed = orjson.loads(text)
                        except Exception:
                            continue

                        messages = parsed if isinstance(parsed, list) else [parsed]
                        for msg in messages:
                            if not isinstance(msg, dict):
                                continue
                            _dispatch_poly(msg, recv_ns, subscribed_assets, engine, pipeline, asset_metadata)
                finally:
                    rotation_task.cancel()

            backoff = 1.0
        except Exception as e:
            if shutdown.is_set():
                break
            logger.warning("poly_ws_error", error=str(e), backoff=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


def _dispatch_poly(msg: dict, recv_ns: int, subscribed: dict, engine: OrderBookEngine,
                   pipeline: PredictionPipeline, metadata: dict) -> None:
    from poly.engine.orderbook import extract_depth_features

    event_type = msg.get("event_type")
    if event_type == "book":
        asset_id = msg.get("asset_id", "")
        if asset_id not in subscribed:
            return
        bids = msg.get("bids", [])
        asks = msg.get("asks", [])
        exchange_ts = int(msg.get("timestamp", "0"))
        features = engine.handle_book(asset_id, bids, asks, exchange_ts)
        if features:
            pipeline.handle_poly_book(asset_id, features, recv_ns, metadata.get(asset_id))

    elif event_type == "price_change":
        exchange_ts = int(msg.get("timestamp", "0"))
        for change in msg.get("price_changes", []):
            asset_id = change.get("asset_id", "")
            if asset_id not in subscribed:
                continue
            side = change.get("side", "BUY").lower()
            price = change.get("price", "0")
            size = change.get("size", "0")
            features = engine.handle_price_change(asset_id, side, price, size, exchange_ts)
            if features:
                pipeline.handle_poly_book(asset_id, features, recv_ns, metadata.get(asset_id))

    elif event_type == "last_trade_price":
        asset_id = msg.get("asset_id", "")
        if asset_id not in subscribed:
            return
        side = msg.get("side", "")
        price = float(msg.get("price", "0"))
        size = float(msg.get("size", "0"))
        pipeline.handle_poly_trade(asset_id, side, price, size, recv_ns)


async def _rotation_loop(ws, config, market_defs, subscribed_assets, asset_metadata,
                         slug_assets, slug_expiry_ts, known_slugs, engine, shutdown) -> None:
    from poly.collector.updown_ws import fetch_market_tokens, _upcoming_timestamps, EXPIRED_MARKET_GRACE_SECONDS

    async with aiohttp.ClientSession() as session:
        while not shutdown.is_set():
            now = int(time.time())
            # Prune expired
            expired = [s for s, exp in slug_expiry_ts.items() if exp + EXPIRED_MARKET_GRACE_SECONDS <= now]
            for slug in expired:
                for aid in slug_assets.get(slug, set()):
                    subscribed_assets.pop(aid, None)
                    asset_metadata.pop(aid, None)
                slug_assets.pop(slug, None)
                slug_expiry_ts.pop(slug, None)
                known_slugs.discard(slug)
            if expired:
                logger.info("pruned_expired", slugs=len(expired))

            # Discover new
            new_assets = []
            for base_slug, period, ahead in market_defs:
                timestamps = _upcoming_timestamps(period, ahead, now)
                for ts in timestamps:
                    if ts + period <= now:
                        continue
                    slug = f"{base_slug}-{ts}"
                    if slug in known_slugs:
                        continue
                    info = await fetch_market_tokens(session, slug)
                    if info is None or not info.get("token_ids"):
                        known_slugs.add(slug)
                        continue
                    known_slugs.add(slug)
                    slug_expiry_ts[slug] = ts + period
                    for row in info.get("metadata_rows", []):
                        aid = str(row.get("asset_id") or "")
                        if aid:
                            asset_metadata[aid] = row
                    for tid in info["token_ids"]:
                        subscribed_assets[tid] = slug
                        slug_assets.setdefault(slug, set()).add(tid)
                        new_assets.append(tid)
                    logger.info("new_market", slug=slug, question=info.get("question", "")[:50])

            if new_assets:
                sub_msg = orjson.dumps({
                    "assets_ids": new_assets,
                    "operation": "subscribe",
                    "type": "market",
                    "custom_feature_enabled": True,
                })
                try:
                    await ws.send(sub_msg)
                    logger.info("subscribed", new=len(new_assets), total=len(subscribed_assets))
                except Exception:
                    pass

            await asyncio.sleep(10)


# ---------------------------------------------------------------------------
# Binance WS
# ---------------------------------------------------------------------------
async def _run_binance_ws(config, pipeline: PredictionPipeline, symbols: list[str],
                          shutdown: asyncio.Event) -> None:
    streams = []
    for sym in symbols:
        streams.extend([
            f"{sym}@depth20@100ms",
            f"{sym}@aggTrade",
        ])

    url = f"{config.binance_ws_url}/stream?streams={'/'.join(streams)}"
    backoff = 1.0

    while not shutdown.is_set():
        try:
            logger.info("binance_ws_connecting", url=url)
            async with websockets.connect(url) as ws:
                logger.info("binance_ws_connected")
                async for message in ws:
                    if shutdown.is_set():
                        break
                    import poly_core
                    recv_ns = poly_core.now_ns()
                    try:
                        envelope = orjson.loads(message)
                    except Exception:
                        continue
                    stream = envelope.get("stream", "")
                    data = envelope.get("data")
                    if data is None:
                        continue

                    if stream.endswith("@aggTrade"):
                        side = "SELL" if data.get("m", False) else "BUY"
                        size = float(data.get("q", "0"))
                        pipeline.handle_binance_trade(side, size, recv_ns)
                    elif "@depth" in stream:
                        bids = data.get("bids", [])
                        asks = data.get("asks", [])
                        if bids and asks:
                            pipeline.handle_binance_depth(bids, asks, recv_ns)
            backoff = 1.0
        except Exception as e:
            if shutdown.is_set():
                break
            logger.warning("binance_ws_error", error=str(e), backoff=backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--model", default="xgboost_classifier", help="Model name (e.g. xgboost_classifier, random_forest_classifier)")
@click.option(
    "--model-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=Path("artifacts/training_reprofit_20260420_21_5m/final_profit_models"),
    show_default=True,
    help="Directory containing <model>.joblib and training_metadata.json.",
)
@click.option(
    "--model-path",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="Explicit model .joblib path. Takes precedence over --model-dir/--model.",
)
@click.option("--unwind-model", default=None, help="Optional unwind regressor model name for two-stage execution policy.")
@click.option(
    "--unwind-model-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Directory containing <unwind-model>.joblib.",
)
@click.option(
    "--unwind-model-path",
    type=click.Path(path_type=Path, file_okay=True, dir_okay=False),
    default=None,
    help="Explicit unwind regressor .joblib path.",
)
@click.option("--threshold", type=float, default=0.6, help="Probability threshold, or expected-profit threshold in two-stage mode.")
@click.option("--min-p-fill", type=float, default=0.0, show_default=True, help="Minimum fill probability in two-stage mode.")
@click.option(
    "--min-pred-unwind-profit",
    type=float,
    default=-1.0,
    show_default=True,
    help="Minimum predicted signed unwind profit in two-stage mode.",
)
@click.option("--sample-interval", type=int, default=100, help="Sampling interval in ms")
@click.option("--horizon", type=int, default=10, help="Monitoring horizon in seconds")
@click.option("--fee-rate", type=float, default=0.072, help="Polymarket fee rate per share")
@click.option("--price-buffer", type=float, default=0.01, help="Price buffer above best_ask for taker fill")
@click.option("--symbols", default="btcusdt", help="Binance symbols")
@click.option(
    "--signal-cooldown",
    type=float,
    default=None,
    help="Minimum seconds between recorded signals for the same market/outcome. Defaults to --horizon.",
)
@click.option("--log-near-threshold", is_flag=True, help="Also log non-signal predictions near the threshold.")
@click.option("--stats-interval", type=int, default=1000, show_default=True, help="Prediction count interval for stats logs.")
@click.option("--min-entry-ask", type=float, default=0.05, show_default=True, help="Skip live signals below this first-leg ask.")
@click.option("--max-entry-ask", type=float, default=0.95, show_default=True, help="Skip live signals above this first-leg ask.")
@click.option("--min-time-to-expiry", type=float, default=20.0, show_default=True, help="Skip live signals with less time to expiry.")
@click.option("--max-spread", type=float, default=0.05, show_default=True, help="Skip live signals when Polymarket spread is wider than this.")
@click.option("--max-entries-per-signal-key", type=int, default=0, show_default=True, help="0 means unlimited; key is market/outcome.")
def main(
    model,
    model_dir,
    model_path,
    unwind_model,
    unwind_model_dir,
    unwind_model_path,
    threshold,
    min_p_fill,
    min_pred_unwind_profit,
    sample_interval,
    horizon,
    fee_rate,
    price_buffer,
    symbols,
    signal_cooldown,
    log_near_threshold,
    stats_interval,
    min_entry_ask,
    max_entry_ask,
    min_time_to_expiry,
    max_spread,
    max_entries_per_signal_key,
):
    """Start live prediction pipeline with outcome monitoring."""
    asyncio.run(run_pipeline(
        model_name=model,
        model_dir=model_dir,
        model_path=model_path,
        unwind_model_name=unwind_model,
        unwind_model_dir=unwind_model_dir,
        unwind_model_path=unwind_model_path,
        threshold=threshold,
        min_p_fill=min_p_fill,
        min_pred_unwind_profit=min_pred_unwind_profit,
        sample_interval_ms=sample_interval,
        horizon_seconds=horizon,
        fee_rate=fee_rate,
        price_buffer=price_buffer,
        symbols=symbols,
        signal_cooldown_seconds=signal_cooldown,
        log_near_threshold=log_near_threshold,
        stats_interval=stats_interval,
        min_entry_ask=min_entry_ask,
        max_entry_ask=max_entry_ask,
        min_time_to_expiry=min_time_to_expiry,
        max_spread=max_spread,
        max_entries_per_signal_key=max_entries_per_signal_key,
    ))


if __name__ == "__main__":
    main()
