#!/usr/bin/env python3
"""ML Signal Server: WebSocket bridge between PredictionPipeline and poly_bot.

Runs the full ML prediction pipeline (Polymarket + Binance WebSocket data →
features → model inference) and broadcasts trading signals to connected
poly_bot clients over a local WebSocket.

Usage:
    python signal_server.py \\
        --model-path artifacts/training_eventdriven_all5m_80m/fill_models/lightgbm_classifier.joblib \\
        --unwind-model-path artifacts/training_eventdriven_all5m_80m/unwind_models/xgboost_regressor.joblib \\
        --threshold 0.04 --min-p-fill 0.8 --min-pred-unwind-profit -0.05 --port 8765

poly_bot connects to ws://localhost:8765 and receives JSON messages:
    {"type":"signal","direction":"BUY_UP","confidence":0.82,"proba":0.82,
     "decisionScore":0.012,"predUnwindProfit":0.003,"assetId":"...","timestamp":...}
    {"type":"prediction","assetId":"...","proba":0.71,"decisionScore":0.008,...}
    {"type":"tick","upAsk":0.45,"downAsk":0.51,"up_asset":"...","down_asset":"..."}
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any

import structlog

# Ensure poly_trade_pipeline is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ]
)
logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# WebSocket server that broadcasts signals to poly_bot clients
# ---------------------------------------------------------------------------
class SignalBroadcaster:
    """Manages connected poly_bot clients and broadcasts ML signals."""

    def __init__(self):
        self._clients: set[asyncio.WebSocketServerProtocol] = set()
        self._server: asyncio.Server | None = None
        self._up_asset_id: str | None = None
        self._down_asset_id: str | None = None
        self._last_up_ask: float | None = None
        self._last_down_ask: float | None = None

    async def start(self, host: str = "127.0.0.1", port: int = 8765):
        import websockets

        self._server = await websockets.serve(
            self._handle_client,
            host,
            port,
            ping_interval=10,
            ping_timeout=5,
        )
        logger.info("signal_server_started", host=host, port=port)

    async def _handle_client(self, ws, path=None):
        self._clients.add(ws)
        logger.info("client_connected", total=len(self._clients))
        try:
            async for msg in ws:
                # Client can send subscription preferences (future extension)
                pass
        except Exception:
            pass
        finally:
            self._clients.discard(ws)
            logger.info("client_disconnected", total=len(self._clients))

    def set_market_assets(self, up_asset_id: str, down_asset_id: str):
        self._up_asset_id = up_asset_id
        self._down_asset_id = down_asset_id

    def update_prices(self, asset_id: str, best_ask: float | None):
        if asset_id == self._up_asset_id:
            self._last_up_ask = best_ask
        elif asset_id == self._down_asset_id:
            self._last_down_ask = best_ask

    def broadcast(self, message: dict):
        if not self._clients:
            return
        data = json.dumps(message, default=float)
        # Fire-and-forget broadcast to all clients
        for ws in list(self._clients):
            try:
                asyncio.ensure_future(ws.send(data))
            except Exception:
                self._clients.discard(ws)

    def broadcast_signal(
        self,
        *,
        direction: str,
        confidence: float,
        proba: float,
        decision_score: float,
        pred_unwind_profit: float | None,
        pred_expected_profit: float | None,
        asset_id: str,
        opposite_asset_id: str | None,
        outcome: str,
        best_bid: float,
        best_ask: float,
        mid: float,
        spread: float,
    ):
        self.broadcast({
            "type": "signal",
            "direction": direction,
            "confidence": confidence,
            "proba": proba,
            "decisionScore": decision_score,
            "predUnwindProfit": pred_unwind_profit,
            "predExpectedProfit": pred_expected_profit,
            "assetId": asset_id,
            "oppositeAssetId": opposite_asset_id,
            "outcome": outcome,
            "bestBid": best_bid,
            "bestAsk": best_ask,
            "mid": mid,
            "spread": spread,
            "upAsk": self._last_up_ask,
            "downAsk": self._last_down_ask,
            "timestamp": int(time.time() * 1000),
        })

    def broadcast_tick(self):
        self.broadcast({
            "type": "tick",
            "upAsk": self._last_up_ask,
            "downAsk": self._last_down_ask,
            "up_asset": self._up_asset_id,
            "down_asset": self._down_asset_id,
            "timestamp": int(time.time() * 1000),
        })

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# ---------------------------------------------------------------------------
# Patched PredictionPipeline that broadcasts signals instead of logging
# ---------------------------------------------------------------------------
class BroadcastingPipeline:
    """Wraps PredictionPipeline and intercepts signals for broadcasting."""

    def __init__(self, pipeline, broadcaster: SignalBroadcaster):
        self._pipeline = pipeline
        self._broadcaster = broadcaster
        # Monkey-patch the _predict method to intercept signals
        self._original_predict = pipeline._predict
        pipeline._predict = self._intercepting_predict
        # Track asset → outcome mapping for signal direction
        self._asset_outcome: dict[str, str] = {}

    def _intercepting_predict(self, asset, recv_ns):
        """Intercept prediction calls to broadcast signals."""
        # Store outcome mapping
        if asset.outcome:
            self._asset_outcome[asset.asset_id] = asset.outcome

        # Call original predict
        self._original_predict(asset, recv_ns)

        # Check if a signal was just produced (look at the last signal)
        # The original method adds signals to self._pipeline.monitor
        # We check if a new signal was added since our last check
        monitor = self._pipeline.monitor
        if monitor.pending:
            latest = monitor.pending[-1]
            logger.info('broadcast_check', pending_count=len(monitor.pending), latest_predict_ns=latest.predict_ns, current_recv_ns=recv_ns, latest_signal_id=latest.signal_id, match=latest.predict_ns == recv_ns)
            if latest.predict_ns == recv_ns:
                # This is a new signal - broadcast it
                outcome = self._asset_outcome.get(latest.asset_id, asset.outcome)
                direction = "BUY_UP" if outcome == "Up" else "BUY_DOWN"

                # Update broadcaster with current asset info
                if outcome == "Up":
                    self._broadcaster.set_market_assets(
                        latest.asset_id,
                        latest.opposite_asset_id or "",
                    )
                elif outcome == "Down":
                    self._broadcaster.set_market_assets(
                        latest.opposite_asset_id or "",
                        latest.asset_id,
                    )

                self._broadcaster.broadcast_signal(
                    direction=direction,
                    confidence=latest.proba,
                    proba=latest.proba,
                    decision_score=latest.decision_score or latest.proba,
                    pred_unwind_profit=latest.pred_unwind_profit,
                    pred_expected_profit=latest.pred_expected_profit,
                    asset_id=latest.asset_id,
                    opposite_asset_id=latest.opposite_asset_id,
                    outcome=outcome,
                    best_bid=latest.best_bid_at_entry,
                    best_ask=latest.best_ask_at_entry,
                    mid=latest.current_mid_at_entry,
                    spread=latest.spread_at_entry,
                )
                logger.info(
                    "signal_broadcast",
                    direction=direction,
                    proba=f"{latest.proba:.4f}",
                    decision_score=f"{latest.decision_score:.6f}" if latest.decision_score else "n/a",
                    outcome=outcome,
                    ask=f"{latest.best_ask_at_entry:.3f}",
                )

        # Broadcast tick with current prices
        self._broadcaster.update_prices(asset.asset_id, asset.best_ask)
        if self._broadcaster._up_asset_id and self._broadcaster._down_asset_id:
            self._broadcaster.broadcast_tick()

    # Delegate all other attributes to the wrapped pipeline
    def __getattr__(self, name):
        return getattr(self._pipeline, name)


# ---------------------------------------------------------------------------
# Main: start pipeline + WebSocket server
# ---------------------------------------------------------------------------
async def run_server(
    model_path: str,
    unwind_model_path: str | None = None,
    first_leg_fill_model_path: str | None = None,
    min_p_first_fill: float = 0.0,
    threshold: float = 0.67,
    min_p_fill: float = 0.75,
    min_pred_unwind_profit: float = 0.0,
    sample_interval_ms: int = 100,
    horizon_seconds: int = 10,
    fee_rate: float = 0.072,
    price_buffer: float = 0.01,
    symbols: str = "btcusdt",
    signal_cooldown_seconds: float | None = None,
    min_entry_ask: float = 0.10,
    max_entry_ask: float = 0.90,
    min_time_to_expiry: float = 20.0,
    max_spread: float = 0.05,
    max_entries_per_signal_key: int = 0,
    host: str = "127.0.0.1",
    port: int = 8765,
    maker_fill_latency_ms: int = 250,
    maker_fill_trade_through_ticks: float = 1.0,
    min_price_change: float = 0.02,
):
    """Run the ML signal server: PredictionPipeline + WebSocket broadcaster."""
    import orjson
    import websockets
    import aiohttp
    from poly.config import get_config
    from poly.engine.orderbook import OrderBookEngine
    from poly.predict.pipeline import PredictionPipeline

    config = get_config()

    # Resolve model paths
    resolved_model_path = Path(model_path)
    if not resolved_model_path.exists():
        logger.error("model_not_found", path=str(resolved_model_path))
        return

    resolved_unwind_path = None
    if unwind_model_path:
        resolved_unwind_path = Path(unwind_model_path)
        if not resolved_unwind_path.exists():
            logger.error("unwind_model_not_found", path=str(resolved_unwind_path))
            return

    resolved_first_leg_path = None
    if first_leg_fill_model_path:
        resolved_first_leg_path = Path(first_leg_fill_model_path)
        if not resolved_first_leg_path.exists():
            logger.error("first_leg_fill_model_not_found", path=str(resolved_first_leg_path))
            return

    # Initialize pipeline
    pipeline = PredictionPipeline(
        model_path=resolved_model_path,
        unwind_model_path=resolved_unwind_path,
        first_leg_fill_model_path=resolved_first_leg_path,
        min_p_first_fill=min_p_first_fill,
        threshold=threshold,
        min_p_fill=min_p_fill,
        min_pred_unwind_profit=min_pred_unwind_profit,
        sample_interval_ms=sample_interval_ms,
        horizon_seconds=horizon_seconds,
        fee_rate=fee_rate,
        price_buffer=price_buffer,
        signal_cooldown_seconds=signal_cooldown_seconds,
        min_entry_ask=min_entry_ask,
        max_entry_ask=max_entry_ask,
        min_time_to_expiry_seconds=min_time_to_expiry,
        max_spread=max_spread,
        max_entries_per_signal_key=max_entries_per_signal_key,
        signal_sample_path=None,  # Disable file logging for server mode
        candidate_sample_path=None,
        maker_fill_latency_ms=maker_fill_latency_ms,
        maker_fill_trade_through_ticks=maker_fill_trade_through_ticks,
        min_price_change=min_price_change,
    )
    engine = OrderBookEngine()

    # Start WebSocket broadcaster
    broadcaster = SignalBroadcaster()
    await broadcaster.start(host, port)

    # Wrap pipeline with broadcasting interceptor
    wrapped = BroadcastingPipeline(pipeline, broadcaster)

    logger.info(
        "ml_signal_server_starting",
        model=str(resolved_model_path),
        unwind_model=str(resolved_unwind_path) if resolved_unwind_path else None,
        first_leg_fill_model=str(resolved_first_leg_path) if resolved_first_leg_path else None,
        threshold=threshold,
        min_p_fill=min_p_fill,
        min_p_first_fill=min_p_first_fill,
        port=port,
    )

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    # Reuse the same WebSocket connection logic from live_predict.py
    from scripts.live_predict import _run_poly_ws, _run_binance_ws

    poly_task = asyncio.create_task(
        _run_poly_ws(config, wrapped, engine, shutdown_event),
        name="poly_ws",
    )
    binance_task = asyncio.create_task(
        _run_binance_ws(config, wrapped, symbols.split(","), shutdown_event),
        name="binance_ws",
    )

    try:
        await asyncio.gather(poly_task, binance_task, return_exceptions=True)
    finally:
        pipeline.close()
        await broadcaster.stop()
        logger.info("ml_signal_server_stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
import click


@click.command()
@click.option(
    "--model-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to fill classifier .joblib model.",
)
@click.option(
    "--unwind-model-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to unwind regressor .joblib model (optional, for two-stage mode).",
)
@click.option(
    "--first-leg-fill-model-path",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to first-leg fill classifier .joblib (optional, for 3-stage mode).",
)
@click.option("--min-p-first-fill", type=float, default=0.0, show_default=True,
              help="Min p(first_leg_fill) to allow signal (3-stage gate).")
@click.option("--threshold", type=float, default=0.67, show_default=True)
@click.option("--min-p-fill", type=float, default=0.75, show_default=True)
@click.option("--min-pred-unwind-profit", type=float, default=0.0, show_default=True)
@click.option("--sample-interval", type=int, default=100, show_default=True, help="Sampling interval ms")
@click.option("--horizon", type=int, default=10, show_default=True, help="Monitoring horizon seconds")
@click.option("--fee-rate", type=float, default=0.072, show_default=True)
@click.option("--price-buffer", type=float, default=0.01, show_default=True)
@click.option("--symbols", default="btcusdt", show_default=True)
@click.option("--signal-cooldown", type=float, default=None)
@click.option("--min-entry-ask", type=float, default=0.10, show_default=True)
@click.option("--max-entry-ask", type=float, default=0.90, show_default=True)
@click.option("--min-time-to-expiry", type=float, default=20.0, show_default=True)
@click.option("--max-spread", type=float, default=0.05, show_default=True)
@click.option("--max-entries-per-signal-key", type=int, default=0, show_default=True)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8765, show_default=True)
@click.option("--maker-fill-latency-ms", type=int, default=250, show_default=True)
@click.option("--maker-fill-trade-through-ticks", type=float, default=1.0, show_default=True)
@click.option("--min-price-change", type=float, default=0.02, show_default=True, help="Min mid price change to trigger inference")
def main(
    model_path,
    unwind_model_path,
    first_leg_fill_model_path,
    min_p_first_fill,
    threshold,
    min_p_fill,
    min_pred_unwind_profit,
    sample_interval,
    horizon,
    fee_rate,
    price_buffer,
    symbols,
    signal_cooldown,
    min_entry_ask,
    max_entry_ask,
    min_time_to_expiry,
    max_spread,
    max_entries_per_signal_key,
    host,
    port,
    maker_fill_latency_ms,
    maker_fill_trade_through_ticks,
    min_price_change,
):
    """ML Signal Server: broadcasts ML predictions to poly_bot via WebSocket."""
    asyncio.run(run_server(
        model_path=str(model_path),
        unwind_model_path=str(unwind_model_path) if unwind_model_path else None,
        first_leg_fill_model_path=str(first_leg_fill_model_path) if first_leg_fill_model_path else None,
        min_p_first_fill=min_p_first_fill,
        threshold=threshold,
        min_p_fill=min_p_fill,
        min_pred_unwind_profit=min_pred_unwind_profit,
        sample_interval_ms=sample_interval,
        horizon_seconds=horizon,
        fee_rate=fee_rate,
        price_buffer=price_buffer,
        symbols=symbols,
        signal_cooldown_seconds=signal_cooldown,
        min_entry_ask=min_entry_ask,
        max_entry_ask=max_entry_ask,
        min_time_to_expiry=min_time_to_expiry,
        max_spread=max_spread,
        max_entries_per_signal_key=max_entries_per_signal_key,
        host=host,
        port=port,
        maker_fill_latency_ms=maker_fill_latency_ms,
        maker_fill_trade_through_ticks=maker_fill_trade_through_ticks,
        min_price_change=min_price_change,
    ))


if __name__ == "__main__":
    main()
