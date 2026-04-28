# Live Execution Issues

Updated: 2026-04-29

Critical problems preventing profitable live execution despite positive shadow results.

## Summary

Signal server shadow: 388 signals, 80.4% profitable, +3.022 total profit.
Polybot real execution: 11 first-leg fills, **0 second-leg hedges completed**, all positions UNWIND.

The prediction model works. The execution pipeline does not.

---

## Issue 1: Second-Leg Hedge Never Succeeds

**Severity: Critical — this is the single biggest blocker.**

`placeSecondSideLimitOrder` returns `undefined` for 10/11 attempts. The one remaining attempt hit HEDGE TIMEOUT.

Root causes:

1. **No local orderbook in polybot** — polybot uses lightweight WS event extraction without maintaining a local BTreeMap orderbook. It does not know the current best bid/ask for the opposite outcome when placing the second-leg maker order.
2. **Stale price data** — by the time polybot tries to place the second-leg order, the market may have moved and the price it quotes is no longer valid.
3. **API response handling** — `placeSecondSideLimitOrder` returning `undefined` suggests the order placement call itself may be failing silently (API error, auth issue, or invalid parameters).

## Issue 2: User WS Fill Events Never Fire

**Severity: High — causes ~10s detection delay.**

Polybot subscribes to Polymarket user WS for fill notifications, but these events never arrive. All fill detection relies on REST polling fallback (`waitFirstLegMatchedFast`), which polls every ~2-3 seconds with a total detection delay of up to 10 seconds.

Impact:
- First-leg fill confirmed late → second-leg placed late → market has moved
- The 10s delay is a significant fraction of the 10s monitoring horizon
- By the time polybot detects the first-leg fill, the window for profitable second-leg placement may have closed

## Issue 3: Orderbook Data Divergence

**Severity: High — causes price mismatch between signal and execution.**

Signal server maintains a Rust BTreeMap orderbook with full L2 depth, seeded by REST snapshot on market open and updated by CLOB WS events. Polybot extracts prices from WS events without maintaining any local orderbook state.

Observed divergence examples:
- Signal server sees ask=0.380, polybot sees ask=0.660 on the same market at similar timestamps
- Polybot may be using stale or incorrect price data for second-leg quotes

This divergence means:
- Signal server shadow profits are computed against accurate prices
- Polybot executes against potentially stale/inaccurate prices
- Second-leg maker quotes may be placed at prices that never fill

## Issue 4: Price Drift Between Signal and Execution

**Severity: Medium — first-leg prices generally match, but drift increases with latency.**

When polybot executes quickly after receiving a signal, first-leg prices match well (see cross-reference table in live_monitoring.md). However:

- Signal server inference to polybot receive: ~26-48ms
- Polybot processing + order placement: ~50-100ms additional
- Total end-to-end: ~76-148ms
- During volatile periods, prices can move significantly in this window

The price drift is currently manageable for first-leg taker buys but becomes critical for second-leg maker quotes where exact pricing matters.

## Issue 5: Shadow Success Does Not Translate to Real Execution

**Severity: High — fundamental gap between paper and live.**

Signal server shadow monitors the orderbook passively. When it "sees" a profitable second-leg maker fill in the shadow, it is observing real market prices without actually placing an order. This means:

- Shadow assumes our maker quote would be at the front of the queue — reality may differ
- Shadow assumes the quote would fill when opposite trades occur — queue position unknown
- Shadow does not account for the latency of real order placement and confirmation
- Shadow does not account for the fact that placing a maker order changes the book

## Recommended Fix Priority

1. **Fix second-leg order placement** — Debug why `placeSecondSideLimitOrder` returns undefined. Add error logging and validate API parameters.
2. **Add local orderbook to polybot** — Maintain a BTreeMap or similar structure from CLOB WS events, with REST seed on market open. Use this for accurate second-leg pricing.
3. **Fix WS fill detection** — Investigate why user WS fill events never fire. Check subscription format, auth tokens, and WS endpoint.
4. **Reduce detection latency** — Until WS fills work, reduce REST polling interval from ~2-3s to ~500ms.
5. **Queue-aware pricing** — For second-leg maker quotes, account for queue position by pricing slightly more aggressively than best bid.

## Related Files

- Signal server: `signal_server.py` — `SignalBroadcaster`, `BroadcastingPipeline`
- Polybot: `poly_bot_copytrade.ts` — `CopytradeArbBot.buyShares()`, `waitFirstLegMatchedFast()`, `runHedgedPairExecution()`
- Pipeline: `poly/predict/pipeline.py` — `handle_poly_book()`, `_predict()`
- Live script: `scripts/live_predict.py` — `_run_poly_ws()`, `_run_binance_ws()`
