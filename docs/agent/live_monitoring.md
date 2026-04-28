# Live Monitoring

This document covers the Ireland live-shadow pipeline and polybot real execution.

## Instance And Runtime

Tracked docs intentionally avoid private SSH material. Use:

```text
docs/private/ireland_lightsail.md
```

Last known facts:

- remote repo: `/home/ubuntu/poly_trade_pipeline`
- Signal server process: `signal_server.py` (WebSocket bridge on port 8765)
- Polybot process: `poly_bot_copytrade.ts` (Node.js execution bot)
- tmux sessions: `signal_server`, `poly_bot`
- Signal server log: `~/poly_trade_pipeline/logs/signal_server_debug.log`
- Polybot log: `~/poly_bot/logs/bot-YYYY-MM-DD.log`
- market focus: BTC 5m
- Binance symbol: `btcusdt`

## Current Online Model

Latest verified live command on Ireland uses `training_eventdriven_all5m_80m` artifacts:

```text
python signal_server.py \
  --model-path artifacts/training_eventdriven_all5m_80m/fill_models/lightgbm_classifier.joblib \
  --unwind-model-path artifacts/training_eventdriven_all5m_80m/unwind_models/xgboost_regressor.joblib \
  --threshold 0.04 --min-p-fill 0.8 --min-pred-unwind-profit -0.05 --port 8765
```

Current market scope is BTC 5m only.

## Signal Server Shadow Results (as of 2026-04-28)

### Overall Stats

| Metric | Value |
|--------|-------|
| Total signals broadcast | 388 |
| Resolved signals | ~388 |
| Profitable (shadow) | 312/388 = 80.4% |
| Shadow total profit | +3.022 |
| Shadow avg profit per signal | +0.00779 |
| Monitoring horizon | 10s |

### Shadow Profit Breakdown

- Success path (second-leg maker fill): profitable in most cases
- Unwind path: directionally favorable first-leg moves sometimes yield positive unwind
- Win rate is high but sample is still small — treat as encouraging, not proven

## Polybot Live Execution Results (as of 2026-04-28)

### Summary

| Metric | Value |
|--------|-------|
| First-leg taker fills | 11 |
| Second-leg maker hedges completed | **0** |
| Second-leg HEDGE TIMEOUT | 1 |
| Second-leg `placeSecondSideLimitOrder` returned undefined | 10/11 |
| All first-leg positions eventually UNWIND | 11/11 |

### First-Leg Fill Details

All 11 first-leg taker buys and their outcomes:

| # | Timestamp | Direction | Ask Price | Gross Shares | Fee (USDC) | Net Shares | Outcome |
|---|-----------|-----------|-----------|-------------|------------|------------|---------|
| 1 | 04-28 08:01 | BUY_DOWN | 0.380 | 5.000 | 0.190 | 4.810 | UNWIND |
| 2 | 04-28 08:01 | BUY_UP | 0.340 | 5.000 | 0.170 | 4.830 | UNWIND |
| 3 | 04-28 08:01 | BUY_DOWN | 0.380 | 5.000 | 0.190 | 4.810 | UNWIND |
| 4 | 04-28 08:01 | BUY_UP | 0.340 | 5.000 | 0.170 | 4.830 | UNWIND |
| 5 | 04-28 08:01 | BUY_DOWN | 0.400 | 5.000 | 0.200 | 4.800 | UNWIND |
| 6 | 04-28 08:01 | BUY_UP | 0.380 | 5.000 | 0.190 | 4.810 | UNWIND |
| 7 | 04-28 08:01 | BUY_DOWN | 0.400 | 5.000 | 0.200 | 4.800 | UNWIND |
| 8 | 04-28 08:01 | BUY_UP | 0.380 | 5.000 | 0.190 | 4.810 | UNWIND |
| 9 | 04-28 08:04 | BUY_DOWN | 0.560 | 5.000 | 0.280 | 4.720 | UNWIND |
| 10 | 04-28 08:07 | BUY_DOWN | 0.580 | 5.000 | 0.290 | 4.710 | UNWIND |
| 11 | 04-28 08:12 | BUY_DOWN | 0.560 | 5.000 | 0.280 | 4.720 | UNWIND |

Notes:
- Polymarket API `size_matched` returns gross shares
- Taker fee = `ask_price * gross_shares` (rate varies; ~3.8% observed)
- Net shares = `gross_shares - fee`
- All positions eventually unwind — no second-leg hedge was completed

### Signal Server vs Polybot Log Cross-Reference

Matching signal server shadow signals to polybot real trades:

| Signal Server Signal | Polybot Trade | Signal Ask | Polybot Executed Ask | Match? |
|---------------------|---------------|-----------|---------------------|--------|
| BUY_DOWN, ask=0.380 | BUY_DOWN, ask=0.380 | 0.380 | 0.380 | Yes |
| BUY_UP, ask=0.340 | BUY_UP, ask=0.340 | 0.340 | 0.340 | Yes |
| BUY_DOWN, ask=0.380 | BUY_DOWN, ask=0.380 | 0.380 | 0.380 | Yes |
| BUY_DOWN, ask=0.400 | BUY_DOWN, ask=0.400 | 0.400 | 0.400 | Yes |
| BUY_DOWN, ask=0.560 | BUY_DOWN, ask=0.560 | 0.560 | 0.560 | Yes |
| BUY_DOWN, ask=0.580 | BUY_DOWN, ask=0.580 | 0.580 | 0.580 | Yes |
| BUY_DOWN, ask=0.560 | BUY_DOWN, ask=0.560 | 0.560 | 0.560 | Yes |

First-leg prices match well between signal server and polybot when executed promptly.

### Orderbook Data Source Comparison

| Aspect | Signal Server | Polybot |
|--------|--------------|---------|
| Polymarket WS | Full L2 orderbook events via CLOB WS | Lightweight WS event extraction |
| Local orderbook | Rust BTreeMap engine, maintained | No local orderbook |
| Initial snapshot | REST seed on market open | None |
| Depth tracking | Full depth (all price levels) | Best bid/ask only |
| Price accuracy | High (maintained BTreeMap) | Dependent on WS event timing |
| Fill detection | Shadow only (no real orders) | REST polling (~10s delay) |

### Inference Latency

| Metric | Value |
|--------|-------|
| Signal server ML inference (p50) | ~23ms |
| E2E signal → polybot receive | ~26-48ms |
| Polybot first-leg placement after signal | ~50-100ms |

## Log Events

The live pipeline writes structured JSONL events.

`pipeline_starting`:

- model paths
- thresholds
- sample interval
- horizon
- entry filters

`signal_open`:

- market/asset/outcome
- entry ask/bid/mid/spread
- first-leg price
- second-leg quote
- success profit estimate
- `p_fill`
- `pred_unwind_profit`
- `pred_expected_profit`
- gates and thresholds

`maker_fill_observed`:

- observed future opposite trade evidence
- quote/fill price
- fill lag

`signal_close`:

- result: `success`, `unwind`, or `no_opposite`
- signed realized profit
- first-leg cost
- second-leg quote/cost
- future same-leg best bid/ask/mid
- fill lag when applicable

`stats`:

- prediction count
- open/closed signals
- rolling profit stats
- score distributions
- gate counts

## Live Profit Semantics

Success branch:

```text
profit = second_leg_size - (first_leg_price + second_leg_size * second_leg_quote)
```

Unwind branch:

```text
profit = second_leg_size * future_same_leg_best_bid - first_leg_price
```

Do not interpret every unwind as a loss. A directionally favorable first-leg move can make unwind positive.

## How To Check Status

On Ireland:

```bash
# Signal server
tmux capture-pane -t signal_server -p -S -120 | tail -120
tail -n 200 ~/poly_trade_pipeline/logs/signal_server_debug.log

# Polybot
tmux capture-pane -t poly_bot -p -S -120 | tail -120
tail -n 200 ~/poly_bot/logs/bot-$(date +%Y-%m-%d).log

# Process check
ps -eo pid,ppid,pcpu,pmem,etime,cmd --sort=-pcpu | grep -E "signal_server|poly_bot|node" | grep -v grep
```

## Main Risk

Signal server shadow shows 80.4% win rate but real execution fails to complete second-leg hedges. The core problem is not prediction quality — it is execution. See [Live Execution Issues](live_execution_issues.md) for detailed analysis.
