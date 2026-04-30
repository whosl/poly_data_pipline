# Current Objectives

Updated: 2026-04-30

## Live Pipeline Status

Current processes on Ireland (108.132.27.76):
- **Signal server**: `signal_server.py` (WebSocket bridge for poly_bot on port 8765)
- **Polybot**: `poly_bot_copytrade.ts` (Node.js execution bot)

**Current deployment: 3-stage s1_xgb + s2_et + s3_et** (`three_stage_eval`)
- Config: threshold=0.05, min_p_fill=0.7, min_p_first_fill=0.5, min_pred_unwind_profit=-0.05
- Offline test: 364 signals, 95.1% win rate, avgP=+0.1009
- p_first used as GATE (not multiplier) — filters out likely non-filling first-leg orders
- Best 3-stage combo out of 12 model combinations evaluated

## 3-Stage Model Architecture

```
Signal generated → Stage 1 gate: p_first_leg_fill >= 0.5?
  → YES → Stage 2+3: expected_profit = p_second * 0.04 + (1-p_second) * pred_unwind
  → Gate: ep >= 0.05 & pred_unwind >= -0.05 & p_second >= 0.7
```

Models (all from `artifacts/three_stage_eval/`):
- **s1_xgb**: XGB classifier on `y_first_leg_fill` (350ms forward-looking fill label)
- **s2_et**: ET classifier on `y_two_leg_entry_binary_10s`
- **s3_et**: ET regressor on `first_unwind_profit_proxy_10s`

## Deployment History

| Date | Model | Config | Shadow | Live | Notes |
|------|-------|--------|--------|------|-------|
| 04-23 | XGB_clf + ET_reg | thr=0.025, pf=0.5 | 33 sig, 84.8% prof, +0.336 | N/A | Shadow-only run |
| 04-25 | XGB_clf + RF_reg | thr=0.05 | 0 signals | N/A | RF_reg too conservative |
| 04-26 | XGB_clf + XGB_reg | thr=0.05 | 1 (lost) | N/A | p_fill too low |
| 04-26 | LGB_clf + XGB_reg | thr=0.05 | 0 signals | N/A | exp_profit < 0.05 |
| 04-27 | RF_clf + RF_reg | thr=0.02 | running | N/A | p_fill max=0.78 |
| 04-28 | LGB_clf + XGB_reg | thr=0.04, pf=0.8 | 388 sig, 80.4% prof, +3.022 | 11 fills, 0 hedges | Execution pipeline broken |
| 04-30 | **s1_xgb+s2_et+s3_et** | thr=0.05, pf=0.7, pf1=0.5 | 364 sig, 95.1% wr, avgP=+0.101 | pending | 3-stage model deployed |

## Current Blockers

1. **Second-leg hedge never succeeds** — `placeSecondSideLimitOrder` returns undefined (10/11)
2. **User WS fill events never fire** — 10s REST polling delay
3. **No local orderbook in polybot** — price divergence between signal server and execution
4. See [Live Execution Issues](live_execution_issues.md) for full analysis and fix priorities

## Reference

- SSH: `ssh -4 -i ~/.ssh/EuKey.pem -o StrictHostKeyChecking=no ubuntu@108.132.27.76`
- Signal server log: `~/poly_trade_pipeline/logs/signal_server_debug.log`
- Polybot log: `~/poly_bot/logs/bot-YYYY-MM-DD.log`
- WebSocket: `ws://127.0.0.1:8765` (localhost only)
