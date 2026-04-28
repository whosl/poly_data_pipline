# Current Objectives

Updated: 2026-04-29

## Live Pipeline Status

Current processes on Ireland (108.132.27.76):
- **Signal server**: `signal_server.py` (WebSocket bridge for poly_bot on port 8765)
- **Polybot**: `poly_bot_copytrade.ts` (Node.js execution bot)

**Current deployment: LGB_clf + XGB_reg** (`training_eventdriven_all5m_80m`)
- Config: threshold=0.04, min_p_fill=0.8, min_pred_unwind_profit=-0.05
- Shadow results: 388 signals, 80.4% profitable, +3.022 total profit
- Real execution: 11 first-leg fills, **0 second-leg hedges**, all UNWIND
- **See [Live Execution Issues](live_execution_issues.md) for blocker analysis**

## Deployment History

| Date | Model | Config | Shadow | Live | Notes |
|------|-------|--------|--------|------|-------|
| 04-23 | XGB_clf + ET_reg | thr=0.025, pf=0.5 | 33 sig, 84.8% prof, +0.336 | N/A | Shadow-only run |
| 04-25 | XGB_clf + RF_reg | thr=0.05 | 0 signals | N/A | RF_reg too conservative |
| 04-26 | XGB_clf + XGB_reg | thr=0.05 | 1 (lost) | N/A | p_fill too low |
| 04-26 | LGB_clf + XGB_reg | thr=0.05 | 0 signals | N/A | exp_profit < 0.05 |
| 04-27 | RF_clf + RF_reg | thr=0.02 | running | N/A | p_fill max=0.78 |
| 04-28 | **LGB_clf + XGB_reg** | thr=0.04, pf=0.8 | 388 sig, 80.4% prof, +3.022 | 11 fills, 0 hedges | Execution pipeline broken |

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
