# Current Objectives

Updated: 2026-04-26

## Live Pipeline Status

Current process: `signal_server.py` (WebSocket bridge for poly_bot on port 8765)
Server: Ireland (108.132.27.76)

**Current deployment: LGB_clf + XGB_reg** (chronological split, 300s purge)
- Model: `artifacts/training_chrono_purge_20260426/`
- Config: threshold=0.05, min_p_fill=0.8, min_unwind=-0.05
- Status: **0 signals after 12,000 predictions** — expected_profit max=0.044, below 0.05 threshold
- p_fill distribution improved (median 0.50 vs 0.30 for XGB_clf), but still no signals

**Previous deployment: XGB_clf + XGB_reg**
- 212,000 predictions → 1 signal (lost -0.0991)
- XGB classifier p_fill too conservative (median 0.30, 99.7% blocked by min_p_fill=0.8)

## Immediate Problems

1. **Live signal production blocked**: Neither XGB_clf nor LGB_clf can consistently produce expected_profit >= 0.05 in live. Options:
   - Lower threshold to 0.03-0.04
   - Try different model combos (RF/ET may have better p_fill distributions)
   - Wait for higher-volatility market conditions

2. **RF/ET training failed on Mac**: sklearn RF/ET too slow for 3M+ rows × 120 features (exact split algorithm). Multiple OOM kills on 16GB Mac. Need to run on machine with more RAM (user PC: 32GB) or Ireland server.

## Active Tasks

### 1. RF/ET Model Training

Script ready: `scripts/train_rf_et.py`
- Balanced sampling: positive labels + equal negatives per date
- Two-pass loading to reduce memory
- 100 trees × depth 12
- Needs machine with 32GB+ RAM to complete

Blocked on: finding suitable hardware to run training.

### 2. Live Threshold Calibration

Consider lowering threshold from 0.05 to 0.03 or 0.04.
- LGB_clf + LGB_reg (backtest combo #4) used threshold=0.04
- Current LGB_clf live max expected_profit = 0.044, so 0.04 would allow some signals

### 3. Feature Reduction

19 zero-importance features removed (125 → 106). Further pruning proposed (GPT V2-V5 plan) but not yet executed. Waiting for live model to work first.

## Recent Experiments Summary

### Chronological Split Training (training_chrono_purge_20260426)

Best backtest: XGB_clf + XGB_reg — val avgP=0.096, test avgP=0.096, test win=92%
But live: almost no signals due to conservative p_fill.

| Combo | Val AvgP | Val n | Test AvgP | Test n | Test Win |
|---|---:|---:|---:|---:|---:|
| XGB_clf + XGB_reg | 0.0962 | 7,037 | 0.0957 | 1,092 | 92.0% |
| XGB_clf + LGB_reg | 0.0780 | 3,201 | 0.0540 | 1,616 | 59.4% |
| LGB_clf + XGB_reg | 0.0697 | 10,400 | 0.0802 | 2,100 | 91.5% |
| LGB_clf + LGB_reg | 0.0471 | 97,499 | 0.0528 | 31,342 | 82.1% |

### Live Deployment History

| Date | Model | Threshold | Signals | Notes |
|------|-------|-----------|---------|-------|
| 04-25 | XGB_clf + RF_reg | 0.05 | 0 | RF_reg too conservative |
| 04-26 | XGB_clf + XGB_reg | 0.05 | 1 (lost) | p_fill too low (median 0.30) |
| 04-26 | LGB_clf + XGB_reg | 0.05 | 0 (running) | p_fill better but exp_profit < 0.05 |

## Reference

- SSH: `ssh ireland`
- Logs: `~/poly_trade_pipeline/logs/signal_server.log`
- WebSocket: `ws://108.132.27.76:8765`
- Chrono training artifacts: `artifacts/training_chrono_purge_20260426/`
- RF/ET training script: `scripts/train_rf_et.py`
- Chrono training script: `scripts/train_full_chrono.py`
