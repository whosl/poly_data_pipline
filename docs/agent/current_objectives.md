# Current Objectives

Updated: 2026-04-27

## Live Pipeline Status

Current process: `signal_server.py` (WebSocket bridge for poly_bot on port 8765)
Server: Ireland (108.132.27.76)

**Current deployment: RF_clf + RF_reg** (combined 8-model evaluation)
- Model: `artifacts/training_combined_20260427/`
- Config: threshold=0.02, min_p_fill=0.65, min_unwind=-0.05
- Training: balanced sampling on 2M rows (666K per date × 3 dates), 105 features
- Backtest: val avgP=0.1048 (n=2820), test avgP=0.0353 (n=1006)
- Live: producing signals, initial results negative (early evaluation, small sample)

**Previous deployment: LGB_clf + XGB_reg**
- 12,000 predictions → 0 signals (expected_profit max=0.044, below 0.05 threshold)

## RF/ET Training — Completed

Trained on Windows PC (i5-13600KF, 32GB RAM):
- Script: `scripts/train_rf_et_win.py` — memory-efficient pipeline
- Phase A: Per-market pyarrow predicate pushdown → partitioned dataset (80M rows, 105 features)
- Phase B: Balanced sampling → numpy arrays with periodic consolidation, capped at 2M total
- 4 models: rf_classifier, et_classifier, rf_regressor, et_regressor
- Training times: RF_clf 1.6min, ET_clf 1.0min, RF_reg 16.8min, ET_reg 12.1min

XGBoost/LightGBM also trained on same data (2M balanced rows):
- Script: `scripts/combined_policy_eval.py`
- Training times: LGB_clf 11s, XGB_clf 11s, LGB_reg 8s, XGB_reg 11s

## Combined 8-Model Policy Evaluation

16 combos (4 classifiers × 4 regressors) evaluated on chronological val/test splits.

**Test results (entries > 100 only — meaningful signal volume):**

| Fill Model | Unwind Model | Val AvgP | Val n | Test AvgP | Test n | Test WR |
|---|---|---:|---:|---:|---:|---:|
| **rf_classifier** | **rf_regressor** | 0.1048 | 2,820 | **0.0353** | **1,006** | 62% |
| xgboost_classifier | rf_regressor | 0.0903 | 3,449 | 0.0271 | 2,112 | 81% |
| lightgbm_classifier | lightgbm_regressor | 0.0881 | 9,070 | 0.0184 | 2,393 | 92% |
| lightgbm_classifier | rf_regressor | 0.0999 | 2,363 | 0.0157 | 1,371 | 93% |

**Key findings:**
- Val overfitting severe: top combos by val avgP (0.32, 0.20) produce 0 test entries
- ET models overfit worst: ET_clf has high val avgP but near-zero test entries
- RF_clf × RF_reg is the best combo for live: highest test avgP with sufficient volume
- RF models have less extreme p_fill distributions (max ~0.78 vs 0.95+ for boosted models)

### Live Deployment History

| Date | Model | Threshold | Signals | Notes |
|------|-------|-----------|---------|-------|
| 04-23 | XGB_clf + ET_reg | 0.025 | 33 | 84.8% profitable, +0.3355 |
| 04-25 | XGB_clf + RF_reg | 0.05 | 0 | RF_reg too conservative |
| 04-26 | XGB_clf + XGB_reg | 0.05 | 1 (lost) | p_fill too low (median 0.30) |
| 04-26 | LGB_clf + XGB_reg | 0.05 | 0 | p_fill better but exp_profit < 0.05 |
| 04-27 | **RF_clf + RF_reg** | 0.02 | running | p_fill max=0.78, threshold lowered to 0.02 |

## Reference

- SSH: `ssh -4 -i ~/.ssh/EuKey.pem -o StrictHostKeyChecking=no ubuntu@108.132.27.76`
- Logs: `~/poly_trade_pipeline/logs/signal_server_rf.log`
- WebSocket: `ws://108.132.27.76:8765`
- Combined eval artifacts: `artifacts/combined_eval/`
- RF/ET training artifacts: `artifacts/training_rf_et_win/`
- Partitioned dataset: `artifacts/training_rf_et_win/alpha_dataset_parts/`
- Training scripts: `scripts/train_rf_et_win.py`, `scripts/combined_policy_eval.py`
