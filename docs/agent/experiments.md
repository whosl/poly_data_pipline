# Experiments

This document summarizes the most important offline and live-shadow results. Treat every number as context, not production proof.

## Early Strict 20260417 Results

Earlier strict no-leak experiments on `20260417` used:

- chronological 70/15/15 split
- 10s purge
- 10s embargo
- validation-selected `p_enter` cutoff
- test evaluated once

Best early candidate:

| Run | Model | Cutoff | Test Entries | Test Avg `final_profit_10s` | Test Success |
| --- | --- | --- | ---: | ---: | ---: |
| BTC-only | RandomForest | `p_enter >= 0.74` | 60 | `+0.01117` | 78.33% |
| Mixed BTC+ETH | RandomForest | `p_enter >= 0.73` | 82 | `+0.00780` | 73.17% |

Caveat: this was one recovered date segment. ETH and mixed runs can be contaminated by BTC-only Binance reference.

## Binance Depth Experiments

Two different top10-ish experiments were run, and they are easy to confuse.

### First Recovered Depth Experiment

Dataset:

- `artifacts/depth20_experiment_20260417/btc_overlap_depth/alpha_dataset.parquet`
- rows: 55,599
- features: 57
- Binance depth features: 16
- overlap period: about 26.6 minutes

Best observed:

| Model | Cutoff | Validation Entries | Validation Avg | Test Entries | Test Avg | Test Success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | `p_enter >= 0.71` | 105 | `+0.003524` | 104 | `+0.013942` | 76.92% |

This suggested depth features might contain signal.

### Later Full Top10 Experiment

Dataset:

- `artifacts/depth_topn_experiment_20260417/btc_top10/alpha_dataset.parquet`
- rows: 55,599
- features: 59
- Binance depth features: 18

Best-ish observed:

| Model | Cutoff | Validation Entries | Validation Avg | Test Entries | Test Avg | Test Success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | `p_enter >= 0.75` | 180 | `-0.003500` | 248 | `-0.007419` | 58.87% |

This did not reproduce the positive result.

### Full Top20 Experiment

Dataset:

- `artifacts/depth_topn_experiment_20260417/btc_top20/alpha_dataset.parquet`
- rows: 55,599
- features: 66
- Binance depth features: 25

Best-ish observed:

| Model | Cutoff | Test Entries | Test Avg | Test Success |
| --- | --- | ---: | ---: | ---: |
| RandomForest | `p_enter >= 0.68` | 423 | about `-0.00286` | about 65.01% |

Conclusion:

- Depth features are used by models.
- Top20 looked better than top10 in that slice, but still negative.
- The sample is too small to decide whether to keep depth20 forever.
- Keep collecting compact depth and rerun on full fresh days.

## Latest BTC 5m Dataset

Main recent dataset:

- `artifacts/training_reprofit_20260420_21_5m/alpha_dataset.parquet`
- rows: 674,889
- markets: 355
- symbol: BTC only
- dates: `20260420`, `20260421`
- feature columns in latest model artifacts: about 109

Artifacts:

- fill models: `artifacts/training_reprofit_20260420_21_5m/fill_models/`
- unwind models: `artifacts/training_reprofit_20260420_21_5m/unwind_models/`
- single-model final profit models: `artifacts/training_reprofit_20260420_21_5m/final_profit_models/`
- two-stage policy selection: `artifacts/training_reprofit_20260420_21_5m/execution_policy_selection/`

## Latest Offline Two-Stage Results

Two-stage means fill classifier plus unwind regressor.

| Fill Model | Unwind Model | Policy | Validation Entries | Validation Avg | Test Entries | Test Avg | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| XGBoost | XGBoost | `exp>=0.020814, p_fill>=0.85, unwind>=0` | 28 | `+0.021085` | 22 | `+0.021517` | Very few entries |
| RandomForest | RandomForest | `exp>=0.020, p_fill>=0.85, unwind>=0` | 160 | `+0.018985` | 151 | `+0.016338` | Moved online most recently |
| LightGBM | LightGBM | `exp>=0.025, p_fill>=0.85, unwind>=0` | not listed | not listed | 84 | `+0.015947` | Positive offline |
| ExtraTrees | ExtraTrees | `exp>=0.020, p_fill>=0.65, unwind>=0` | not listed | not listed | 186 | `+0.014028` | Lower p_fill gate |

These are optimistic relative to live-shadow results.

## Offline Single-Model Live-Like Results

With `max_entries_per_signal_key=3`, earlier single-model cutoffs looked good offline:

| Model | Cutoff | Test Entries | Test Avg | Test Success |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | `0.721893` | 34 | `+0.022248` | 97.1% |
| RandomForest | `0.774243` | 36 | `+0.022007` | 100% |
| ExtraTrees | `0.580` | 51 | `+0.017016` | not listed |
| LightGBM | `0.730` | 181 | `+0.015497` | not listed |

Live-shadow contradicted these results; see [Live Monitoring](live_monitoring.md).

## Interpretation

The current evidence says:

- the pipeline can find plausible positive offline slices
- test-set results are sensitive to data window and label definition
- live fill rates are much worse than offline labels imply
- expected-profit policies need better fill and unwind labels

Do not infer production edge from any single table above.

## training_eventdriven_20260423: Event-Driven Sampled Book

Dataset:
- `artifacts/training_eventdriven_20260423/alpha_dataset.parquet`
- rows: 3,091,181
- columns: 181
- feature columns: 118
- symbol: BTC only
- periods: 5m = 1,630,456 rows, 15m = 1,460,725 rows
- dates: `20260420`, `20260421`
- split: chronological 70/15/15

Sampled book inputs:
- `data/normalized/20260420/poly_sampled_book.parquet`: 7,234,818 rows
- `data/normalized/20260421/poly_sampled_book.parquet`: 7,245,926 rows
- source `poly_l2_book` rows: 64,243,394 on 20260420 and 58,878,879 on 20260421

Event-driven defaults in `scripts/build_sampled_book.py`:
- Tier 1, any change triggers: `best_bid`, `best_ask`, `total_bid_levels`, `total_ask_levels`
- Tier 2, magnitude-filtered: `depth_top1/3/5/10/20_imbalance`, `cum_bid/ask_depth_top10/20`
- `min_gap_ms=0`, so no debounce
- tier2 magnitude thresholds still apply by default: `0.001` for imbalance, `10.0` for cumulative depth

Artifacts:
- fill models: `artifacts/training_eventdriven_20260423/fill_models/`
- unwind models: `artifacts/training_eventdriven_20260423/unwind_models/`

Selected live-shadow policy:

```text
fill model: xgboost_classifier
unwind model: extra_trees_regressor
threshold = 0.025
min_p_fill = 0.5
min_pred_unwind_profit = -0.05
market scope = btc-updown-5m only
```

Early Ireland live-shadow verification:

| Window | Profitable | Success Path | Total Profit | Avg Profit |
| --- | ---: | ---: | ---: | ---: |
| first 23 resolved | 19/23 = 82.6% | 17/23 = 73.9% | `+0.1848` | `+0.0080` |
| first 33 resolved | 28/33 = 84.8% | 26/33 = 78.8% | `+0.3355` | `+0.0102` |

This run is encouraging but is still a small live sample. Continue candidate calibration before treating it as stable edge.

### 2026-04-23 5m-Only Retrain and Live Switch Candidate

New 5m-only dataset:

- `artifacts/training_eventdriven_20260423_5m/alpha_dataset.parquet`
- rows: 1,630,456
- symbol: BTC only
- markets: 355
- assets: 710
- feature columns: 118
- split: chronological 70/15/15

Single-model fill cutoff selection under live-like gating:

| Fill Model | Selected Cutoff | Validation Entries | Validation Avg | Test Entries | Test Avg | Test Total | Test Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ExtraTrees | `0.795457` | 100 | `+0.015266` | 96 | `+0.016448` | `+1.5790` | 96.88% |
| LightGBM | `0.900000` | 136 | `+0.018166` | 139 | `+0.015011` | `+2.0865` | 92.81% |
| RandomForest | `0.880000` | 157 | `+0.019616` | 149 | `+0.014950` | `+2.2276` | 91.28% |
| XGBoost | `0.800000` | 101 | `+0.019942` | 109 | `+0.007305` | `+0.7962` | 88.99% |

Unwind regressor quality on the same 5m-only split:

| Unwind Model | MAE | RMSE | Rank Corr |
| --- | ---: | ---: | ---: |
| ExtraTrees | `0.05760` | `0.08369` | `0.1990` |
| XGBoost | `0.05767` | `0.08364` | `0.1963` |
| RandomForest | `0.05778` | `0.08384` | `0.1937` |
| LightGBM | `0.05931` | `0.08562` | `0.1720` |

Best two-stage live-like policies on the same 5m-only split:

| Fill Model | Unwind Model | Policy | Validation Entries | Validation Avg | Test Entries | Test Avg | Test Total | Test Positive |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RandomForest | XGBoost | `exp>=0.030, p_fill>=0.65, unwind>=0` | 30 | `+0.024661` | 19 | `+0.022954` | `+0.4361` | 89.47% |
| XGBoost | ExtraTrees | `exp>=0.02692, p_fill>=0.75, unwind>=0` | 39 | `+0.022782` | 50 | `+0.019643` | `+0.9822` | 96.00% |
| LightGBM | ExtraTrees | `exp>=0.025, p_fill>=0.85, unwind>=0` | 85 | `+0.019572` | 83 | `+0.019424` | `+1.6122` | 92.77% |
| LightGBM | XGBoost | `exp>=0.028608, p_fill>=0.75, unwind>=0` | 46 | `+0.020969` | 33 | `+0.017772` | `+0.5865` | 84.85% |

Operational decision after this run:

- keep `btc-updown-5m` only on live
- switch live fill model to `lightgbm_classifier`
- keep live unwind model as `extra_trees_regressor`
- set live policy to:
  - `threshold = 0.025`
  - `min_p_fill = 0.85`
  - `min_pred_unwind_profit = 0.0`
  - `max_entries_per_signal_key = 3`

Reasoning:

- `RandomForest + XGBoost` had the highest test average profit, but only 19 test entries.
- `LightGBM + ExtraTrees` kept positive average profit with a larger sample and is the better first live replacement.

Pre-switch Ireland live snapshot for the older event-driven pair
(`xgboost_classifier + extra_trees_regressor`, `threshold=0.025`, `min_p_fill=0.5`, `min_pred_unwind_profit=-0.05`):

- around `331` resolved signals
- accuracy about `71.3%`
- total profit about `+1.5502`
- the run remained highly sensitive to unwind losses

## training_20260422: EWMA + Winsorize + Time-Bucket

Dataset:
- `artifacts/training_20260422/alpha_dataset_5m.parquet`
- rows: 674,889
- features: 118 (including 9 EWMA features)
- markets: 355
- symbol: BTC only
- dates: `20260420`, `20260421`
- split: chronological 70/15/15, purge=0, embargo=0
- preprocessing: winsorize (q0.005/q0.995), time-bucket 250ms downsampling

Artifacts:
- fill models: `artifacts/training_20260422/fill_models/`
- unwind models: `artifacts/training_20260422/unwind_models/`

### Fill Classifier Results (target: y_two_leg_entry_10s)

The table below is from `artifacts/training_20260422/fill_evaluation/summary_metrics.json`.
An earlier version of this doc had a transcription error where XGBoost `p>=0.6`
showed fewer entries than `p>=0.7`; the artifact itself is monotonic.

**ExtraTrees classifier:**
| Threshold | Entries | Win Rate | Avg Profit |
| ---: | ---: | ---: | ---: |
| 0.5 | 48860 | 38.1% | -0.0197 |
| 0.6 | 17276 | 45.8% | -0.0067 |
| 0.7 | 3426 | 57.8% | +0.0176 |

**LightGBM classifier:**
| Threshold | Entries | Win Rate | Avg Profit |
| ---: | ---: | ---: | ---: |
| 0.5 | 50569 | 37.5% | -0.0214 |
| 0.6 | 26017 | 42.9% | -0.0138 |
| 0.7 | 10921 | 49.6% | -0.0026 |

**RandomForest classifier:**
| Threshold | Entries | Win Rate | Avg Profit |
| ---: | ---: | ---: | ---: |
| 0.5 | 46734 | 38.6% | -0.0191 |
| 0.6 | 19284 | 45.4% | -0.0085 |
| 0.7 | 6191 | 54.1% | +0.0075 |

**XGBoost classifier:**
| Threshold | Entries | Win Rate | Avg Profit |
| ---: | ---: | ---: | ---: |
| 0.5 | 10052 | 50.6% | -0.0006 |
| 0.6 | 4240 | 57.0% | +0.0109 |
| 0.7 | 1540 | 64.9% | +0.0313 |

Best: XGBoost at p>=0.7 (1540 entries, 64.9% win, EV +0.0313).

### Unwind Regressor Results (target: first_unwind_profit_proxy_10s)

| Model | MAE | RMSE | Rank Corr | Pred>=0 entries | Pred>=0 win | Pred>=0 avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ExtraTrees | 0.058 | 0.084 | 0.201 | 3676 | 54.7% | +0.0163 |
| RandomForest | 0.058 | 0.085 | 0.189 | 5463 | 48.8% | +0.0075 |
| XGBoost | 0.058 | 0.084 | 0.193 | 5828 | 48.2% | +0.0085 |
| LightGBM | 0.060 | 0.086 | 0.174 | 6660 | 44.2% | +0.0014 |

Best: ExtraTrees (rank_corr 0.201, best win rate at pred>=0).

### Live Results (Ireland)

Single-model RF at threshold=0.67: 39 signals in 10 minutes, 7.7% accuracy (3/39 profitable), total profit -1.23.

Two-stage RF+RF at threshold=0.005, min_p_fill=0.7, min_pred_unwind_profit=0.0: 0 signals after 2000 predictions. Live p_fill max 0.638, pred_unwind_profit max -0.009.

## training_eventdriven_20260423_5m: Live 4-Run Results (2026-04-24)

Model: XGBoost fill classifier + ExtraTrees unwind regressor
Artifacts: `artifacts/training_eventdriven_20260423_5m/`
Policy: `threshold=0.027, min_p_fill=0.75, min_pred_unwind_profit=0.0`
Market scope: `btc-updown-5m` only

### Per-Run Breakdown

| Run | Time Window (UTC) | Signals | Win | Unwind | Accuracy | Profit |
|-----|-------------------|---------|-----|--------|----------|--------|
| 1 | 01:05~01:34 | 7 | 7 | 0 | 100% | +0.1760 |
| 2 | 02:11~02:34 | 6 | 4 | 2 | 66.7% | +0.2046 |
| 3 | 02:57~03:21 | 11 | 10 | 1 | 90.9% | -0.0272 |
| 4 | 03:42~06:13 | 32 | 29 | 3 | 90.6% | +0.5993 |
| **Combined** | | **56** | **50** | **6** | **89.3%** | **+0.9527** |

### Price Bucket Analysis

| Bucket | N | Win | Unwind | Acc | Profit | AvgP | MaxLoss |
|--------|---|-----|--------|-----|--------|------|---------|
| 0.10-0.25 | 19 | 19 | 0 | 100% | +0.5141 | +0.0271 | +0.0082 |
| 0.25-0.35 | 15 | 12 | 3 | 80.0% | +0.3752 | +0.0250 | -0.1075 |
| 0.35-0.45 | 7 | 7 | 0 | 100% | +0.1599 | +0.0228 | +0.0048 |
| 0.45-0.55 | 10 | 9 | 1 | 90.0% | +0.0773 | +0.0077 | -0.0727 |
| 0.55-0.70 | 2 | 1 | 1 | 50.0% | -0.2465 | -0.1232 | -0.2465 |
| 0.70-0.90 | 3 | 2 | 1 | 66.7% | +0.0726 | +0.0242 | -0.0047 |

Key observations:

- Profit is concentrated in the 0.10-0.35 entry ask range (34 of 56 signals, +0.8893 profit)
- 0.45-0.55 is marginal: high accuracy (90%) but low avg profit (+0.0077) due to unfavorable risk-reward
- 0.55+ is negative on this sample: 5 signals, -0.1739 combined, with one large unwind loss at -0.2465
- The mechanism: higher entry ask → lower `success_profit_estimate` (because `second_leg_quote = 0.96 - best_ask`) → same unwind loss magnitude → worse risk-reward ratio

### Unwind Detail

| Signal | Entry Ask | Mid | p_fill | Score | Profit | Unwind Price | Price Move |
|--------|-----------|-----|--------|-------|--------|--------------|------------|
| S000006 | 0.32 | 0.320 | 0.795 | 0.030 | +0.0248 | 0.35 | +0.04 |
| S000017 | 0.33 | 0.325 | 0.810 | 0.028 | -0.1075 | 0.23 | -0.10 |
| S000024 | 0.54 | 0.545 | 0.784 | 0.027 | -0.0727 | 0.47 | -0.08 |
| S000034 | 0.26 | 0.260 | 0.867 | 0.049 | -0.0086 | 0.25 | -0.01 |
| S000037 | 0.63 | 0.630 | 0.757 | 0.030 | -0.2465 | 0.39 | -0.25 |
| S000051 | 0.75 | 0.750 | 0.826 | 0.031 | -0.0047 | 0.75 | -0.01 |

Signal S000037 is the worst unwind: entered at ask 0.63, mid moved from 0.63 to 0.39 (24-cent adverse move), resulting in -0.2465 loss. This confirms high-entry-ask tail risk.

### Operational Implications

- The two-stage XGBoost+ExtraTrees model with `min_p_fill=0.75` produces signals with 89% accuracy and positive expected value
- Entry ask price is a strong predictor of per-signal profitability
- Consider lowering `max_entry_ask` to 0.55 to cut the negative tail
- 56 signals is still a small sample; continue accumulating before making aggressive changes

## training_8models_20260425: Full 8-Model Retrain (2026-04-25)

Trained all 8 model types (RF, ExtraTrees, LightGBM, XGBoost) on full 5-day pure 5m BTC dataset.

Dataset:
- source: `artifacts/training_eventdriven_parallel_btc5m_20260420_24/alpha_dataset_parts/`
- dates: 20260420-20260424, pure 5m BTC
- sampled: train=1,500,000, val=200,000, test=200,000
- features: 120

Training script: `scripts/train_all_models_partitioned.py`

Artifacts: `artifacts/training_8models_20260425/`

### Individual Model Performance

Fill classifiers (AUC):

| Model | Accuracy | AUC |
| --- | ---: | ---: |
| lightgbm_classifier | 0.630 | 0.698 |
| xgboost_classifier | 0.721 | 0.701 |

Unwind regressors:

| Model | MAE | R2 | Rank Corr |
| --- | ---: | ---: | ---: |
| lightgbm_regressor | 0.064 | 0.047 | 0.203 |
| xgboost_regressor | 0.064 | 0.051 | 0.204 |

### Full 16-Combination Policy Selection

All fill×unwind pairs, sorted by val avg_profit:

| # | Fill + Unwind | exp>= | p_fill>= | unwind>= | val_avgP | val_n | val_win | test_avgP | test_n | test_win |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LGB_clf + RF_reg | 0.05 | 0.8 | -0.05 | 0.177 | 14 | 100% | — | 0 | — |
| 2 | XGB_clf + RF_reg | 0.05 | 0.6 | -0.05 | 0.143 | 44 | 93.2% | 0.105 | 42 | 90.5% |
| 3 | RF_clf + LGB_reg | 0.05 | 0.8 | -0.05 | 0.122 | 37 | 91.9% | 0.077 | 12 | 91.7% |
| 4 | XGB_clf + LGB_reg | 0.05 | 0.75 | -0.05 | 0.118 | 24 | 100% | 0.136 | 12 | 75.0% |
| 5 | ET_clf + RF_reg | 0.04 | 0.8 | -0.05 | 0.113 | 59 | 91.5% | 0.092 | 38 | 100% |
| 6 | ET_clf + XGB_reg | 0.05 | 0.5 | -0.05 | 0.109 | 21 | 100% | 0.156 | 30 | 83.3% |
| 7 | XGB_clf + ET_reg | 0.04 | 0.8 | -0.05 | 0.104 | 57 | 93.0% | 0.143 | 51 | 100% |
| 8 | XGB_clf + XGB_reg | 0.05 | 0.7 | -0.05 | 0.103 | 12 | 100% | 0.352 | 9 | 100% |
| 9 | ET_clf + ET_reg | 0.04 | 0.8 | -0.05 | 0.101 | 63 | 96.8% | 0.103 | 45 | 100% |
| 10 | ET_clf + LGB_reg | 0.05 | 0.75 | -0.05 | 0.100 | 68 | 95.6% | 0.102 | 32 | 87.5% |
| 11 | RF_clf + RF_reg | 0.05 | 0.8 | -0.05 | 0.095 | 13 | 100% | — | 0 | — |
| 12 | RF_clf + XGB_reg | 0.05 | 0.5 | -0.05 | 0.086 | 10 | 100% | 0.213 | 12 | 83.3% |
| 13 | LGB_clf + ET_reg | 0.04 | 0.8 | -0.05 | 0.078 | 158 | 86.1% | 0.113 | 117 | 94.9% |
| 14 | RF_clf + ET_reg | 0.04 | 0.8 | -0.05 | 0.078 | 158 | 86.1% | 0.115 | 116 | 95.7% |
| 15 | LGB_clf + XGB_reg | 0.04 | 0.8 | -0.05 | 0.069 | 604 | 86.8% | 0.073 | 491 | 85.3% |
| 16 | LGB_clf + LGB_reg | 0.05 | 0.8 | -0.05 | 0.068 | 31 | 83.9% | 0.190 | 12 | 91.7% |

Note: min_unwind=0.00 vs -0.05 was tested; results are identical because all selected entries already have pred_unwind >= 0.

### Live Deployment: Combo #2 (XGB_clf + RF_reg)

Deployed with threshold=0.05, min_p_fill=0.6, min_unwind=-0.05.

Result after 10,000 predictions: **0 signals**. Live expected_profit max 0.044, below 0.05 threshold. RF regressor is too conservative in live — it underpredicts relative to the offline validation.

Promising alternatives for next deployment:
- **#5 ET_clf + RF_reg** (threshold=0.04): val 0.113, test 0.092, 100% win rate on test
- **#7 XGB_clf + ET_reg** (threshold=0.04): val 0.104, test 0.143, 100% win rate on test, 51 test entries

## training_eventdriven_all5m_80m: Full 80M Row Retrain (2026-04-25)

Retrained on all available 5m BTC Up/Down data using GPT-processed 80M row partitioned Hive dataset.

Dataset:
- source: `data/normalized/` partitioned Hive-style (`date=YYYYMMDD/market_id=xxx.parquet`)
- 80M total rows across 656 parquet files, 183 columns, 12GB on disk
- dates: `20260420` through `20260424`
- split: train=20260420-20260422 (500K sampled per date = 1.5M), val=20260423 (200K), test=20260424 (200K)
- sampling: file-by-file PyArrow fragment loading with step-based downsampling (avoids OOM)
- feature columns: 120

Training script: `scripts/train_fast.py` (LightGBM + XGBoost only, no sklearn RF/ET)

Models trained (8 total):
- fill classifiers: `lightgbm_classifier`, `xgboost_classifier`
- unwind regressors: `lightgbm_regressor`, `xgboost_regressor`
- fill regressors: `lightgbm_regressor`, `xgboost_regressor` (auxiliary)
- unwind classifiers: `lightgbm_classifier`, `xgboost_classifier` (auxiliary)

Artifacts: `artifacts/training_eventdriven_all5m_80m/`

### Policy Selection Results (all 4 fill×unwind combinations)

| Fill Model | Unwind Model | Val AvgP | Val n | Val Win | Test AvgP | Test n | Test Win | Config |
|---|---|---:|---:|---:|---:|---:|---:|---|
| LGB_clf | LGB_reg | 0.1721 | 23 | 78.3% | 0.1561 | 17 | 94.1% | exp>=0.05, p_fill>=0.8, unwind>=-0.05 |
| XGB_clf | LGB_reg | 0.0963 | 17 | 94.1% | 0.1167 | 12 | 75.0% | exp>=0.05, p_fill>=0.75, unwind>=-0.05 |
| XGB_clf | XGB_reg | 0.0812 | 21 | 76.2% | 0.2075 | 36 | 80.6% | exp>=0.05, p_fill>=0.5, unwind>=-0.05 |
| **LGB_clf** | **XGB_reg** | **0.0716** | **802** | **81.9%** | **0.0857** | **530** | **87.5%** | **exp>=0.04, p_fill>=0.8, unwind>=-0.05** |

### Selected Combination: LGB_clf + XGB_reg

Selected for live deployment despite not having the highest avg_profit, because it has 35-40x more samples than the next-best combos (802 val / 530 test vs 17-23 val / 12-36 test), making it statistically far more reliable.

Thresholds:
- `expected_profit_threshold = 0.04` (up from 0.025/0.027)
- `min_p_fill = 0.8` (up from 0.5/0.75)
- `min_pred_unwind_profit = -0.05` (unchanged)

Test performance: 530 entries, 87.5% win rate, avg_profit=0.0857, total_profit=45.42

Compared to previous live model (XGB_clf + ET_reg, training_eventdriven_20260423_5m):
- Higher thresholds → fewer but higher-quality signals
- Much larger training dataset (1.5M sampled from 80M vs 1.6M from single parquet)
- More dates in training (5 days vs 2 days)
- LightGBM fill classifier instead of XGBoost

## training_chrono_purge_20260426: Chronological Split with 300s Purge

First proper chronological train/val/test split with 300s purge/embargo gap.

Dataset:
- source: `artifacts/training_eventdriven_parallel_btc5m_20260420_24/alpha_dataset_parts/`
- Train: 20260420-20260422 (57M rows, incremental 100 rounds/date × 3 = 300 total)
- Val: 20260423 (15.4M rows, 300s embargo + purge)
- Test: 20260424 (7.6M rows, 300s embargo)
- Features: 125 (before removal), 106 (after zero-importance removal)
- Purge/embargo: 300s between each split boundary

Training script: `scripts/train_full_chrono.py`

### 19 Zero-Importance Features Removed

Cross-model zero-importance features identified by checking gain=0 across all 4 models (LGB_clf, XGB_clf, LGB_reg, XGB_reg). Removed from both offline features and live pipeline:

Removed: `depth_top1_imbalance`, `imbalance_bucket`, `spread_bucket`, `price_bucket`, `min_order_size`, `maker_base_fee`, `taker_base_fee`, `top3/5/10_imbalance`, `cum_bid/ask_depth_topN_proxy`, `depth_level_imbalance_proxy`, `bid/ask_depth_slope`, `binance_spread`, `poly_trade_count_recent`, `poly_aggressive_buy/sell_volume_recent`, `poly_signed_volume_recent`

Result: 125 → 106 features.

### Policy Selection Results (4 combinations)

| Fill | Unwind | Val AvgP | Val n | Val Win | Test AvgP | Test n | Test Win | Config |
|---|---|---:|---:|---:|---:|---:|---:|---|
| XGB_clf | XGB_reg | 0.0962 | 7,037 | 85.9% | 0.0957 | 1,092 | 92.0% | exp>=0.05, p_fill>=0.8, unwind>=-0.05 |
| XGB_clf | LGB_reg | 0.0780 | 3,201 | 75.3% | 0.0540 | 1,616 | 59.4% | exp>=0.05, p_fill>=0.8, unwind>=-0.05 |
| LGB_clf | XGB_reg | 0.0697 | 10,400 | 74.7% | 0.0802 | 2,100 | 91.5% | exp>=0.05, p_fill>=0.8, unwind>=-0.05 |
| LGB_clf | LGB_reg | 0.0471 | 97,499 | 80.4% | 0.0528 | 31,342 | 82.1% | exp>=0.04, p_fill>=0.8, unwind>=-0.05 |

### Live Deployment: XGB_clf + XGB_reg

Deployed with threshold=0.05, min_p_fill=0.8, min_unwind=-0.05.

**Result: nearly zero signals.** After 212,000 predictions over ~4 hours:
- Signals: 1 (lost, profit -0.0991)
- p_fill distribution: median=0.30, p90=0.52 — XGB classifier too conservative for 0.8 gate
- blocked_p_fill: 99.7-100% of samples per window

### Live Deployment: LGB_clf + XGB_reg

Switched to LGB fill classifier (p_fill distribution much better: median=0.50, p90=0.72).

**Result: still 0 signals after 12,000 predictions.** expected_profit max=0.044, below 0.05 threshold. Market was calm (Saturday morning ET).

Status: LGB_clf + XGB_reg still running on Ireland. Need either market volatility or lower threshold (0.03-0.04) to produce signals.

## RF/ET Training Attempt (2026-04-26)

Attempted to train sklearn RandomForest and ExtraTrees models on the same chronological split.

**Goal:** Test if RF/ET produce better p_fill distributions for live signal production.

### Approach: Balanced Sampling

Strategy: keep all positive labels (y=1) + equal number of randomly sampled negatives per date.

Two-pass loading to avoid OOM:
1. Pass 1: load only label columns, determine positive/negative indices
2. Pass 2: load feature columns only for sampled recv_ns range

### Problems Encountered

1. **First attempt (6M target, importance-weighted):** OOM when concatenating 57M rows from 3 dates
2. **Second attempt (per-date importance sampling):** OOM during Polars concat, killed before training
3. **Third attempt (per-date sampling, 3M target):** sklearn RF training extremely slow — 23+ hours for 300 trees × 6M rows × 120 features. sklearn's exact split algorithm (sort-based) doesn't scale.
4. **Fourth attempt (balanced pos+neg, 3M target):** OOM again during data loading
5. **Fifth attempt (two-pass balanced, 3M target):** Process stuck after Phase 1 loading — Polars threads sleeping, possibly OOM during to_numpy() conversion on Mac (16GB RAM)

### Key Takeaway

sklearn RF/ET with exact splits does not scale to millions of rows × 100+ features. The fundamental issue:
- sklearn uses exact split (sorts all feature values at each node) — O(n × m × log(n)) per node
- LightGBM/XGBoost use histogram-based splits — much faster on large data
- Mac's 16GB RAM is insufficient for the data loading + numpy conversion step

### Next Steps

- Run RF/ET training on machine with more RAM (user's PC: i5-13600KF, 32GB) or Ireland server
- Consider using histogram-based RF alternatives (e.g., `cuml` RandomForest on GPU, or reduced feature set)
- Training script ready: `scripts/train_rf_et.py` with balanced sampling and two-pass loading
