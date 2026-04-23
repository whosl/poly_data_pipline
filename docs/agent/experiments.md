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
