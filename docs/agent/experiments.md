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
