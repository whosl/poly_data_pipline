# Training Strategy

The current modeling objective is short-horizon tradable edge, not final market resolution.

## Target Strategy

The intended strategy is a two-leg round trip:

1. First leg: taker buy the selected Polymarket outcome at current `best_ask`.
2. Second leg: maker buy the opposite outcome within a short horizon, currently 10 seconds.
3. Desired condition: `first_leg_ask + future_opposite_maker_fill_price <= 0.96`.

The `0.96` cap is a simplified edge/cost rule. Since one Up share plus one Down share pays out `1.00`, buying both legs for `<= 0.96` leaves about `0.04` gross edge before detailed fee/slippage assumptions.

## Current Profit Definition

The latest training/live formula is:

```text
entry_price = best_ask
first_leg_price = entry_price + price_buffer
fee_per_share = fee_rate * first_leg_price * (1 - first_leg_price)
second_leg_size = 1 - fee_per_share
second_leg_quote = max_total_price - entry_price
```

Success branch:

```text
success_profit = second_leg_size - (first_leg_price + second_leg_size * second_leg_quote)
```

Failure/unwind branch:

```text
unwind_profit = second_leg_size * future_same_leg_best_bid - first_leg_price
```

Final realized training target:

```text
final_profit_10s = success_profit if y_two_leg_entry_10s == "enter" else unwind_profit
y_final_profit_entry_10s = "enter" if final_profit_10s > 0 else "skip"
```

Important: unwind profit is signed. It can be positive.

Example:

```text
entry_price = 0.55
future_same_leg_best_bid = 0.58
```

Even if the second leg does not fill, selling the first leg at the future bid may produce positive PnL after fee/share-size adjustment.

## Modeling Layers

| Layer | Purpose | Target | Status |
| --- | --- | --- | --- |
| Layer 1 alpha | Predict short-horizon markout/direction | `markout_10s_bps`, `y_cls_10s` | Implemented for sanity checks |
| Layer 2 entry | Decide whether first-leg taker entry is worth doing | `y_final_profit_entry_10s` | Current main offline target |
| Layer 3 exit/fill | Predict second-leg maker fill probability and quality | fill / time-to-fill / unwind labels | Partially derived; needs queue-aware replay |

## Two-Stage Execution Policy

The live/offline two-stage policy separates fill probability from unwind quality:

```text
p_fill = fill classifier probability
pred_unwind_profit = unwind regressor prediction
pred_expected_profit = p_fill * success_profit + (1 - p_fill) * pred_unwind_profit
```

Current live signal gates:

- `pred_expected_profit >= threshold`
- `p_fill >= min_p_fill`
- `pred_unwind_profit >= min_pred_unwind_profit`
- `min_entry_ask <= entry_ask <= max_entry_ask`
- spread gate
- time-to-expiry gate
- opposite asset available
- signal cooldown
- max entries per signal key

The current verified Ireland live-shadow policy (`training_eventdriven_20260423` artifacts) uses:

```text
fill model = xgboost_classifier
unwind model = extra_trees_regressor
threshold = 0.025
min_p_fill = 0.5
min_pred_unwind_profit = -0.05
min_entry_ask = 0.10
max_entry_ask = 0.90
min_time_to_expiry = 20
max_spread = 0.05
signal_cooldown = 10 (defaults to horizon)
max_entries_per_signal_key = 0 (unlimited)
POLY_UPDOWN_MARKETS = btc-updown-5m
```

The previous `training_20260422` RF+RF policy used `threshold=0.005, min_p_fill=0.7, min_pred_unwind_profit=0.0` and produced 0 signals after 2000 predictions.

Previous policy (training_reprofit_20260420_21_5m artifacts) used `threshold=0.020, min_p_fill=0.85` but produced 0 signals because live p_fill max was ~0.74.

## Split Discipline

This is time-series data.

Use chronological splits only:

- train: earliest 70%
- validation: next 15%
- test: final 15%

For strict 10s experiments, use:

```text
purge = 10s
embargo = 10s
```

Select thresholds on validation, then freeze and apply to test. Do not tune cutoffs or top-K on test.

## Baseline Models

Keep models simple until labels are reliable:

- logistic regression
- SGD logistic classifier
- GaussianNB
- RandomForest
- ExtraTrees
- LightGBM
- XGBoost

Do not add deep learning yet. The bottleneck is label realism and offline/live alignment, not model class.

## Core Commands

### Latest training_eventdriven_20260423 pipeline

Build event-driven sampled book:

```bash
python scripts/build_sampled_book.py --mode event-driven --overwrite --dates 20260420 20260421
```

Observed sampled rows:

```text
20260420: 7,234,818
20260421: 7,245,926
```

Build features:

```bash
python scripts/build_features.py \
  --data-dir data \
  --dates 20260420 20260421 \
  --output-dir artifacts/training_eventdriven_20260423 \
  --sample-interval-ms 100
```

Train fill classifiers:

```bash
python scripts/train_alpha_model.py \
  --dataset artifacts/training_eventdriven_20260423/alpha_dataset.parquet \
  --output-dir artifacts/training_eventdriven_20260423/fill_models \
  --target-reg first_unwind_profit_proxy_10s \
  --target-cls y_two_leg_entry_10s \
  --models random_forest_classifier extra_trees_classifier lightgbm_classifier xgboost_classifier
```

Train unwind regressors:

```bash
python scripts/train_alpha_model.py \
  --dataset artifacts/training_eventdriven_20260423/alpha_dataset.parquet \
  --output-dir artifacts/training_eventdriven_20260423/unwind_models \
  --target-reg first_unwind_profit_proxy_10s \
  --target-cls y_two_leg_entry_10s \
  --models random_forest_regressor extra_trees_regressor lightgbm_regressor xgboost_regressor
```

### Previous training_20260422 pipeline

Build sampled book (time-bucket 250ms):

```bash
python scripts/build_sampled_book.py --mode time-bucket --sample-interval-ms 250 --overwrite --dates 20260420 20260421
```

Build features with EWMA + winsorize:

```bash
python scripts/build_features.py \
  --data-dir data \
  --dates 20260420 20260421 \
  --output-dir artifacts/training_20260422 \
  --sample-interval-ms 100
```

Filter to 5m only:

```bash
python scripts/filter_dataset.py \
  --input artifacts/training_20260422/alpha_dataset.parquet \
  --output artifacts/training_20260422/alpha_dataset_5m.parquet \
  --period 5m
```

Train fill classifiers:

```bash
python scripts/train_alpha_model.py \
  --dataset artifacts/training_20260422/alpha_dataset_5m.parquet \
  --output-dir artifacts/training_20260422/fill_models \
  --target-reg first_unwind_profit_proxy_10s \
  --target-cls y_two_leg_entry_10s \
  --models random_forest_classifier extra_trees_classifier lightgbm_classifier xgboost_classifier
```

Train unwind regressors:

```bash
python scripts/train_alpha_model.py \
  --dataset artifacts/training_20260422/alpha_dataset_5m.parquet \
  --output-dir artifacts/training_20260422/unwind_models \
  --target-reg first_unwind_profit_proxy_10s \
  --target-cls y_two_leg_entry_10s \
  --models random_forest_regressor extra_trees_regressor lightgbm_regressor xgboost_regressor
```

### Previous training_reprofit_20260420_21_5m pipeline

```bash
python scripts/build_features.py \
  --data-dir data \
  --dates 20260420 20260421 \
  --output-dir artifacts/training_reprofit_20260420_21_5m \
  --sample-interval-ms 100
```

Select two-stage policy:

```bash
python scripts/select_execution_policy.py \
  --run btc5m \
    artifacts/training_reprofit_20260420_21_5m/alpha_dataset.parquet \
    artifacts/training_reprofit_20260420_21_5m/alpha_dataset.parquet \
    artifacts/training_reprofit_20260420_21_5m/fill_models \
    random_forest_classifier \
    artifacts/training_reprofit_20260420_21_5m/unwind_models \
    random_forest_regressor \
  --output-dir artifacts/training_reprofit_20260420_21_5m/execution_policy_selection \
  --split-purge-ms 10000 \
  --split-embargo-ms 10000 \
  --selection-mode live \
  --max-entries-per-signal-key 3 \
  --min-validation-entries 25
```
