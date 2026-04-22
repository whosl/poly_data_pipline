# Project History

This document records the major user requests already handled in this thread. It is intentionally practical: what changed, why it changed, and what a new agent should not redo blindly.

## Setup And Data Inspection

- Cloned `git@github.com:whosl/poly_data_pipline.git` into the local workspace.
- Set up the Python virtualenv from README instructions.
- Installed dependencies, including XGBoost.
- Built the Rust `poly_core` extension with `maturin`.
- Inspected `data/` multiple times as the user replaced local datasets.
- Confirmed the repo writes three layers:
  - raw gzip JSONL under `data/raw_feed`
  - normalized parquet under `data/normalized`
  - research/label parquet under `data/research`

## First Training Pipeline

Built the initial reusable training pipeline on top of normalized/research parquet, not raw websocket messages.

Important modules:

- `poly/training/config.py`
- `poly/training/io.py`
- `poly/training/features.py`
- `poly/training/labels.py`
- `poly/training/splits.py`
- `poly/training/models.py`
- `poly/training/evaluation.py`

Important scripts:

- `scripts/build_features.py`
- `scripts/build_labels.py`
- `scripts/train_alpha_model.py`
- `scripts/evaluate_alpha_model.py`
- `scripts/select_entry_cutoffs.py`
- `scripts/select_execution_policy.py`

The training pipeline supports:

- 100ms sample construction
- normalized/research parquet discovery
- feature generation
- 1s/3s/5s/10s labels
- chronological train/validation/test splits
- purge/embargo
- validation-selected cutoffs
- baseline classifiers and regressors

## Target Reframing

The user first asked whether the prediction target should be something like `markout_10s > 0.04`.

The target was reframed to the actual strategy question:

```text
first_leg_ask + future_opposite_maker_fill_price <= 0.96
```

Then the repo added strategy-oriented labels:

- `future_opposite_maker_fill_price_10s`
- `two_leg_total_price_10s`
- `two_leg_edge_10s`
- `y_two_leg_entry_10s`
- `first_unwind_loss_proxy_10s`
- `first_unwind_profit_proxy_10s`
- `final_profit_10s`
- `y_final_profit_entry_10s`

The key lesson: `success` means second-leg maker opportunity happened within horizon, not that the market resolved correctly.

## Model Families Added

Classifier support includes:

- logistic regression
- SGD logistic classifier
- GaussianNB
- RandomForest
- ExtraTrees
- LightGBM
- XGBoost

Regressor support includes:

- linear/ridge style baselines where available
- RandomForest regressor
- ExtraTrees regressor
- LightGBM regressor
- XGBoost regressor

The user later preferred classifier-focused runs for final-profit entry decisions, then added a two-stage fill-classifier plus unwind-regressor policy.

## Symbol Experiments

The pipeline was extended to evaluate:

- BTC-only train/test
- ETH-only train/test
- train BTC, test ETH
- train ETH, test BTC
- mixed BTC+ETH

Be careful with old ETH/mixed results. Earlier datasets often had only BTCUSDT Binance reference, which can make ETH rows misleading.

## Binance Depth Work

Originally, `btcusdt@depth20@100ms` was collected but its core value was discarded. The Binance API did provide full `bids` and `asks`; our collector/normalizer kept only:

- best bid
- best ask
- spread
- midpoint
- level counts

Added:

- `poly/collector/binance_depth.py`
- compact top-N depth features
- normalized schema extensions
- training feature ingestion for Binance depth

Relevant commits:

- `d6af588 Add Binance depth feature extraction`
- `1294526 Compact Binance depth raw collection`

After compaction, the collector still subscribes depth20@100ms but no longer saves raw `bids`/`asks` arrays for depth messages. It saves compact derived features instead.

## Raw Recovery

Some local gzip raw files were truncated/corrupt. Recovery logic was added so one damaged gzip member does not kill the entire normalizer.

Recovered depth artifacts:

- `artifacts/depth20_recovered/20260417/binance_depth20_features.parquet`
- `artifacts/depth20_recovered/20260418/binance_depth20_features.parquet`

Approximate recovered Binance depth:

- 20260417: 15,658 depth20 rows, about 26.6 minutes
- 20260418: 79,194 depth20 rows, about 161 minutes

## Collection Load And BTC-Only Default

Tokyo Lightsail CPU pressure was investigated. After Binance depth compaction, the main load was Polymarket L2 volume, not Binance depth.

The default Polymarket markets were changed to BTC-only:

```text
POLY_UPDOWN_MARKETS="btc-updown-5m,btc-updown-15m"
```

Relevant commit:

- `24f3185 Default Polymarket updown markets to BTC only`

## Subscription Growth Hotfix

Tokyo logs showed `updown_subscribed total_assets` increasing over time. The fix:

- tracks slug expiry
- unsubscribes expired assets
- prunes subscribed asset maps
- removes local order book state for expired assets
- ignores messages for no-longer-subscribed assets
- writes Polymarket raw only after relevance filtering

Relevant commit:

- `d13926c Prune expired UpDown subscriptions`

Expected healthy BTC-only behavior:

```text
updown_subscribed new_assets=6 total_assets=6
updown_pruned_expired assets=2 slugs=1 total_assets=4
```

## Live Prediction And Monitoring

An Ireland Lightsail live-shadow monitor was set up. It is not an execution engine.

It logs:

- `pipeline_starting`
- `signal_open`
- `maker_fill_observed`
- `signal_close`
- `stats`

The live pipeline was refactored to record cleaner details:

- decision scores
- fill probability
- predicted unwind profit
- expected profit
- entry price
- second-leg quote
- success/unwind outcome
- signed realized profit
- future same-leg bid/ask/mid
- rolling score distributions
- gate counts

## Profit Formula Fixes

The user caught that unwind was being treated as pure loss. It is now signed:

```text
unwind_profit = second_leg_size * future_same_leg_best_bid - first_leg_price
```

This can be positive when the first leg moves favorably even if the second leg never fills.

The current formula is documented in [Training Strategy](training_strategy.md).
