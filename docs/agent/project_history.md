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

## EWMA Features

Added 9 EWMA (Exponential Weighted Moving Average) features to capture trend/momentum signals:

| Source column | Fast (span=5) | Slow (span=20) | Diff (fast-slow) |
| --- | --- | --- | --- |
| `realized_vol_short` | `realized_vol_short_ewma_fast` | `realized_vol_short_ewma_slow` | `realized_vol_short_ewma_diff` |
| `depth_top10_imbalance` | `depth_top10_imbalance_ewma_fast` | `depth_top10_imbalance_ewma_slow` | `depth_top10_imbalance_ewma_diff` |
| `binance_return_1s` | `binance_return_1s_ewma_fast` | `binance_return_1s_ewma_slow` | `binance_return_1s_ewma_diff` |

Implementation: `poly/training/features.py` `add_ewma_features()` with `_EWMA_SOURCE_COLUMNS` dict. Called in `build_date_dataset()` after `add_binance_features()`.

Live pipeline tracks EWMA state incrementally via `AssetState` dataclass in `poly/predict/pipeline.py`, using `_ewma_update()` helper. Avoids lookback bias by computing EWMA only from current + past values.

## Winsorize Preprocessing

Added winsorization (quantile clipping) for features: clips values to [q0.005, q0.995] bounds fitted on training data. Prevents extreme outliers from distorting tree splits.

Implemented in `poly/training/features.py`. Quantile bounds stored alongside model artifacts for live inference.

## Data Filtering: Time-Bucket Downsampling

Original event-driven downsampling with tiered magnitude filtering was too slow on 123M+ row datasets (>1.5 hours without completing). Switched to time-bucket mode:

- 250ms buckets, keep last row per bucket per (market, outcome)
- Compression: 123M → 1.39M rows (1.1% ratio)
- Runs in seconds

Event-driven mode is still defined in `scripts/build_sampled_book.py` with tiered filtering (Tier1: best_bid/ask/total_levels always kept; Tier2: depth imbalance/cum_depth filtered by magnitude threshold) but is impractical for full-day datasets.

Relevant command:

```bash
python scripts/build_sampled_book.py --mode time-bucket --sample-interval-ms 250 --overwrite --dates 20260420 20260421
```

## WebSocket Ping Fix

Ireland had 442 WS errors in 14.5 hours (connections lasting ~10s each). Tokyo had 0 errors over the same period.

Root cause investigation:
1. First attempt: enabled default library ping (removed `ping_interval=None`). Result: 68s connections. Server ignores WS ping frames, library times out.
2. Second attempt: `ping_interval=None` + manual text "PING" heartbeat every 10s. Result: ~10s connections. Server rejects text "PING", closes connection.
3. Final fix: `ping_interval=None` + no heartbeat. Result: 2-3 minute connections.

The Polymarket server neither responds to WebSocket protocol-level ping frames nor accepts text "PING" messages. Tokyo works better because it subscribes to hundreds of markets ensuring constant message flow, preventing idle timeouts.

Affected files:
- `poly/collector/updown_ws.py`
- `poly/collector/market_ws.py`
- `poly/collector/user_ws.py`
- `scripts/live_predict.py`

Relevant commits: `756511d`, `f7a0b4f`

## New Training Run: training_20260422

New dataset with EWMA features and winsorize preprocessing:
- `artifacts/training_20260422/alpha_dataset.parquet`: 1,340,767 rows, 181 columns (5m: 674,889, 15m: 665,878)
- `artifacts/training_20260422/alpha_dataset_5m.parquet`: 674,889 rows, 118 feature columns, 222MB
- Feature count: 118 (up from ~109 in training_reprofit_20260420_21_5m)
- Split: chronological 70/15/15, purge=0, embargo=0

Fill classifiers: RF, ExtraTrees, LightGBM, XGBoost
Unwind regressors: RF, ExtraTrees, LightGBM, XGBoost

Best offline results:
- Fill: XGBoost (p>=0.7: 1540 entries, 64.9% win, EV +0.0313)
- Unwind: ExtraTrees (rank_corr 0.201, MAE 0.058)

## Live Two-Stage Model Switch (Ireland)

Deployed RF classifier + RF regressor from training_20260422:

```bash
python scripts/live_predict.py \
  --model-path artifacts/training_20260422/fill_models/random_forest_classifier.joblib \
  --unwind-model-path artifacts/training_20260422/unwind_models/random_forest_regressor.joblib \
  --threshold 0.005 \
  --min-p-fill 0.7 \
  --min-pred-unwind-profit 0.0
```

Result after 2000 predictions: 0 signals. Live p_fill max 0.638 (below 0.7), pred_unwind_profit max -0.009 (below 0.0). Train/test distribution shift suspected.

Prior single-model RF run (threshold=0.67, same artifacts) produced 39 signals in 10 minutes but only 7.7% accuracy (3/39 profitable, total profit -1.23).

## Event-Driven Downsampling Optimization

Replaced time-bucket (250ms) downsampling with optimized event-driven mode using vectorized Polars window functions. Key insight: instead of iterating rows, use `shift()` + `ne_missing()` to detect state changes in a single vectorized pass.

Default event columns:
- Tier 1, any change triggers: `best_bid`, `best_ask`, `total_bid_levels`, `total_ask_levels`
- Tier 2, change triggers only above magnitude threshold: `depth_top1_imbalance`, `depth_top3_imbalance`, `depth_top5_imbalance`, `depth_top10_imbalance`, `depth_top20_imbalance`, `cum_bid_depth_top10`, `cum_ask_depth_top10`, `cum_bid_depth_top20`, `cum_ask_depth_top20`

Compression results on 20260420+20260421 (123.1M raw rows: 64.2M on 20260420, 58.9M on 20260421):
- Pure event-driven: 82.3% retention (too much, depth features change constantly)
- With 1ms debounce: 37.8%
- With 5ms debounce: 10.7%
- With 10ms debounce: 4.8%

Final approach: event-driven with default settings (`min_gap_ms=0`, no debounce; tier2 still uses default magnitude thresholds of `0.001` for imbalance and `10.0` for cumulative depth). Generated 14.5M rows total: 7,234,818 rows for 20260420 and 7,245,926 rows for 20260421, about 10.4x more data than time-bucket.

Command:
```bash
python scripts/build_sampled_book.py --mode event-driven --overwrite --dates 20260420 20260421
```

## New Training Run: training_eventdriven_20260423

Trained on event-driven sampled book data:
- `artifacts/training_eventdriven_20260423/alpha_dataset.parquet`: 3,091,181 rows × 181 columns
- 5m subset: 1,630,456 rows
- 15m subset: 1,460,725 rows
- Feature columns: 118
- Dates: 20260420 + 20260421
- Split: chronological 70/15/15

Fill classifiers trained: RF, ExtraTrees, LightGBM, XGBoost
Unwind regressors trained: RF, ExtraTrees, LightGBM, XGBoost

Best models selected:
- Fill: **XGBoost classifier** (AUC=0.7573, p>=0.5: 88.2% win rate, sharpe +11.05)
- Unwind: **ExtraTrees regressor** (R2=0.0478, rank_corr=0.231)

Optimal thresholds from execution policy analysis:
- `threshold=0.025`, `min_p_fill=0.5`, `min_pred_unwind_profit=-0.05`

## Event-Driven Model Deployment to Ireland

Deployed XGBoost fill classifier + ExtraTrees unwind regressor from training_eventdriven_20260423 to Ireland via SCP:

```bash
# Fill model
scp -4 -i /Users/wenzhuolin/Downloads/EuKey.pem \
  artifacts/training_eventdriven_20260423/fill_models/xgboost_classifier.joblib \
  artifacts/training_eventdriven_20260423/fill_models/training_metadata.json \
  ubuntu@108.132.27.76:~/poly_trade_pipeline/artifacts/training_eventdriven_20260423/fill_models/

# Unwind model
scp -4 -i /Users/wenzhuolin/Downloads/EuKey.pem \
  artifacts/training_eventdriven_20260423/unwind_models/extra_trees_regressor.joblib \
  artifacts/training_eventdriven_20260423/unwind_models/training_metadata.json \
  ubuntu@108.132.27.76:~/poly_trade_pipeline/artifacts/training_eventdriven_20260423/unwind_models/
```

sklearn version mismatch warning (local 1.8.0 vs Ireland 1.7.2) — non-critical, models load and run correctly.

## 5m-Only Market Subscription

Model was trained exclusively on 5m market data. Changed Ireland pipeline to subscribe only to 5m markets via environment variable:

```bash
POLY_UPDOWN_MARKETS=btc-updown-5m
```

This limits live Polymarket subscription to the active BTC 5m Up/Down market. Recent logs after deployment showed about 17-19 model predictions/sec.

Pipeline launch command on Ireland:
```bash
cd ~/poly_trade_pipeline && POLY_UPDOWN_MARKETS=btc-updown-5m nohup .venv/bin/python scripts/live_predict.py \
  --model-path artifacts/training_eventdriven_20260423/fill_models/xgboost_classifier.joblib \
  --unwind-model-path artifacts/training_eventdriven_20260423/unwind_models/extra_trees_regressor.joblib \
  --threshold 0.025 --min-p-fill 0.5 --min-pred-unwind-profit -0.05 \
  --sample-interval 100 --min-entry-ask 0.10 --max-entry-ask 0.90 \
  --signal-sample-path logs/live_signal_samples.jsonl \
  --candidate-sample-path logs/live_candidate_samples.jsonl \
  --candidate-sample-interval-ms 1000 \
  --maker-fill-latency-ms 250 --maker-fill-trade-through-ticks 1.0 \
  > /tmp/live_predict.log 2>&1 &
```

## Live Results: Event-Driven Model on Ireland (2026-04-23)

First 23 signals after deployment (5m markets only):

| Metric | Value |
|---|---|
| Total signals | 23 |
| Profitable win rate | 82.6% (19/23) |
| Total profit | +0.1848 |
| Avg profit | +0.0080 |
| Success path rate | 73.9% (17/23) |
| Unwind path rate | 26.1% (6/23) |

Notable: early signals (S000001-008) had 2 large unwind losses totaling -0.19, but subsequent signals (S000009-023) showed consistent profitability with total +0.271 across 15 signals.

Direction accuracy: 88.9% (8/9 verifiable from early signals). The model's directional prediction is strong; main risk is second-leg fill failure on unwind path.

Later verification from the same `/tmp/live_predict.log` showed 33 resolved signals:
- Profitable: 28/33 = 84.8%
- Success path: 26/33 = 78.8%
- Total profit: +0.3355
- Avg profit: +0.0102

Treat this as a small live-shadow sample, not proof of stable edge.

## SSH Connectivity: IPv4 Direct Access

Ireland and Tokyo Lightsail instances are accessible via IPv4 when Clash TUN mode is disabled on the local Mac.

**Critical**: Clash TUN mode (utun4 interface) intercepts all traffic including IPv4. Must be disabled for direct SSH.

Ireland IPv4:
```bash
ssh -4 -i /Users/wenzhuolin/Downloads/EuKey.pem -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@108.132.27.76
```

Tokyo IPv4:
```bash
ssh -4 -i /Users/wenzhuolin/Downloads/LightsailDefaultKey-ap-northeast-1.pem -o ConnectTimeout=10 -o StrictHostKeyChecking=no ubuntu@16.171.27.253
```

See `docs/private/ireland_lightsail.md` and `docs/private/tokyo_lightsail.md` for full connection details.

## RF/ET Training on Windows (2026-04-27)

User migrated to Windows PC (i5-13600KF, 32GB RAM) to train RF/ET models that previously OOM'd on Mac (16GB).

### Phase A: Partitioned Dataset Build

- Script: `scripts/train_rf_et_win.py` — memory-efficient per-market pipeline
- Used pyarrow predicate pushdown to load one market at a time from `poly_sampled_book` (28M rows/date)
- Proper canonicalization: `canonicalize_binance_book()` for Binance features
- Result: 80,326,087 rows across 5 dates, 105 features
- Output: `artifacts/training_rf_et_win/alpha_dataset_parts/date=YYYYMMDD/market_id=xxx.parquet`

### Phase B: Balanced Sampling + Training

Multiple memory issues resolved:
1. Polars concat of 11M+ balanced rows → segfault. Fix: convert to numpy immediately per file
2. Accumulated numpy arrays across 182 markets → OOM. Fix: periodic consolidation every 25 files
3. Cross-date accumulation → OOM during 3rd date. Fix: early-stop when rows exceed 2× max, incremental cross-date concat

Final config: 2M total train rows (666K/date × 3 dates), balanced sampling (all positives + equal negatives).

Training results (100 estimators, max_depth=12):
| Model | Time |
|---|---|
| rf_classifier | 1.6 min |
| et_classifier | 1.0 min |
| rf_regressor | 16.8 min |
| et_regressor | 12.1 min |

### XGBoost + LightGBM Training

Same 2M balanced dataset used to train 4 additional models:
- lightgbm_classifier (11s), xgboost_classifier (11s)
- lightgbm_regressor (8s), xgboost_regressor (11s)

### Combined 8-Model Policy Evaluation

Script: `scripts/combined_policy_eval.py` — trains LGB/XGB, loads RF/ET, batched predictions, full policy grid.

16 combos evaluated. Best test-result combos (entries > 100):
- **rf_clf × rf_reg**: test avgP=0.0353, n=1,006 (best)
- xgb_clf × rf_reg: test avgP=0.0271, n=2,112
- lgb_clf × lgb_reg: test avgP=0.0184, n=2,393
- lgb_clf × rf_reg: test avgP=0.0157, n=1,371

Key insight: Val overfitting severe — combos with highest val avgP (0.32, 0.20) produce 0 test entries.

### Signal Server Deployment (2026-04-27)

8 models pushed to Ireland (`artifacts/training_combined_20260427/`).
Signal server restarted with rf_clf × rf_reg:
- Initial config: threshold=0.05, min_p_fill=0.8 → 0 signals (RF max p_fill=0.78)
- Adjusted: threshold=0.02, min_p_fill=0.65 → signals producing
- Log: `~/poly_trade_pipeline/logs/signal_server_rf.log`
