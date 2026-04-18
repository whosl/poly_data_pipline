# Training Pipeline Agent Guide

This document is for coding agents taking over the training work in this repo. It explains the current objective, data flow, feature set, known limitations, and the next feature/modeling tasks.

## Current Objective

The immediate goal is to train and evaluate a short-horizon microstructure alpha model for Polymarket BTC/ETH Up/Down markets.

Do not train a final market-resolution model yet. The current target is:

- Primary model: predict 10 second markout / tradable short-horizon edge.
- Next layer: decide whether a first-leg taker entry is worth doing after cost.
- Later layer: estimate second-leg maker fill probability and exit quality.

The intended long-term strategy is two-leg:

1. First leg: taker entry.
2. Second leg: maker exit on the opposite side.

The current code does not implement a strategy engine. It only builds offline datasets, baseline models, and evaluation reports.

## Important Files

Training library:

- `poly/training/config.py`: dataclasses for dataset and training configs.
- `poly/training/io.py`: parquet discovery and safe loading.
- `poly/training/features.py`: sample construction, feature generation, label alignment, feature-column inference.
- `poly/training/labels.py`: alpha labels, entry labels, second-leg fill-label design note.
- `poly/training/splits.py`: chronological train/validation/test and walk-forward helper.
- `poly/training/models.py`: baseline model training.
- `poly/training/evaluation.py`: prediction metrics and trading-usefulness evaluation.

Scripts:

- `scripts/build_features.py`
- `scripts/build_labels.py`
- `scripts/train_alpha_model.py`
- `scripts/evaluate_alpha_model.py`

Recovery and normalization:

- `poly/storage/recover.py`: scans concatenated gzip JSONL files and skips corrupt gzip members.
- `poly/normalize/poly_norm.py`: normalizes Polymarket raw feed.
- `poly/normalize/binance_norm.py`: normalizes Binance raw feed.
- `poly/normalize/labels.py`: builds research markout labels and regime buckets.

## Data Flow

The intended pipeline is:

1. Raw collection writes gzip JSONL under `data/raw_feed/YYYYMMDD`.
2. Normalization writes parquet under `data/normalized/YYYYMMDD`.
3. Research labels write parquet under `data/research/YYYYMMDD`.
4. Training feature builder reads normalized/research parquet only.
5. Model training reads the engineered training parquet.
6. Evaluation reads the engineered training parquet plus saved model artifacts.

Avoid training directly from raw JSONL. Raw recovery is allowed only to reconstruct normalized/research layers.

## Raw Recovery Context

The local sample data had corrupt/truncated gzip members. The previous stdlib `gzip.open` path stopped at the first bad member and recovered only a few seconds of data.

`poly/storage/recover.py` now scans for gzip member headers and continues after corrupt members. On the recovered `20260417` sample, this recovered about 26 minutes of usable raw data:

- Polymarket raw JSONL: about 1.64M lines.
- Binance raw JSONL: about 447k lines.
- `poly_l2_book`: about 2.98M rows.
- `poly_trades`: about 31.9k rows.
- `binance_best_bid_ask`: about 396k rows.
- 10s engineered training rows: about 106k rows.

If normalized/research files look too small, rerun:

```bash
source .venv/bin/activate
python -m poly.main normalize 20260417
python -m poly.main labels 20260417
```

## Standard Commands

Build the default 10s alpha dataset:

```bash
source .venv/bin/activate
python scripts/build_features.py \
  --data-dir data \
  --dates 20260417 \
  --output-dir artifacts/training \
  --basename alpha_dataset
```

Train baselines:

```bash
python scripts/train_alpha_model.py \
  --dataset artifacts/training/alpha_dataset.parquet \
  --output-dir artifacts/training/models
```

Evaluate:

```bash
python scripts/evaluate_alpha_model.py \
  --dataset artifacts/training/alpha_dataset.parquet \
  --model-dir artifacts/training/models \
  --output-dir artifacts/training/evaluation
```

Export labels only:

```bash
python scripts/build_labels.py \
  --dataset artifacts/training/alpha_dataset.parquet \
  --output-dir artifacts/training/labels
```

## Dataset Construction

Current sampling:

- Event-level book data is downsampled to one sample per `sample_interval_ms`.
- Default sample interval is 100ms.
- Sampling happens per `market_id` and `asset_id`.
- If the upstream `market` field is missing in book data, `market_id` falls back to `asset:<asset_id>`.

Current label alignment:

- Future mids are generated from book snapshots using as-of joins.
- Horizons generated: 1s, 3s, 5s, 10s.
- Primary target: `y_reg_10s = markout_10s_bps`.
- Classification target: `y_cls_10s` in `down`, `neutral`, `up`.
- Default classification threshold is 5 bps.
- Entry label: `y_entry = 1` if `markout_10s_bps > entry_threshold_bps`.
- Default entry threshold is 8 bps.

Splitting:

- Chronological only.
- Default split: earliest 70% train, next 15% validation, final 15% test.
- Do not use random splits for this data.

## Current Feature List

After fixing leakage, enabling single-symbol Binance timestamp joins, adding categorical regime buckets, and joining Polymarket metadata, the current verified feature set is 42 features on `20260417`. Without metadata artifacts it falls back to the earlier 37-feature set.

Polymarket top-of-book and price:

- `best_bid`
- `best_ask`
- `current_mid`
- `current_spread`
- `relative_spread`
- `current_microprice`

Polymarket imbalance/depth proxies:

- `top1_imbalance`
- `total_bid_levels`
- `total_ask_levels`
- `top3_imbalance`
- `top5_imbalance`
- `top10_imbalance`
- `cum_bid_depth_topN_proxy`
- `cum_ask_depth_topN_proxy`
- `depth_level_imbalance_proxy`

Important limitation: `top3_imbalance`, `top5_imbalance`, and `top10_imbalance` are currently fallbacks to top1 imbalance because normalized L2 does not store per-level sizes. Likewise cumulative depth is a level-count proxy, not true size depth.

Polymarket book event activity:

- `book_update_count_100ms`
- `book_update_count_500ms`
- `book_update_count_1s`
- `spread_widen_count_recent`
- `spread_narrow_count_recent`
- `realized_vol_short`

Polymarket short return:

- `poly_return_1s`

Binance reference features:

- `binance_mid`
- `binance_spread`
- `binance_return_tick`
- `binance_return_100ms`
- `binance_return_500ms`
- `binance_return_1s`
- `binance_return_3s`
- `binance_recent_trade_imbalance`
- `lead_lag_binance_minus_poly_1s`
- `lead_lag_binance_minus_poly_500ms`

Regime/metadata/research buckets and volatility:

- `time_to_expiry_seconds`
- `tick_size`
- `min_order_size`
- `maker_base_fee`
- `taker_base_fee`
- `imbalance_bucket`
- `spread_bucket`
- `price_bucket`
- `vol_bucket`
- `vol_60s`

The four bucket features are categorical and are one-hot encoded by `poly/training/models.py`.

`volume_24h` and `liquidity` are carried through metadata when the source provides them, but they are not guaranteed to be training features. On the CLOB condition-id backfill path they are currently null and are dropped by the `max_null_fraction` filter.

Market metadata is fetched with:

```bash
python scripts/fetch_polymarket_metadata.py --data-dir data --dates 20260417
```

or through the CLI:

```bash
python -m poly.main metadata 20260417
```

This writes `data/normalized/YYYYMMDD/poly_market_metadata.parquet`. It is one row per `asset_id`, with `market_id`, `condition_id`, `slug`, `outcome`, `symbol`, `period`, `expiry_ns`, tick/min-size/fee fields, and liquidity/volume snapshots. Re-run `python -m poly.main normalize YYYYMMDD --source polymarket` after fetching metadata to enrich normalized book/trade/BBA parquet. The training builder also joins this metadata directly if the normalized book was produced before the metadata artifact existed.

## Binance Join Logic

Current logic:

- If normalized Binance data contains only one symbol, join Binance features globally by `recv_ns` using backward as-of join.
- If normalized Binance data contains multiple symbols, join by `symbol` and `recv_ns`.

This is intentional. The recovered sample currently only has `btcusdt`, and Polymarket book rows often lack a clean `symbol`, so strict symbol joins would drop most Binance features.

Future improvement:

- Preserve market slug/symbol in Polymarket normalized book rows so BTC and ETH markets can be matched explicitly to `btcusdt` and `ethusdt`.

## Features That Still Need To Be Added

### True Polymarket Depth Features

The normalized L2 table currently stores only:

- best bid/ask
- midpoint
- microprice
- imbalance
- total bid/ask level counts

It does not store actual top-N price/size levels. Because of this, many intended LOB features are proxies.

Add normalized columns or a separate parquet table for top-N depth:

- bid price/size levels 1..10
- ask price/size levels 1..10
- top3/top5/top10 true imbalance
- cumulative bid depth top N
- cumulative ask depth top N
- bid depth slope
- ask depth slope
- notional depth near touch
- distance-weighted depth

### Better Queue/Depletion Proxies

Current queue feature is not meaningful enough. Add:

- best-bid size change over 100ms/500ms/1s
- best-ask size change over 100ms/500ms/1s
- quote pull indicator
- replenish indicator
- top-of-book depletion by side
- number of consecutive updates at same best price

These require actual top-level size in normalized data.

### Polymarket Trade-Flow Features

The code has initial trade-flow feature plumbing, but these features may be dropped if null fraction is too high or if market/asset alignment is poor.

Still needed:

- recent trade count over 100ms / 500ms / 1s
- aggressive buy volume
- aggressive sell volume
- signed volume imbalance
- recent VWAP deviation from current mid
- sweep indicator based on multi-level trade-through
- consecutive buy run length
- consecutive sell run length

Important: confirm Polymarket trade `side` semantics. If side means maker side rather than aggressor side, signed volume must be inverted or re-derived.

### Better Binance Reference Features

Current Binance features are mostly mid returns and trade imbalance. Add:

- Binance book imbalance from `bookTicker` size fields if collected.
- Binance depth imbalance from depth20.
- Binance realized volatility over 1s/3s/10s.
- Binance trade count and signed volume over exact time windows.
- Binance jump indicators.
- BTC/ETH symbol-aware matching once Polymarket symbol is reliable.
- lead/lag residual: Polymarket move expected from Binance beta minus actual Polymarket move.

### Regime / State Features

Currently `time_to_expiry_seconds`, `expiry_bucket`, and `market_phase` are often `unknown` because normalized book rows do not preserve market slug or expiry.

Fix upstream first:

- Preserve Polymarket market slug in `poly_l2_book`.
- Preserve period: 5m vs 15m.
- Preserve underlying: BTC vs ETH.
- Preserve expiry timestamp.
- Map asset_id to outcome side if possible.

Then add:

- `time_to_expiry_seconds`
- expiry bucket: `>180s`, `60-180s`, `30-60s`, `10-30s`, `<10s`
- market phase: early/mid/tail
- period: 5m vs 15m
- underlying: BTC vs ETH
- outcome side: up/down token if known
- price regime: near 0/1, near 0.5, etc.

## Labels Still Needed

### Entry Label

The current entry label is simple:

- `y_entry = markout_10s_bps > entry_threshold_bps`

Improve it to include real costs:

- taker fee/cost
- half/full spread assumptions
- expected slippage
- minimum safety margin
- market-specific minimum tick/price constraints

### Second-Leg Maker Fill Labels

Not implemented yet. Add a candidate-level label builder that simulates an opposite-side maker quote after first-leg taker entry.

Generate:

- `fill_1s`
- `fill_3s`
- `fill_5s`
- `fill_10s`
- `time_to_fill_ms`
- `realized_second_leg_price`
- `best_possible_exit_price_within_horizon`
- `forced_exit_price_if_not_filled`

These must be derived from future book/trade evolution, not guessed from current state.

Recommended conservative fill logic:

1. At candidate time, place maker quote on the opposite side at configured price.
2. Replay future book/trade events chronologically.
3. Mark fill only if trade-through or conservative queue depletion evidence reaches the quote.
4. If not filled by horizon, use forced exit price from future best bid/ask.

## Current Evaluation Interpretation

There was an earlier leakage bug: `realized_edge_after_entry_cost_bps` was accidentally included as a feature. Any report that includes that feature is invalid.

Valid reports must have feature list excluding:

- `realized_edge_after_entry_cost_bps`
- any `future_mid_*`
- any `markout_*`
- any `y_*`

Recent no-leak results showed weak but nonzero signal:

- Linear/ridge rank correlation around 0.22 on the recovered single-date test split before timestamp-joined Binance features.
- With timestamp-joined Binance features, linear/ridge positive predictions selected fewer entries but improved average realized markout in this one recovered sample.
- LightGBM did not clearly improve in this small one-date sample.

Do not treat these results as production evidence. They are a pipeline sanity check only.

Why caution is required:

- Current dataset is one recovered date segment.
- It covers only about 26 minutes.
- Market metadata is incomplete.
- Markout distribution is very fat-tailed.
- Features still contain proxies where true depth is needed.
- There is no second-leg fill simulation yet.

## Leakage Rules

Never include these columns as features:

- `future_mid_*`
- `markout_*`
- `y_*`
- `realized_edge_after_entry_cost_bps`
- any label or cost-adjusted realized edge derived from future data

Be careful with:

- bucket columns generated using full-day quantiles. These are currently included as categorical features for the baseline, but they can leak distributional information across splits. Prefer train-fit bucket boundaries for strict experiments.
- rolling features that are not grouped by asset/market.
- Binance joins that use future timestamps. Use backward as-of joins only for features.
- future-mid labels that cross market boundaries. Always group by `market_id` and `asset_id`.

## Immediate Next Tasks

1. Preserve Polymarket market metadata in normalized book rows.
   - market slug
   - underlying BTC/ETH
   - period 5m/15m
   - expiry timestamp
   - outcome token side

2. Store true top-N book depth in normalized parquet.
   - Add top 10 levels or nested/list columns.
   - Rebuild true top3/top5/top10 imbalance.

3. Make trade-flow features survive into the final dataset.
   - Confirm side semantics.
   - Align trades by asset and market.
   - Use time-based rolling windows.

4. Improve Binance features.
   - Add depth imbalance and exact time-window trade imbalance.
   - Add symbol-aware matching once Polymarket symbol is fixed.

5. Add strict no-leak validation.
   - Unit test that feature columns exclude label-like names.
   - Unit test that future labels are generated with forward as-of and features with backward as-of.

6. Add walk-forward reports.
   - Multiple windows.
   - Per-date metrics.
   - Per-market lifecycle metrics.

7. Implement second-leg maker fill label builder.
   - Keep it separate from the alpha model.
   - Use future book/trade replay, not heuristics alone.

## Suggested Agent Workflow

When taking a new task:

1. Run `git status --short`.
2. Check `artifacts/training*/alpha_dataset_metadata.json` only as local diagnostics; artifacts are gitignored.
3. Read `poly/training/features.py` before changing feature logic.
4. If changing normalized schemas, update both collector-time writers and raw normalizers.
5. Rebuild normalized/research after schema changes.
6. Rebuild features and inspect metadata feature list.
7. Train and evaluate.
8. Report both ML metrics and trading-usefulness metrics.
9. Treat any perfect score as suspicious until leakage is disproven.
