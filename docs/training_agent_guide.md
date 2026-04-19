# Training Pipeline Agent Guide

This document is for coding agents taking over the training work in this repo. It explains the current objective, data flow, feature set, known limitations, and the next feature/modeling tasks.

## Current Objective

The immediate goal is to train and evaluate a short-horizon microstructure entry model for Polymarket BTC/ETH Up/Down markets.

Do not train a final market-resolution model yet. This repo is not trying to predict whether the market eventually resolves Up or Down. The current question is narrower and more tradable:

> Given the current Polymarket order book, recent flow, Binance reference move, and market regime, should we take the first leg now because the next 10 seconds are likely to give us a profitable second-leg maker opportunity?

The intended long-term strategy is two-leg:

1. First leg: taker buy the current asset at or near current `best_ask`.
2. Second leg: maker buy the opposite outcome at a favorable price within a short horizon.
3. Desired round trip: `first_leg_ask + future_opposite_maker_fill_price <= 0.96`.

The `0.96` threshold is a configurable simplification. It means the two bought outcomes leave about `0.04` gross edge against the `1.00` payoff identity, with room for fee, slippage, and safety margin. The current simplified finalProfit assumes:

- success profit: `+0.02`
- failure loss: `-max(first_leg_ask - future_best_bid_10s, 0)`

In other words, a failed second leg is unwound by selling the first leg at the future same-asset best bid proxy. The actual dataset column is `future_best_bid_10s`; it is same-leg because each row is already keyed by `market_id` and `asset_id`.

The current modeling layers are:

| Layer | Purpose | Target | Status |
|-------|---------|--------|--------|
| Layer 1 alpha | Predict short-horizon microstructure edge | `markout_10s_bps`, `y_cls_10s` | Implemented, use as sanity check |
| Layer 2 entry | Decide whether first-leg taker entry is worth doing | `y_final_profit_entry_10s` | Current main training target |
| Layer 3 exit/fill | Predict second-leg maker fill probability and exit quality | fill/time/quality labels | Partially derived, still needs conservative replay |

The current code does not implement a strategy engine. It only builds offline datasets, baseline models, and evaluation reports.

## Mental Model For New Agents

Keep these rules in mind before changing code:

- The model output should be interpreted as `p_enter`, not as final Up/Down resolution probability.
- A positive sample means "this first-leg taker entry led to a profitable two-leg opportunity within the horizon", not "the market resolved in this direction".
- Features must be observable at decision time. Labels can use future data; features cannot.
- Thresholds must be selected on validation, then frozen and applied to test. Do not select top-K or cutoffs by looking at test.
- Chronological splits are mandatory. For 10s labels, strict experiments should use 10s purge and 10s embargo.
- Treat perfect or near-perfect results as leakage until proven otherwise.

## Important Files

Training library:

- `poly/training/config.py`: dataclasses for dataset and training configs.
- `poly/training/io.py`: parquet discovery and safe loading.
- `poly/training/features.py`: sample construction, feature generation, label alignment, feature-column inference.
- `poly/training/labels.py`: alpha labels, entry labels, second-leg fill-label design note.
- `poly/training/splits.py`: chronological train/validation/test and walk-forward helper.
- `poly/training/models.py`: baseline model training.
- `poly/training/evaluation.py`: prediction metrics and trading-usefulness evaluation.
- `scripts/select_entry_cutoffs.py`: validation-selected entry cutoff evaluation.

Scripts:

- `scripts/build_features.py`
- `scripts/build_labels.py`
- `scripts/train_alpha_model.py`
- `scripts/evaluate_alpha_model.py`
- `scripts/select_entry_cutoffs.py`

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

Strict two-leg finalProfit classifier training:

```bash
python scripts/train_alpha_model.py \
  --dataset artifacts/training/alpha_dataset.parquet \
  --output-dir artifacts/training/strict_models \
  --target-reg final_profit_10s \
  --target-cls y_final_profit_entry_10s \
  --sample-weight-col final_profit_weight_10s \
  --split-purge-ms 10000 \
  --split-embargo-ms 10000
```

Evaluate:

```bash
python scripts/evaluate_alpha_model.py \
  --dataset artifacts/training/alpha_dataset.parquet \
  --model-dir artifacts/training/models \
  --output-dir artifacts/training/evaluation
```

Select `p_enter` cutoff on validation, then apply it to test:

```bash
python scripts/select_entry_cutoffs.py \
  --output-dir artifacts/training/cutoff_selection \
  --split-purge-ms 10000 \
  --split-embargo-ms 10000 \
  --target-profit final_profit_10s \
  --target-cls y_final_profit_entry_10s \
  --run btc_only artifacts/training/btc_only/alpha_dataset.parquet artifacts/training/btc_only/alpha_dataset.parquet artifacts/training/btc_only/strict_models
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
- Legacy alpha regression target: `y_reg_10s = markout_10s_bps`.
- Legacy alpha classification target: `y_cls_10s` in `down`, `neutral`, `up`.
- Default classification threshold is 5 bps.
- Entry label: `y_entry = 1` if `markout_10s_bps > entry_threshold_bps`.
- Default entry threshold is 8 bps.
- Current main target: `y_final_profit_entry_10s`.
- Current main profit metric: `final_profit_10s`.

Current two-leg label derivation:

1. For each sample, first leg price is current `best_ask`.
2. Find the opposite outcome asset for the same market.
3. Look forward within the configured horizon, currently 10s.
4. Derive `future_opposite_maker_fill_price_10s` from future opposite-side trade evidence.
5. Compute `two_leg_total_price_10s = best_ask + future_opposite_maker_fill_price_10s`.
6. Set `y_two_leg_entry_10s = enter` if `two_leg_total_price_10s <= two_leg_max_total_price`, default `0.96`.
7. Derive same-leg unwind loss from future book: `first_unwind_loss_proxy_10s = max(best_ask - future_best_bid_10s, 0)`.
8. Set `final_profit_10s = +0.02` on two-leg success, otherwise `-first_unwind_loss_proxy_10s`.
9. Set `y_final_profit_entry_10s = enter` when `final_profit_10s > 0`, else `skip`.

The current second-leg fill derivation is intentionally conservative-ish but not final. It uses observed future trade evidence on the opposite asset; it is not yet a full queue-position replay.

Splitting:

- Chronological only.
- Default split: earliest 70% train, next 15% validation, final 15% test.
- Strict experiments should pass `--split-purge-ms 10000 --split-embargo-ms 10000` for 10s labels.
- Do not use random splits for this data.

## Current Feature List

After fixing leakage, enabling single-symbol Binance timestamp joins, adding categorical regime buckets, and joining Polymarket metadata, the current verified strict feature set is 41 features on `20260417`.

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

### Existing Labels

The dataset currently carries both legacy alpha labels and strategy-entry labels.

Legacy alpha labels:

- `future_mid_1s`, `future_mid_3s`, `future_mid_5s`, `future_mid_10s`
- `markout_1s`, `markout_3s`, `markout_5s`, `markout_10s`
- `markout_1s_bps`, `markout_3s_bps`, `markout_5s_bps`, `markout_10s_bps`
- `y_reg_10s`
- `y_cls_10s`
- `y_entry`
- `y_entry_after_cost`

Book-derived future labels:

- `future_best_bid_1s`, `future_best_bid_3s`, `future_best_bid_5s`, `future_best_bid_10s`
- `future_best_ask_1s`, `future_best_ask_3s`, `future_best_ask_5s`, `future_best_ask_10s`

Two-leg strategy labels:

- `opposite_asset_id`
- `future_opposite_maker_fill_price_10s`
- `future_opposite_maker_fill_recv_ns_10s`
- `two_leg_total_price_10s`
- `two_leg_edge_10s`
- `two_leg_realized_edge_10s`
- `y_two_leg_entry_10s`
- `y_two_leg_entry_binary_10s`
- `first_unwind_loss_proxy_10s`
- `final_profit_10s`
- `y_final_profit_entry_10s`
- `y_final_profit_entry_binary_10s`
- `final_profit_weight_10s`

Use `y_final_profit_entry_10s` as the current primary classification target. Use `final_profit_10s` as the main trading usefulness metric.

### Entry Label Improvements

The current markout-only `y_entry` is simple:

- `y_entry = markout_10s_bps > entry_threshold_bps`

The current strategy entry label is closer to the desired strategy, but still simplified. Improve it to include:

- taker fee/cost
- half/full spread assumptions
- expected slippage
- minimum safety margin
- market-specific minimum tick/price constraints
- configurable success profit instead of fixed `+0.02`
- real unwind price if the first leg cannot exit at future best bid size

### Second-Leg Maker Fill Labels

Partially implemented as future opposite trade evidence. The next upgrade is a candidate-level label builder that simulates an opposite-side maker quote after first-leg taker entry.

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

There were earlier leakage bugs: `realized_edge_after_entry_cost_bps` and later future book/second-leg outcome columns were accidentally eligible as features. Any report that includes those fields is invalid.

Valid reports must have feature list excluding:

- `realized_edge_after_entry_cost_bps`
- any `future_mid_*`
- any `future_best_bid_*`
- any `future_best_ask_*`
- any `markout_*`
- any `y_*`
- any `future_opposite_maker_fill_*`
- any `two_leg_*`
- any `first_unwind_loss_proxy_*`
- any `final_profit_*`

Current strict no-leak finalProfit experiment:

- Target: `y_final_profit_entry_10s`, where success means `first_leg_ask + future_opposite_maker_fill_price_10s <= 0.96`.
- Profit label: `final_profit_10s = +0.02` on success, otherwise `-max(best_ask - future_best_bid_10s, 0)`.
- Split: chronological 70/15/15 with 10s purge and 10s embargo.
- Cutoff selection: choose `p_enter` threshold on validation only, then apply unchanged to test.
- Best current candidate on `20260417`: BTC-only RandomForest, validation-selected `p_enter >= 0.74`, test 60 entries, average `final_profit_10s` about `+0.01117`, success rate about `78.3%`.
- Mixed RandomForest is the main backup: validation-selected `p_enter >= 0.73`, test 82 entries, average `final_profit_10s` about `+0.00780`, success rate about `73.2%`.
- ETH-only remains weak/negative in the current sample.

How to read these numbers:

- `test_entries` is the number of test rows where model probability exceeded the validation-selected `p_enter` cutoff.
- `success_rate` is second-leg strategy success, not final market-resolution accuracy.
- `avg final_profit_10s` is in Polymarket price/probability units. `+0.01117` means about `+1.117 cents` per selected entry under the simplified payoff assumptions.
- `total final_profit_10s` is just selected rows times realized simplified profit. It is not position-size-adjusted PnL.
- A model with fewer entries and positive average can still be more relevant than a high-accuracy classifier, because the trading question is whether top-confidence entries have positive expected value.

Do not treat these results as production evidence. They are a pipeline sanity check only.

Why caution is required:

- Current dataset is one recovered date segment.
- It covers one date segment, not a broad market sample.
- Market metadata is incomplete.
- FinalProfit distribution is very regime-sensitive and entry counts are still small.
- Features still contain proxies where true depth is needed.
- Second-leg fill is currently derived from future opposite-side trade evidence, not a full conservative queue simulation.

## What A New Agent Should Do Next

The most useful continuation sequence is:

1. Add tests around `infer_feature_columns` so future/label columns cannot silently enter training.
2. Add tests around strict split purge/embargo behavior.
3. Re-run strict BTC-only/mixed training when more dates are available.
4. Add walk-forward cutoff selection across dates.
5. Upgrade second-leg fill labels from future trade evidence to conservative replay:
   - quote price chosen at candidate time
   - future opposite book/trade replay
   - queue depletion or trade-through evidence
   - time-to-fill and forced-exit labels
6. Add true top-N book depth to normalized parquet and replace top-N proxies.
7. Only after those are stable, begin wiring a replay/backtest strategy engine.

Do not spend time optimizing model hyperparameters before the fill labels and validation design are stronger. The current bottleneck is label realism and leakage-proof evaluation, not model complexity.

## Leakage Rules

Never include these columns as features:

- `future_mid_*`
- `future_best_bid_*`
- `future_best_ask_*`
- `markout_*`
- `y_*`
- `future_opposite_maker_fill_*`
- `two_leg_*`
- `first_unwind_loss_proxy_*`
- `final_profit_*`
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

5. Harden strict no-leak validation.
   - Unit test that feature columns exclude label-like names.
   - Unit test that future labels are generated with forward as-of and features with backward as-of.
   - Keep using validation-selected cutoffs; do not choose thresholds from test top-K.

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
