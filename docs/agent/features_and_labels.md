# Features And Labels

This document is the leakage map. Read it before changing feature generation or target construction.

## Data Sources

Training should use repository outputs:

- normalized parquet under `data/normalized/YYYYMMDD`
- research parquet under `data/research/YYYYMMDD`
- metadata parquet when available

Do not train directly from raw JSONL unless reconstructing missing normalized/research layers.

## Current Feature Groups

### Polymarket Top Of Book / Price

- `best_bid`
- `best_ask`
- `current_mid`
- `current_spread`
- `relative_spread`
- `current_microprice`

### Polymarket Imbalance / Depth Proxies

- `top1_imbalance`
- `total_bid_levels`
- `total_ask_levels`
- `top3_imbalance`
- `top5_imbalance`
- `top10_imbalance`
- `cum_bid_depth_topN_proxy`
- `cum_ask_depth_topN_proxy`
- `depth_level_imbalance_proxy`

Current limitation: true Polymarket top-N price/size levels are not preserved in normalized L2. Some top-N fields are therefore proxies or fallbacks, not true depth features.

### Polymarket Book Activity

- `book_update_count_100ms`
- `book_update_count_500ms`
- `book_update_count_1s`
- `spread_widen_count_recent`
- `spread_narrow_count_recent`
- `realized_vol_short`

### Polymarket EWMA Features

Added in training_20260422. 9 features derived from Exponential Weighted Moving Averages of 3 source columns, each producing fast (span=5), slow (span=20), and diff (fast-slow) variants:

- `realized_vol_short_ewma_fast`
- `realized_vol_short_ewma_slow`
- `realized_vol_short_ewma_diff`
- `depth_top10_imbalance_ewma_fast`
- `depth_top10_imbalance_ewma_slow`
- `depth_top10_imbalance_ewma_diff`
- `binance_return_1s_ewma_fast`
- `binance_return_1s_ewma_slow`
- `binance_return_1s_ewma_diff`

Implementation: `poly/training/features.py` `add_ewma_features()`. Live pipeline tracks state incrementally in `poly/predict/pipeline.py` via `AssetState` dataclass.

### Polymarket Short Return

- `poly_return_1s`

### Polymarket Trade Flow

The pipeline has plumbing for trade-flow features, but they may be dropped if null coverage is poor. The intended set is:

- recent trade count over 100ms / 500ms / 1s
- aggressive buy volume
- aggressive sell volume
- signed volume imbalance
- recent VWAP deviation from current mid
- sweep indicator
- consecutive buy run length
- consecutive sell run length

Confirm Polymarket trade `side` semantics before relying on signed flow.

### Binance Reference

Base Binance features:

- `binance_mid`
- `binance_spread`
- `binance_return_tick`
- `binance_return_100ms`
- `binance_return_500ms`
- `binance_return_1s`
- `binance_return_3s`
- `binance_recent_trade_imbalance`

Lead-lag:

- `lead_lag_binance_minus_poly_1s`
- `lead_lag_binance_minus_poly_500ms`

Compact Binance depth features when available:

- `binance_microprice`
- `binance_imbalance`
- `binance_depth_top1_imbalance`
- `binance_depth_top3_imbalance`
- `binance_depth_top5_imbalance`
- `binance_depth_top10_imbalance`
- `binance_depth_top20_imbalance`
- `binance_cum_bid_depth_top1`
- `binance_cum_bid_depth_top3`
- `binance_cum_bid_depth_top5`
- `binance_cum_bid_depth_top10`
- `binance_cum_bid_depth_top20`
- `binance_cum_ask_depth_top1`
- `binance_cum_ask_depth_top3`
- `binance_cum_ask_depth_top5`
- `binance_cum_ask_depth_top10`
- `binance_cum_ask_depth_top20`
- `binance_bid_depth_slope_top10`
- `binance_bid_depth_slope_top20`
- `binance_ask_depth_slope_top10`
- `binance_ask_depth_slope_top20`
- `binance_near_touch_bid_notional_5`
- `binance_near_touch_bid_notional_10`
- `binance_near_touch_bid_notional_20`
- `binance_near_touch_ask_notional_5`
- `binance_near_touch_ask_notional_10`
- `binance_near_touch_ask_notional_20`

### Regime / Metadata

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
- `symbol`
- `period`
- `outcome`

`volume_24h` and `liquidity` may be carried by metadata when available, but they are often null and may be dropped by feature filtering.

## Binance Join Logic

Current behavior:

- if normalized Binance data has one symbol, join globally by `recv_ns` with backward as-of
- if normalized Binance data has multiple symbols, join by `symbol` and `recv_ns`

This exists because some recovered Polymarket rows lacked reliable symbols while Binance had only BTCUSDT. For mixed BTC/ETH experiments, ensure symbol mapping is correct before trusting the result.

## Current Label Columns

Legacy alpha labels:

- `future_mid_1s`
- `future_mid_3s`
- `future_mid_5s`
- `future_mid_10s`
- `markout_1s`
- `markout_3s`
- `markout_5s`
- `markout_10s`
- `markout_1s_bps`
- `markout_3s_bps`
- `markout_5s_bps`
- `markout_10s_bps`
- `y_reg_10s`
- `y_cls_10s`
- `y_entry`
- `y_entry_after_cost`

Future book labels:

- `future_best_bid_1s`
- `future_best_bid_3s`
- `future_best_bid_5s`
- `future_best_bid_10s`
- `future_best_ask_1s`
- `future_best_ask_3s`
- `future_best_ask_5s`
- `future_best_ask_10s`

Two-leg labels:

- `opposite_asset_id`
- `future_opposite_maker_fill_price_10s`
- `future_opposite_maker_fill_recv_ns_10s`
- `two_leg_total_price_10s`
- `two_leg_edge_10s`
- `two_leg_realized_edge_10s`
- `y_two_leg_entry_10s`
- `y_two_leg_entry_binary_10s`
- `first_unwind_loss_proxy_10s`
- `first_unwind_profit_proxy_10s`
- `final_profit_10s`
- `y_final_profit_entry_10s`
- `y_final_profit_entry_binary_10s`
- `final_profit_weight_10s`

## Leakage Rules

Never use these as features:

- `future_*`
- `markout_*`
- `two_leg_*`
- `final_profit_*`
- `final_profit_weight_*`
- `first_unwind_loss_proxy_*`
- `first_unwind_profit_proxy_*`
- `y_*`
- `realized_edge_after_entry_cost_bps`
- any realized cost-adjusted edge derived from future data

Be careful with:

- bucket columns fitted on full-day data
- rolling features not grouped by market/asset
- forward as-of joins accidentally used for features
- labels crossing market boundaries

Feature inference is implemented in `poly/training/features.py`. Add tests around it before broadening feature rules.

## Missing Or Weak Features

Highest-priority feature work:

- true Polymarket top-N depth, not proxies
- best-bid/best-ask size changes over 100ms/500ms/1s
- quote pull/replenish/depletion proxies
- queue position estimates
- stronger trade-flow features with verified side semantics
- Binance exact-window trade count and signed volume
- Binance jump/volatility indicators
- symbol-aware BTC/ETH reference matching
- separate 5m/15m and Up/Down regime features

Do not spend time polishing hyperparameters until these data/label issues are stronger.
