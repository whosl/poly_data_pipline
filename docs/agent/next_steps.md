# Next Steps

This is the recommended priority order for a new agent.

## 1. Reconcile Offline Labels With Live Outcomes

Compare live `signal_open` rows to what the offline label builder would assign at the same market/asset/timestamp.

Focus on:

- maker-fill false positives
- fill-lag mismatch
- whether future opposite trade evidence really proves our maker quote would fill
- unwind tail misses
- score distribution drift between offline validation and live

This is the highest-value task.

## 2. Build Queue-Aware Second-Leg Fill Labels

Current fill labels are too optimistic. Upgrade them to a conservative replay:

1. At candidate time, place an opposite-side maker quote at the intended price.
2. Track book state and displayed size ahead of the quote.
3. Replay future trades/book updates chronologically.
4. Mark fill only on conservative trade-through or queue depletion evidence.
5. Emit:
   - `fill_1s`
   - `fill_3s`
   - `fill_5s`
   - `fill_10s`
   - `time_to_fill_ms`
   - `realized_second_leg_price`
   - `best_possible_exit_price_within_horizon`
   - `forced_exit_price_if_not_filled`

Do not guess these from current state only.

## 3. Add Validation-Tested Live Filters

Candidate filters:

- entry price band, e.g. `0.20 <= entry_ask <= 0.80`
- stricter max spread
- one entry per market/outcome instead of three
- separate thresholds for Up and Down outcomes
- separate thresholds for early/mid/tail market phase
- stronger minimum predicted unwind profit

Each filter must be selected on validation and then frozen for test. Do not tune by live or test anecdotes alone.

## 4. Store True Polymarket Top-N Depth

Normalized Polymarket L2 currently lacks full top-N price/size levels. Add either columns or a nested/list table for:

- bid price/size levels 1..10
- ask price/size levels 1..10
- true top3/top5/top10 imbalance
- cumulative bid/ask depth
- slope
- near-touch notional
- queue/depletion proxies

This should replace current proxy depth features.

## 5. Harden Leakage Tests

Add tests that fail if feature inference includes:

- `future_*`
- `markout_*`
- `two_leg_*`
- `final_profit_*`
- `first_unwind_*`
- `y_*`
- realized edge fields

Also test purge/embargo boundaries.

## 6. Collect More Fresh Data

Let Tokyo collect longer BTC-only compact-depth data.

Then rebuild:

```bash
python -m poly.main metadata YYYYMMDD
python -m poly.main normalize YYYYMMDD --source polymarket
python -m poly.main normalize YYYYMMDD --source binance
python -m poly.main labels YYYYMMDD
```

Rerun strict experiments across full days and walk-forward windows.

## 7. Compare Depth Policies Again

Only after enough fresh data:

- no Binance depth
- top10 compact depth
- top20 compact depth
- depth10@100ms collection
- depth20@100ms collection

Do not delete depth collection solely because one small recovered slice was negative.

## 8. Improve Feature Calibration

Track and compare offline vs live distributions:

- `p_fill`
- `pred_unwind_profit`
- `pred_expected_profit`
- entry price
- spread
- time to expiry
- outcome side
- market phase

If distributions drift, add calibration or reject out-of-distribution live samples.

## 9. Only Then Tune Models

Model complexity is not the current bottleneck. Once labels and validation are stronger, revisit:

- probability calibration
- class/sample weighting
- monotonic constraints where sensible
- per-regime models
- hyperparameter tuning
