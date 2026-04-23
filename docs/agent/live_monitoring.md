# Live Monitoring

This document covers the Ireland live-shadow pipeline. It is for evaluation and monitoring only; it does not place real orders.

## Instance And Runtime

Tracked docs intentionally avoid private SSH material. Use:

```text
docs/private/ireland_lightsail.md
```

Last known facts:

- remote repo: `/home/ubuntu/poly_trade_pipeline`
- tmux session: `poly_live_predict`
- log file: `/home/ubuntu/poly_trade_pipeline/logs/live_predict.log`
- live script: `scripts/live_predict.py`
- market focus: BTC 5m
- Binance symbol: `btcusdt`

## Current Online Model

Latest live command (training_20260422 artifacts with EWMA + winsorize):

```text
python scripts/live_predict.py \
  --model-path artifacts/training_20260422/fill_models/random_forest_classifier.joblib \
  --unwind-model-path artifacts/training_20260422/unwind_models/random_forest_regressor.joblib \
  --threshold 0.005 \
  --min-p-fill 0.7 \
  --min-pred-unwind-profit 0.0 \
  --sample-interval 100 \
  --horizon 10 \
  --stats-interval 1000 \
  --min-entry-ask 0.05 \
  --max-entry-ask 0.95 \
  --min-time-to-expiry 20 \
  --max-spread 0.05 \
  --symbols btcusdt
```

Previous policy used `training_reprofit_20260420_21_5m` artifacts with `threshold=0.020, min_p_fill=0.85` but produced 0 signals because live p_fill max was ~0.74.

Best offline models (not yet deployed): XGBoost fill classifier (p>=0.7: 64.9% win, EV +0.0313) + ExtraTrees unwind regressor (rank_corr 0.201).

## Log Events

The live pipeline writes structured JSONL events.

`pipeline_starting`:

- model paths
- thresholds
- sample interval
- horizon
- entry filters

`signal_open`:

- market/asset/outcome
- entry ask/bid/mid/spread
- first-leg price
- second-leg quote
- success profit estimate
- `p_fill`
- `pred_unwind_profit`
- `pred_expected_profit`
- gates and thresholds

`maker_fill_observed`:

- observed future opposite trade evidence
- quote/fill price
- fill lag

`signal_close`:

- result: `success`, `unwind`, or `no_opposite`
- signed realized profit
- first-leg cost
- second-leg quote/cost
- future same-leg best bid/ask/mid
- fill lag when applicable

`stats`:

- prediction count
- open/closed signals
- rolling profit stats
- score distributions
- gate counts

## Live Profit Semantics

Success branch:

```text
profit = second_leg_size - (first_leg_price + second_leg_size * second_leg_quote)
```

Unwind branch:

```text
profit = second_leg_size * future_same_leg_best_bid - first_leg_price
```

Do not interpret every unwind as a loss. A directionally favorable first-leg move can make unwind positive.

## Recent Live Results

Single-model XGBoost and RandomForest both looked good offline but lost money live.

Observed live failure pattern:

- maker fill rate much lower than offline labels
- success branch around `+0.02`
- unwind losses often around `-0.03` to `-0.05`
- some extreme price entries had much worse tails

### Previous RF+RF (training_reprofit_20260420_21_5m)

```text
pred_expected_profit >= 0.020
p_fill >= 0.85
pred_unwind_profit >= 0
```

Result: zero signals after ~9,000 predictions. Live p_fill max ~0.74.

### XGBoost two-stage strict (older artifacts)

```text
pred_expected_profit >= 0.020814
p_fill >= 0.85
pred_unwind_profit >= 0
```

Result: almost no signals.

### XGBoost two-stage probe (older artifacts)

```text
pred_expected_profit >= 0.010
p_fill >= 0.65
pred_unwind_profit >= -0.02
```

Result: produced signals but negative; ~50 closed signals averaged about `-0.0259`.

### Single-model RF (training_20260422, threshold=0.67)

Result: 39 signals in 10 minutes, 7.7% accuracy (3/39 profitable), total profit -1.23.

### Two-stage RF+RF (training_20260422, threshold=0.005)

```text
pred_expected_profit >= 0.005
p_fill >= 0.7
pred_unwind_profit >= 0.0
```

Result: 0 signals after 2000 predictions. Live p_fill max 0.638 (below 0.7), pred_unwind_profit max -0.009 (below 0.0). Train/test distribution shift suspected.

## How To Check Status

On Ireland:

```bash
tmux capture-pane -t poly_live_predict -p -S -120 | tail -120
ps -eo pid,ppid,pcpu,pmem,etime,cmd --sort=-pcpu | grep -E "live_predict|python" | grep -v grep
tail -n 200 logs/live_predict.log
```

Useful local analysis:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path("logs/live_predict.log")
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
print(rows[-5:])
PY
```

## Main Risk

Offline/live calibration is currently poor. The next improvement should compare live `signal_open` rows against offline labels at the same timestamps and identify why live maker fills are missing.
