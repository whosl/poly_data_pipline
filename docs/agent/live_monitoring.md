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

Latest verified live command on Ireland uses `training_eventdriven_20260423` artifacts:

```text
POLY_UPDOWN_MARKETS=btc-updown-5m \
python scripts/live_predict.py \
  --model-path artifacts/training_eventdriven_20260423/fill_models/xgboost_classifier.joblib \
  --unwind-model-path artifacts/training_eventdriven_20260423/unwind_models/extra_trees_regressor.joblib \
  --threshold 0.025 \
  --min-p-fill 0.5 \
  --min-pred-unwind-profit -0.05 \
  --sample-interval 100 \
  --min-entry-ask 0.10 \
  --max-entry-ask 0.90 \
  --signal-sample-path logs/live_signal_samples.jsonl \
  --candidate-sample-path logs/live_candidate_samples.jsonl \
  --candidate-sample-interval-ms 1000 \
  --maker-fill-latency-ms 250 \
  --maker-fill-trade-through-ticks 1.0
```

Current market scope is BTC 5m only. Recent verification showed `new_market` events only for `btc-updown-5m-*`, with no `btc-updown-15m` in `/tmp/live_predict.log`.

Historical policy notes:

- `training_reprofit_20260420_21_5m` with `threshold=0.020, min_p_fill=0.85` produced 0 signals because live p_fill max was ~0.74.
- `training_20260422` RF+RF with `threshold=0.005, min_p_fill=0.7, min_pred_unwind_profit=0.0` produced 0 signals after 2000 predictions.

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

### Event-Driven XGBoost+ExtraTrees (training_eventdriven_20260423)

Parsed from Ireland `/tmp/live_predict.log`:

| Window | Profitable | Success Path | Total Profit | Avg Profit |
| --- | ---: | ---: | ---: | ---: |
| first 23 resolved | 19/23 = 82.6% | 17/23 = 73.9% | `+0.1848` | `+0.0080` |
| first 30 resolved | 25/30 = 83.3% | 23/30 = 76.7% | `+0.2611` | `+0.0087` |
| first 33 resolved | 28/33 = 84.8% | 26/33 = 78.8% | `+0.3355` | `+0.0102` |

This is promising but still a small live-shadow sample.

### Earlier Live Runs

Single-model XGBoost and RandomForest looked good offline but lost money live.

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

The latest event-driven live-shadow run is positive so far, but calibration is not proven. Keep collecting resolved `live_candidate_samples.jsonl`, run `scripts/analyze_live_calibration.py`, and compare live outcomes against offline labels at the same market/asset/timestamp.
