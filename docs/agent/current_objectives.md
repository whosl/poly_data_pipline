# Current Objectives

Updated: 2026-04-23

## Live Pipeline Status

Model: `training_eventdriven_20260423_5m` (XGBoost fill + ExtraTrees unwind)
Server: Ireland (108.132.27.76)
Status: **running well**

| Metric | Value |
|--------|-------|
| Predictions | 354,000 |
| Signals | 148 |
| Resolved | 148 |
| Accuracy | 90.5% |
| Total PnL | +2.4720 |
| Win rate | 93.3% (126W / 9L) |
| Uptime | 6h 43min |
| Prediction rate | 14.6/sec |
| CPU | 56-60% |
| Memory | 289MB |

## Gate Flow Analysis (latest 1000-sample batch)

```
samples: 1960
├── pre_model_skip: 960 (49%)
│   ├── blocked_entry_filter: 959
│   └── blocked_cooldown: 1
└── model inference: 1000 (51%)
    ├── blocked_threshold: 1000 (100%)
    ├── blocked_p_fill: 1000 (100%)
    ├── blocked_unwind: 986 (98.6%)
    └── pass_unwind: 14 (1.4%)
```

## Current Objectives

### 1. Monitor live calibration

Accumulate 500+ resolved signals, then run calibration analysis:

```bash
python scripts/analyze_live_calibration.py \
  --samples logs/live_candidate_samples.jsonl \
  --output-dir artifacts/live_calibration/<timestamp>
```

Compare live `p_fill`, `pred_unwind_profit`, `pred_expected_profit` distributions against offline validation.

### 2. Optimize CPU usage

Current 60% CPU is driven by model inference on samples that all get blocked by post-model gates. Options:

- **Increase `sample_interval`** from 100ms to 200-500ms (reduces prediction rate proportionally)
- Post-model gates (threshold, p_fill, unwind) cannot be moved to pre-model because they depend on model output

### 3. Collect more signal samples

Continue running to accumulate signal samples for offline analysis. Current `live_signal_samples.jsonl` has 25MB+ of data.

### 4. Validate unwind regressor accuracy

The unwind regressor is the critical filter. When `pred_unwind_profit > 0`, actual win rate is 76.5%. When negative, only 26.9%. Verify this calibration holds as more data accumulates.

### 5. Reconcile offline vs live labels

Use `live_candidate_samples.jsonl` and `live_signal_samples.jsonl` to compare:

- maker-fill false positives
- fill-lag mismatch
- score distribution drift
- unwind tail misses

## Reference

- Pipeline config: `--threshold 0.027 --min-p-fill 0.75 --min-pred-unwind-profit 0.0 --sample-interval 100`
- SSH: `ssh -4 -i ~/Downloads/EuKey.pem -o ConnectTimeout=15 -o ProxyCommand=none ubuntu@108.132.27.76`
- Logs: `/tmp/live_predict.log` (current), `~/poly_trade_pipeline/logs/live_predict.log` (historical)
- Signal samples: `~/poly_trade_pipeline/logs/live_signal_samples.jsonl`
