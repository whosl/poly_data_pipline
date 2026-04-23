# Agent Manual

This is the top-level handoff for a fresh coding agent working on `poly_data_pipline` / `poly_trade_pipeline`.

The repo is not a finished trading engine. It is a data collection, normalization, label generation, feature engineering, baseline training, offline evaluation, and live-shadow monitoring system for Polymarket Up/Down short-horizon microstructure research.

Start here, then follow the subdocs:

- [Project History](agent/project_history.md): what the user asked for and what has already been done.
- [Training Strategy](agent/training_strategy.md): current objective, target strategy, profit formula, splits, and model families.
- [Features And Labels](agent/features_and_labels.md): current feature columns, label columns, leakage rules, and missing features.
- [Experiments](agent/experiments.md): key offline results, depth experiments, top10/top20 notes, and known caveats.
- [Collection And Ops](agent/collection_and_ops.md): collector behavior, default markets, Binance streams, S3/data notes, and Tokyo status.
- [Live Monitoring](agent/live_monitoring.md): Ireland live-shadow pipeline, logging semantics, current online model, and live failures.
- [Next Steps](agent/next_steps.md): the highest-value work from here.

Private instance connection notes are intentionally gitignored:

- `docs/private/tokyo_lightsail.md`
- `docs/private/ireland_lightsail.md`

Do not paste private key contents into tracked docs, chat, logs, or commits.

## Repository Facts

- GitHub: `git@github.com:whosl/poly_data_pipline.git`
- Local workspace used in this thread: `/Users/wenzhuolin/dev/poly/poly_trade_pipeline`
- Main branch: `master`
- Latest pushed commit at the time this manual was rewritten: `86d8195 Add two-stage live execution policy logging`
- Important current local state: there may be uncommitted live/shadow changes in `poly/predict/pipeline.py` and docs.

Always start with:

```bash
git status --short
```

Never reset or overwrite unknown changes.

## Current Goal In One Screen

The user is trying to prove whether Polymarket BTC/ETH Up/Down markets contain a stable, tradable 5s/10s microstructure edge.

Do not train a final market-resolution model. The model is not trying to predict who wins at expiry. The current target strategy is:

1. First leg: taker buy the selected outcome at or near current `best_ask`.
2. Second leg: maker buy the opposite outcome within the horizon.
3. Entry is interesting when `first_leg_ask + future_opposite_maker_fill_price <= 0.96`.

The current live/two-stage decision estimates:

```text
p_fill = classifier probability that the second-leg maker condition fills
pred_unwind_profit = regressor estimate for failure/unwind branch
success_profit = second_leg_size - (first_leg_price + second_leg_size * second_leg_quote)
pred_expected_profit = p_fill * success_profit + (1 - p_fill) * pred_unwind_profit
```

A two-stage live signal currently requires:

```text
pred_expected_profit >= threshold
p_fill >= min_p_fill
pred_unwind_profit >= min_pred_unwind_profit
entry filters pass
cooldown / max-entry gates pass
```

The most recent live policy on Ireland uses the `training_20260422` artifacts with EWMA features and winsorize preprocessing:

```text
RandomForest classifier + RandomForest unwind regressor
pred_expected_profit >= 0.005
p_fill >= 0.7
pred_unwind_profit >= 0.0
```

This produced zero signals because live `p_fill` max was ~0.638 (below 0.7) and `pred_unwind_profit` max was ~-0.009 (below 0.0). A prior single-model RF run at threshold=0.67 produced 39 signals in 10 minutes but only 7.7% accuracy.

Best offline models (training_20260422): XGBoost fill classifier (p>=0.7: 64.9% win, EV +0.0313) + ExtraTrees unwind regressor (rank_corr 0.201). Not yet deployed live.

## Non-Negotiable Rules

- Chronological splits only. Do not use random train/test splits.
- Choose cutoffs on validation, freeze them, then test once.
- Use 10s purge and 10s embargo for strict 10s-label experiments.
- No `future_*`, `markout_*`, `two_leg_*`, `final_profit_*`, `first_unwind_*`, `y_*`, or realized edge columns may enter features.
- Treat good offline results as provisional until live-shadow outcomes and label realism agree.
- Report trading usefulness metrics, not only generic ML metrics.
- For server operations, use the private ignored docs and avoid printing secrets.

## Current Bottom Line

Offline models can produce attractive validation/test slices, especially on BTC 5m data. Live-shadow runs have not yet confirmed the edge. The major gap is not model horsepower; it is label realism and offline/live alignment:

- maker fill labels are too optimistic
- queue position is not modeled
- live fill rate is much lower than offline labels imply
- unwind tail losses are underpredicted
- extreme price regimes are dangerous

The next agent should spend its energy on queue-aware second-leg labels, offline/live reconciliation, and validation-tested filters before tuning more models.
