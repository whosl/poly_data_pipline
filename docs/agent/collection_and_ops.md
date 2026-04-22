# Collection And Ops

This document covers collection defaults, data layout, Tokyo behavior, and operational cautions.

## Default Local Collection

Default command:

```bash
source .venv/bin/activate
source "$HOME/.cargo/env"
python -m poly.main collect --symbols btcusdt
```

Default Polymarket markets:

```bash
POLY_UPDOWN_MARKETS="btc-updown-5m,btc-updown-15m"
```

This default was chosen because Polymarket L2 volume, not Binance compact depth, was the main CPU/IO load on Tokyo.

To re-enable ETH:

```bash
export POLY_UPDOWN_MARKETS="btc-updown-5m,btc-updown-15m,eth-updown-5m,eth-updown-15m"
```

## UpDown Subscription Logic

For each configured market family:

- compute current and next slugs
- discover token IDs via Gamma API
- subscribe to those asset IDs on Polymarket market websocket
- refresh every 10 seconds
- subscribe next markets before current expiry
- prune expired markets/assets

Expected healthy BTC-only logs:

```text
updown_ws_connecting markets=['btc-updown-5m', 'btc-updown-15m']
updown_subscribed new_assets=6 total_assets=6
updown_pruned_expired assets=2 slugs=1 total_assets=4
```

`total_assets` should stay bounded. If it grows without bound, check the expiry pruning path.

## Expiry And Noise

Markets are subscribed before they start and pruned after expiry. There can be a short overlap where old and new market assets are both known. The current training builder should rely on `market_id`, `asset_id`, expiry/time-to-expiry, and chronological labels to avoid mixing adjacent markets.

If investigating start-of-market noise:

- inspect raw `polymarket_market.jsonl.gz`
- group by slug/source if available
- verify messages after expiry are ignored or pruned
- verify normalized rows preserve enough market metadata

## Binance Streams

Current default for `btcusdt`:

- `btcusdt@bookTicker`
- `btcusdt@aggTrade`
- `btcusdt@depth20@100ms`

Depth behavior after compact-depth change:

- continue subscribing `depth20@100ms`
- compute compact depth features at collection time
- do not save raw `bids`/`asks` arrays for depth messages
- write compact raw envelopes
- write normalized `binance_l2_book.parquet`

This reduces CPU/disk relative to saving raw depth arrays, while keeping top-N depth features for training.

## Data Layout

```text
data/
  raw_feed/YYYYMMDD/
    polymarket_market.jsonl.gz
    polymarket_user.jsonl.gz
    binance_spot.jsonl.gz
  normalized/YYYYMMDD/
    poly_l2_book.parquet
    poly_trades.parquet
    poly_best_bid_ask.parquet
    poly_market_metadata.parquet
    binance_l2_book.parquet
    binance_trades.parquet
    binance_best_bid_ask.parquet
  research/YYYYMMDD/
    poly_markout_labels.parquet
    poly_enriched_book.parquet
```

## Normalize / Metadata / Labels

Typical rebuild:

```bash
python -m poly.main metadata 20260420
python -m poly.main normalize 20260420 --source polymarket
python -m poly.main normalize 20260420 --source binance
python -m poly.main labels 20260420
```

If raw gzip files are truncated, the recovery path scans gzip members and skips damaged members where possible.

## Tokyo Lightsail

Tracked docs intentionally avoid private SSH details. Use:

```text
docs/private/tokyo_lightsail.md
```

Last known Tokyo state:

- repo path: `/home/ubuntu/poly_data_pipeline/poly_data_pipline`
- tmux session: `poly`
- command: `python -m poly.main collect --symbols btcusdt`
- default markets: BTC 5m and BTC 15m
- remote had an unrelated local modification in `poly/metadata/polymarket.py`; inspect before overwriting

Tokyo S3 context from the user:

- region: `ap-northeast-1`
- bucket: `poly-trade-data-2026`
- IAM user: `s3user`
- credentials live on the server in `~/.aws/credentials`
- S3 auto-upload cron was disabled earlier
- S3 data files had previously been deleted

Do not place AWS credentials in tracked docs.

## Safe Operational Habits

- Use `kill -INT` for collectors so writers can flush.
- Avoid `kill -9` unless explicitly approved.
- Run `git status --short --branch` before pulling on servers.
- Use `git pull --ff-only` when deploying.
- Compile-check after pulling:

```bash
python -m compileall poly scripts
```

- Artifacts, data, virtualenvs, and `docs/private/` are ignored.
