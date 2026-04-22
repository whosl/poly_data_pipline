# Polymarket 低延迟量化数据采集系统

面向 Polymarket BTC/ETH Up/Down 市场的低延迟数据采集与标准化管线，用于短周期微观结构研究。

## 功能

- **Polymarket Up/Down 市场采集**：默认自动发现、订阅、轮换 BTC updown-5m/15m 市场，可用 `POLY_UPDOWN_MARKETS` 打开 ETH
- Polymarket 实时 L2 订单簿 + 成交采集（Market WebSocket）
- Polymarket 自有订单/成交回报（User WebSocket，需认证）
- Binance BTC/ETH 参考行情（bookTicker / aggTrade / depth20）
- 纳秒级本地时间戳（Rust 扩展）
- 三层存储：raw JSONL → 标准化 Parquet → 研究 labels Parquet
- 标准化管线 + 1s/3s/5s/10s markout 标签生成
- 数据回放（按原速或倍速）

## 架构

```
Python (asyncio)  ←→  Rust (PyO3)
  I/O 和编排            订单簿引擎 / 纳秒时间戳 / Decimal 精度计算
```

## 快速开始

### 前置条件

- Python >= 3.10
- Rust (通过 rustup 安装)

### 安装

```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 编译 Rust 扩展
maturin develop --release

# 验证
python -c "import poly_core; print(poly_core.now_ns())"
```

### 配置

通过环境变量配置，也可写入 `.env` 文件：

```bash
# 可选（user channel 需要认证，不设置则跳过）
export POLY_API_KEY=""
export POLY_API_SECRET=""
export POLY_API_PASSPHRASE=""

# 可选
export POLY_DATA_DIR="./data"                    # 数据存储目录
export POLY_BINANCE_SYMBOLS="btcusdt"            # Binance 交易对
export POLY_UPDOWN_MARKETS="btc-updown-5m,btc-updown-15m"  # Polymarket Up/Down 市场；默认 BTC 5m/15m
export POLY_WATCHDOG_TIMEOUT="120"               # 静默冻结检测超时（秒）
export POLY_RAW_FLUSH_INTERVAL="5"               # 原始数据 flush 间隔（秒）
```

## 使用

### 采集数据

```bash
# 启动所有采集器（Polymarket Up/Down + Binance）
python -m poly.main collect --symbols btcusdt

# Ctrl+C 优雅停止，自动 flush 所有数据
```

采集器启动后：
1. 自动计算 `POLY_UPDOWN_MARKETS` 配置的当前和下一期市场 slug（默认 `btc-updown-5m` / `btc-updown-15m`，如 `btc-updown-5m-1776409800`）
2. 通过 Gamma API 查询市场 token IDs
3. 连接 Polymarket market WebSocket，订阅订单簿和成交
4. 每 10 秒检查是否有新市场需要订阅
5. 到期市场自动轮换，提前 60-120 秒订阅下一期
6. 同时连接 Binance WebSocket，订阅 BTC 行情
7. 原始消息实时写入 gzip JSONL，标准化数据写入 Parquet

#### 采集的市场

默认只采 BTC 5m/15m，以降低 Polymarket L2 book 的 CPU 和落盘压力。需要恢复 ETH 时可设置：

```bash
export POLY_UPDOWN_MARKETS="btc-updown-5m,btc-updown-15m,eth-updown-5m,eth-updown-15m"
```

| 市场类型 | 周期 | 提前订阅 |
|---------|------|---------|
| btc-updown-5m | 5 分钟 | 60 秒 |
| btc-updown-15m | 15 分钟 | 120 秒 |
| eth-updown-5m | 5 分钟 | 60 秒，可选 |
| eth-updown-15m | 15 分钟 | 120 秒，可选 |

### 标准化

```bash
# 对指定日期的原始数据运行标准化
python -m poly.main normalize 20260417

# 可指定来源
python -m poly.main normalize 20260417 --source binance
python -m poly.main normalize 20260417 --source polymarket
```

Normalizer 和 replay 会扫描 gzip member；如果采集中途被 kill 导致某个 gzip member 损坏，会跳过坏段并继续恢复后续可读 JSONL。

### 生成研究标签

```bash
python -m poly.main metadata 20260417
python -m poly.main normalize 20260417 --source polymarket
python -m poly.main labels 20260417
```

生成内容：
- `poly_market_metadata.parquet` — `asset_id -> market_id/condition_id/slug/outcome/symbol/period/expiry` 以及 tick/min size/fee/liquidity/volume metadata
- `future_mid_1s / 3s / 5s / 10s` — 未来 midprice
- `markout_1s / 3s / 5s / 10s` — markout 收益率
- `markout_*s_bps` — markout（基点）
- `imbalance_bucket / spread_bucket / price_bucket` — 分位桶

### 训练目标和目标策略

给 coding agent 的完整分层文档从这里开始：

- [`docs/agent_manual.md`](docs/agent_manual.md) — 总览入口
- [`docs/agent/training_strategy.md`](docs/agent/training_strategy.md) — 当前训练目标、收益公式、两阶段策略
- [`docs/agent/features_and_labels.md`](docs/agent/features_and_labels.md) — 特征/标签/泄漏规则
- [`docs/agent/experiments.md`](docs/agent/experiments.md) — 关键实验结果和 caveat
- [`docs/agent/live_monitoring.md`](docs/agent/live_monitoring.md) — Ireland live-shadow 监控说明

训练管线优先使用 `data/normalized` 和 `data/research` 里的 Parquet，不直接从 raw JSONL 训练。当前训练目标不是预测 Polymarket 最终结算方向，而是验证 BTC/ETH Up/Down 市场里是否存在 5s/10s 级别的可交易微观结构 edge。

目标策略是 two-leg round trip：

1. 第一腿：用 taker 直接买入当前 asset，成交价近似为当前 `best_ask`。
2. 第二腿：在 opposite outcome 上挂 maker，目标是在 10 秒内用更好的价格补上反向腿。
3. 交易判定：如果 `first_leg_ask + future_opposite_maker_fill_price <= 0.96`，认为这次 entry 有足够空间 cover fee/slippage/safety margin。
4. 当前收益按实际两腿/撤退公式计算：成功时用 `second_leg_size - (first_leg_price + second_leg_size * second_leg_quote)`；失败时用同腿未来 bid 撤退，`unwind_profit = second_leg_size * future_best_bid_10s - first_leg_price`。这里的 `future_best_bid_10s` 是当前样本同一个 asset 的未来 bid label。

因此当前最重要的模型不是“最终涨跌模型”，而是 `p_enter` classifier：给定当前订单簿、Binance 参考行情、regime 状态，预测这笔 first-leg taker entry 后，第二腿 maker 是否有机会在 10 秒内让 round trip 成立。

当前训练分三层推进：

| 层 | 目标 | 当前状态 |
|----|------|----------|
| Layer 1 alpha | 预测 1s/3s/5s/10s markout 或短周期方向 | 已实现，主要用于 sanity check |
| Layer 2 entry | 预测 first-leg taker entry 是否值得做 | 已实现为 `y_final_profit_entry_10s` classifier，是当前主线 |
| Layer 3 exit/fill | 预测 second-leg maker fill 概率、fill 质量、失败强平价格 | 有派生标签雏形，仍需更保守的 queue/fill replay |

关键标签定义：

| 字段 | 含义 |
|------|------|
| `future_mid_10s`, `markout_10s_bps` | 传统 10 秒 markout 标签，用来做 alpha baseline |
| `future_best_bid_10s`, `future_best_ask_10s` | 当前 asset 未来 10 秒 book label，只能用于标签/评估，禁止作为特征 |
| `future_opposite_maker_fill_price_10s` | 10 秒内 opposite asset 上可观察到的 maker fill 价格证据 |
| `two_leg_total_price_10s` | `best_ask + future_opposite_maker_fill_price_10s` |
| `y_two_leg_entry_10s` | `two_leg_total_price_10s <= 0.96` 时为 `enter` |
| `first_unwind_loss_proxy_10s` | 第二腿失败时，第一腿用未来同腿 best bid 撤退的 loss proxy；对应 profit 可以为正 |
| `first_unwind_profit_proxy_10s` | `second_leg_size * future_best_bid_10s - first_leg_price`，signed unwind PnL |
| `final_profit_10s` | 成功用 two-leg success profit，失败用 signed unwind profit |
| `y_final_profit_entry_10s` | 当前主训练分类目标：`final_profit_10s > 0` 为 `enter` |

训练时只能使用当下可见 feature。所有 `future_*`、`markout_*`、`two_leg_*`、`final_profit_*`、`y_*` 列都属于未来标签/评估列，禁止进模型。

给后续 coding agent 的详细接手说明见 [`docs/agent_manual.md`](docs/agent_manual.md)。

当前无泄漏版 feature list（单 Binance symbol 时，Binance reference 按时间戳 asof join）：

| 分组 | 特征 |
|------|------|
| Polymarket top-of-book / price | `best_bid`, `best_ask`, `current_mid`, `current_spread`, `relative_spread`, `current_microprice` |
| Polymarket imbalance / depth proxy | `top1_imbalance`, `total_bid_levels`, `total_ask_levels`, `top3_imbalance`, `top5_imbalance`, `top10_imbalance`, `cum_bid_depth_topN_proxy`, `cum_ask_depth_topN_proxy`, `depth_level_imbalance_proxy` |
| Polymarket book event activity | `book_update_count_100ms`, `book_update_count_500ms`, `book_update_count_1s`, `spread_widen_count_recent`, `spread_narrow_count_recent`, `realized_vol_short` |
| Polymarket short return | `poly_return_1s` |
| Binance reference | `binance_mid`, `binance_spread`, `binance_return_tick`, `binance_return_100ms`, `binance_return_500ms`, `binance_return_1s`, `binance_return_3s`, `binance_recent_trade_imbalance` |
| Binance compact depth | `binance_microprice`, `binance_imbalance`, `binance_depth_top1_imbalance`, `binance_depth_top3_imbalance`, `binance_depth_top5_imbalance`, `binance_depth_top10_imbalance`, `binance_depth_top20_imbalance`, `binance_cum_bid_depth_top1/3/5/10/20`, `binance_cum_ask_depth_top1/3/5/10/20`, `binance_bid_depth_slope_top10/20`, `binance_ask_depth_slope_top10/20`, `binance_near_touch_bid_notional_5/10/20`, `binance_near_touch_ask_notional_5/10/20` |
| Lead-lag | `lead_lag_binance_minus_poly_1s`, `lead_lag_binance_minus_poly_500ms` |
| Regime / metadata / research buckets | `time_to_expiry_seconds`, `tick_size`, `min_order_size`, `maker_base_fee`, `taker_base_fee`, `imbalance_bucket`, `spread_bucket`, `price_bucket`, `vol_bucket`, `vol_60s`; `volume_24h`/`liquidity` are carried when non-null |

注意：`top3_imbalance/top5_imbalance/top10_imbalance` 和 `cum_*_depth_topN_proxy` 目前是 proxy，因为 normalized L2 还没有保存真实 top-N 价量档位。`realized_edge_after_entry_cost_bps` 是由未来 markout 派生的标签类字段，禁止作为训练特征。

```bash
# 0) 可选但推荐：补齐 Polymarket metadata，再重新 normalize/labels
python scripts/fetch_polymarket_metadata.py --data-dir data --dates 20260417
python -m poly.main normalize 20260417 --source polymarket
python -m poly.main labels 20260417

# 1) 构造 100ms 采样的特征 + 对齐标签数据集
python scripts/build_features.py \
  --data-dir data \
  --dates 20260417 \
  --output-dir artifacts/training \
  --sample-interval-ms 100 \
  --classification-theta-bps 5 \
  --entry-threshold-bps 8

# 2) 可选：单独导出 labels 便于检查分布
python scripts/build_labels.py \
  --dataset artifacts/training/alpha_dataset.parquet \
  --output-dir artifacts/training

# 3) 训练第一层 alpha baseline（默认按时间顺序 70/15/15 切分）
python scripts/train_alpha_model.py \
  --dataset artifacts/training/alpha_dataset.parquet \
  --output-dir artifacts/training/models

# 4) 评估预测质量和交易可用性
python scripts/evaluate_alpha_model.py \
  --dataset artifacts/training/alpha_dataset.parquet \
  --model-dir artifacts/training/models \
  --output-dir artifacts/training/evaluation \
  --taker-cost-bps 0
```

严格做 two-leg entry 实验时，不要用 test top-K 反推阈值。先用 validation set 选择 `p_enter` cutoff，再把这个 cutoff 原样打到 test set。因为 10 秒标签会和边界附近样本重叠，建议同时开启 10 秒 purge/embargo。

当前推荐主实验：

```bash
# 训练 finalProfit entry classifier，第一腿 ask，第二腿 opposite maker fill，总价格 <= 0.96
python scripts/train_alpha_model.py \
  --dataset artifacts/training/alpha_dataset.parquet \
  --output-dir artifacts/training/strict_models \
  --target-reg final_profit_10s \
  --target-cls y_final_profit_entry_10s \
  --sample-weight-col final_profit_weight_10s \
  --split-purge-ms 10000 \
  --split-embargo-ms 10000

# 用 validation 选 cutoff，再固定到 test
python scripts/select_entry_cutoffs.py \
  --output-dir artifacts/training/cutoff_selection \
  --split-purge-ms 10000 \
  --split-embargo-ms 10000 \
  --target-profit final_profit_10s \
  --target-cls y_final_profit_entry_10s \
  --run btc_only artifacts/training/btc_only/alpha_dataset.parquet artifacts/training/btc_only/alpha_dataset.parquet artifacts/training/btc_only/strict_models
```

当前 `20260417` strict no-leak sanity check 里，最好的候选是 BTC-only RandomForest：

| 实验 | 模型 | validation 选出的 cutoff | test entries | test avg `final_profit_10s` | test success rate |
|------|------|--------------------------|--------------|-----------------------------|-------------------|
| BTC-only | RandomForest | `p_enter >= 0.74` | 60 | `+0.01117` | 78.33% |
| mixed BTC+ETH | RandomForest | `p_enter >= 0.73` | 82 | `+0.00780` | 73.17% |

这些结果只能证明 pipeline 有希望，不是生产策略证据。当前数据仍是单日期样本，second-leg fill 还是基于未来 opposite trade evidence 的简化派生，不是完整 queue simulation。

产物：
- `alpha_dataset.parquet/csv` — 特征、10s markout 回归目标、三分类目标、entry-worthiness label
- `alpha_dataset_metadata.json` — 输入 schema、坏文件报告、特征列、标签列、日期范围、采样配置
- `models/*.joblib` — 默认训练 7 个 classifier：logistic、SGD logistic、GaussianNB、RandomForest、ExtraTrees、LightGBM、XGBoost
- `evaluation/summary_metrics.json` — MAE/RMSE/rank correlation、precision/recall/F1、阈值 entry EV
- `evaluation/*_by_*_bucket.csv` 和 `*.png` — 按 spread/imbalance/expiry/price 等 regime 的表现
- `cutoff_selection/validation_selected_cutoffs_test_results.csv` — validation-selected cutoff 在 test 上的最终表现

### 回放

```bash
# 按原速回放
python -m poly.main replay polymarket 20260417

# 2 倍速
python -m poly.main replay binance 20260417 --speed 2.0

# 最快速度（不等待）
python -m poly.main replay binance 20260417 --speed 0
```

### 查看状态

```bash
python -m poly.main status
```

### 删除数据

```bash
# 删除某天全部数据（会二次确认）
python -m poly.main purge 20260417

# 跳过确认
python -m poly.main purge 20260417 --yes

# 只删某一层
python -m poly.main purge 20260417 --layer raw
python -m poly.main purge 20260417 --layer normalized
python -m poly.main purge 20260417 --layer research
```

## 数据目录结构

```
data/
├── metadata/
│   └── markets.db                          # SQLite 市场元数据
├── raw_feed/
│   └── YYYYMMDD/
│       ├── polymarket_market.jsonl.gz      # Polymarket 市场原始消息
│       ├── polymarket_user.jsonl.gz        # Polymarket 用户原始消息
│       └── binance_spot.jsonl.gz           # Binance 原始消息
├── normalized/
│   └── YYYYMMDD/
│       ├── poly_l2_book.parquet            # Polymarket L2 订单簿快照
│       ├── poly_trades.parquet             # Polymarket 公共成交
│       ├── poly_best_bid_ask.parquet       # Polymarket 最优买卖价
│       ├── poly_orders.parquet             # Polymarket 自有订单
│       ├── poly_user_trades.parquet        # Polymarket 自有成交
│       ├── binance_l2_book.parquet         # Binance L2 订单簿
│       ├── binance_trades.parquet          # Binance 成交
│       └── binance_best_bid_ask.parquet    # Binance 最优买卖价
└── research/
    └── YYYYMMDD/
        ├── binance_markout_labels.parquet  # Binance markout 标签
        ├── poly_markout_labels.parquet     # Polymarket markout 标签
        └── poly_enriched_book.parquet      # Polymarket 增强订单簿
```

## 数据源

| 数据源 | 端点 | 用途 |
|--------|------|------|
| Gamma API | `https://gamma-api.polymarket.com` | Up/Down 市场发现（按 slug 查询） |
| Polymarket Market WS | `wss://ws-subscriptions-clob.polymarket.com/ws/market` | 实时订单簿和成交 |
| Polymarket User WS | `wss://ws-subscriptions-clob.polymarket.com/ws/user` | 自有订单和成交回报 |
| Binance WS | `wss://stream.binance.com:9443` | BTC 参考行情 |

## 技术栈

- **Rust**: PyO3, rust_decimal, BTreeMap 订单簿引擎
- **Python**: asyncio, websockets, aiohttp, pydantic, pyarrow, polars, orjson, structlog, click

## 关键设计

- **数据不丢**：每条 WS 消息接收时立即打纳秒时间戳，5 秒 flush + fsync
- **订单簿精度**：Rust 使用 Decimal 运算，避免浮点误差
- **自动轮换**：Up/Down 市场到期前自动订阅下一期，5m 提前 60s、15m 提前 120s
- **市场标识**：数据 source 字段包含 slug（如 `polymarket:btc-updown-5m-1776409800`），便于区分不同市场和周期
- **截断容忍**：normalizer 能处理被 kill 截断的 gzip 文件
- **消息格式兼容**：自动处理 Polymarket WS 返回 dict 或 list 格式
