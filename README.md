# Polymarket 低延迟量化数据采集系统

面向 Polymarket BTC/ETH Up/Down 市场的低延迟数据采集与标准化管线，用于短周期微观结构研究。

## 功能

- **Polymarket Up/Down 市场采集**：自动发现、订阅、轮换 btc/eth updown-5m/15m 市场
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
1. 自动计算 btc/eth updown-5m/15m 当前和下一期市场 slug（如 `btc-updown-5m-1776409800`）
2. 通过 Gamma API 查询市场 token IDs
3. 连接 Polymarket market WebSocket，订阅订单簿和成交
4. 每 10 秒检查是否有新市场需要订阅
5. 到期市场自动轮换，提前 60-120 秒订阅下一期
6. 同时连接 Binance WebSocket，订阅 BTC 行情
7. 原始消息实时写入 gzip JSONL，标准化数据写入 Parquet

#### 采集的市场

| 市场类型 | 周期 | 提前订阅 |
|---------|------|---------|
| btc-updown-5m | 5 分钟 | 60 秒 |
| btc-updown-15m | 15 分钟 | 120 秒 |
| eth-updown-5m | 5 分钟 | 60 秒 |
| eth-updown-15m | 15 分钟 | 120 秒 |

### 标准化

```bash
# 对指定日期的原始数据运行标准化
python -m poly.main normalize 20260417

# 可指定来源
python -m poly.main normalize 20260417 --source binance
python -m poly.main normalize 20260417 --source polymarket
```

### 生成研究标签

```bash
python -m poly.main labels 20260417
```

生成内容：
- `future_mid_1s / 3s / 5s / 10s` — 未来 midprice
- `markout_1s / 3s / 5s / 10s` — markout 收益率
- `markout_*s_bps` — markout（基点）
- `imbalance_bucket / spread_bucket / price_bucket` — 分位桶

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
