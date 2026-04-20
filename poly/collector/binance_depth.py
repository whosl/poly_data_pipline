"""Helpers for deriving compact Binance depth features."""

from __future__ import annotations


def depth_features(bids: list, asks: list, max_depth: int = 20) -> dict[str, float | int | None]:
    bid_levels = parse_levels(bids, max_depth)
    ask_levels = parse_levels(asks, max_depth)
    if not bid_levels or not ask_levels:
        return {}

    top_bid_price, top_bid_size = bid_levels[0]
    top_ask_price, top_ask_size = ask_levels[0]
    top_size_total = top_bid_size + top_ask_size
    microprice = None
    if top_size_total > 0:
        microprice = ((top_ask_price * top_bid_size) + (top_bid_price * top_ask_size)) / top_size_total

    result: dict[str, float | int | None] = {
        "microprice": microprice,
        "imbalance": imbalance(sum_sizes(bid_levels, 1), sum_sizes(ask_levels, 1)),
        "total_bid_levels": len(bid_levels),
        "total_ask_levels": len(ask_levels),
    }
    for n in [1, 3, 5, 10, 20]:
        if max_depth < n:
            continue
        result[f"depth_top{n}_imbalance"] = imbalance(sum_sizes(bid_levels, n), sum_sizes(ask_levels, n))
        result[f"cum_bid_depth_top{n}"] = sum_sizes(bid_levels, n)
        result[f"cum_ask_depth_top{n}"] = sum_sizes(ask_levels, n)
    for n in [10, 20]:
        if max_depth < n:
            continue
        result[f"bid_depth_slope_top{n}"] = depth_slope(bid_levels, n)
        result[f"ask_depth_slope_top{n}"] = depth_slope(ask_levels, n)
    for n in [5, 10, 20]:
        if max_depth < n:
            continue
        result[f"near_touch_bid_notional_{n}"] = sum_notional(bid_levels, n)
        result[f"near_touch_ask_notional_{n}"] = sum_notional(ask_levels, n)
    return result


def parse_levels(levels: list, max_depth: int) -> list[tuple[float, float]]:
    parsed: list[tuple[float, float]] = []
    for level in levels[:max_depth]:
        if len(level) < 2:
            continue
        try:
            price = float(level[0])
            size = float(level[1])
        except (TypeError, ValueError):
            continue
        parsed.append((price, size))
    return parsed


def sum_sizes(levels: list[tuple[float, float]], n: int) -> float:
    return float(sum(size for _, size in levels[:n]))


def sum_notional(levels: list[tuple[float, float]], n: int) -> float:
    return float(sum(price * size for price, size in levels[:n]))


def imbalance(bid_size: float, ask_size: float) -> float | None:
    total = bid_size + ask_size
    if total <= 0:
        return None
    return float((bid_size - ask_size) / total)


def depth_slope(levels: list[tuple[float, float]], n: int) -> float | None:
    selected = levels[:n]
    if len(selected) < 2:
        return None
    first = selected[0][1]
    last = selected[-1][1]
    return float((last - first) / (len(selected) - 1))
