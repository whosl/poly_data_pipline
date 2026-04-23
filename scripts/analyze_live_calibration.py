#!/usr/bin/env python3
"""Analyze live p_fill calibration from shadow candidate samples."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


P_FILL_BUCKETS = [
    ("<0.50", None, 0.50),
    ("0.50-0.60", 0.50, 0.60),
    ("0.60-0.70", 0.60, 0.70),
    ("0.70-0.80", 0.70, 0.80),
    ("0.80+", 0.80, None),
]
EXPECTED_PROFIT_BUCKETS = [
    ("<0.000", None, 0.0),
    ("0.000-0.005", 0.0, 0.005),
    ("0.005-0.010", 0.005, 0.010),
    ("0.010-0.020", 0.010, 0.020),
    ("0.020+", 0.020, None),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True, help="live_candidate_samples.jsonl")
    parser.add_argument("--signal-samples", type=Path, default=None, help="Optional live_signal_samples.jsonl")
    parser.add_argument(
        "--include-all-signal-samples",
        action="store_true",
        help="Do not align signal samples to the candidate sample time range.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path, event: str | None = None) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event is None or row.get("event") == event:
                rows.append(row)
    return rows


def nested(row: dict, *keys: str):
    cur = row
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def as_float(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def bucket(value: float | None, buckets: list[tuple[str, float | None, float | None]]) -> str:
    if value is None:
        return "missing"
    for label, lo, hi in buckets:
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return label
    return "other"


def entry_ask_bucket(row: dict) -> str:
    value = as_float(nested(row, "entry", "best_ask"))
    return bucket(
        value,
        [
            ("<0.20", None, 0.20),
            ("0.20-0.40", 0.20, 0.40),
            ("0.40-0.60", 0.40, 0.60),
            ("0.60-0.80", 0.60, 0.80),
            ("0.80+", 0.80, None),
        ],
    )


def tte_bucket(row: dict) -> str:
    value = as_float(nested(row, "entry", "time_to_expiry_seconds"))
    return bucket(
        value,
        [
            ("<30s", None, 30.0),
            ("30-60s", 30.0, 60.0),
            ("60-180s", 60.0, 180.0),
            (">=180s", 180.0, None),
        ],
    )


def slug_period(row: dict) -> str:
    slug = str(row.get("slug") or "")
    if "-updown-5m-" in slug:
        return "5m"
    if "-updown-15m-" in slug:
        return "15m"
    return "unknown"


def summarize(rows: list[dict], group_fn) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_fn(row)].append(row)
    output = []
    for group, items in sorted(groups.items()):
        profits = [as_float(nested(r, "labels", "final_profit_10s")) for r in items]
        profits = [p for p in profits if p is not None]
        y_fill = [1 if nested(r, "labels", "y_two_leg_entry_10s") == "enter" else 0 for r in items]
        y_final = [1 if nested(r, "labels", "y_final_profit_entry_10s") == "enter" else 0 for r in items]
        p_fill = [as_float(nested(r, "prediction", "p_fill")) for r in items]
        p_fill = [p for p in p_fill if p is not None]
        output.append(
            {
                "bucket": group,
                "n": len(items),
                "fill_rate": mean(y_fill) if y_fill else None,
                "final_profit_positive_rate": mean(y_final) if y_final else None,
                "avg_final_profit": mean(profits) if profits else None,
                "total_final_profit": sum(profits) if profits else None,
                "avg_p_fill": mean(p_fill) if p_fill else None,
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_metrics(rows: list[dict]) -> dict[str, object]:
    profits = [as_float(nested(r, "labels", "final_profit_10s")) for r in rows]
    profits = [p for p in profits if p is not None]
    y_fill = [1 if nested(r, "labels", "y_two_leg_entry_10s") == "enter" else 0 for r in rows]
    y_final = [1 if nested(r, "labels", "y_final_profit_entry_10s") == "enter" else 0 for r in rows]
    p_fill_pairs = []
    for row in rows:
        p_fill = as_float(nested(row, "prediction", "p_fill"))
        if p_fill is not None:
            p_fill_pairs.append((p_fill, 1 if nested(row, "labels", "y_two_leg_entry_10s") == "enter" else 0))
    return {
        "resolved_count": len(rows),
        "two_leg_success_rate": mean(y_fill) if y_fill else None,
        "final_profit_positive_rate": mean(y_final) if y_final else None,
        "avg_final_profit": mean(profits) if profits else None,
        "total_final_profit": sum(profits) if profits else None,
        "brier_score": mean([(p - y) ** 2 for p, y in p_fill_pairs]) if p_fill_pairs else None,
        "label_counts": dict(Counter(nested(r, "labels", "y_final_profit_entry_10s") for r in rows)),
        "two_leg_counts": dict(Counter(nested(r, "labels", "y_two_leg_entry_10s") for r in rows)),
    }


def main() -> None:
    args = parse_args()
    signal_samples = args.signal_samples or args.samples.with_name("live_signal_samples.jsonl")
    candidate_rows = load_jsonl(args.samples, "live_candidate_resolved_sample")
    signal_rows = load_jsonl(signal_samples, "live_signal_resolved_sample")
    signal_since_ns = None
    if candidate_rows and not args.include_all_signal_samples:
        candidate_timestamps = [as_int(row.get("timestamp_ns")) for row in candidate_rows]
        candidate_timestamps = [ts for ts in candidate_timestamps if ts is not None]
        if candidate_timestamps:
            signal_since_ns = min(candidate_timestamps)
            signal_rows = [row for row in signal_rows if (as_int(row.get("timestamp_ns")) or 0) >= signal_since_ns]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "calibration_by_p_fill_bucket.csv": summarize(
            candidate_rows, lambda r: bucket(as_float(nested(r, "prediction", "p_fill")), P_FILL_BUCKETS)
        ),
        "by_pred_expected_profit_bucket.csv": summarize(
            candidate_rows,
            lambda r: bucket(as_float(nested(r, "prediction", "pred_expected_profit")), EXPECTED_PROFIT_BUCKETS),
        ),
        "by_entry_ask_bucket.csv": summarize(candidate_rows, entry_ask_bucket),
        "by_time_to_expiry_bucket.csv": summarize(candidate_rows, tte_bucket),
        "by_slug_period.csv": summarize(candidate_rows, slug_period),
    }
    for filename, rows in tables.items():
        write_csv(args.output_dir / filename, rows)

    summary = {
        "candidate_samples": {
            "path": str(args.samples),
            **summary_metrics(candidate_rows),
        },
        "live_policy": {
            "path": str(signal_samples),
            "signal_since_timestamp_ns": signal_since_ns,
            "signal_count": len(signal_rows),
            **summary_metrics(signal_rows),
        },
        "outputs": sorted(tables),
    }
    with (args.output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
