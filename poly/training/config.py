"""Configuration objects for offline microstructure training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class DatasetConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("artifacts/training")
    dates: list[str] = field(default_factory=list)
    sample_interval_ms: int = 100
    min_spacing_ms: int = 100
    horizon_seconds: int = 10
    classification_theta_bps: float = 5.0
    entry_threshold_bps: float = 8.0
    strategy_entry_threshold_price: float = 0.04
    two_leg_max_total_price: float = 0.96
    two_leg_no_fill_edge: float = -1.0
    two_leg_maker_fill_trade_side: str = "SELL"
    maker_fill_latency_ms: int = 250
    maker_fill_trade_through_ticks: float = 1.0
    fee_rate: float = 0.072
    price_buffer: float = 0.01
    first_leg_fill_validation_ms: int = 0
    taker_cost_bps: float = 0.0
    slippage_buffer_bps: float = 2.0
    safety_margin_bps: float = 1.0
    join_tolerance_ms: int = 500
    max_null_fraction: float = 0.35
    sample_mode: str = "time-bucket"
    book_event_source: str = "book"
    feature_workers: int = 1
    symbols: list[str] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    dataset_path: Path
    output_dir: Path = Path("artifacts/training/models")
    target_reg: str = "y_reg_10s"
    target_cls: str = "y_cls_10s"
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    split_purge_ms: int = 0
    split_embargo_ms: int = 0
    random_seed: int = 42
    sample_weight_col: str | None = None
    winsorize_lower: float | None = None
    winsorize_upper: float | None = None
    models: list[str] = field(
        default_factory=lambda: [
            "logistic_regression",
            "sgd_logistic_classifier",
            "gaussian_nb_classifier",
            "random_forest_classifier",
            "extra_trees_classifier",
            "lightgbm_classifier",
            "xgboost_classifier",
        ]
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def dataclass_to_json_dict(obj: object) -> dict[str, Any]:
    data = asdict(obj)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data
