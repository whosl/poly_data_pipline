"""Offline evaluation with ML metrics and trading-usefulness tables."""

from __future__ import annotations

from pathlib import Path
import math

import joblib
import numpy as np
import polars as pl

from poly.training.config import save_json
from poly.training.features import infer_feature_columns
from poly.training.splits import chronological_split, split_ranges


def evaluate_dataset_models(
    dataset_path: Path,
    model_dir: Path,
    output_dir: Path,
    target_reg: str = "y_reg_10s",
    target_cls: str = "y_cls_10s",
    taker_cost_bps: float = 0.0,
    prediction_thresholds: list[float] | None = None,
    split_purge_ms: int = 0,
    split_embargo_ms: int = 0,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = pl.read_parquet(str(dataset_path))
    splits = chronological_split(dataset, purge_ms=split_purge_ms, embargo_ms=split_embargo_ms)
    eval_df = splits["test"]
    if eval_df.is_empty():
        raise ValueError("test split is empty")

    feature_columns = infer_feature_columns(dataset)
    thresholds = prediction_thresholds or [0.0, 1.0, 2.0, 5.0, 10.0]
    reports: dict[str, object] = {
        "dataset_path": str(dataset_path),
        "model_dir": str(model_dir),
        "split_purge_ms": split_purge_ms,
        "split_embargo_ms": split_embargo_ms,
        "split_ranges": split_ranges(splits),
        "models": {},
    }

    for path in sorted(model_dir.glob("*.joblib")):
        payload = joblib.load(path)
        model = payload["model"]
        task = payload["task"]
        model_features = payload.get("feature_columns", feature_columns)
        x_eval = eval_df.select(model_features).to_pandas()
        model_report: dict[str, object] = {"task": task, "target": payload.get("target")}
        if task == "regression":
            pred = model.predict(x_eval)
            scored = eval_df.with_columns(pl.Series("prediction", pred))
            model_report.update(regression_metrics(scored, target_reg))
            model_report["trading_thresholds"] = threshold_eval(scored, target_reg, thresholds, taker_cost_bps)
            write_bucket_tables(scored, output_dir, path.stem, target_reg)
            plot_regression(scored, output_dir / f"{path.stem}_prediction_vs_realized.png", target_reg)
            plot_confidence(scored, output_dir / f"{path.stem}_confidence_buckets.png", target_reg)
        elif task == "classification":
            pred = model.predict(x_eval)
            scored = eval_df.with_columns(pl.Series("prediction_class", pred))
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_eval)
                classes = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["model"].classes_)
                positive_class = choose_positive_class(classes)
                if positive_class in classes:
                    positive_idx = classes.index(positive_class)
                    scored = scored.with_columns(pl.Series("prediction_positive_proba", proba[:, positive_idx]))
                    if positive_class == "up":
                        scored = scored.with_columns(pl.Series("prediction_up_proba", proba[:, positive_idx]))
                scored = scored.with_columns(pl.Series("prediction_confidence", proba.max(axis=1)))
                model_report["positive_class"] = positive_class
            model_report.update(classification_metrics(scored, target_cls))
            if "prediction_positive_proba" in scored.columns:
                model_report["trading_thresholds"] = proba_threshold_eval(scored, target_reg, "prediction_positive_proba", taker_cost_bps)
                plot_confidence(scored.rename({"prediction_positive_proba": "prediction"}), output_dir / f"{path.stem}_confidence_buckets.png", target_reg)
        reports["models"][path.stem] = model_report

    save_json(reports, output_dir / "summary_metrics.json")
    return reports


def regression_metrics(df: pl.DataFrame, target: str) -> dict[str, float | None]:
    clean = df.drop_nulls([target, "prediction"])
    if clean.is_empty():
        return {"mae": None, "rmse": None, "rank_correlation": None}
    y = clean[target].to_numpy()
    pred = clean["prediction"].to_numpy()
    err = pred - y
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err * err))),
        "rank_correlation": rank_corr(pred, y),
    }


def classification_metrics(df: pl.DataFrame, target: str) -> dict[str, object]:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    clean = df.drop_nulls([target, "prediction_class"])
    if clean.is_empty():
        return {"precision_macro": None, "recall_macro": None, "f1_macro": None, "confusion_matrix": []}
    y = clean[target].to_list()
    pred = clean["prediction_class"].to_list()
    preferred = ["down", "neutral", "up", "skip", "enter"]
    observed = list(dict.fromkeys(y + pred))
    labels = [label for label in preferred if label in observed]
    labels.extend([label for label in observed if label not in labels])
    return {
        "precision_macro": float(precision_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def choose_positive_class(classes: list[object]) -> object | None:
    for label in ["enter", "up", 1, "1", True]:
        if label in classes:
            return label
    return classes[-1] if classes else None


def threshold_eval(
    df: pl.DataFrame,
    target: str,
    thresholds: list[float],
    taker_cost_bps: float,
) -> list[dict[str, float | int]]:
    rows = []
    for threshold in thresholds:
        selected = df.filter(pl.col("prediction") >= threshold).drop_nulls([target])
        rows.append(selection_stats(selected, target, taker_cost_bps, {"prediction_threshold": threshold}))
    return rows


def proba_threshold_eval(
    df: pl.DataFrame,
    target: str,
    proba_col: str,
    taker_cost_bps: float,
) -> list[dict[str, float | int]]:
    rows = []
    for threshold in [0.40, 0.50, 0.60, 0.70, 0.80]:
        selected = df.filter(pl.col(proba_col) >= threshold).drop_nulls([target])
        rows.append(selection_stats(selected, target, taker_cost_bps, {"probability_threshold": threshold}))
    return rows


def selection_stats(
    selected: pl.DataFrame,
    target: str,
    taker_cost_bps: float,
    prefix: dict[str, float],
) -> dict[str, float | int]:
    if selected.is_empty():
        return {
            **prefix,
            "candidate_entries": 0,
            "avg_realized_target": None,
            "median_realized_target": None,
            "success_rate": None,
            "avg_realized_markout_bps": None,
            "win_rate": None,
            "ev_after_taker_cost_bps": None,
        }
    realized = selected[target]
    avg_target = float(realized.mean())
    success_rate = float((realized > 0).mean())
    return {
        **prefix,
        "candidate_entries": selected.height,
        "avg_realized_target": avg_target,
        "median_realized_target": float(realized.median()),
        "success_rate": success_rate,
        "avg_realized_markout_bps": avg_target,
        "win_rate": success_rate,
        "ev_after_taker_cost_bps": float(avg_target - taker_cost_bps),
    }


def write_bucket_tables(df: pl.DataFrame, output_dir: Path, model_name: str, target: str) -> None:
    bucket_cols = [c for c in ["spread_bucket", "imbalance_bucket", "expiry_bucket", "price_bucket", "market_phase"] if c in df.columns]
    for col in bucket_cols:
        table = (
            df.drop_nulls([target, "prediction", col])
            .group_by(col)
            .agg(
                [
                    pl.len().alias("rows"),
                    pl.col(target).mean().alias("avg_realized_markout_bps"),
                    (pl.col(target) > 0).mean().alias("win_rate"),
                    pl.col("prediction").mean().alias("avg_prediction"),
                ]
            )
            .sort(col)
        )
        table.write_csv(str(output_dir / f"{model_name}_by_{col}.csv"))


def plot_regression(df: pl.DataFrame, path: Path, target: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    clean = df.drop_nulls([target, "prediction"])
    if clean.is_empty():
        return
    sample = clean.sample(min(clean.height, 5000), seed=42) if clean.height > 5000 else clean
    plt.figure(figsize=(7, 5))
    plt.scatter(sample["prediction"].to_numpy(), sample[target].to_numpy(), alpha=0.25, s=8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("prediction")
    plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_confidence(df: pl.DataFrame, path: Path, target: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if "prediction" not in df.columns:
        return
    clean = df.drop_nulls([target, "prediction"])
    if clean.height < 10:
        return
    buckets = clean.with_columns(
        pl.col("prediction").qcut(10, labels=[f"q{i}" for i in range(10)], allow_duplicates=True).alias("prediction_bucket")
    )
    agg = buckets.group_by("prediction_bucket").agg(pl.col(target).mean().alias("avg_markout")).sort("prediction_bucket")
    plt.figure(figsize=(8, 4))
    plt.bar(agg["prediction_bucket"].to_list(), agg["avg_markout"].to_list())
    plt.xticks(rotation=45)
    plt.ylabel(f"avg {target}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def rank_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2:
        return None
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    if np.std(ar) == 0 or np.std(br) == 0:
        return None
    return float(np.corrcoef(ar, br)[0, 1])
