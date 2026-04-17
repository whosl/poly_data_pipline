"""Baseline model training for alpha regression/classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import importlib.util

import joblib
import polars as pl
import structlog

from poly.training.config import TrainConfig, dataclass_to_json_dict, save_json
from poly.training.features import infer_feature_columns
from poly.training.splits import chronological_split, split_ranges

logger = structlog.get_logger()


@dataclass
class TrainedArtifacts:
    output_dir: Path
    feature_columns: list[str]
    model_paths: dict[str, str]
    metadata: dict[str, object]
    splits: dict[str, pl.DataFrame]


def train_baselines(config: TrainConfig) -> TrainedArtifacts:
    ensure_sklearn()
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    dataset = pl.read_parquet(str(config.dataset_path))
    feature_columns = infer_feature_columns(dataset)
    if not feature_columns:
        raise ValueError("no numeric feature columns found")

    splits = chronological_split(
        dataset,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    train_df = splits["train"].drop_nulls([config.target_reg, config.target_cls])
    if train_df.is_empty():
        raise ValueError("training split is empty after dropping missing targets")

    preprocessor = ColumnTransformer(
        [("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), feature_columns)],
        remainder="drop",
    )
    tree_preprocessor = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median"), feature_columns)],
        remainder="drop",
    )

    model_specs: dict[str, tuple[str, object]] = {
        "linear_regression": ("regression", Pipeline([("prep", preprocessor), ("model", LinearRegression())])),
        "ridge_regression": ("regression", Pipeline([("prep", preprocessor), ("model", Ridge(alpha=1.0, random_state=config.random_seed))])),
        "logistic_regression": (
            "classification",
            Pipeline(
                [
                    ("prep", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=config.random_seed,
                        ),
                    ),
                ]
            ),
        ),
    }

    if "lightgbm_regressor" in config.models or "lightgbm_classifier" in config.models:
        lgb = import_optional("lightgbm")
        if lgb is not None:

            model_specs["lightgbm_regressor"] = (
                "regression",
                Pipeline(
                    [
                        ("prep", tree_preprocessor),
                        (
                            "model",
                            lgb.LGBMRegressor(
                                n_estimators=300,
                                learning_rate=0.03,
                                random_state=config.random_seed,
                                verbosity=-1,
                            ),
                        ),
                    ]
                ),
            )
            model_specs["lightgbm_classifier"] = (
                "classification",
                Pipeline(
                    [
                        ("prep", tree_preprocessor),
                        (
                            "model",
                            lgb.LGBMClassifier(
                                n_estimators=300,
                                learning_rate=0.03,
                                class_weight="balanced",
                                random_state=config.random_seed,
                                verbosity=-1,
                            ),
                        ),
                    ]
                ),
            )
        else:
            logger.warning("optional_model_unavailable", model="lightgbm")

    if "xgboost_regressor" in config.models or "xgboost_classifier" in config.models:
        xgb = import_optional("xgboost")
        if xgb is not None:

            model_specs["xgboost_regressor"] = (
                "regression",
                Pipeline(
                    [
                        ("prep", tree_preprocessor),
                        (
                            "model",
                            xgb.XGBRegressor(
                                n_estimators=300,
                                learning_rate=0.03,
                                max_depth=4,
                                random_state=config.random_seed,
                            ),
                        ),
                    ]
                ),
            )
            model_specs["xgboost_classifier"] = (
                "classification",
                Pipeline(
                    [
                        ("prep", tree_preprocessor),
                        (
                            "model",
                            xgb.XGBClassifier(
                                n_estimators=300,
                                learning_rate=0.03,
                                max_depth=4,
                                random_state=config.random_seed,
                                eval_metric="mlogloss",
                            ),
                        ),
                    ]
                ),
            )
        else:
            logger.warning("optional_model_unavailable", model="xgboost")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model_paths: dict[str, str] = {}
    trained: dict[str, object] = {}
    for name, (task, pipeline) in model_specs.items():
        if name not in config.models:
            continue
        target = config.target_reg if task == "regression" else config.target_cls
        fit_df = train_df.drop_nulls([target])
        if fit_df.is_empty():
            logger.warning("skip_empty_target", model=name, target=target)
            continue
        x_train = fit_df.select(feature_columns).to_pandas()
        y_train = fit_df[target].to_pandas()
        pipeline.fit(x_train, y_train)
        path = config.output_dir / f"{name}.joblib"
        joblib.dump(
            {
                "model": pipeline,
                "task": task,
                "feature_columns": feature_columns,
                "target": target,
                "config": dataclass_to_json_dict(config),
            },
            path,
        )
        model_paths[name] = str(path)
        trained[name] = pipeline
        importance = feature_importance(pipeline, feature_columns)
        if importance:
            pl.DataFrame(importance).write_csv(str(config.output_dir / f"{name}_feature_importance.csv"))

    metadata = {
        "config": dataclass_to_json_dict(config),
        "dataset_path": str(config.dataset_path),
        "rows": dataset.height,
        "feature_columns": feature_columns,
        "model_paths": model_paths,
        "split_ranges": split_ranges(splits),
    }
    save_json(metadata, config.output_dir / "training_metadata.json")
    return TrainedArtifacts(config.output_dir, feature_columns, model_paths, metadata, splits)


def ensure_sklearn() -> None:
    if importlib.util.find_spec("sklearn") is None:
        raise RuntimeError("scikit-learn is required for baseline training; install requirements.txt")


def import_optional(module_name: str):
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        logger.warning("optional_model_import_failed", module=module_name, error=str(exc))
        return None


def feature_importance(pipeline: object, feature_columns: list[str]) -> list[dict[str, object]]:
    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline
    values = None
    if hasattr(model, "feature_importances_"):
        values = list(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if getattr(coef, "ndim", 1) > 1:
            values = list(abs(coef).mean(axis=0))
        else:
            values = list(abs(coef))
    if values is None:
        return []
    return sorted(
        [{"feature": feature, "importance": float(value)} for feature, value in zip(feature_columns, values)],
        key=lambda x: x["importance"],
        reverse=True,
    )
