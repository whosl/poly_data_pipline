"""Baseline model training for alpha regression/classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib
import importlib.util

import joblib
import numpy as np
import polars as pl
import structlog
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from poly.training.config import TrainConfig, dataclass_to_json_dict, save_json
from poly.training.features import infer_feature_columns
from poly.training.splits import chronological_split, split_ranges

logger = structlog.get_logger()


class LGBBoosterWrapper:
    """Wraps a LightGBM Booster with sklearn-like predict_proba / predict."""
    def __init__(self, booster, n_features, task="classification"):
        self.booster = booster
        self.n_features = n_features
        self.task = task

    def predict_proba(self, X):
        raw = self.booster.predict(X)
        if raw.ndim == 1:
            return np.column_stack([1 - raw, raw])
        return raw

    def predict(self, X):
        if self.task == "classification":
            return (self.booster.predict(X) >= 0.5).astype(int)
        return self.booster.predict(X)


class Winsorizer(TransformerMixin, BaseEstimator):
    """Clip features to quantile bounds fitted on training data.

    Parameters
    ----------
    lower_quantile : float
        Lower bound quantile (e.g. 0.005 for 0.5th percentile).
    upper_quantile : float
        Upper bound quantile (e.g. 0.995 for 99.5th percentile).
    """

    def __init__(self, lower_quantile: float = 0.005, upper_quantile: float = 0.995):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lower_ = np.nanpercentile(X, self.lower_quantile * 100, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper_quantile * 100, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.clip(X, self.lower_, self.upper_)


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
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    dataset = pl.read_parquet(str(config.dataset_path))
    feature_columns = infer_feature_columns(dataset)
    if not feature_columns:
        raise ValueError("no feature columns found")
    categorical_columns = [col for col in feature_columns if is_categorical_feature(col, dataset.schema[col])]
    numeric_columns = [col for col in feature_columns if col not in categorical_columns]

    splits = chronological_split(
        dataset,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        purge_ms=config.split_purge_ms,
        embargo_ms=config.split_embargo_ms,
    )
    train_df = splits["train"].drop_nulls([config.target_reg, config.target_cls])
    if train_df.is_empty():
        raise ValueError("training split is empty after dropping missing targets")

    winsorize_quantiles = None
    if config.winsorize_lower is not None and config.winsorize_upper is not None:
        winsorize_quantiles = (config.winsorize_lower, config.winsorize_upper)
        logger.info("winsorize_enabled", lower=config.winsorize_lower, upper=config.winsorize_upper)

    preprocessor = make_preprocessor(numeric_columns, categorical_columns, scale_numeric=True, winsorize_quantiles=winsorize_quantiles)
    tree_preprocessor = make_preprocessor(numeric_columns, categorical_columns, scale_numeric=False, winsorize_quantiles=winsorize_quantiles)

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
        "sgd_logistic_classifier": (
            "classification",
            Pipeline(
                [
                    ("prep", preprocessor),
                    (
                        "model",
                        SGDClassifier(
                            loss="log_loss",
                            alpha=0.0001,
                            max_iter=1000,
                            tol=1e-3,
                            class_weight="balanced",
                            random_state=config.random_seed,
                        ),
                    ),
                ]
            ),
        ),
        "gaussian_nb_classifier": (
            "classification",
            Pipeline([("prep", tree_preprocessor), ("model", GaussianNB())]),
        ),
        "random_forest_classifier": (
            "classification",
            Pipeline(
                [
                    ("prep", tree_preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=100,
                            max_depth=8,
                            min_samples_leaf=50,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=config.random_seed,
                        ),
                    ),
                ]
            ),
        ),
        "random_forest_regressor": (
            "regression",
            Pipeline(
                [
                    ("prep", tree_preprocessor),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=100,
                            max_depth=8,
                            min_samples_leaf=50,
                            n_jobs=-1,
                            random_state=config.random_seed,
                        ),
                    ),
                ]
            ),
        ),
        "extra_trees_classifier": (
            "classification",
            Pipeline(
                [
                    ("prep", tree_preprocessor),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=100,
                            max_depth=8,
                            min_samples_leaf=50,
                            class_weight="balanced",
                            n_jobs=-1,
                            random_state=config.random_seed,
                        ),
                    ),
                ]
            ),
        ),
        "extra_trees_regressor": (
            "regression",
            Pipeline(
                [
                    ("prep", tree_preprocessor),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=100,
                            max_depth=8,
                            min_samples_leaf=50,
                            n_jobs=-1,
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
                            LabelEncodedClassifier(
                                xgb.XGBClassifier(
                                    n_estimators=300,
                                    learning_rate=0.03,
                                    max_depth=4,
                                    random_state=config.random_seed,
                                    eval_metric="mlogloss",
                                )
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
        required_cols = [target]
        if config.sample_weight_col and config.sample_weight_col in train_df.columns:
            required_cols.append(config.sample_weight_col)
        fit_df = train_df.drop_nulls(required_cols)
        if fit_df.is_empty():
            logger.warning("skip_empty_target", model=name, target=target)
            continue
        x_train = fit_df.select(feature_columns).to_pandas()
        y_train = fit_df[target].to_pandas()
        sample_weight = None
        if config.sample_weight_col and config.sample_weight_col in fit_df.columns:
            sample_weight = fit_df[config.sample_weight_col].to_pandas()
        fit_model(pipeline, x_train, y_train, sample_weight)
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
        "numeric_feature_columns": numeric_columns,
        "categorical_feature_columns": categorical_columns,
        "model_paths": model_paths,
        "sample_weight_col": config.sample_weight_col,
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


def fit_model(pipeline: object, x_train, y_train, sample_weight) -> None:
    if sample_weight is None:
        pipeline.fit(x_train, y_train)
        return
    try:
        pipeline.fit(x_train, y_train, model__sample_weight=sample_weight)
    except TypeError as exc:
        logger.warning("sample_weight_unsupported", error=str(exc))
        pipeline.fit(x_train, y_train)


def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_numeric: bool,
    winsorize_quantiles: tuple[float, float] | None = None,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    transformers = []
    if numeric_columns:
        steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if winsorize_quantiles is not None:
            lower, upper = winsorize_quantiles
            steps.append(("winsorizer", Winsorizer(lower_quantile=lower, upper_quantile=upper)))
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def is_categorical_feature(col: str, dtype: pl.DataType) -> bool:
    categorical_feature_names = {
        "imbalance_bucket",
        "spread_bucket",
        "price_bucket",
        "vol_bucket",
    }
    return col in categorical_feature_names or dtype in {pl.String, pl.Categorical, pl.Enum}


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Wrap classifiers that require numeric labels while exposing original labels."""

    _estimator_type = "classifier"

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y, sample_weight=None):
        from sklearn.preprocessing import LabelEncoder

        self.label_encoder_ = LabelEncoder()
        encoded = self.label_encoder_.fit_transform(y)
        if sample_weight is None:
            self.estimator.fit(x, encoded)
        else:
            self.estimator.fit(x, encoded, sample_weight=sample_weight)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, x):
        encoded = self.estimator.predict(x)
        return self.label_encoder_.inverse_transform(encoded.astype(int))

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)

    @property
    def feature_importances_(self):
        if not hasattr(self.estimator, "feature_importances_"):
            raise AttributeError("wrapped estimator has no feature_importances_")
        return self.estimator.feature_importances_

    @property
    def coef_(self):
        if not hasattr(self.estimator, "coef_"):
            raise AttributeError("wrapped estimator has no coef_")
        return self.estimator.coef_


def feature_importance(pipeline: object, feature_columns: list[str]) -> list[dict[str, object]]:
    model = pipeline.named_steps.get("model") if hasattr(pipeline, "named_steps") else pipeline
    names = feature_columns
    if hasattr(pipeline, "named_steps") and "prep" in pipeline.named_steps:
        try:
            names = [
                name.replace("numeric__", "").replace("categorical__", "")
                for name in pipeline.named_steps["prep"].get_feature_names_out()
            ]
        except Exception:
            names = feature_columns
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
        [{"feature": feature, "importance": float(value)} for feature, value in zip(names, values)],
        key=lambda x: x["importance"],
        reverse=True,
    )
