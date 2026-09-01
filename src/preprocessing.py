"""Leakage-safe preprocessing utilities for the Telco churn dataset.

The functions in this module keep feature engineering deterministic and fit all
learned preprocessing steps on the training split only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

SERVICE_COLUMNS: tuple[str, ...] = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_services",
)

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    ID_COLUMN,
    TARGET_COLUMN,
    *NUMERIC_FEATURES[:-1],
    *CATEGORICAL_FEATURES[:-1],
)


def load_telco_data(path: str | Path) -> pd.DataFrame:
    """Load the raw Telco churn CSV without changing its values."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _normalize_senior_citizen(series: pd.Series) -> pd.Series:
    """Return the binary senior flag as readable Yes/No categories."""
    mapping = {0: "No", 1: "Yes", "0": "No", "1": "Yes", "No": "No", "Yes": "Yes"}
    normalized = series.map(mapping)
    if normalized.isna().any():
        invalid = sorted(series.loc[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unexpected SeniorCitizen values: {invalid}")
    return normalized


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric types and add tenure-band and service-count features."""
    _require_columns(frame, REQUIRED_COLUMNS)
    engineered = frame.copy()

    engineered["TotalCharges"] = pd.to_numeric(
        engineered["TotalCharges"], errors="coerce"
    )
    engineered["SeniorCitizen"] = _normalize_senior_citizen(
        engineered["SeniorCitizen"]
    )
    engineered["tenure_group"] = pd.cut(
        engineered["tenure"],
        bins=[-1, 12, 24, 48, 60, float("inf")],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-60 months", "61+ months"],
    ).astype("object")
    engineered["num_services"] = (
        engineered.loc[:, SERVICE_COLUMNS].eq("Yes").sum(axis=1).astype("int64")
    )
    return engineered


def prepare_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Engineer predictors, remove the identifier, and encode Churn as 0/1."""
    engineered = engineer_features(frame)
    target = engineered[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if target.isna().any():
        invalid = sorted(
            engineered.loc[target.isna(), TARGET_COLUMN].astype(str).unique().tolist()
        )
        raise ValueError(f"Unexpected Churn values: {invalid}")

    features = engineered.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    expected_features = set(NUMERIC_FEATURES).union(CATEGORICAL_FEATURES)
    unexpected = sorted(set(features.columns).difference(expected_features))
    missing = sorted(expected_features.difference(features.columns))
    if unexpected or missing:
        raise ValueError(
            f"Feature contract mismatch. Missing: {missing}; unexpected: {unexpected}"
        )
    return features, target.astype("int8").rename(TARGET_COLUMN)


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train/test split."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(features) != len(target):
        raise ValueError("Features and target must contain the same number of rows")
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build numeric and categorical transformations without fitting them."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return transformed feature names from a fitted preprocessor."""
    return preprocessor.get_feature_names_out().tolist()
