
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_rul_model.pkl"

FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"

PROCESSED_TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "test_features.csv"
)


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_model():
    """Load the existing Random Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    try:
        return joblib.load(MODEL_PATH)

    except Exception as exc:
        raise RuntimeError(
            f"Could not load Random Forest model: {exc}"
        ) from exc


# --------------------------------------------------
# LOAD FEATURE COLUMNS
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_feature_columns():
    """Load the original model feature columns."""

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Feature columns file not found: {FEATURE_COLUMNS_PATH}"
        )

    try:
        columns = joblib.load(FEATURE_COLUMNS_PATH)

        if hasattr(columns, "tolist"):
            columns = columns.tolist()

        return list(columns)

    except Exception as exc:
        raise RuntimeError(
            f"Could not load feature columns: {exc}"
        ) from exc


# --------------------------------------------------
# LOAD PROCESSED TEST DATASET
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_test_dataset():
    """Load the processed NASA test dataset."""

    if not PROCESSED_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Processed test dataset not found: {PROCESSED_TEST_PATH}"
        )

    try:
        df = pd.read_csv(PROCESSED_TEST_PATH)

    except Exception as exc:
        raise RuntimeError(
            f"Could not read test dataset: {exc}"
        ) from exc

    required_columns = {"engine_id", "cycle"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain engine_id and cycle columns."
        )

    return df


# --------------------------------------------------
# PREDICT RUL
# --------------------------------------------------

def predict_rul(features: dict) -> float:
    """Predict remaining useful life from prepared features."""

    model = load_model()
    feature_columns = load_feature_columns()

    missing_features = [
        column
        for column in feature_columns
        if column not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    try:
        input_data = pd.DataFrame(
            [[features[column] for column in feature_columns]],
            columns=feature_columns
        )

        prediction = model.predict(input_data)[0]

        return max(0.0, float(prediction))

    except Exception as exc:
        raise RuntimeError(
            f"RUL prediction failed: {exc}"
        ) from exc


# --------------------------------------------------
# PREDICT FOR SELECTED ENGINE
# --------------------------------------------------

def predict_engine_rul(engine_id: int) -> dict:
    """Predict RUL using the latest processed row of an engine."""

    df = load_test_dataset()

    feature_columns = load_feature_columns()

    missing_columns = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing model features: {missing_columns}"
        )

    engine_rows = df[df["engine_id"] == engine_id]

    if engine_rows.empty:
        raise ValueError(
            f"Engine ID {engine_id} was not found."
        )

    latest_row = engine_rows.sort_values(
        "cycle"
    ).iloc[-1]

    features = {
        column: float(latest_row[column])
        for column in feature_columns
    }

    predicted_rul = predict_rul(features)

    return {
        "engine_id": int(engine_id),
        "cycle": int(latest_row["cycle"]),
        "predicted_rul": round(predicted_rul, 2),
        "unit": "cycles"
    }


# --------------------------------------------------
# GET AVAILABLE ENGINE IDS
# --------------------------------------------------

def get_available_engines() -> list:
    """Return all available engine IDs."""

    df = load_test_dataset()

    return sorted(
        df["engine_id"].dropna().astype(int).unique().tolist()
    )
