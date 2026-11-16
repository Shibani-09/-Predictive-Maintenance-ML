
from io import BytesIO
from functools import lru_cache
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import COLUMNS
from src.feature_engineering import add_change_features, add_rolling_features


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


# --------------------------------------------------
# UPLOADED RAW SENSOR CSV PREDICTIONS
# --------------------------------------------------

RAW_SENSOR_COLUMNS = [
    column for column in COLUMNS if column.startswith("sensor_")
]
MIN_HISTORY_CYCLES = 5


def load_uploaded_sensor_csv(filename: str, content: bytes) -> pd.DataFrame:
    """Read an uploaded CSV containing raw C-MAPSS sensor records."""

    if Path(filename).suffix.lower() != ".csv":
        raise ValueError("Unsupported file format. Please upload a .csv file.")

    if not content:
        raise ValueError("The uploaded CSV file is empty.")

    try:
        return pd.read_csv(BytesIO(content))
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV is empty or has no header.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"The uploaded CSV could not be parsed: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("The uploaded CSV must use UTF-8 text encoding.") from exc
    except OSError as exc:
        raise ValueError(f"The uploaded CSV could not be read: {exc}") from exc


def validate_uploaded_sensor_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize raw C-MAPSS rows before feature engineering."""

    if dataframe.empty:
        raise ValueError("The CSV contains no data rows.")

    expected_columns = list(COLUMNS)
    missing_columns = [
        column for column in expected_columns if column not in dataframe.columns
    ]
    unexpected_columns = [
        str(column)
        for column in dataframe.columns
        if column not in expected_columns
    ]
    if missing_columns or unexpected_columns:
        details = []
        if missing_columns:
            details.append("missing columns: " + ", ".join(missing_columns))
        if unexpected_columns:
            details.append("unexpected columns: " + ", ".join(unexpected_columns))
        raise ValueError(
            "The CSV must use the raw C-MAPSS schema (" + "; ".join(details) + ")."
        )

    validated = dataframe.loc[:, expected_columns].copy()
    missing_values = validated.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        details = ", ".join(
            f"{column} ({count})"
            for column, count in missing_values.items()
        )
        raise ValueError(f"The CSV contains missing values: {details}.")

    invalid_values = []
    for column in expected_columns:
        converted = pd.to_numeric(validated[column], errors="coerce")
        invalid_count = int((converted.isna() & validated[column].notna()).sum())
        if invalid_count:
            invalid_values.append(f"{column} ({invalid_count})")
        validated[column] = converted

    if invalid_values:
        raise ValueError(
            "All C-MAPSS columns must contain numeric values; invalid data in: "
            + ", ".join(invalid_values)
            + "."
        )

    finite_values = np.isfinite(validated.to_numpy(dtype=float))
    if not finite_values.all():
        invalid_columns = [
            column
            for column in expected_columns
            if not np.isfinite(validated[column].to_numpy(dtype=float)).all()
        ]
        raise ValueError(
            "The CSV contains non-finite numeric values in: "
            + ", ".join(invalid_columns)
            + "."
        )

    for column in ("engine_id", "cycle"):
        values = validated[column].to_numpy(dtype=float)
        if ((values <= 0) | (values % 1 != 0)).any():
            raise ValueError(
                f"The {column} column must contain positive whole numbers."
            )

    duplicate_cycles = validated.duplicated(
        subset=["engine_id", "cycle"], keep=False
    )
    if duplicate_cycles.any():
        raise ValueError(
            "The CSV contains duplicate engine_id/cycle records."
        )

    insufficient_history = []
    for engine_id, engine_rows in validated.groupby("engine_id", sort=False):
        cycles = np.sort(engine_rows["cycle"].to_numpy(dtype=int))
        if len(cycles) < MIN_HISTORY_CYCLES:
            insufficient_history.append(str(int(engine_id)))
        elif not np.all(np.diff(cycles[-MIN_HISTORY_CYCLES:]) == 1):
            insufficient_history.append(str(int(engine_id)))

    if insufficient_history:
        raise ValueError(
            "Each engine needs at least five observations with consecutive latest "
            "cycles for the rolling features. Check engine IDs: "
            + ", ".join(insufficient_history)
            + "."
        )

    return validated.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)


def predict_uploaded_engines(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Engineer raw CSV rows and predict RUL at each engine's latest cycle."""

    sensor_data = validate_uploaded_sensor_data(dataframe)
    feature_columns = load_feature_columns()

    if len(feature_columns) != 64 or len(set(feature_columns)) != 64:
        raise ValueError(
            "The saved feature list must contain exactly 64 unique model features."
        )

    active_sensors = [
        column
        for column in RAW_SENSOR_COLUMNS
        if column in feature_columns
    ]
    expected_features = (
        ["cycle", "setting_1", "setting_2", "setting_3"]
        + active_sensors
        + [
            feature
            for sensor in active_sensors
            for feature in (
                f"{sensor}_rolling_mean",
                f"{sensor}_rolling_std",
            )
        ]
        + [f"{sensor}_change" for sensor in active_sensors]
    )
    if feature_columns != expected_features:
        raise ValueError(
            "The saved feature list does not match the existing C-MAPSS "
            "feature-engineering pipeline."
        )

    engineered = add_rolling_features(sensor_data, active_sensors)
    engineered = add_change_features(engineered, active_sensors)

    missing_features = [
        column for column in feature_columns if column not in engineered.columns
    ]
    if missing_features:
        raise ValueError(
            "Feature engineering did not produce required model features: "
            + ", ".join(missing_features)
            + "."
        )

    latest_rows = (
        engineered.sort_values(["engine_id", "cycle"])
        .groupby("engine_id", sort=True)
        .tail(1)
    )

    predictions = []
    for _, row in latest_rows.iterrows():
        features = {
            column: float(row[column])
            for column in feature_columns
        }
        predicted_rul = predict_rul(features)
        if not math.isfinite(predicted_rul):
            raise ValueError(
                f"The model returned a non-finite RUL for engine "
                f"{int(row['engine_id'])}."
            )
        predictions.append(
            {
                "engine_id": int(row["engine_id"]),
                "cycle": int(row["cycle"]),
                "predicted_rul": round(predicted_rul, 2),
            }
        )

    return pd.DataFrame(
        predictions,
        columns=["engine_id", "cycle", "predicted_rul"],
    )
