"""Simple, leakage-aware preprocessing for the academic ML experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "server_temperature_data.csv"
TARGET = "CPU_Temperature"
RISK_TARGET = "Temperature_Risk"
BASE_FEATURES = [
    "CPU_Usage", "CPU_Frequency", "Memory_Usage", "Disk_Usage",
    "Process_Count", "Thread_Count", "GPU_Temperature",
]


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load, type-convert, sort, and de-duplicate the source records."""
    frame = pd.read_csv(path)
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    numeric_columns = [column for column in frame.columns if column != "Timestamp"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates().sort_values("Timestamp").reset_index(drop=True)
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add past-only features and an approximately five-minute future target."""
    result = frame.copy().sort_values("Timestamp").reset_index(drop=True)
    if result["Timestamp"].notna().sum() > 1:
        seconds = result["Timestamp"].diff().dt.total_seconds().median()
        rows_for_five_minutes = max(1, int(round(300 / seconds))) if seconds > 0 else 1
    else:
        rows_for_five_minutes = 1

    # Shift first so each rolling value uses only information available before t.
    result["CPU_Temperature_Lag_1"] = result[TARGET].shift(1)
    result["CPU_Temperature_Rolling_Mean_5"] = result[TARGET].shift(1).rolling(5, min_periods=1).mean()
    result["CPU_Usage_Lag_1"] = result["CPU_Usage"].shift(1)
    result["CPU_Temperature_Change"] = result[TARGET].diff().shift(1)
    result["CPU_Usage_Change"] = result["CPU_Usage"].diff().shift(1)
    result["Future_CPU_Temperature"] = result[TARGET].shift(-rows_for_five_minutes)
    result.attrs["five_minute_rows"] = rows_for_five_minutes
    return result


def add_risk_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result[RISK_TARGET] = pd.cut(
        result[TARGET], bins=[-np.inf, 70, 80, np.inf],
        labels=["Safe", "Moderate Risk", "High Risk"], right=False,
    )
    return result


def remove_constant_features(frame: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [column for column in feature_columns if column in frame and frame[column].nunique(dropna=False) > 1]


def fit_input_preprocessor(X_train: pd.DataFrame, scale: str | None = None):
    """Fit imputation and optional scaling on training rows only."""
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(X_train)
    scaler = None
    if scale == "minmax":
        scaler = MinMaxScaler().fit(train_values)
    elif scale == "standard":
        scaler = StandardScaler().fit(train_values)
    return imputer, scaler


def transform_inputs(X: pd.DataFrame, imputer, scaler=None):
    values = imputer.transform(X)
    return scaler.transform(values) if scaler is not None else values


def experiment_frame(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Return rows with a known current target; feature missingness is imputed later."""
    frame = add_features(load_raw_data(path))
    frame = add_risk_labels(frame)
    return frame[frame[TARGET].notna()].reset_index(drop=True)


if __name__ == "__main__":
    data = experiment_frame()
    print(f"Rows after duplicate removal and target filtering: {len(data)}")
    print(f"Source shape: {data.shape}")
    print(data.describe(numeric_only=True).round(2).to_string())
    print(f"Approximate five-minute shift: {data.attrs.get('five_minute_rows', 'see add_features') } rows")
