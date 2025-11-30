# agentic_energy/forecast_mcp/feature_engineering.py
from typing import List

import numpy as np
import pandas as pd
import torch

from agentic_energy.schemas import ForecastFeatures  # your shared pydantic schemas
from .models import FEATURE_ORDER


def features_to_dataframe(features: List[ForecastFeatures]) -> pd.DataFrame:
    """Convert ForecastFeatures list to a DataFrame in the correct FEATURE_ORDER."""
    rows = []
    for feat in features:
        row = {
            "temperature": feat.temperature,
            "radiation_direct_horizontal": feat.radiation_direct_horizontal,
            "radiation_diffuse_horizontal": feat.radiation_diffuse_horizontal,
            "is_weekday": feat.is_weekday,
            "is_holiday": feat.is_holiday,
        }
        # month dummies: month_1 .. month_11 (Month 12 → all zeros)
        for m in range(1, 12):
            row[f"month_{m}"] = 1 if feat.month == m else 0
        rows.append(row)

    df = pd.DataFrame(rows)
    return df[FEATURE_ORDER]


def predict_rf(rf_dict: dict, features_df: pd.DataFrame, hours: List[int]) -> np.ndarray:
    """Make per-hour forecasts using RF models stored in rf_dict['models'][hour]."""
    preds = []
    feature_names = rf_dict["metadata"]["features"]

    for i, hour in enumerate(hours):
        if hour not in rf_dict["models"]:
            raise ValueError(f"No RF model available for hour {hour}")
        model = rf_dict["models"][hour]
        X = features_df.iloc[i:i+1][feature_names].values
        preds.append(model.predict(X)[0])

    return np.array(preds)


def predict_lstm(lstm_dict: dict, features_df: pd.DataFrame) -> np.ndarray:
    """Make forecasts using an LSTM model + scalers from lstm_dict."""
    model = lstm_dict["model"]
    scaler_X = lstm_dict["scaler_X"]
    scaler_y = lstm_dict["scaler_y"]
    seq_length = lstm_dict["seq_length"]
    feature_names = lstm_dict["features"]

    X = features_df[feature_names].values
    X_scaled = scaler_X.transform(X)

    sequences = []
    for i in range(len(X_scaled)):
        if i < seq_length - 1:
            padding = np.zeros((seq_length - i - 1, X_scaled.shape[1]))
            seq = np.vstack([padding, X_scaled[: i + 1]])
        else:
            seq = X_scaled[i - seq_length + 1 : i + 1]
        sequences.append(seq)

    sequences = np.array(sequences)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(sequences)
        preds_scaled = model(X_tensor).numpy().flatten()

    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    return preds
