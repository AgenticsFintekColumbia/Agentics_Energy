import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import holidays
from mcp.server.fastmcp import FastMCP

from agentic_energy.schemas import ForecastRequest, ForecastResponse, ForecastFeatures
from .models import MODELS, FEATURE_ORDER, load_models
from .predictions import features_to_dataframe, predict_rf, predict_lstm

# ---------------------------------------------------------------------
# Path configuration (package-root based, with env overrides)
# ---------------------------------------------------------------------

# tools.py -> forecast_mcp/ -> agentic_energy/
PKG_ROOT = Path(__file__).resolve().parents[1]

# Default locations, overridable via env vars
DATA_DIR = Path(os.getenv("FORECAST_DATA_DIR", PKG_ROOT / "forecast/data"))
MODELS_DIR = Path(os.getenv("FORECAST_MODELS_DIR", PKG_ROOT / "forecast/trained_models"))

DATA_IT_PATH = Path(os.getenv("ITALY_DATA_PATH", DATA_DIR / "data_IT.csv"))

# Initialize MCP server so server.py can import it
mcp = FastMCP("Forecast")


@mcp.tool()
def forecast_predict(args: ForecastRequest) -> ForecastResponse:
    """
    Generate forecasts for energy prices or consumption.

    Args:
      - target: 'prices' or 'consumption'
      - model_type: 'RF' or 'LSTM'
      - features: List[ForecastFeatures]
      - timestamps: Optional[List[str]]

    Returns:
      ForecastResponse with predictions and metadata.
    """
    try:
        if args.target not in {"prices", "consumption"}:
            return ForecastResponse(
                predictions=[],
            )

        if args.model_type not in {"RF", "LSTM"}:
            return ForecastResponse(
                predictions=[],
            )

        model_key = f"{args.model_type.lower()}_{args.target}"
        if MODELS.get(model_key) is None:
            return ForecastResponse(
                predictions=[],
            )

        features_df = features_to_dataframe(args.features)

        if args.model_type == "RF":
            hours = [feat.hour for feat in args.features]
            preds = predict_rf(MODELS[model_key], features_df, hours)
        else:
            preds = predict_lstm(MODELS[model_key], features_df)

        return ForecastResponse(
            predictions=preds.tolist(),
        )

    except Exception as e:
        return ForecastResponse(
            predictions=[],
        )


@mcp.tool()
def forecast_check_models() -> Dict[str, Any]:
    """
    Check which models are loaded and available.
    """
    status: Dict[str, str] = {}
    for key, model in MODELS.items():
        if model is None:
            status[key] = "Not loaded"
        elif key.startswith("rf_"):
            status[key] = f"Loaded ({len(model['models'])} hour models)"
        else:
            status[key] = f"Loaded (seq_length={model['seq_length']})"

    return {
        # "status": "success",
        "models": status,
        "feature_order": FEATURE_ORDER,
    }


@mcp.tool()
def forecast_for_date(date: str, target: str = "prices", model_type: str = "LSTM") -> Dict[str, Any]:
    """
    Get forecast for a specific date from Italy data, along with actuals and error metrics.

    Args:
      - date: 'YYYY-MM-DD'
      - target: 'prices' or 'consumption'
      - model_type: 'RF' or 'LSTM'
    """
    try:
        # -----------------------------------------------------------------
        # Locate data_IT.csv in a path-agnostic way
        # -----------------------------------------------------------------
        if not DATA_IT_PATH.exists():
            return {
                "predictions": [],
                "actual": [],
            }

        df = pd.read_csv(DATA_IT_PATH, parse_dates=["timestamps"])

        timestamps = df["timestamps"]
        df["hour"] = timestamps.dt.hour + 1
        df["month"] = timestamps.dt.month
        df["is_weekday"] = timestamps.dt.dayofweek.isin(range(5)).astype(int)

        it_holidays = holidays.Italy()
        df["is_holiday"] = timestamps.dt.date.map(lambda d: int(d in it_holidays))

        # Filter to the requested date
        target_date = pd.to_datetime(date).date()
        day_mask = timestamps.dt.date == target_date
        day_data = df[day_mask].copy()

        if day_data.empty:
            return {
                "predictions": [],
                "actual": [],
            }

        # Build features
        features: List[ForecastFeatures] = []
        for _, row in day_data.iterrows():
            features.append(
                ForecastFeatures(
                    temperature=float(row["temperature"]),
                    radiation_direct_horizontal=float(row["radiation_direct_horizontal"]),
                    radiation_diffuse_horizontal=float(row["radiation_diffuse_horizontal"]),
                    hour=int(row["hour"]),
                    month=int(row["month"]),
                    is_weekday=int(row["is_weekday"]),
                    is_holiday=int(row["is_holiday"]),
                )
            )

        req = ForecastRequest(
            target=target,
            model_type=model_type,
            features=features,
            timestamps=day_data["timestamps"].astype(str).tolist(),
        )
        resp = forecast_predict(req)

        # if resp.status != "success":
        #     return {
        #         "predictions": [],
        #         "actual": [],
        #     }

        actual_col = "prices" if target == "prices" else "consumption"
        actual_values = day_data[actual_col].tolist()

        preds = np.array(resp.predictions)
        actuals = np.array(actual_values)

        return {
            "date": date,
            "predictions": resp.predictions,
            "actual": actual_values,
        }

    except Exception as e:
        import traceback

        return {
            "predictions": [],
            "actual": [],
        }


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Load models from the package-root-based directory (path-agnostic)
    # -----------------------------------------------------------------
    candidate_dirs = [MODELS_DIR]

    loaded = False
    for d in candidate_dirs:
        d = Path(d)
        if d.exists():
            print(f"📂 Found models directory: {d}", file=sys.stderr)
            load_models(d)
            loaded = True
            break

    if not loaded:
        print(
            "⚠️  No models directory found.\n"
            f"   Expected default: {MODELS_DIR}\n"
            "   Set FORECAST_MODELS_DIR to override.",
            file=sys.stderr,
        )

    # Start MCP server
    print("🚀 Starting Forecast MCP Server...", file=sys.stderr)
    mcp.run(transport="stdio")
