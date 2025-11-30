# agentic_energy/forecast_mcp/models.py
import pickle
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Global model registry
MODELS: Dict[str, Any] = {
    "rf_prices": None,
    "rf_consumption": None,
    "lstm_prices": None,
    "lstm_consumption": None,
}

# Fixed feature order (for RF/LSTM)
FEATURE_ORDER = [
    "temperature",
    "radiation_direct_horizontal",
    "radiation_diffuse_horizontal",
    "is_weekday",
    "is_holiday",
    "month_1",
    "month_2",
    "month_3",
    "month_4",
    "month_5",
    "month_6",
    "month_7",
    "month_8",
    "month_9",
    "month_10",
    "month_11",
]


def _load_rf_model(models_path: Path, name: str, key: str) -> None:
    try:
        path = models_path / f"{name}.pkl"
        if not path.exists():
            print(f"⚠️  RF model not found: {path}", file=sys.stderr)
            return
        with open(path, "rb") as f:
            MODELS[key] = pickle.load(f)
        print(f"✅ Loaded {name} ({len(MODELS[key]['models'])} hour models)", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error loading {name}: {e}", file=sys.stderr)


def _load_lstm_model(models_path: Path, name: str, key: str) -> None:
    try:
        path = models_path / f"{name}.pkl"
        if not path.exists():
            print(f"⚠️  LSTM model not found: {path}", file=sys.stderr)
            return
        with open(path, "rb") as f:
            lstm_dict = pickle.load(f)

        config = lstm_dict["model_config"]
        model = LSTMModel(
            input_size=config["input_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )
        model.load_state_dict(lstm_dict["model_state_dict"])
        model.eval()

        MODELS[key] = {
            "model": model,
            "scaler_X": lstm_dict["scaler_X"],
            "scaler_y": lstm_dict["scaler_y"],
            "seq_length": lstm_dict["seq_length"],
            "features": lstm_dict["metadata"]["features"],
        }
        print(f"✅ Loaded {name} (seq_length={lstm_dict['seq_length']})", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error loading {name}: {e}", file=sys.stderr)


def load_models(models_dir: str = "trained_models") -> None:
    """Load all RF/LSTM models from the given directory into MODELS."""
    models_path = Path(models_dir)
    print(f"📂 Loading models from: {models_path.absolute()}", file=sys.stderr)

    if not models_path.exists():
        print(f"⚠️  Models directory not found: {models_path}", file=sys.stderr)
        return

    # RF models
    _load_rf_model(models_path, "rf_prices", "rf_prices")
    _load_rf_model(models_path, "rf_consumption", "rf_consumption")

    # LSTM models
    _load_lstm_model(models_path, "lstm_prices", "lstm_prices")
    _load_lstm_model(models_path, "lstm_consumption", "lstm_consumption")

    print("🎯 Model loading complete.\n", file=sys.stderr)
