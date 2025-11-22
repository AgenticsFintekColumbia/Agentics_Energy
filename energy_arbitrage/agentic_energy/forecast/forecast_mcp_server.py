
"""
Standalone Forecast MCP Server - FIXED VERSION
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import holidays

# Initialize MCP server
mcp = FastMCP("Forecast")

# Schemas (copied here to avoid package import)
class ForecastFeatures(BaseModel):
    temperature: float
    radiation_direct_horizontal: float
    radiation_diffuse_horizontal: float
    hour: int = Field(ge=1, le=24)
    month: int = Field(ge=1, le=12)
    is_weekday: int = Field(ge=0, le=1)
    is_holiday: int = Field(ge=0, le=1)

class ForecastRequest(BaseModel):
    target: str
    model_type: str
    features: List[ForecastFeatures]
    timestamps: Optional[List[str]] = None

class ForecastResponse(BaseModel):
    status: str
    message: Optional[str] = None
    target: str
    model_type: str
    predictions: List[float]
    timestamps: Optional[List[str]] = None
    num_predictions: int

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Global storage
MODELS = {
    'rf_prices': None,
    'rf_consumption': None,
    'lstm_prices': None,
    'lstm_consumption': None
}

FEATURE_ORDER = [
    'temperature', 'radiation_direct_horizontal', 'radiation_diffuse_horizontal',
    'is_weekday', 'is_holiday',
    'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
    'month_7', 'month_8', 'month_9', 'month_10', 'month_11'
]

def load_models(models_dir: str = "trained_models"):
    """Load all pickle models"""
    models_path = Path(models_dir)
    
    print(f"📂 Loading models from: {models_path.absolute()}", file=sys.stderr)
    
    if not models_path.exists():
        print(f"⚠️  Models directory not found: {models_path}", file=sys.stderr)
        return
    
    # Load RF
    try:
        with open(models_path / "rf_prices.pkl", 'rb') as f:
            MODELS['rf_prices'] = pickle.load(f)
        print(f"✅ Loaded RF prices", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error loading RF prices: {e}", file=sys.stderr)
    
    try:
        with open(models_path / "rf_consumption.pkl", 'rb') as f:
            MODELS['rf_consumption'] = pickle.load(f)
        print(f"✅ Loaded RF consumption", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error loading RF consumption: {e}", file=sys.stderr)
    
    # Load LSTM
    for target in ['prices', 'consumption']:
        try:
            with open(models_path / f"lstm_{target}.pkl", 'rb') as f:
                lstm_dict = pickle.load(f)
            
            config = lstm_dict['model_config']
            model = LSTMModel(config['input_size'], config['hidden_size'], 
                             config['num_layers'], config['dropout'])
            model.load_state_dict(lstm_dict['model_state_dict'])
            model.eval()
            
            MODELS[f'lstm_{target}'] = {
                'model': model,
                'scaler_X': lstm_dict['scaler_X'],
                'scaler_y': lstm_dict['scaler_y'],
                'seq_length': lstm_dict['seq_length'],
                'features': lstm_dict['metadata']['features']
            }
            print(f"✅ Loaded LSTM {target}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error loading LSTM {target}: {e}", file=sys.stderr)

def features_to_dataframe(features: List[ForecastFeatures]) -> pd.DataFrame:
    """Convert features to DataFrame"""
    data = []
    for feat in features:
        row = {
            'temperature': feat.temperature,
            'radiation_direct_horizontal': feat.radiation_direct_horizontal,
            'radiation_diffuse_horizontal': feat.radiation_diffuse_horizontal,
            'is_weekday': feat.is_weekday,
            'is_holiday': feat.is_holiday,
        }
        for m in range(1, 12):
            row[f'month_{m}'] = 1 if feat.month == m else 0
        data.append(row)
    
    df = pd.DataFrame(data)
    return df[FEATURE_ORDER]

def predict_rf(rf_dict: dict, features_df: pd.DataFrame, hours: List[int]) -> np.ndarray:
    """RF predictions"""
    predictions = []
    for i, hour in enumerate(hours):
        model = rf_dict['models'][hour]
        X = features_df.iloc[i:i+1][rf_dict['metadata']['features']].values
        predictions.append(model.predict(X)[0])
    return np.array(predictions)

def predict_lstm(lstm_dict: dict, features_df: pd.DataFrame) -> np.ndarray:
    """LSTM predictions"""
    model = lstm_dict['model']
    scaler_X = lstm_dict['scaler_X']
    scaler_y = lstm_dict['scaler_y']
    seq_length = lstm_dict['seq_length']
    
    X = features_df[lstm_dict['features']].values
    X_scaled = scaler_X.transform(X)
    
    sequences = []
    for i in range(len(X_scaled)):
        if i < seq_length - 1:
            padding = np.zeros((seq_length - i - 1, X_scaled.shape[1]))
            seq = np.vstack([padding, X_scaled[:i+1]])
        else:
            seq = X_scaled[i-seq_length+1:i+1]
        sequences.append(seq)
    
    sequences = np.array(sequences)
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(sequences)
        predictions_scaled = model(X_tensor).numpy().flatten()
    
    return scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

@mcp.tool()
def forecast_predict(args: ForecastRequest) -> ForecastResponse:
    """Generate forecasts"""
    try:
        model_key = f"{args.model_type.lower()}_{args.target}"
        if MODELS[model_key] is None:
            return ForecastResponse(
                status="error",
                message=f"Model not loaded: {model_key}",
                target=args.target,
                model_type=args.model_type,
                predictions=[],
                num_predictions=0
            )
        
        features_df = features_to_dataframe(args.features)
        
        if args.model_type == 'RF':
            hours = [feat.hour for feat in args.features]
            predictions = predict_rf(MODELS[model_key], features_df, hours)
        else:
            predictions = predict_lstm(MODELS[model_key], features_df)
        
        return ForecastResponse(
            status="success",
            target=args.target,
            model_type=args.model_type,
            predictions=predictions.tolist(),
            timestamps=args.timestamps,
            num_predictions=len(predictions)
        )
    except Exception as e:
        return ForecastResponse(
            status="error",
            message=str(e),
            target=args.target,
            model_type=args.model_type,
            predictions=[],
            num_predictions=0
        )

@mcp.tool()
def forecast_check_models() -> dict:
    """Check which models are loaded and available"""
    status = {}
    for key, model in MODELS.items():
        if model is None:
            status[key] = "Not loaded"
        elif key.startswith('rf_'):
            status[key] = f"Loaded ({len(model['models'])} hour models)"
        else:  # LSTM
            status[key] = f"Loaded (seq_length={model['seq_length']})"
    
    return {
        "status": "success",
        "models": status,
        "feature_order": FEATURE_ORDER
    }

@mcp.tool()
def forecast_for_date(date: str, target: str = "prices", model_type: str = "LSTM") -> dict:
    """Forecast for a specific date with actual data comparison"""
    try:
        # Find data file
        possible_paths = [
            "agentic_energy/data/data_IT.csv",
            "./agentic_energy/data/data_IT.csv",
            "C:/Users/16467/OneDrive/Desktop/Columbia/Agentics/Another/Agentics_for_EnergyArbitrage_Battery/energy_arbitrage/agentic_energy/data/data_IT.csv",
            "data_IT.csv",
            "../data_IT.csv",
            "../../data_IT.csv"
        ]
        
        data_path = None
        for path in possible_paths:
            if Path(path).exists():
                data_path = path
                break
        
        if data_path is None:
            return {
                "status": "error",
                "message": "data_IT.csv not found. Searched: " + str(possible_paths)
            }
        
        # Load data
        df = pd.read_csv(data_path, parse_dates=['timestamps'])
        df['date'] = pd.to_datetime(df['timestamps']).dt.date
        
        # Filter for requested date
        target_date = pd.to_datetime(date).date()
        day_data = df[df['date'] == target_date].copy()
        
        if len(day_data) == 0:
            return {
                "status": "error",
                "message": f"No data found for {date}"
            }
        
        # Prepare features
        day_data['hour'] = pd.to_datetime(day_data['timestamps']).dt.hour + 1
        day_data['month'] = pd.to_datetime(day_data['timestamps']).dt.month
        day_data['is_weekday'] = pd.to_datetime(day_data['timestamps']).dt.dayofweek < 5
        
        it_holidays = holidays.Italy()
        day_data['is_holiday'] = day_data['timestamps'].apply(
            lambda x: pd.to_datetime(x).date() in it_holidays
        )
        
        # Create features
        features = []
        for _, row in day_data.iterrows():
            feat = ForecastFeatures(
                # temperature=row['t2m [C]'],
                # radiation_direct_horizontal=row['ssrd [Wh/m2]'],
                # radiation_diffuse_horizontal=row['fdir [Wh/m2]'],
                temperature=row['temperature'],
                radiation_direct_horizontal=row['radiation_direct_horizontal'],
                radiation_diffuse_horizontal=row['radiation_diffuse_horizontal'],
                hour=int(row['hour']),
                month=int(row['month']),
                is_weekday=int(row['is_weekday']),
                is_holiday=int(row['is_holiday'])
            )
            features.append(feat)
        
        # Make prediction
        req = ForecastRequest(
            target=target,
            model_type=model_type,
            features=features,
            timestamps=day_data['timestamps'].astype(str).tolist()
        )
        
        response = forecast_predict(req)
        
        if response.status != "success":
            return {
                "status": "error",
                "message": response.message
            }
        
        # Get actual values
        # actual_col = 'PUN [EUR/MWh]' if target == 'prices' else 'Load [GWh]'
        actual_col = 'prices' if target == 'prices' else 'consumption'
        actual_values = day_data[actual_col].tolist()
        
        # Calculate metrics
        predictions = np.array(response.predictions)
        actuals = np.array(actual_values)
        
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        return {
            "status": "success",
            "date": date,
            "target": target,
            "model_type": model_type,
            "predictions": response.predictions,
            "actual": actual_values,
            "mae": float(mae),
            "rmse": float(rmse),
            "num_predictions": len(predictions)
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Error: {str(e)}\n{traceback.format_exc()}"
        }

# ============================================================================
# MAIN - THIS WAS COMMENTED OUT IN YOUR VERSION!
# ============================================================================

if __name__ == "__main__":
    # Load models at startup
    models_dir = os.getenv("FORECAST_MODELS_DIR", "trained_models")
    
    # Try multiple possible paths
    possible_paths = [
        models_dir,
        "trained_new",
        "agentic_energy/trained_models",
        "agentic_energy/trained_new",
        "./agentic_energy/trained_new",
        "C:/Users/16467/OneDrive/Desktop/Columbia/Agentics/Another/Agentics_for_EnergyArbitrage_Battery/energy_arbitrage/agentic_energy/trained_models",
        "../trained_models",
        "../trained_new"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"📂 Found models directory: {path}", file=sys.stderr)
            load_models(path)
            break
    else:
        print("⚠️  No models directory found. Please set FORECAST_MODELS_DIR environment variable.", file=sys.stderr)
        print(f"   Searched paths: {possible_paths}", file=sys.stderr)
    
    # Start MCP server - THIS IS CRITICAL!
    print("🚀 Starting Forecast MCP Server...", file=sys.stderr)
    mcp.run(transport="stdio")



# """
# Standalone Forecast MCP Server - no package imports
# """
# import os
# import sys
# import pickle
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from pathlib import Path
# from typing import List, Optional
# from mcp.server.fastmcp import FastMCP
# from pydantic import BaseModel, Field
# import holidays

# # Initialize MCP server
# mcp = FastMCP("Forecast")

# # Schemas (copied here to avoid package import)
# class ForecastFeatures(BaseModel):
#     temperature: float
#     radiation_direct_horizontal: float
#     radiation_diffuse_horizontal: float
#     hour: int = Field(ge=1, le=24)
#     month: int = Field(ge=1, le=12)
#     is_weekday: int = Field(ge=0, le=1)
#     is_holiday: int = Field(ge=0, le=1)

# class ForecastRequest(BaseModel):
#     target: str
#     model_type: str
#     features: List[ForecastFeatures]
#     timestamps: Optional[List[str]] = None

# class ForecastResponse(BaseModel):
#     status: str
#     message: Optional[str] = None
#     target: str
#     model_type: str
#     predictions: List[float]
#     timestamps: Optional[List[str]] = None
#     num_predictions: int

# # LSTM Model
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
#                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
#         self.fc = nn.Linear(hidden_size, 1)
        
#     def forward(self, x):
#         batch_size = x.size(0)
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out

# # Global storage
# MODELS = {
#     'rf_prices': None,
#     'rf_consumption': None,
#     'lstm_prices': None,
#     'lstm_consumption': None
# }

# FEATURE_ORDER = [
#     'temperature', 'radiation_direct_horizontal', 'radiation_diffuse_horizontal',
#     'is_weekday', 'is_holiday',
#     'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
#     'month_7', 'month_8', 'month_9', 'month_10', 'month_11'
# ]

# def load_models(models_dir: str = "trained_models"):
#     """Load all pickle models"""
#     models_path = Path(models_dir)
    
#     print(f"📂 Loading models from: {models_path.absolute()}")
    
#     if not models_path.exists():
#         print(f"⚠️  Models directory not found: {models_path}")
#         return
    
#     # Load RF
#     try:
#         with open(models_path / "rf_prices.pkl", 'rb') as f:
#             MODELS['rf_prices'] = pickle.load(f)
#         print(f"✅ Loaded RF prices")
#     except Exception as e:
#         print(f"❌ Error loading RF prices: {e}")
    
#     try:
#         with open(models_path / "rf_consumption.pkl", 'rb') as f:
#             MODELS['rf_consumption'] = pickle.load(f)
#         print(f"✅ Loaded RF consumption")
#     except Exception as e:
#         print(f"❌ Error loading RF consumption: {e}")
    
#     # Load LSTM
#     for target in ['prices', 'consumption']:
#         try:
#             with open(models_path / f"lstm_{target}.pkl", 'rb') as f:
#                 lstm_dict = pickle.load(f)
            
#             config = lstm_dict['model_config']
#             model = LSTMModel(config['input_size'], config['hidden_size'], 
#                              config['num_layers'], config['dropout'])
#             model.load_state_dict(lstm_dict['model_state_dict'])
#             model.eval()
            
#             MODELS[f'lstm_{target}'] = {
#                 'model': model,
#                 'scaler_X': lstm_dict['scaler_X'],
#                 'scaler_y': lstm_dict['scaler_y'],
#                 'seq_length': lstm_dict['seq_length'],
#                 'features': lstm_dict['metadata']['features']
#             }
#             print(f"✅ Loaded LSTM {target}")
#         except Exception as e:
#             print(f"❌ Error loading LSTM {target}: {e}")

# def features_to_dataframe(features: List[ForecastFeatures]) -> pd.DataFrame:
#     """Convert features to DataFrame"""
#     data = []
#     for feat in features:
#         row = {
#             'temperature': feat.temperature,
#             'radiation_direct_horizontal': feat.radiation_direct_horizontal,
#             'radiation_diffuse_horizontal': feat.radiation_diffuse_horizontal,
#             'is_weekday': feat.is_weekday,
#             'is_holiday': feat.is_holiday,
#         }
#         for m in range(1, 12):
#             row[f'month_{m}'] = 1 if feat.month == m else 0
#         data.append(row)
    
#     df = pd.DataFrame(data)
#     return df[FEATURE_ORDER]

# def predict_rf(rf_dict: dict, features_df: pd.DataFrame, hours: List[int]) -> np.ndarray:
#     """RF predictions"""
#     predictions = []
#     for i, hour in enumerate(hours):
#         model = rf_dict['models'][hour]
#         X = features_df.iloc[i:i+1][rf_dict['metadata']['features']].values
#         predictions.append(model.predict(X)[0])
#     return np.array(predictions)

# def predict_lstm(lstm_dict: dict, features_df: pd.DataFrame) -> np.ndarray:
#     """LSTM predictions"""
#     model = lstm_dict['model']
#     scaler_X = lstm_dict['scaler_X']
#     scaler_y = lstm_dict['scaler_y']
#     seq_length = lstm_dict['seq_length']
    
#     X = features_df[lstm_dict['features']].values
#     X_scaled = scaler_X.transform(X)
    
#     sequences = []
#     for i in range(len(X_scaled)):
#         if i < seq_length - 1:
#             padding = np.zeros((seq_length - i - 1, X_scaled.shape[1]))
#             seq = np.vstack([padding, X_scaled[:i+1]])
#         else:
#             seq = X_scaled[i-seq_length+1:i+1]
#         sequences.append(seq)
    
#     sequences = np.array(sequences)
    
#     model.eval()
#     with torch.no_grad():
#         X_tensor = torch.FloatTensor(sequences)
#         predictions_scaled = model(X_tensor).numpy().flatten()
    
#     return scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

# @mcp.tool()
# def forecast_predict(args: ForecastRequest) -> ForecastResponse:
#     """Generate forecasts"""
#     try:
#         model_key = f"{args.model_type.lower()}_{args.target}"
#         if MODELS[model_key] is None:
#             return ForecastResponse(
#                 status="error",
#                 message=f"Model not loaded: {model_key}",
#                 target=args.target,
#                 model_type=args.model_type,
#                 predictions=[],
#                 num_predictions=0
#             )
        
#         features_df = features_to_dataframe(args.features)
        
#         if args.model_type == 'RF':
#             hours = [feat.hour for feat in args.features]
#             predictions = predict_rf(MODELS[model_key], features_df, hours)
#         else:
#             predictions = predict_lstm(MODELS[model_key], features_df)
        
#         return ForecastResponse(
#             status="success",
#             target=args.target,
#             model_type=args.model_type,
#             predictions=predictions.tolist(),
#             timestamps=args.timestamps,
#             num_predictions=len(predictions)
#         )
#     except Exception as e:
#         return ForecastResponse(
#             status="error",
#             message=str(e),
#             target=args.target,
#             model_type=args.model_type,
#             predictions=[],
#             num_predictions=0
#         )

# @mcp.tool()
# def forecast_for_date(date: str, target: str = "prices", model_type: str = "LSTM") -> dict:
#     """Forecast for a specific date"""
#     try:
#         # Find data file
#         possible_paths = [
#             "agentic_energy/data/data_IT.csv",
#             "data_IT.csv",
#             r"C:\Users\16467\OneDrive\Desktop\Columbia\Agentics\Another\Agentics_for_EnergyArbitrage_Battery\energy_arbitrage\agentic_energy\data\data_IT.csv"
#         ]
        
#         df = None
#         for path in possible_paths:
#             try:
#                 df = pd.read_csv(path)
#                 break
#             except:
#                 continue
        
#         if df is None:
#             return {"status": "error", "message": "Could not find data_IT.csv", "predictions": [], "actual": []}
        
#         # Process data
#         timestamps = pd.to_datetime(df['timestamps'])
#         df['hour'] = timestamps.dt.hour + 1
#         df['month'] = timestamps.dt.month
#         df['is_weekday'] = timestamps.dt.dayofweek.isin([0, 1, 2, 3, 4]).astype(int)
#         it_holidays = holidays.IT(years=[2018, 2019])
#         df['is_holiday'] = timestamps.apply(lambda x: 1 if x.date() in it_holidays else 0)
        
#         # Get day data
#         date_obj = pd.to_datetime(date)
#         day_data = df[timestamps.dt.date == date_obj.date()].copy()
        
#         if len(day_data) == 0:
#             return {"status": "error", "message": f"No data for {date}", "predictions": [], "actual": []}
        
#         # Build features
#         features = []
#         for idx, row in day_data.iterrows():
#             features.append(ForecastFeatures(
#                 temperature=float(row['temperature']),
#                 radiation_direct_horizontal=float(row['radiation_direct_horizontal']),
#                 radiation_diffuse_horizontal=float(row['radiation_diffuse_horizontal']),
#                 hour=int(row['hour']),
#                 month=int(row['month']),
#                 is_weekday=int(row['is_weekday']),
#                 is_holiday=int(row['is_holiday'])
#             ))
        
#         request = ForecastRequest(
#             target=target,
#             model_type=model_type,
#             features=features,
#             timestamps=day_data['timestamps'].tolist()
#         )
        
#         response = forecast_predict(request)
#         actual = day_data[target].tolist()
        
#         mae = float(np.mean(np.abs(np.array(response.predictions) - np.array(actual))))
#         rmse = float(np.sqrt(np.mean((np.array(response.predictions) - np.array(actual))**2)))
        
#         return {
#             "status": "success",
#             "date": date,
#             "target": target,
#             "model_type": model_type,
#             "predictions": response.predictions,
#             "actual": actual,
#             "timestamps": response.timestamps,
#             "mae": mae,
#             "rmse": rmse,
#             "num_hours": len(response.predictions)
#         }
#     except Exception as e:
#         return {"status": "error", "message": str(e), "predictions": [], "actual": []}

# if __name__ == "__main__":
#     # Load models at startup
#     possible_dirs = ["trained_models", "agentic_energy/trained_models", "../trained_models"]
#     for d in possible_dirs:
#         if Path(d).exists():
#             load_models(d)
#             break
    
#     print("🚀 Starting Forecast MCP Server...")
#     mcp.run(transport="stdio")



# """
# Forecast MCP Server - Serves price and consumption forecasting tools
# Loads pre-trained RF and LSTM models from pickle files
# """
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from pathlib import Path
# from typing import List, Optional
# from mcp.server.fastmcp import FastMCP

# from agentic_energy.schemas import ForecastRequest, ForecastResponse, ForecastFeatures

# # Initialize MCP server
# mcp = FastMCP("Forecast")

# # ============================================================================
# # LSTM MODEL DEFINITION
# # ============================================================================

# class LSTMModel(nn.Module):
#     """LSTM model for time series forecasting"""
#     def __init__(self, input_size, hidden_size, num_layers, dropout):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         self.lstm = nn.LSTM(
#             input_size, hidden_size, num_layers,
#             batch_first=True, 
#             dropout=dropout if num_layers > 1 else 0
#         )
#         self.fc = nn.Linear(hidden_size, 1)
        
#     def forward(self, x):
#         batch_size = x.size(0)
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out

# # ============================================================================
# # GLOBAL MODEL STORAGE
# # ============================================================================

# MODELS = {
#     'rf_prices': None,
#     'rf_consumption': None,
#     'lstm_prices': None,
#     'lstm_consumption': None
# }

# FEATURE_ORDER = [
#     'temperature',
#     'radiation_direct_horizontal',
#     'radiation_diffuse_horizontal',
#     'is_weekday',
#     'is_holiday',
#     'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
#     'month_7', 'month_8', 'month_9', 'month_10', 'month_11'
# ]

# # ============================================================================
# # MODEL LOADING FUNCTIONS
# # ============================================================================

# def load_models(models_dir: str = "trained_models"):
#     """Load all pickle models at server startup"""
#     models_path = Path(models_dir)
    
#     print(f"📂 Loading models from: {models_path.absolute()}")
    
#     # Check if directory exists
#     if not models_path.exists():
#         print(f"⚠️  Models directory not found: {models_path}")
#         print(f"   Please place pickle files in: {models_path.absolute()}")
#         return
    
#     # Load Random Forest models
#     try:
#         rf_prices_path = models_path / "rf_prices.pkl"
#         if rf_prices_path.exists():
#             with open(rf_prices_path, 'rb') as f:
#                 MODELS['rf_prices'] = pickle.load(f)
#             print(f"✅ Loaded RF prices model ({len(MODELS['rf_prices']['models'])} hour models)")
#         else:
#             print(f"⚠️  RF prices model not found: {rf_prices_path}")
#     except Exception as e:
#         print(f"❌ Error loading RF prices: {e}")
    
#     try:
#         rf_consumption_path = models_path / "rf_consumption.pkl"
#         if rf_consumption_path.exists():
#             with open(rf_consumption_path, 'rb') as f:
#                 MODELS['rf_consumption'] = pickle.load(f)
#             print(f"✅ Loaded RF consumption model ({len(MODELS['rf_consumption']['models'])} hour models)")
#         else:
#             print(f"⚠️  RF consumption model not found: {rf_consumption_path}")
#     except Exception as e:
#         print(f"❌ Error loading RF consumption: {e}")
    
#     # Load LSTM models
#     try:
#         lstm_prices_path = models_path / "lstm_prices.pkl"
#         if lstm_prices_path.exists():
#             with open(lstm_prices_path, 'rb') as f:
#                 lstm_dict = pickle.load(f)
            
#             # Reconstruct LSTM model
#             config = lstm_dict['model_config']
#             model = LSTMModel(
#                 input_size=config['input_size'],
#                 hidden_size=config['hidden_size'],
#                 num_layers=config['num_layers'],
#                 dropout=config['dropout']
#             )
#             model.load_state_dict(lstm_dict['model_state_dict'])
#             model.eval()
            
#             MODELS['lstm_prices'] = {
#                 'model': model,
#                 'scaler_X': lstm_dict['scaler_X'],
#                 'scaler_y': lstm_dict['scaler_y'],
#                 'seq_length': lstm_dict['seq_length'],
#                 'features': lstm_dict['metadata']['features']
#             }
#             print(f"✅ Loaded LSTM prices model (seq_length={lstm_dict['seq_length']})")
#         else:
#             print(f"⚠️  LSTM prices model not found: {lstm_prices_path}")
#     except Exception as e:
#         print(f"❌ Error loading LSTM prices: {e}")
    
#     try:
#         lstm_consumption_path = models_path / "lstm_consumption.pkl"
#         if lstm_consumption_path.exists():
#             with open(lstm_consumption_path, 'rb') as f:
#                 lstm_dict = pickle.load(f)
            
#             # Reconstruct LSTM model
#             config = lstm_dict['model_config']
#             model = LSTMModel(
#                 input_size=config['input_size'],
#                 hidden_size=config['hidden_size'],
#                 num_layers=config['num_layers'],
#                 dropout=config['dropout']
#             )
#             model.load_state_dict(lstm_dict['model_state_dict'])
#             model.eval()
            
#             MODELS['lstm_consumption'] = {
#                 'model': model,
#                 'scaler_X': lstm_dict['scaler_X'],
#                 'scaler_y': lstm_dict['scaler_y'],
#                 'seq_length': lstm_dict['seq_length'],
#                 'features': lstm_dict['metadata']['features']
#             }
#             print(f"✅ Loaded LSTM consumption model (seq_length={lstm_dict['seq_length']})")
#         else:
#             print(f"⚠️  LSTM consumption model not found: {lstm_consumption_path}")
#     except Exception as e:
#         print(f"❌ Error loading LSTM consumption: {e}")
    
#     print("🎯 Model loading complete!\n")

# # ============================================================================
# # FEATURE PROCESSING
# # ============================================================================

# def features_to_dataframe(features: List[ForecastFeatures]) -> pd.DataFrame:
#     """Convert ForecastFeatures list to DataFrame with proper feature encoding"""
#     data = []
    
#     for feat in features:
#         row = {
#             'temperature': feat.temperature,
#             'radiation_direct_horizontal': feat.radiation_direct_horizontal,
#             'radiation_diffuse_horizontal': feat.radiation_diffuse_horizontal,
#             'is_weekday': feat.is_weekday,
#             'is_holiday': feat.is_holiday,
#         }
        
#         # Add month dummy variables (month_1 through month_11)
#         for m in range(1, 12):
#             row[f'month_{m}'] = 1 if feat.month == m else 0
        
#         data.append(row)
    
#     df = pd.DataFrame(data)
    
#     # Ensure feature order matches training
#     df = df[FEATURE_ORDER]
    
#     return df

# # ============================================================================
# # FORECASTING FUNCTIONS
# # ============================================================================

# def predict_rf(rf_dict: dict, features_df: pd.DataFrame, hours: List[int]) -> np.ndarray:
#     """Make predictions using Random Forest models"""
#     predictions = []
    
#     for i, hour in enumerate(hours):
#         if hour not in rf_dict['models']:
#             raise ValueError(f"No RF model available for hour {hour}")
        
#         model = rf_dict['models'][hour]
#         feature_names = rf_dict['metadata']['features']
        
#         # Get features for this row
#         X = features_df.iloc[i:i+1][feature_names].values
#         pred = model.predict(X)[0]
#         predictions.append(pred)
    
#     return np.array(predictions)


# def predict_lstm(lstm_dict: dict, features_df: pd.DataFrame) -> np.ndarray:
#     """Make predictions using LSTM model"""
#     model = lstm_dict['model']
#     scaler_X = lstm_dict['scaler_X']
#     scaler_y = lstm_dict['scaler_y']
#     seq_length = lstm_dict['seq_length']
#     feature_names = lstm_dict['features']
    
#     # Get features in correct order
#     X = features_df[feature_names].values
    
#     # Scale features
#     X_scaled = scaler_X.transform(X)
    
#     # Build sequences (pad early points)
#     sequences = []
#     for i in range(len(X_scaled)):
#         if i < seq_length - 1:
#             # Pad with zeros for early points
#             padding = np.zeros((seq_length - i - 1, X_scaled.shape[1]))
#             seq = np.vstack([padding, X_scaled[:i+1]])
#         else:
#             seq = X_scaled[i-seq_length+1:i+1]
#         sequences.append(seq)
    
#     sequences = np.array(sequences)
    
#     # Make predictions
#     model.eval()
#     with torch.no_grad():
#         X_tensor = torch.FloatTensor(sequences)
#         predictions_scaled = model(X_tensor).numpy().flatten()
    
#     # Inverse transform
#     predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
#     return predictions

# # ============================================================================
# # MCP TOOLS
# # ============================================================================

# @mcp.tool()
# def forecast_predict(args: ForecastRequest) -> ForecastResponse:
#     """
#     Generate forecasts for energy prices or consumption.
    
#     Args:
#         args: ForecastRequest containing:
#             - target: 'prices' or 'consumption'
#             - model_type: 'RF' or 'LSTM'
#             - features: List of feature records
#             - timestamps: Optional list of timestamps
    
#     Returns:
#         ForecastResponse with predictions and metadata
#     """
#     try:
#         # Validate target
#         if args.target not in ['prices', 'consumption']:
#             return ForecastResponse(
#                 status="error",
#                 message=f"Invalid target: {args.target}. Must be 'prices' or 'consumption'",
#                 target=args.target,
#                 model_type=args.model_type,
#                 predictions=[],
#                 num_predictions=0
#             )
        
#         # Validate model type
#         if args.model_type not in ['RF', 'LSTM']:
#             return ForecastResponse(
#                 status="error",
#                 message=f"Invalid model_type: {args.model_type}. Must be 'RF' or 'LSTM'",
#                 target=args.target,
#                 model_type=args.model_type,
#                 predictions=[],
#                 num_predictions=0
#             )
        
#         # Check if model is loaded
#         model_key = f"{args.model_type.lower()}_{args.target}"
#         if MODELS[model_key] is None:
#             return ForecastResponse(
#                 status="error",
#                 message=f"Model not loaded: {model_key}. Please check models directory.",
#                 target=args.target,
#                 model_type=args.model_type,
#                 predictions=[],
#                 num_predictions=0
#             )
        
#         # Convert features to DataFrame
#         features_df = features_to_dataframe(args.features)
        
#         # Make predictions based on model type
#         if args.model_type == 'RF':
#             # Extract hours for RF predictions
#             hours = [feat.hour for feat in args.features]
#             predictions = predict_rf(MODELS[model_key], features_df, hours)
#         else:  # LSTM
#             predictions = predict_lstm(MODELS[model_key], features_df)
        
#         return ForecastResponse(
#             status="success",
#             target=args.target,
#             model_type=args.model_type,
#             predictions=predictions.tolist(),
#             timestamps=args.timestamps,
#             num_predictions=len(predictions)
#         )
        
#     except Exception as e:
#         return ForecastResponse(
#             status="error",
#             message=f"Forecast error: {str(e)}",
#             target=args.target,
#             model_type=args.model_type,
#             predictions=[],
#             num_predictions=0
#         )


# @mcp.tool()
# def forecast_check_models() -> dict:
#     """
#     Check which models are loaded and available.
    
#     Returns:
#         Dictionary showing model availability status
#     """
#     status = {}
#     for key, model in MODELS.items():
#         if model is None:
#             status[key] = "Not loaded"
#         elif key.startswith('rf_'):
#             status[key] = f"Loaded ({len(model['models'])} hour models)"
#         else:  # LSTM
#             status[key] = f"Loaded (seq_length={model['seq_length']})"
    
#     return {
#         "status": "success",
#         "models": status,
#         "feature_order": FEATURE_ORDER
#     }


# @mcp.tool()
# def forecast_for_date(date: str, target: str = "prices", model_type: str = "LSTM") -> dict:
#     """
#     Get forecast for a specific date from Italy data.
    
#     Args:
#         date: Date string 'YYYY-MM-DD' (e.g., '2019-06-15')
#         target: 'prices' or 'consumption'
#         model_type: 'RF' or 'LSTM'
    
#     Returns:
#         Dictionary with predictions, actual values, and metrics
#     """
#     import holidays
    
#     try:
#         # Load data
#         data_path = os.getenv("ITALY_DATA_PATH", "agentic_energy/data/data_IT.csv")
        
#         # Try multiple possible paths
#         possible_paths = [
#             data_path,
#             "data_IT.csv",
#             "../data_IT.csv",
#             "agentic_energy/data/data_IT.csv",
#             r"C:\Users\16467\OneDrive\Desktop\Columbia\Agentics\Another\Agentics_for_EnergyArbitrage_Battery\energy_arbitrage\agentic_energy\data\data_IT.csv"
#         ]
        
#         df = None
#         for path in possible_paths:
#             try:
#                 df = pd.read_csv(path)
#                 break
#             except:
#                 continue
        
#         if df is None:
#             return {
#                 "status": "error",
#                 "message": "Could not find data_IT.csv. Set ITALY_DATA_PATH environment variable.",
#                 "predictions": [],
#                 "actual": []
#             }
        
#         # Process temporal features
#         timestamps = pd.to_datetime(df['timestamps'])
#         df['hour'] = timestamps.dt.hour + 1  # Hour 1-24
#         df['month'] = timestamps.dt.month
#         df['is_weekday'] = timestamps.dt.dayofweek.isin([0, 1, 2, 3, 4]).astype(int)
        
#         # Add holidays
#         it_holidays = holidays.IT(years=[2018, 2019])
#         df['is_holiday'] = timestamps.apply(lambda x: 1 if x.date() in it_holidays else 0)
        
#         # Get data for this date
#         date_obj = pd.to_datetime(date)
#         day_data = df[timestamps.dt.date == date_obj.date()].copy()
        
#         if len(day_data) == 0:
#             return {
#                 "status": "error",
#                 "message": f"No data found for {date}",
#                 "predictions": [],
#                 "actual": []
#             }
        
#         # Build features
#         features = []
#         for idx, row in day_data.iterrows():
#             features.append(ForecastFeatures(
#                 temperature=float(row['temperature']),
#                 radiation_direct_horizontal=float(row['radiation_direct_horizontal']),
#                 radiation_diffuse_horizontal=float(row['radiation_diffuse_horizontal']),
#                 hour=int(row['hour']),
#                 month=int(row['month']),
#                 is_weekday=int(row['is_weekday']),
#                 is_holiday=int(row['is_holiday'])
#             ))
        
#         # Create request
#         request = ForecastRequest(
#             target=target,
#             model_type=model_type,
#             features=features,
#             timestamps=day_data['timestamps'].tolist()
#         )
        
#         # Get forecast
#         response = forecast_predict(request)
        
#         # Get actual values
#         actual_values = day_data[target].tolist()
        
#         # Calculate metrics
#         predictions_arr = np.array(response.predictions)
#         actual_arr = np.array(actual_values)
#         mae = float(np.mean(np.abs(predictions_arr - actual_arr)))
#         rmse = float(np.sqrt(np.mean((predictions_arr - actual_arr)**2)))
        
#         return {
#             "status": "success",
#             "date": date,
#             "target": target,
#             "model_type": model_type,
#             "predictions": response.predictions,
#             "actual": actual_values,
#             "timestamps": response.timestamps,
#             "mae": mae,
#             "rmse": rmse,
#             "num_hours": len(response.predictions)
#         }
        
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": str(e),
#             "predictions": [],
#             "actual": []
#         }

# # ============================================================================
# # MAIN
# # ============================================================================

# if __name__ == "__main__":
#     # Load models at startup
#     # Adjust path as needed - look for trained_models or trained_new directory
#     models_dir = os.getenv("FORECAST_MODELS_DIR", "trained_models")
    
#     # Try multiple possible paths
#     possible_paths = [
#         models_dir,
#         "trained_new",
#         "agentic_energy/trained_models",
#         "agentic_energy/trained_new",
#         "../trained_models",
#         "../trained_new"
#     ]
    
#     for path in possible_paths:
#         if Path(path).exists():
#             print(f"📂 Found models directory: {path}")
#             load_models(path)
#             break
#     else:
#         print("⚠️  No models directory found. Please set FORECAST_MODELS_DIR environment variable.")
#         print(f"   Searched paths: {possible_paths}")
    
#     # Start MCP server
#     print("🚀 Starting Forecast MCP Server...")
#     mcp.run(transport="stdio")



# """
# Forecast MCP Server - Serves price and consumption forecasting tools
# Loads pre-trained RF and LSTM models from pickle files
# """
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from pathlib import Path
# from typing import List, Optional
# from mcp.server.fastmcp import FastMCP

# from agentic_energy.schemas import ForecastRequest, ForecastResponse, ForecastFeatures

# # Initialize MCP server
# mcp = FastMCP("Forecast")

# # ============================================================================
# # LSTM MODEL DEFINITION
# # ============================================================================

# class LSTMModel(nn.Module):
#     """LSTM model for time series forecasting"""
#     def __init__(self, input_size, hidden_size, num_layers, dropout):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         self.lstm = nn.LSTM(
#             input_size, hidden_size, num_layers,
#             batch_first=True, 
#             dropout=dropout if num_layers > 1 else 0
#         )
#         self.fc = nn.Linear(hidden_size, 1)
        
#     def forward(self, x):
#         batch_size = x.size(0)
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out

# # ============================================================================
# # GLOBAL MODEL STORAGE
# # ============================================================================

# MODELS = {
#     'rf_prices': None,
#     'rf_consumption': None,
#     'lstm_prices': None,
#     'lstm_consumption': None
# }

# FEATURE_ORDER = [
#     'temperature',
#     'radiation_direct_horizontal',
#     'radiation_diffuse_horizontal',
#     'is_weekday',
#     'is_holiday',
#     'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6',
#     'month_7', 'month_8', 'month_9', 'month_10', 'month_11'
# ]

# # ============================================================================
# # MODEL LOADING FUNCTIONS
# # ============================================================================

# def load_models(models_dir: str = "trained_models"):
#     """Load all pickle models at server startup"""
#     models_path = Path(models_dir)
    
#     print(f"📂 Loading models from: {models_path.absolute()}")
    
#     # Check if directory exists
#     if not models_path.exists():
#         print(f"⚠️  Models directory not found: {models_path}")
#         print(f"   Please place pickle files in: {models_path.absolute()}")
#         return
    
#     # Load Random Forest models
#     try:
#         rf_prices_path = models_path / "rf_prices.pkl"
#         if rf_prices_path.exists():
#             with open(rf_prices_path, 'rb') as f:
#                 MODELS['rf_prices'] = pickle.load(f)
#             print(f"✅ Loaded RF prices model ({len(MODELS['rf_prices']['models'])} hour models)")
#         else:
#             print(f"⚠️  RF prices model not found: {rf_prices_path}")
#     except Exception as e:
#         print(f"❌ Error loading RF prices: {e}")
    
#     try:
#         rf_consumption_path = models_path / "rf_consumption.pkl"
#         if rf_consumption_path.exists():
#             with open(rf_consumption_path, 'rb') as f:
#                 MODELS['rf_consumption'] = pickle.load(f)
#             print(f"✅ Loaded RF consumption model ({len(MODELS['rf_consumption']['models'])} hour models)")
#         else:
#             print(f"⚠️  RF consumption model not found: {rf_consumption_path}")
#     except Exception as e:
#         print(f"❌ Error loading RF consumption: {e}")
    
#     # Load LSTM models
#     try:
#         lstm_prices_path = models_path / "lstm_prices.pkl"
#         if lstm_prices_path.exists():
#             with open(lstm_prices_path, 'rb') as f:
#                 lstm_dict = pickle.load(f)
            
#             # Reconstruct LSTM model
#             config = lstm_dict['model_config']
#             model = LSTMModel(
#                 input_size=config['input_size'],
#                 hidden_size=config['hidden_size'],
#                 num_layers=config['num_layers'],
#                 dropout=config['dropout']
#             )
#             model.load_state_dict(lstm_dict['model_state_dict'])
#             model.eval()
            
#             MODELS['lstm_prices'] = {
#                 'model': model,
#                 'scaler_X': lstm_dict['scaler_X'],
#                 'scaler_y': lstm_dict['scaler_y'],
#                 'seq_length': lstm_dict['seq_length'],
#                 'features': lstm_dict['metadata']['features']
#             }
#             print(f"✅ Loaded LSTM prices model (seq_length={lstm_dict['seq_length']})")
#         else:
#             print(f"⚠️  LSTM prices model not found: {lstm_prices_path}")
#     except Exception as e:
#         print(f"❌ Error loading LSTM prices: {e}")
    
#     try:
#         lstm_consumption_path = models_path / "lstm_consumption.pkl"
#         if lstm_consumption_path.exists():
#             with open(lstm_consumption_path, 'rb') as f:
#                 lstm_dict = pickle.load(f)
            
#             # Reconstruct LSTM model
#             config = lstm_dict['model_config']
#             model = LSTMModel(
#                 input_size=config['input_size'],
#                 hidden_size=config['hidden_size'],
#                 num_layers=config['num_layers'],
#                 dropout=config['dropout']
#             )
#             model.load_state_dict(lstm_dict['model_state_dict'])
#             model.eval()
            
#             MODELS['lstm_consumption'] = {
#                 'model': model,
#                 'scaler_X': lstm_dict['scaler_X'],
#                 'scaler_y': lstm_dict['scaler_y'],
#                 'seq_length': lstm_dict['seq_length'],
#                 'features': lstm_dict['metadata']['features']
#             }
#             print(f"✅ Loaded LSTM consumption model (seq_length={lstm_dict['seq_length']})")
#         else:
#             print(f"⚠️  LSTM consumption model not found: {lstm_consumption_path}")
#     except Exception as e:
#         print(f"❌ Error loading LSTM consumption: {e}")
    
#     print("🎯 Model loading complete!\n")

# # ============================================================================
# # FEATURE PROCESSING
# # ============================================================================

# def features_to_dataframe(features: List[ForecastFeatures]) -> pd.DataFrame:
#     """Convert ForecastFeatures list to DataFrame with proper feature encoding"""
#     data = []
    
#     for feat in features:
#         row = {
#             'temperature': feat.temperature,
#             'radiation_direct_horizontal': feat.radiation_direct_horizontal,
#             'radiation_diffuse_horizontal': feat.radiation_diffuse_horizontal,
#             'is_weekday': feat.is_weekday,
#             'is_holiday': feat.is_holiday,
#         }
        
#         # Add month dummy variables (month_1 through month_11)
#         for m in range(1, 12):
#             row[f'month_{m}'] = 1 if feat.month == m else 0
        
#         data.append(row)
    
#     df = pd.DataFrame(data)
    
#     # Ensure feature order matches training
#     df = df[FEATURE_ORDER]
    
#     return df

# # ============================================================================
# # FORECASTING FUNCTIONS
# # ============================================================================

# def predict_rf(rf_dict: dict, features_df: pd.DataFrame, hours: List[int]) -> np.ndarray:
#     """Make predictions using Random Forest models"""
#     predictions = []
    
#     for i, hour in enumerate(hours):
#         if hour not in rf_dict['models']:
#             raise ValueError(f"No RF model available for hour {hour}")
        
#         model = rf_dict['models'][hour]
#         feature_names = rf_dict['metadata']['features']
        
#         # Get features for this row
#         X = features_df.iloc[i:i+1][feature_names].values
#         pred = model.predict(X)[0]
#         predictions.append(pred)
    
#     return np.array(predictions)


# def predict_lstm(lstm_dict: dict, features_df: pd.DataFrame) -> np.ndarray:
#     """Make predictions using LSTM model"""
#     model = lstm_dict['model']
#     scaler_X = lstm_dict['scaler_X']
#     scaler_y = lstm_dict['scaler_y']
#     seq_length = lstm_dict['seq_length']
#     feature_names = lstm_dict['features']
    
#     # Get features in correct order
#     X = features_df[feature_names].values
    
#     # Scale features
#     X_scaled = scaler_X.transform(X)
    
#     # Build sequences (pad early points)
#     sequences = []
#     for i in range(len(X_scaled)):
#         if i < seq_length - 1:
#             # Pad with zeros for early points
#             padding = np.zeros((seq_length - i - 1, X_scaled.shape[1]))
#             seq = np.vstack([padding, X_scaled[:i+1]])
#         else:
#             seq = X_scaled[i-seq_length+1:i+1]
#         sequences.append(seq)
    
#     sequences = np.array(sequences)
    
#     # Make predictions
#     model.eval()
#     with torch.no_grad():
#         X_tensor = torch.FloatTensor(sequences)
#         predictions_scaled = model(X_tensor).numpy().flatten()
    
#     # Inverse transform
#     predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
#     return predictions

# # ============================================================================
# # MCP TOOLS
# # ============================================================================

# @mcp.tool()
# def forecast_predict(args: ForecastRequest) -> ForecastResponse:
#     """
#     Generate forecasts for energy prices or consumption.
    
#     Args:
#         args: ForecastRequest containing:
#             - target: 'prices' or 'consumption'
#             - model_type: 'RF' or 'LSTM'
#             - features: List of feature records
#             - timestamps: Optional list of timestamps
    
#     Returns:
#         ForecastResponse with predictions and metadata
#     """
#     try:
#         # Validate target
#         if args.target not in ['prices', 'consumption']:
#             return ForecastResponse(
#                 status="error",
#                 message=f"Invalid target: {args.target}. Must be 'prices' or 'consumption'",
#                 target=args.target,
#                 model_type=args.model_type,
#                 predictions=[],
#                 num_predictions=0
#             )
        
#         # Validate model type
#         if args.model_type not in ['RF', 'LSTM']:
#             return ForecastResponse(
#                 status="error",
#                 message=f"Invalid model_type: {args.model_type}. Must be 'RF' or 'LSTM'",
#                 target=args.target,
#                 model_type=args.model_type,
#                 predictions=[],
#                 num_predictions=0
#             )
        
#         # Check if model is loaded
#         model_key = f"{args.model_type.lower()}_{args.target}"
#         if MODELS[model_key] is None:
#             return ForecastResponse(
#                 status="error",
#                 message=f"Model not loaded: {model_key}. Please check models directory.",
#                 target=args.target,
#                 model_type=args.model_type,
#                 predictions=[],
#                 num_predictions=0
#             )
        
#         # Convert features to DataFrame
#         features_df = features_to_dataframe(args.features)
        
#         # Make predictions based on model type
#         if args.model_type == 'RF':
#             # Extract hours for RF predictions
#             hours = [feat.hour for feat in args.features]
#             predictions = predict_rf(MODELS[model_key], features_df, hours)
#         else:  # LSTM
#             predictions = predict_lstm(MODELS[model_key], features_df)
        
#         return ForecastResponse(
#             status="success",
#             target=args.target,
#             model_type=args.model_type,
#             predictions=predictions.tolist(),
#             timestamps=args.timestamps,
#             num_predictions=len(predictions)
#         )
        
#     except Exception as e:
#         return ForecastResponse(
#             status="error",
#             message=f"Forecast error: {str(e)}",
#             target=args.target,
#             model_type=args.model_type,
#             predictions=[],
#             num_predictions=0
#         )


# @mcp.tool()
# def forecast_check_models() -> dict:
#     """
#     Check which models are loaded and available.
    
#     Returns:
#         Dictionary showing model availability status
#     """
#     status = {}
#     for key, model in MODELS.items():
#         if model is None:
#             status[key] = "Not loaded"
#         elif key.startswith('rf_'):
#             status[key] = f"Loaded ({len(model['models'])} hour models)"
#         else:  # LSTM
#             status[key] = f"Loaded (seq_length={model['seq_length']})"
    
#     return {
#         "status": "success",
#         "models": status,
#         "feature_order": FEATURE_ORDER
#     }

# # ============================================================================
# # MAIN
# # ============================================================================

# if __name__ == "__main__":
#     # Load models at startup
#     # Adjust path as needed - look for trained_models or trained_new directory
#     models_dir = os.getenv("FORECAST_MODELS_DIR", "trained_models")
    
#     # Try multiple possible paths
#     possible_paths = [
#         models_dir,
#         "trained_new",
#         "agentic_energy/trained_models",
#         "agentic_energy/trained_new",
#         "../trained_models",
#         "../trained_new"
#     ]
    
#     for path in possible_paths:
#         if Path(path).exists():
#             print(f"📂 Found models directory: {path}")
#             load_models(path)
#             break
#     else:
#         print("⚠️  No models directory found. Please set FORECAST_MODELS_DIR environment variable.")
#         print(f"   Searched paths: {possible_paths}")
    
#     # Start MCP server
#     print("🚀 Starting Forecast MCP Server...")
#     mcp.run(transport="stdio")