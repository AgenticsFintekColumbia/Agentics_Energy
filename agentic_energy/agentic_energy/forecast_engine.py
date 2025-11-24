"""
Energy Forecasting Engine using Agentics Framework

This module provides utilities to load trained LSTM models and generate forecasts
for energy price and consumption data with MSE error metrics.
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
from datetime import datetime, timedelta

from agentics import AG
from .schemas import EnergyDataRecord, ForecastResult, ForecastMetrics, ForecastRecord
from .data_loader import EnergyDataLoader


# Force CPU usage (no CUDA)
device = torch.device('cpu')
print(f"Forecast Engine using device: {device}")


class LSTMForecaster(nn.Module):
    """LSTM-based forecasting model (matches training architecture)"""
    
    def __init__(self, lookback, horizon, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        
        self.lookback = lookback
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class ForecastEngine:
    """Engine for loading models and generating forecasts"""
    
    def __init__(self, models_dir: Union[str, Path] = None):
        """
        Initialize the forecast engine
        
        Args:
            models_dir: Directory containing trained .pkl model files.
                       If None, defaults to 'trained_models' folder inside agentic_energy module.
        """
        if models_dir is None:
            # Default to trained_models folder inside agentic_energy (same as data/)
            self.models_dir = Path(__file__).parent / "trained_models"
        else:
            self.models_dir = Path(models_dir)
        
        # Verify the directory exists
        if not self.models_dir.exists():
            print(f"Warning: Models directory not found at {self.models_dir}")
            print(f"         Please ensure trained models are in this location or specify correct path.")
        
        self.loaded_models = {}
        self.region_configs = {
            'CAISO': {'lookback': 96, 'horizon': 24},
            'ERCOT': {'lookback': 96, 'horizon': 24},
            'GERMANY': {'lookback': 96, 'horizon': 24},
            'ITALY': {'lookback': 96, 'horizon': 24},
            'NEWYORK': {'lookback': 1152, 'horizon': 288}
        }
        
    def _get_model_path(self, region: str, target: str) -> Path:
        """Get the path to a trained model file"""
        region = region.upper()
        
        # Map region to file name pattern
        region_file_map = {
            'CAISO': 'CAISO_data',
            'ERCOT': 'Ercot_energy_data',
            'GERMANY': 'Germany_energy_Data',
            'ITALY': 'Italy_data',
            'NEWYORK': 'NewYork_energy_data'
        }
        
        file_prefix = region_file_map.get(region)
        if not file_prefix:
            raise ValueError(f"Unknown region: {region}")
        
        model_filename = f"{file_prefix}_{target}_model.pkl"
        return self.models_dir / model_filename
    
    def load_model(self, region: str, target: str) -> Dict:
        """Load a trained model for a specific region and target"""
        model_key = f"{region}_{target}"
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        model_path = self._get_model_path(region, target)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Try multiple strategies to load model on CPU
        model_package = None
        last_error = None
        
        # Strategy 1: Explicit torch.device('cpu')
        try:
            with open(model_path, 'rb') as f:
                model_package = torch.load(
                    f, 
                    map_location=torch.device('cpu'),
                    weights_only=False
                )
        except Exception as e1:
            last_error = e1
            
            # Strategy 2: Override _rebuild_tensor to force CPU
            try:
                # Save original function
                original_rebuild_tensor = torch._utils._rebuild_tensor
                
                # Create CPU-forcing wrapper
                def cpu_rebuild_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                    # Force storage to CPU
                    if hasattr(storage, 'cpu'):
                        storage = storage.cpu()
                    return original_rebuild_tensor(storage, storage_offset, size, stride, requires_grad, backward_hooks)
                
                # Monkey patch temporarily
                torch._utils._rebuild_tensor = cpu_rebuild_tensor
                
                try:
                    with open(model_path, 'rb') as f:
                        model_package = torch.load(f, weights_only=False)
                finally:
                    # Restore original function
                    torch._utils._rebuild_tensor = original_rebuild_tensor
                    
            except Exception as e2:
                last_error = e2
                
                # Strategy 3: Custom unpickler class
                try:
                    import io
                    
                    class CPUUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):
                            if module == 'torch.storage' and name == '_load_from_bytes':
                                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                            else:
                                return super().find_class(module, name)
                    
                    with open(model_path, 'rb') as f:
                        model_package = CPUUnpickler(f).load()
                    
                    # Manually move all tensors in state_dict to CPU
                    if 'model_state_dict' in model_package:
                        new_state_dict = {}
                        for key, value in model_package['model_state_dict'].items():
                            if isinstance(value, torch.Tensor):
                                new_state_dict[key] = value.cpu() if value.is_cuda else value
                            else:
                                new_state_dict[key] = value
                        model_package['model_state_dict'] = new_state_dict
                        
                except Exception as e3:
                    raise RuntimeError(
                        f"All loading strategies failed for {model_path}.\n"
                        f"The model was saved with CUDA tensors and cannot be loaded on CPU.\n"
                        f"Please re-train or re-save the model on a CPU-only environment.\n"
                        f"Last error: {e3}"
                    )
        
        if model_package is None:
            raise RuntimeError(f"Failed to load model from {model_path}: {last_error}")
        
        # Recreate model
        model = LSTMForecaster(**model_package['model_config'])
        model.load_state_dict(model_package['model_state_dict'])
        model.eval()
        model.to(device)
        
        print(f"Successfully loaded {model_key}")
        
        # Optionally compile with JIT for optimization
        try:
            sample_input = torch.randn(1, model_package['model_config']['lookback'])
            model = torch.jit.trace(model, sample_input)
            print(f"  - Compiled with TorchScript JIT")
        except Exception as e:
            print(f"  - Using eager mode (JIT compilation skipped)")
        
        # Store complete package
        model_package['model'] = model
        self.loaded_models[model_key] = model_package
        
        return model_package
    
    def _validate_inference_period(self, ag_data: AG, start_date: str, end_date: str) -> None:
        """Validate that the inference period is in the second half of the dataset"""
        if not ag_data.states:
            raise ValueError("No data available in AG object")
        
        # Get all timestamps and sort
        all_timestamps = sorted([pd.to_datetime(s.timestamps) for s in ag_data.states])
        
        # Find the midpoint
        midpoint_idx = len(all_timestamps) // 2
        midpoint_date = all_timestamps[midpoint_idx].date()
        
        # Convert requested dates
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        
        # Validate
        if start < midpoint_date:
            raise ValueError(
                f"Forecast period starts at {start} but must be in second half of data "
                f"(starting from {midpoint_date}). Cannot provide inference for first half."
            )
        
        print(f"Validation passed: Forecast period {start} to {end} is in second half (>= {midpoint_date})")
    
    def _prepare_sequence(self, data: np.ndarray, lookback: int, 
                         mean: float, std: float) -> torch.Tensor:
        """Prepare a sequence for inference"""
        if len(data) < lookback:
            raise ValueError(f"Need at least {lookback} data points, got {len(data)}")
        
        # Take last lookback points
        sequence = data[-lookback:]
        
        # Normalize
        sequence_normalized = (sequence - mean) / (std + 1e-8)
        
        return torch.FloatTensor(sequence_normalized).unsqueeze(0).to(device)
    
    def _denormalize(self, predictions: np.ndarray, mean: float, std: float) -> np.ndarray:
        """Denormalize predictions"""
        return predictions * std + mean
    
    async def generate_forecasts(
        self,
        ag_data: AG,
        start_date: str,
        end_date: str,
        targets: List[str] = ['prices', 'consumption']
    ) -> AG:
        """
        Generate forecasts for specified period and targets
        
        Args:
            ag_data: Agentics object with energy data
            start_date: Start date for forecast period (YYYY-MM-DD)
            end_date: End date for forecast period (YYYY-MM-DD)
            targets: List of targets to forecast ['prices', 'consumption']
            
        Returns:
            AG object with ForecastResult states
        """
        # Validate period is in second half
        self._validate_inference_period(ag_data, start_date, end_date)
        
        # Get region
        region = ag_data.states[0].region if ag_data.states else None
        if not region:
            raise ValueError("Region not found in data")
        
        # Get config for region
        config = self.region_configs.get(region)
        if not config:
            raise ValueError(f"No configuration for region: {region}")
        
        lookback = config['lookback']
        horizon = config['horizon']
        
        # Convert data to dataframe for easier manipulation
        df = pd.DataFrame([
            {
                'timestamp': pd.to_datetime(s.timestamps),
                'prices': s.prices,
                'consumption': s.consumption
            }
            for s in ag_data.states
        ]).sort_values('timestamp')
        
        # Filter to before start_date (for historical context)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get historical data up to start_date
        historical_df = df[df['timestamp'] < start_dt].copy()
        
        # Get actual values for the forecast period
        actual_df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].copy()
        
        if len(historical_df) < lookback:
            raise ValueError(
                f"Not enough historical data. Need {lookback} points, got {len(historical_df)}"
            )
        
        forecast_results = []
        
        for target in targets:
            if target not in ['prices', 'consumption']:
                print(f"Warning: Skipping unknown target '{target}'")
                continue
            
            print(f"\nGenerating {target} forecast for {region}...")
            
            # Load model
            try:
                model_package = self.load_model(region, target)
                model = model_package['model']
                normalization = model_package['normalization']
            except FileNotFoundError as e:
                print(f"Warning: {e}. Skipping {target}.")
                continue
            
            # Prepare historical data
            historical_values = historical_df[target].values
            actual_values = actual_df[target].values
            
            # Sequential inference
            predictions = []
            current_data = historical_values.copy()
            
            num_forecasts = len(actual_values) // horizon + (1 if len(actual_values) % horizon else 0)
            
            for i in range(num_forecasts):
                # Prepare input sequence
                input_seq = self._prepare_sequence(
                    current_data,
                    lookback,
                    normalization['mean'],
                    normalization['std']
                )
                
                # Generate forecast
                with torch.no_grad():
                    pred_normalized = model(input_seq).cpu().numpy()[0]
                
                # Denormalize
                pred = self._denormalize(
                    pred_normalized,
                    normalization['mean'],
                    normalization['std']
                )
                
                predictions.extend(pred)
                
                # Update current_data with actual values for next iteration
                start_idx = i * horizon
                end_idx = min((i + 1) * horizon, len(actual_values))
                
                if end_idx <= len(actual_values):
                    current_data = np.concatenate([
                        current_data,
                        actual_values[start_idx:end_idx]
                    ])
            
            # Trim predictions to match actual length
            predictions = np.array(predictions[:len(actual_values)])
            
            # Calculate MSE
            mse = np.mean((predictions - actual_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actual_values))
            
            # Create forecast records
            forecast_records = []
            for idx, (timestamp, actual, pred) in enumerate(zip(
                actual_df['timestamp'],
                actual_values,
                predictions
            )):
                forecast_records.append(ForecastRecord(
                    timestamp=timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    actual=float(actual),
                    predicted=float(pred),
                    error=float(pred - actual)
                ))
            
            # Create metrics
            metrics = ForecastMetrics(
                mse=float(mse),
                rmse=float(rmse),
                mae=float(mae),
                num_predictions=len(predictions)
            )
            
            # Create result
            result = ForecastResult(
                region=region,
                target=target,
                start_date=start_date,
                end_date=end_date,
                lookback=lookback,
                horizon=horizon,
                metrics=metrics,
                forecasts=forecast_records
            )
            
            forecast_results.append(result)
            
            print(f"  MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        # Return as Agentics object
        return AG(atype=ForecastResult, states=forecast_results)
    
    async def get_forecast_for_day(
        self,
        ag_data: AG,
        date: str,
        targets: List[str] = ['prices', 'consumption']
    ) -> AG:
        """
        Generate forecast for a specific day
        
        Args:
            ag_data: Agentics object with energy data
            date: Date to forecast (YYYY-MM-DD)
            targets: List of targets to forecast
            
        Returns:
            AG object with ForecastResult states
        """
        # For NewYork (5-minute data), forecast the whole day
        # For others (hourly), forecast the whole day
        region = ag_data.states[0].region if ag_data.states else None
        
        date_obj = pd.to_datetime(date)
        start_date = date_obj.strftime('%Y-%m-%d')
        end_date = (date_obj + timedelta(days=1) - timedelta(seconds=1)).strftime('%Y-%m-%d')
        
        return await self.generate_forecasts(ag_data, start_date, end_date, targets)