# agentic_energy_app/data_utils.py

import asyncio
from typing import Tuple, List, Sequence, Optional

import pandas as pd
import numpy as np

from agentic_energy.data_loader import EnergyDataLoader
from agentic_energy.schemas import DayInputs, PlotResponse, BatteryParams
from agentic_energy.milp.milp_mcp_server import records_to_arrays

from agentic_energy.mcp_clients import run_price_forecast_plot


async def _load_energy_day_async(
    region: str = "ITALY",
    date_str: str = "2018-01-01",
    forecast_type: str = "LSTM",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Async helper for EnergyDataLoader."""
    actual_obj = EnergyDataLoader(region=region, data_version="actual")
    actual_obj.load_region_data()
    actual = await actual_obj.get_filtered_data(date_str, date_str)

    forecast_obj = EnergyDataLoader(
        region=region,
        data_version="forecast",
        forecast_type=forecast_type,
    )
    forecast_obj.load_region_data()
    forecast = await forecast_obj.get_filtered_data(date_str, date_str)

    return actual, forecast


def load_energy_day(
    region: str,
    date_str: str,
    forecast_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sync wrapper."""
    return asyncio.run(_load_energy_day_async(region, date_str, forecast_type))



def make_day_inputs_from_forecast(
    prices: List[float],
    demand: List[float],
    dt_hours: float = 1.0,
) -> DayInputs:
    """Create DayInputs using forecast prices and (actual) demand."""
    return DayInputs(
        prices_buy=prices,
        prices_sell=prices,
        demand_MW=demand,
        allow_export=True,
        dt_hours=dt_hours,
    )


def run_forecast_step(
    region: str,
    date_str: str,
    forecast_type: str,
    forecast_plot_path: str,
) -> Tuple[DayInputs, pd.DataFrame, pd.DataFrame, PlotResponse]:
    """
    High-level helper used by Streamlit app:

    1) Load actual + forecast from EnergyDataLoader
    2) Extract prices + demand
    3) Build DayInputs
    4) Call visualization MCP to make price forecast plot
    """
    actual_df, forecast_df = load_energy_day(region, date_str, forecast_type)
    forecast_prices, forecast_demand = records_to_arrays(forecast_df)
    actual_prices, actual_demand = records_to_arrays(actual_df)

    dt_hours = 1.0  # 🔧 infer from timestamps if you want
    day_inputs = make_day_inputs_from_forecast(forecast_prices,forecast_demand, dt_hours)

    plot = run_price_forecast_plot(
        prices=forecast_prices,
        dt_hours=dt_hours,
        out_path=forecast_plot_path,
        title=f"Price Forecast - {region} {date_str} ({forecast_type})",
    )

    return day_inputs, actual_df, forecast_df, plot

