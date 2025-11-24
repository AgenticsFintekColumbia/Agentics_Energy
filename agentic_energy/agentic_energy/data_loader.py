"""
Energy Data Loader using Agentics Framework

This module provides utilities to load and process energy market data from various regions
(CAISO, ERCOT, Germany, Italy, NewYork) using the Agentics framework for structured data handling.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from agentics import AG
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union, Tuple, Literal
import pandas as pd
from datetime import datetime
from .schemas import EnergyDataRecord, MetricStats, SummaryStats, DateRange, BatteryParams
import numpy as np


class EnergyDataLoader:
    """
    Energy Data Loader using Agentics framework for structured energy market data loading.

    Parameters
    ----------
    region : str
        Region name (CAISO, ERCOT, GERMANY, ITALY, NEWYORK).
    data_dir : Union[str, Path] | None
        Base directory containing CSV files. Defaults to <this_file>/data.
    data_version : Literal["actual", "forecast"]
        Whether to load actuals or forecasts. Defaults to "actual".
    forecast_type : Optional[Literal["LSTM", "NOISE", "RF"]]
        Required when data_version == "forecast". Ignored for "actual".
    """
    
    def __init__(
            self, 
            region: str, 
            data_dir: Union[str, Path] = None,
            data_version: Literal["actual", "forecast"] = "actual",
            forecast_type: Optional[Literal["LSTM", "NOISE", "RF"]] = None,
        ):
        """
        Initialize the data loader
        
        Args:
            region: Region name (CAISO, ERCOT, GERMANY, ITALY, NEWYORK)
            data_dir: Path to the directory containing CSV files
        """
        if data_dir is None:
            # Default to current directory's data folder
            self.data_dir = Path(__file__).parent / "data"
        else:
            self.data_dir = Path(data_dir)
            
        self.region = region.upper()
        self.data_version = data_version.lower()
        self.forecast_type = forecast_type.upper() if forecast_type is not None else None

        # --- Supported base files for ACTUALS ---
        self.available_actuals: Dict[str, str] = {
            "CAISO":   "CAISO_data.csv",
            "ERCOT":   "Ercot_energy_data.csv",
            "GERMANY": "Germany_energy_Data.csv",
            "ITALY":   "Italy_data_actual.csv",
            "ITALY_TEST": "Italy_data.csv",
            "NEWYORK": "NewYork_energy_data.csv",
        }

        # --- Supported forecast files by region ---
        # You mentioned only Italy variants; we can extend this dict later for other regions.
        self.available_forecasts: Dict[str, Dict[str, str]] = {
            "ITALY": {
                "LSTM":  "Italy_data_forecast_LSTM.csv",
                "NOISE": "Italy_data_forecast_NOISE.csv",
                "RF":    "Italy_data_forecast_RF.csv",
            },
            # Add other regions here if/when forecasts exist.
            # e.g., "CAISO": {"LSTM": "...", "NOISE": "...", "RF": "..."}
            "NEWYORK": {
                "LSTM":  "NewYork_data_forecast_LSTM.csv",
                "NOISE": "NewYork_data_forecast_NOISE.csv",
                "RF":    "NewYork_data_forecast_RF.csv",
            },
            "CAISO": {
                "LSTM":  "CAISO_data_forecast_LSTM.csv",
                "NOISE": "CAISO_data_forecast_NOISE.csv",
                "RF":    "CAISO_data_forecast_RF.csv",
            },
            "ERCOT": {
                "LSTM":  "Ercot_data_forecast_LSTM.csv",
                "NOISE": "Ercot_data_forecast_NOISE.csv",
                "RF":    "Ercot_data_forecast_RF.csv",
            },
            "GERMANY": {
                "LSTM":  "Germany_data_forecast_LSTM.csv",
                "NOISE": "Germany_data_forecast_NOISE.csv",
                "RF":    "Germany_data_forecast_RF.csv",
            },
        }

        self.data: Optional[AG] = None

        # --- Validate upfront to catch config issues early ---
        self._validate_init()

    def _validate_init(self):
        # Region support
        known_regions = set(self.available_actuals.keys()) | set(self.available_forecasts.keys())
        if self.region not in known_regions:
            raise ValueError(
                f"Region '{self.region}' not supported. Available: {sorted(known_regions)}"
            )

        if self.data_version not in {"actual", "forecast"}:
            raise ValueError("data_version must be either 'actual' or 'forecast'.")

        if self.data_version == "forecast":
            # Region must have forecast mapping
            if self.region not in self.available_forecasts:
                raise ValueError(
                    f"Forecasts are not configured for region '{self.region}'. "
                    f"Available forecast regions: {sorted(self.available_forecasts.keys())}"
                )
            # forecast_type must be provided & valid
            if not self.forecast_type:
                raise ValueError(
                    "forecast_type is required when data_version='forecast' "
                    "(choose one of: 'LSTM', 'NOISE', 'RF')."
                )
            valid_types = set(self.available_forecasts[self.region].keys())
            if self.forecast_type not in valid_types:
                raise ValueError(
                    f"Unsupported forecast_type '{self.forecast_type}' for region '{self.region}'. "
                    f"Supported: {sorted(valid_types)}"
                )

    def _resolve_filename(self) -> Path:
        """Return the Path to the CSV file based on data_version/forecast_type."""
        if self.data_version == "actual":
            fname = self.available_actuals.get(self.region)
            if not fname:
                # Defensive: shouldn't happen because of _validate_init.
                raise ValueError(f"No actuals file registered for region '{self.region}'.")
            return self.data_dir / fname

        # Forecast case
        region_map = self.available_forecasts.get(self.region, {})
        fname = region_map.get(self.forecast_type or "")
        if not fname:
            raise ValueError(
                f"No forecast file for region '{self.region}' and type '{self.forecast_type}'."
            )
        return self.data_dir / fname

    # ----------------------------
    # Public API
    # ----------------------------
    def load_region_data(self) -> AG:
        """
        Load data for the configured region/data_version/forecast_type using Agentics.

        Raises
        ------
        FileNotFoundError
            If the resolved CSV does not exist.
        ValueError
            If the configuration is invalid.
        """
        file_path = self._resolve_filename()
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        energy_data = AG.from_csv(file_path, atype=EnergyDataRecord)

        # Stamp region on each record
        for state in energy_data.states:
            state.region = self.region
            # Optional: annotate provenance
            if getattr(state, "source_kind", None) is not None:
                state.source_kind = self.data_version
            if getattr(state, "forecast_type", None) is not None:
                state.forecast_type = self.forecast_type if self.data_version == "forecast" else None

        self.data = energy_data
        return self.data

    async def get_filtered_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        price_range: Optional[Tuple[float, float]] = None,
    ) -> AG:
        """
        Efficiently filter region data using vectorized masking inside areduce.
        Assumes each state has attributes: timestamps, prices.
        """

        has_date_filter = bool(start_date or end_date)
        start_day = np.datetime64(pd.to_datetime(start_date).date()) if start_date else None
        end_day   = np.datetime64(pd.to_datetime(end_date).date())   if end_date   else None

        has_price_filter = bool(price_range)
        if has_price_filter:
            min_price, max_price = price_range

        async def _filter_reduce(states: list):
            if not states:
                return []
            ts_arr = np.array([s.timestamps for s in states], dtype="datetime64[ns]")
            day_arr = ts_arr.astype("datetime64[D]")

            pr_arr = None
            if has_price_filter:
                pr_arr = pd.to_numeric([getattr(s, "prices", np.nan) for s in states], errors="coerce").to_numpy()

            mask = np.ones(len(states), dtype=bool)
            if has_date_filter:
                if start_day is not None:
                    mask &= (day_arr >= start_day)
                if end_day is not None:
                    mask &= (day_arr <= end_day)
            if has_price_filter:
                mask &= np.isfinite(pr_arr)
                mask &= (pr_arr >= min_price) & (pr_arr <= max_price)

            if mask.all():
                return states
            if not mask.any():
                return []
            idx = np.nonzero(mask)[0]
            return [states[i] for i in idx]

        if self.data is None:
            raise RuntimeError("No data loaded. Call load_region_data() first.")

        filtered_states = await self.data.areduce(_filter_reduce)
        return AG(atype=EnergyDataRecord, states=filtered_states)

    @staticmethod
    async def get_summary_stats_from_ag(
        ag_data: AG, column: Optional[str] = None
    ) -> SummaryStats | Dict:
        prices = np.array([s.prices for s in ag_data.states if s.prices is not None], dtype=float)
        consumption = np.array([s.consumption for s in ag_data.states if s.consumption is not None], dtype=float)
        timestamps = [s.timestamps for s in ag_data.states if getattr(s, "timestamps", None)]

        async def summarize(arr: np.ndarray) -> MetricStats:
            if arr.size == 0:
                return MetricStats()
            return MetricStats(
                count=int(arr.size),
                min=float(np.min(arr)),
                max=float(np.max(arr)),
                avg=float(np.mean(arr)),
                median=float(np.median(arr)),
                p25=float(np.percentile(arr, 25)),
                p75=float(np.percentile(arr, 75)),
                std=float(np.std(arr)),
                var=float(np.var(arr)),
            )

        stats_obj = SummaryStats(
            region=ag_data[0].region if len(ag_data.states) else None,
            total_records=len(ag_data.states),
            date_range=DateRange(
                start=min(timestamps) if timestamps else None,
                end=max(timestamps) if timestamps else None,
            ),
            prices=await summarize(prices),
            consumption=await summarize(consumption),
        )

        if column:
            if column not in ["prices", "consumption"]:
                raise ValueError("Column must be 'prices' or 'consumption'.")
            return AG(atype=MetricStats, states=[getattr(stats_obj, column)])

        return AG(atype=SummaryStats, states=[stats_obj])


class BatteryDataLoader:
    """
    Battery Data Loader — computes battery parameters from load statistics.
    """

    def __init__(self, 
            load_stats: Dict[str, float], 
            duration_hours: float = 4.0,
            soc_init=0.5,
            soc_min=0.0,
            soc_max=1.0,
            eta_c=0.95,
            eta_d=0.95,
            soc_target=0.5
        ):
        """
        Args:
            load_stats (dict): Must contain 'p25' and 'p75' in MW.
            duration_hours (float): Hours battery should sustain IQR deviation.
        """
        if 'p25' not in load_stats or 'p75' not in load_stats:
            raise ValueError("load_stats must include 'p25' and 'p75' values in MW.")
        self.load_stats = load_stats
        self.duration_hours = duration_hours
        self.soc_init = soc_init
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.soc_target = soc_target

    def compute_battery_params(self) -> BatteryParams:
        """
        Compute capacity and charge/discharge limits from IQR of load statistics.
        Converts MW → MW and MWh → MWh internally.
        """
        p25, p75 = self.load_stats['p25'], self.load_stats['p75']

        # Interquartile load deviation
        iqr_range_MW = p75 - p25
        capacity_MWh = iqr_range_MW * self.duration_hours
        cmax_MW = capacity_MWh / self.duration_hours
        dmax_MW = cmax_MW

        return BatteryParams(
            capacity_MWh=round(capacity_MWh, 2),
            cmax_MW=round(cmax_MW, 2),
            dmax_MW=round(dmax_MW, 2),
            soc_init=self.soc_init,
            soc_min=self.soc_min,
            soc_max=self.soc_max,
            eta_c=self.eta_c,
            eta_d=self.eta_d,
            soc_target=self.soc_target  
        )

    def summary(self) -> Dict[str, float]:
        """Return computed specs as readable summary."""
        params = self.compute_battery_params()
        return {
            "Capacity (MWh)": params.capacity_MWh,
            "Charge Power (MW)": params.cmax_MW,
            "Discharge Power (MW)": params.dmax_MW,
            "Efficiency (Charge/Discharge)": (params.eta_c, params.eta_d),
            "Duration (hours)": self.duration_hours
        }


    
    # async def get_summary_stats_from_ag(ag_data: AG, column: Optional[str] = None) -> SummaryStats | Dict:
    #     """
    #     Compute summary statistics (min, max, avg, median, percentiles, std, var)
    #     and return as Pydantic schema (SummaryStats).
    #     """

    #     # source = AG(
    #     #     atype = EnergyDataRecord,
    #     #     verbose_agent = True
    #     #     state = ag_data.states
    #     # )
    #     if column:
    #         answer = await(
    #             AG(
    #                 atype = SummaryStats,
    #                 verbose_agent = True,
    #                 instructions = f"Compute summary statistics for the '{column}' column only. "
    #             ) 
    #             << ag_data(column)
    #         )
    #         return answer
    #     else:
    #         answer = await(
    #             AG(
    #                 atype = SummaryStats,
    #                 verbose_agent = True,
    #                 instructions = "Compute summary statistics for all relevant columns."
    #             ) 
    #             << ag_data
    #         )
    #         return answer
        
    