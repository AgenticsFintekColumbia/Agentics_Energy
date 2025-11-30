"""
Build optimization dataset from _data_actual.csv

For each day in the CSV:
    * build EnergyDataRecord list
    * call milp_solve_from_records(...) to get the optimal schedule
    * write a JSONL dataset with fields:
        - "user_message": compact JSON with battery + time series
        - "assistant_message": JSON with ground-truth SoC (and cost)

You will later wrap these into ChatML in Colab, e.g.:

<|im_start|>system
  ...fixed description of the problem...
<|im_end|>
<|im_start|>user
  {user_message}
<|im_end|>
<|im_start|>assistant
  {assistant_message}
<|im_end|>
"""
import asyncio
import json
from pathlib import Path
from typing import List
from datetime import datetime
import numpy as np

import pandas as pd
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

# ---- adjust these imports to your actual package layout ----
from agentic_energy.milp.milp_mcp_server import records_to_arrays, solve_daily_milp
from agentic_energy.data_loader import EnergyDataLoader, BatteryDataLoader
from agentic_energy.schemas import BatteryParams, DayInputs, EnergyDataRecord, SolveFromRecordsRequest# ------------------------------------------------------------


# ---------- CONFIG ----------
REGION = "ITALY"
if "TEST" in REGION:
    OUTPUT_JSONL = Path(f"test_qwen_soc_{REGION}.jsonl")
else:
    OUTPUT_JSONL = Path(f"train_qwen_soc_{REGION}.jsonl")

DATAPOINTS_in_HOUR = 1  #  data is hourly
DT_HOURS = 1.0/DATAPOINTS_in_HOUR          #  data is hourly
ALLOW_EXPORT = True    # set True if you want export in the MILP
MAX_DAYS = None         # or set to an int to limit dataset size (e.g., 200)
DATA_POINTS =  24 * DATAPOINTS_in_HOUR      # number of data points per day (24 for hourly data)

# ---------- HELPERS TO BUILD MESSAGES ----------

def build_user_message(
    battery: BatteryParams,
    df_day: pd.DataFrame,
    dt_hours: float,
    allow_export: bool,
) -> str:
    """
    Build the 'user_message' as a compact JSON string:
    instance-specific data only (battery params + time series).
    """
    prices, demand = records_to_arrays(df_day)
    payload = {
        "capacity_MWh": battery.capacity_MWh,
        "max_c_MW": battery.cmax_MW,
        "max_d_MW": battery.dmax_MW,
        "eta_c": battery.eta_c,
        "eta_d": battery.eta_d,
        "soc0": battery.soc_init,
        "soc_min": battery.soc_min,
        "soc_max": battery.soc_max,
        "dt_hours": dt_hours,
        "allow_export": bool(allow_export),
        "prices_buy": prices,
        "prices_sell": prices,   # symmetric here; adapt if you have different series
        "demand_MW": demand,
    }

    # Compact JSON (no extra spaces) to reduce tokens
    return json.dumps(payload, separators=(",", ":"))


def build_assistant_message(sol, include_cost: bool = True) -> str:
    """
    Build the 'assistant_message' as a JSON string.

    Here we only keep:
      - soc: list of T+1 SoC values
      - cost: scalar objective value (optional but recommended)
    """
    if getattr(sol, "soc", None) is None:
        raise ValueError("SolveResponse has no 'soc' field; cannot build label.")

    payload = {
        "soc": sol.soc,
    }

    if include_cost and hasattr(sol, "objective_cost"):
        payload["cost"] = sol.objective_cost

    return json.dumps(payload, separators=(",", ":"))


# ---------- MAIN DATASET BUILDER ----------

async def main():
    _obj = EnergyDataLoader(region=REGION)
    _data = _obj.load_region_data()
    stats = await EnergyDataLoader.get_summary_stats_from_ag(_data)
    timedelta = datetime.strptime(stats.states[0].date_range.end, "%Y-%m-%d %H:%M:%S") - datetime.strptime(stats.states[0].date_range.start, "%Y-%m-%d %H:%M:%S")
    days = timedelta.days + 1
        
    batt = BatteryDataLoader(
            load_stats={
                "p25":stats.states[0].consumption.p25,
                "p75":stats.states[0].consumption.p75
            },
            duration_hours=4,
            soc_init=0.5,
            soc_min=0.0,
            soc_max=1.0,
            eta_c = 0.95,
            eta_d = 0.95,
            soc_target=0.5,
        )
    batterydetails = batt.compute_battery_params()


    print(f"Found {days} unique days in {REGION}.")
    print(f"Building dataset for {days} days...")

    n_written = 0
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f_out:
        for i in range(len(_data)//DATA_POINTS):
            print(f"Solving day {i+1} of {len(_data)//DATA_POINTS}")
            records = SolveFromRecordsRequest(
                battery=batterydetails,
                records=_data[i*DATA_POINTS:(i+1)*DATA_POINTS],
                dt_hours=DT_HOURS,
                allow_export=ALLOW_EXPORT,
                solver=None,
                solver_opts=None
            )

            prices, demand = records_to_arrays(records.records)
            day = DayInputs(
                prices_buy=prices,
                demand_MW=demand,
                prices_sell=prices,
                allow_export=records.allow_export,
                dt_hours=records.dt_hours
            )
            sol = solve_daily_milp(records.battery, day, records.solver, records.solver_opts)

            # Build user_message and assistant_message
            user_message = build_user_message(
                battery=batterydetails,
                df_day=_data[i*DATA_POINTS:(i+1)*DATA_POINTS],
                dt_hours=DT_HOURS,
                allow_export=ALLOW_EXPORT,
            )
            
            batterydetails.soc_init= np.round((np.random.rand() + 1e-12) % 1,2)

            assistant_message = build_assistant_message(sol, include_cost=True)

            date_str = _data[i*DATA_POINTS].timestamps.split(" ")[0]
            example = {
                "date": date_str,
                "user_message": user_message,
                "assistant_message": assistant_message,
            }

            f_out.write(json.dumps(example) + "\n")
            n_written += 1

    print(f"Done. Wrote {n_written} examples to {OUTPUT_JSONL}")


if __name__ == "__main__":
    asyncio.run(main())
