"""
Dataset builder with forecast + actual + multi-day history window.

For each day D:
  - history = last H days, each with forecast + actual arrays
  - today_forecast = forecast arrays for day D
  - assistant = SoC schedule from MILP using today_forecast only
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# --- your imports ---
from agentic_energy.milp.milp_mcp_server import records_to_arrays, solve_daily_milp
from agentic_energy.data_loader import EnergyDataLoader, BatteryDataLoader
from agentic_energy.schemas import (
    BatteryParams, DayInputs, EnergyDataRecord, SolveFromRecordsRequest
)

# ---------------------

REGION = "ITALY"
if "TEST" in REGION:
    OUTPUT_JSONL = Path(f"test_with_history_qwen_soc_{REGION}.jsonl")
else:
    OUTPUT_JSONL = Path(f"train_with_history_qwen_soc_{REGION}.jsonl")

HISTORY_DAYS = 2   # <--- how many past days to include
DT_HOURS = 1.0
DATA_POINTS = 24


# ------------------------------------------------------------
def build_history_block(
    data_actual, data_forecast, day_index
):
    """
    Build list of history of past H days.
    Each entry includes forecast + actual arrays.
    """
    history = []
    for h in range(1, HISTORY_DAYS + 1):
        past_day = day_index - h
        if past_day < 0:
            break

        start = past_day * DATA_POINTS
        end   = (past_day + 1) * DATA_POINTS

        records_a = data_actual[start:end]
        records_f = data_forecast[start:end]

        prices_f, demand_f = records_to_arrays(records_f)
        prices_a, demand_a = records_to_arrays(records_a)

        history.append({
            "date": records_a[0].timestamps.split(" ")[0],
            "prices_buy_forecast": prices_f,
            "prices_buy_actual":   prices_a,
            "prices_sell_forecast": prices_f,
            "prices_sell_actual":   prices_a,
            "demand_forecast_MW": demand_f,
            "demand_actual_MW":   demand_a,
        })

    # reverse chronological (old → recent)
    history.reverse()
    return history


def build_user_message(
    battery, day_records_forecast, day_records_actual, history
):
    prices_f, demand_f = records_to_arrays(day_records_forecast)

    payload = {
        "battery": {
            "capacity_MWh": battery.capacity_MWh,
            "max_c_MW": battery.cmax_MW,
            "max_d_MW": battery.dmax_MW,
            "eta_c": battery.eta_c,
            "eta_d": battery.eta_d,
            "soc0": battery.soc_init,
            "soc_min": battery.soc_min,
            "soc_max": battery.soc_max,
            "dt_hours": DT_HOURS,
            "allow_export": True
        },

        # ### HISTORY ADDED HERE ###
        "history": history,

        # today's forecast
        "today_forecast": {
            "prices_buy_forecast": prices_f,
            "prices_sell_forecast": prices_f,
            "demand_forecast_MW":  demand_f
        }
    }

    return json.dumps(payload, separators=(",", ":"))


def build_assistant_message(sol):
    return json.dumps({
        "soc": sol.soc,
        "cost_forecast": float(sol.objective_cost)
    }, separators=(",", ":"))


# ------------------------------------------------------------

async def main():
    loader_actual   = EnergyDataLoader(region=REGION)
    data_actual     = loader_actual.load_region_data()

    loader_forecast = EnergyDataLoader(region=REGION, data_version="forecast", forecast_type="RF")
    data_forecast   = loader_forecast.load_region_data()

    assert len(data_actual) == len(data_forecast)

    # battery setup (unchanged)
    stats = await EnergyDataLoader.get_summary_stats_from_ag(data_actual)
    batt_loader = BatteryDataLoader(
        load_stats={
            "p25": stats.states[0].consumption.p25,
            "p75": stats.states[0].consumption.p75
        },
        duration_hours=4,
        soc_init=0.5,
        soc_min=0.0,
        soc_max=1.0,
        eta_c=0.95,
        eta_d=0.95,
        soc_target=0.5
    )
    battery = batt_loader.compute_battery_params()

    n_days = len(data_actual) // DATA_POINTS

    with OUTPUT_JSONL.open("w") as f_out:
        for day_index in range(n_days):
            print(f"Building day {day_index+1}/{n_days}")

            start = day_index * DATA_POINTS
            end   = (day_index + 1) * DATA_POINTS

            day_records_actual = data_actual[start:end]
            day_records_forecast = data_forecast[start:end]

            # === HISTORY (forecast + actual for past days)
            history = build_history_block(data_actual, data_forecast, day_index)

            # === MILP solved using FORECAST ONLY ===
            req = SolveFromRecordsRequest(
                battery=battery,
                records=day_records_actual,
                dt_hours=DT_HOURS,
                allow_export=True
            )
            prices_f, demand_f = records_to_arrays(day_records_actual)

            sol = solve_daily_milp(
                battery,
                DayInputs(
                    prices_buy=prices_f, prices_sell=prices_f, demand_MW=demand_f,
                    dt_hours=DT_HOURS, allow_export=True
                ),
                solver=None, solver_opts=None
            )

            # === MESSAGES ===
            user_msg = build_user_message(
                battery,
                day_records_forecast,
                day_records_actual,
                history
            )
            assistant_msg = build_assistant_message(sol)

            example = {
                "date": day_records_actual[0].timestamps.split(" ")[0],
                "user_message": user_msg,
                "assistant_message": assistant_msg
            }

            f_out.write(json.dumps(example) + "\n")

            # randomize SoC for next day
            battery.soc_init = np.round(np.random.rand(), 2)

    print(f"Dataset complete → {OUTPUT_JSONL}")


if __name__ == "__main__":
    asyncio.run(main())
