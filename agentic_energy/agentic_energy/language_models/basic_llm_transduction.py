# milp_mcp_client_direct.py
import os, sys, asyncio
import warnings
import contextlib
import io
from dotenv import load_dotenv
from agentics import AG
import numpy as np

from agentic_energy.schemas import (
    BatteryParams, DayInputs, SolveRequest, SolveResponse,
)
# Point to your server file
# SERVER_PATH = os.getenv("MCP_MILP_SERVER", "milp_mcp_server.py")

# Comprehensive error suppression for CrewAI stream issues
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")


load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")

async def client_calling(data_records: DayInputs = None, battery_details: BatteryParams= None):
    # Start server and get tool adapter
    try:
        if data_records is None or len(data_records) == 0:
            hours = np.arange(24)
            prices_buy = 50 + 20*np.sin(2*np.pi*hours/24)
            demand = 5 + 2*np.cos(2*np.pi*hours/24)
            prices_sell = prices_buy  # or different if allowing export
            day = DayInputs(
                prices_buy=prices_buy,
                demand_MW=demand,
                prices_sell=prices_sell,
                allow_export=True,
                dt_hours=1.0
            )

        if battery_details is None:
            battery_details = BatteryParams(
                capacity_MWh=20.0, soc_init=0.5, soc_min=0.10, soc_max=0.90,
                cmax_MW=6.0, dmax_MW=6.0, eta_c=0.95, eta_d=0.95, soc_target=0.5
            )
        
        req = SolveRequest(
            battery=battery_details,
            day=day,
            solver=None,
            solver_opts=None
            # solver_opts = {
            #     "TimeLimit": 300,        # Maximum solve time in seconds
            #     "MIPGap": 0.01,         # Stop when gap between best solution and bound < 1%
            #     "Threads": 4,           # Number of threads to use
            #     "OutputFlag": 1,        # 1 = show solver output, 0 = silent
            #     "LogToConsole": 1       # Print log to console
            # }
        )

    # TODO: work with prompt template: instructions + solverequest data
        source = AG(
            atype = SolveRequest,
            states = [req],
            # prompt_template = '''
            #     You are solving a daily battery scheduling optimization problem using Mixed Integer Linear Programming (MILP). 
            #     You are given a request object containing:
            #     - Hourly energy prices for buying and selling electricity.
            #     - Hourly electricity demand from a building or system.
            #     - Battery technical parameters including capacity_MWh: {battery_details.capacity_MWh}, charge/discharge power limits cmax_MW: {battery_details.cmax_MW}, dmax_MW: {battery_details.dmax_MW}, efficiencies - eta_c: {battery_details.eta_c}, eta_d: {battery_details.eta_d}, and state-of-charge soc_max: {battery_details.soc_max}, soc_min: {battery_details.soc_min} bounds.

            #     Your task is to:
            #     1. Determine the hourly charge , discharge, grid import, grid export, and SoC schedule for 24 hours.
            #     2. Minimize the total operational cost:
            #         total_cost = Σ_t [ (price_buy[t] {req.day.prices_buy} * import_MW[t] - price_sell[t] {req.day.prices_sell} * export_MW[t]) * dt_hours {req.day.dt_hours} ]
            #     3. Ensure all constraints are satisfied:
            #     - SoC at time t = SoC at time t-1 + (eta_c {battery_details.eta_c} * charge_MW[t]  - discharge_MW[t] / eta_d {battery_details.eta_d}) * dt_hours {req.day.dt_hours} / capacity_MWh {battery_details.capacity_MWh}
            #     - soc_min {battery_details.soc_min} ≤ SoC_t ≤ soc_max {battery_details.soc_max} for all t
            #     - 0 ≤ charge_MW[t]  ≤ cmax_MW {battery_details.cmax_MW}
            #     - 0 ≤ discharge_MW[t] ≤ dmax_MW {battery_details.dmax_MW}
            #     - import_MW[t] = max(0, demand_MW[t] + charge_MW[t]  - discharge_MW[t] - export_MW[t])
            #     - export_MW[t] ≥ 0 only if allow_export {req.day.allow_export} = True
            #     - initialize the soc variable at soc_init {battery_details.soc_init} at t=0, where t is the first hour of the day.
            #     - The battery SoC at the end of the day should reach soc_target {battery_details.soc_target} (if provided), else soc_init {battery_details.soc_init}.
            #     - Assume the battery can either charge  or discharge or stay idle in a given hour, not both. So try to schedule the battery in such a way.

            #     4. Output a JSON-compatible SolveResponse object with:
            #     - status: "success" or "failure"
            #     - message: optional diagnostic
            #     - objective_cost: the minimized total cost
            #     - charge_MW: list of hourly charge values (MW)
            #     - discharge_MW: list of hourly discharge values (MW)
            #     - import_MW: list of hourly grid import values (MW)
            #     - export_MW: list of hourly grid export values (MW)
            #     - soc: list of hourly state of charge values (fraction of capacity between 0 and 1)

            #     Make sure the final schedule satisfies all physical constraints and the objective function is minimized.
            # ''',
        )
        target = AG(
            atype=SolveResponse,
            max_iter=1,
            verbose_agent=True,
            reasoning=True,
            # '''Minimize the total objective cost i.e. price sell times grid export subtracted from price buy times grid import, 
            # given the battery constraints of following the rates and efficiencies of charging and discharging and staying with state of charge limits  for all the 24 timestamps,
            # and return the SolveResponse object with a goal to at least fulfill the demand_MW at each timestamp using the grid import and the battery
            # and maximize profit by selling the excess as grid export by taking advantage of the price variation.'''

            # TODO:may be feed data directly to the instructions


            instructions=f'''
                You are solving a daily battery scheduling optimization problem using Mixed Integer Linear Programming (MILP). 
                You are given a request object containing:
                - Hourly energy prices for buying and selling electricity.
                - Hourly electricity demand from a building or system.
                - Battery technical parameters including capacity_MWh: {battery_details.capacity_MWh}, charge/discharge power limits cmax_MW: {battery_details.cmax_MW}, dmax_MW: {battery_details.dmax_MW}, efficiencies - eta_c: {battery_details.eta_c}, eta_d: {battery_details.eta_d}, and state-of-charge soc_max: {battery_details.soc_max}, soc_min: {battery_details.soc_min} bounds.

                Your task is to:
                1. Determine the hourly charge , discharge, grid import, grid export, and SoC schedule for 24 hours.
                2. Minimize the total operational cost:
                    total_cost = Σ_t [ (price_buy[t] {req.day.prices_buy} * import_MW[t] - price_sell[t] {req.day.prices_sell} * export_MW[t]) * dt_hours {req.day.dt_hours} ]
                3. Ensure all constraints are satisfied:
                - SoC at time t = SoC at time t-1 + (eta_c {battery_details.eta_c} * charge_MW[t]  - discharge_MW[t] / eta_d {battery_details.eta_d}) * dt_hours {req.day.dt_hours} / capacity_MWh {battery_details.capacity_MWh}
                - soc_min {battery_details.soc_min} ≤ SoC_t ≤ soc_max {battery_details.soc_max} for all t
                - 0 ≤ charge_MW[t]  ≤ cmax_MW {battery_details.cmax_MW}
                - 0 ≤ discharge_MW[t] ≤ dmax_MW {battery_details.dmax_MW}
                - import_MW[t] = max(0, demand_MW[t] + charge_MW[t]  - discharge_MW[t] - export_MW[t])
                - export_MW[t] ≥ 0 only if allow_export {req.day.allow_export} = True
                - initialize the soc variable at soc_init {battery_details.soc_init} at t=0, where t is the first hour of the day.
                - The battery SoC at the end of the day should reach soc_target {battery_details.soc_target} (if provided), else soc_init {battery_details.soc_init}.
                - Assume the battery can either charge  or discharge or stay idle in a given hour, not both. So try to schedule the battery in such a way.

                4. Output a JSON-compatible SolveResponse object with:
                - status: "success" or "failure"
                - message: optional diagnostic
                - objective_cost: the minimized total cost
                - charge_MW: list of hourly charge values (MW)
                - discharge_MW: list of hourly discharge values (MW)
                - import_MW: list of hourly grid import values (MW)
                - export_MW: list of hourly grid export values (MW)
                - soc: list of hourly state of charge values (fraction of capacity between 0 and 1)

                Make sure the final schedule satisfies all physical constraints and the objective function is minimized.
            ''',
        )

        arr_res = await (target << source)

        # target = await source.amap(lambda x: milp_solve.call(x.model_dump()))

        print("\n— Arrays call —")
        print(arr_res.pretty_print())

        # help me convert arr_res to a Dictionary and then store it in some json file
        # which  i  can later load to analyze the results
    
        return arr_res, day, battery_details
        

    except Exception as e:
        print(f"💥 MCP client error: {e}")
        import traceback
        traceback.print_exc()
