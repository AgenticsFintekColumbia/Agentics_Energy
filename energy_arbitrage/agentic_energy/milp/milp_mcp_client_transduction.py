# milp_mcp_client_direct.py
import os, sys, asyncio
import warnings
import contextlib
import io
from dotenv import load_dotenv
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter
import json
from agentics import Agentics as AG
import numpy as np

from agentic_energy.schemas import (
    BatteryParams, DayInputs, SolveRequest, SolveResponse,
    EnergyDataRecord, SolveFromRecordsRequest,
)
# Point to your server file
# SERVER_PATH = os.getenv("MCP_MILP_SERVER", "milp_mcp_server.py")

# Comprehensive error suppression for CrewAI stream issues
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")

# Suppress stderr during cleanup to hide the CrewAI stream error
@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    old_stderr = sys.stderr
    sys.stderr = mystderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr

load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")

params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "agentic_energy.milp.milp_mcp_server"],
    env=os.environ,
)

async def main():
    # Start server and get tool adapter
    try:
        with MCPServerAdapter(params) as tools:
            print("✅ Connected to MCP server")
            print("🛠️  Available tools:", [t.name for t in tools])

            def get_tool(name: str):
                for t in tools:
                    if t.name == name:
                        return t
                raise RuntimeError(f"Tool {name!r} not found")

            milp_solve = get_tool("milp_solve")

            # -------- A) Arrays path (SolveRequest) --------

            hours = np.arange(24)
            prices_buy = 50 + 20*np.sin(2*np.pi*hours/24)
            demand = 5 + 2*np.cos(2*np.pi*hours/24)
            prices_sell = prices_buy  # or different if allowing export
            req = SolveRequest(
                battery=BatteryParams(
                    capacity_MWh=20.0, soc_init=0.5, soc_min=0.10, soc_max=0.90,
                    cmax_MW=6.0, dmax_MW=6.0, eta_c=0.95, eta_d=0.95, soc_target=0.5
                ),
                day=DayInputs(
                    prices_buy=prices_buy,
                    demand_MW=demand,
                    prices_sell=prices_sell,
                    allow_export=True,
                    dt_hours=1.0
                ),
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


            source = AG(
                atype = SolveRequest,
                tools = [milp_solve],
                states = [req]
            )
            target = AG(
                atype=SolveResponse,
                tools=[milp_solve],
                max_iter=1,
                verbose_agent=False,
                reasoning=False,
                # '''Minimize the total objective cost i.e. price sell times grid export subtracted from price buy times grid import, 
                # given the battery constraints of following the rates and efficiencies of charging and discharging and staying with state of charge limits  for all the 24 timestamps,
                # and return the SolveResponse object with a goal to at least fulfill the demand_MW at each timestamp using the grid import and the battery
                # and maximize profit by selling the excess as grid export by taking advantage of the price variation.'''
                instructions='''
                    You are solving a daily battery scheduling optimization problem using Mixed Integer Linear Programming (MILP). 
                    You are given a request object containing:
                    - Hourly energy prices for buying and selling electricity.
                    - Hourly electricity demand from a building or system.
                    - Battery technical parameters including capacity_MWh, charge/discharge power limits cmax_MW and dmax_MW, efficiencies - eta_c and eta_d, and state-of-charge soc_max, soc_min bounds.

                    Your task is to:
                    1. Determine the hourly charge, discharge, grid import, grid export, and SoC schedule for 24 hours.
                    2. Minimize the total operational cost:
                        total_cost = Σ_t [ (price_buy[t] * import_MW[t] - price_sell[t] * export_MW[t]) * dt_hours ]
                    3. Ensure all constraints are satisfied:
                    - SoC at time t = SoC at time t-1 + (eta_c * charge_MW[t] - discharge_MW[t] / eta_d) * dt_hours / capacity_MWh
                    - soc_min ≤ SoC_t ≤ soc_max for all t
                    - 0 ≤ charge_MW[t] ≤ cmax_MW
                    - 0 ≤ discharge_MW[t] ≤ dmax_MW
                    - import_MW[t] = max(0, demand_MW[t] + charge_MW[t] - discharge_MW[t] - export_MW[t])
                    - export_MW[t] ≥ 0 only if allow_export = True
                    - initialize the soc variable at soc_init at t=0, where t is the first hour of the day.
                    - The battery SoC at the end of the day should reach soc_target (if provided), else soc_init.
                    - Assume the battery can either charge or discharge or stay idle in a given hour, not both. So try to schedule the battery in such a way.

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

    except Exception as e:
        print(f"💥 MCP client error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        # Run the main function
        asyncio.run(main())
        print("\n🎉 Client completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n💥 Application error: {e}")
    finally:
        # Suppress any cleanup errors from CrewAI
        with suppress_stderr():
            import time
            time.sleep(0.2)  # Give time for cleanup
        print("👋 Goodbye!")

