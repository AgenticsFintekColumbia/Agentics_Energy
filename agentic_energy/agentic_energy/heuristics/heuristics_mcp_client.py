# agentic_energy/heuristics/heuristic_mcp_client_direct.py

import os
import sys
import asyncio
import warnings
import contextlib
import io
import json

from dotenv import load_dotenv
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

from agentic_energy.schemas import (
    BatteryParams,
    DayInputs,
    SolveRequest,
    SolveResponse,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")


@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
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
    # NOTE: adjust this module path if your server file lives elsewhere
    args=["-m", "agentic_energy.heuristics.heuristic_mcp_server"],
    env=os.environ,
)


async def main():
    print("⚡ Starting MCP Heuristic Battery Client...")

    try:
        with MCPServerAdapter(params) as tools:
            print("✅ Connected to Heuristic MCP server")
            print("🛠️  Available tools:", [t.name for t in tools])

            def get_tool(name: str):
                for t in tools:
                    if t.name == name:
                        return t
                raise RuntimeError(f"Tool {name!r} not found")

            # Grab both tools
            heuristic_time_solve = get_tool("heuristic_time_solve")
            heuristic_quantile_solve = get_tool("heuristic_quantile_solve")

            # ------------------------------------------------------------------
            # 1. Build a test SolveRequest (same style as MILP client)
            # ------------------------------------------------------------------
            battery = BatteryParams(
                capacity_MWh=20.0,
                soc_init=0.5,
                soc_min=0.10,
                soc_max=0.90,
                cmax_MW=6.0,
                dmax_MW=6.0,
                eta_c=0.95,
                eta_d=0.95,
                soc_target=0.5,
            )

            day = DayInputs(
                prices_buy=[0.12] * 6 + [0.15] * 6 + [0.22] * 6 + [0.16] * 6,
                demand_MW=[0.9] * 24,
                allow_export=False,
                dt_hours=1.0,
            )

            # Example 1: time-window heuristic
            req_time = SolveRequest(
                battery=battery,
                day=day,
                solver=None,
                solver_opts={
                    "mode": "time",
                    # You could override windows here if desired:
                    # "charge_windows": [(2, 6), (10, 16), (20, 22)],
                    # "discharge_windows": [(0, 2), (6, 10), (16, 20), (22, 24)],
                },
            )

            print("\n⏰ Running time-window heuristic optimization...")

            call_fn_time = (
                getattr(heuristic_time_solve, "call", None)
                or getattr(heuristic_time_solve, "run", None)
                or getattr(heuristic_time_solve, "__call__", None)
            )
            if call_fn_time is None:
                raise RuntimeError("heuristic_time_solve tool has no callable interface")

            raw_time = call_fn_time(solverequest=req_time.model_dump())

            try:
                if isinstance(raw_time, dict):
                    res_time = SolveResponse(**raw_time)
                elif isinstance(raw_time, str):
                    parsed = json.loads(raw_time)
                    res_time = SolveResponse(**parsed)
                else:
                    res_time = SolveResponse.model_validate(raw_time)

                print("✅ Time heuristic successful!")
                print(f"📈 Status: {res_time.status}")
                print(f"💰 Objective cost: {res_time.objective_cost:.4f}")

                if res_time.charge_MW and res_time.discharge_MW:
                    total_charge = sum(res_time.charge_MW)
                    total_discharge = sum(res_time.discharge_MW)
                    print(f"🔋 Total charging:   {total_charge:.2f} MWh")
                    print(f"⚡ Total discharging: {total_discharge:.2f} MWh")
                    print(f"SoC_length", len(res_time.soc) if res_time.soc else "N/A")

            except Exception as parse_error:
                print(f"❌ Error parsing time-heuristic response: {parse_error}")
                print(f"🔍 Raw response type: {type(raw_time)}")
                print(f"🔍 Raw response: {raw_time}")

            # ------------------------------------------------------------------
            # 2. Quantile-based heuristic on the same day
            # ------------------------------------------------------------------
            req_quant = SolveRequest(
                battery=battery,
                day=day,
                solver=None,
                solver_opts={
                    "mode": "quantile",
                    "low_q": 0.30,
                    "high_q": 0.70,
                },
            )

            print("\n📊 Running quantile-based heuristic optimization...")

            call_fn_quant = (
                getattr(heuristic_quantile_solve, "call", None)
                or getattr(heuristic_quantile_solve, "run", None)
                or getattr(heuristic_quantile_solve, "__call__", None)
            )
            if call_fn_quant is None:
                raise RuntimeError("heuristic_quantile_solve tool has no callable interface")

            raw_quant = call_fn_quant(solverequest=req_quant.model_dump())

            try:
                if isinstance(raw_quant, dict):
                    res_quant = SolveResponse(**raw_quant)
                elif isinstance(raw_quant, str):
                    parsed = json.loads(raw_quant)
                    res_quant = SolveResponse(**parsed)
                else:
                    res_quant = SolveResponse.model_validate(raw_quant)

                print("✅ Quantile heuristic successful!")
                print(f"📈 Status: {res_quant.status}")
                print(f"💰 Objective cost: {res_quant.objective_cost:.4f}")

                if res_quant.charge_MW and res_quant.discharge_MW:
                    total_charge = sum(res_quant.charge_MW)
                    total_discharge = sum(res_quant.discharge_MW)
                    print(f"🔋 Total charging:   {total_charge:.2f} MWh")
                    print(f"⚡ Total discharging: {total_discharge:.2f} MWh")
                    print(f"SoC_length", len(res_quant.soc) if res_quant.soc else "N/A")


            except Exception as parse_error:
                print(f"❌ Error parsing quantile-heuristic response: {parse_error}")
                print(f"🔍 Raw response type: {type(raw_quant)}")
                print(f"🔍 Raw response: {raw_quant}")


    except Exception as e:
        print(f"💥 MCP heuristic client error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("\n🎉 Heuristic client completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n💥 Application error: {e}")
    finally:
        with suppress_stderr():
            import time

            time.sleep(0.2)
        print("👋 Goodbye!")
