import os, sys, asyncio
import warnings
import contextlib
import io
import json
from typing import List, Optional

from dotenv import load_dotenv, find_dotenv
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

from agentic_energy.schemas import (
    BatteryParams, DayInputs, SolveRequest, SolveResponse,
    ReasoningRequest, ReasoningResponse,
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


load_dotenv(find_dotenv())
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")

params = StdioServerParameters(
    command=sys.executable,
    # Adjust module path if you place the server elsewhere:
    args=["-m", "agentic_energy.reasoning.reasoning_server"],
    env=os.environ,
)


async def main():
    print("🧠 Starting MCP Battery Reasoning Client...")

    try:
        with MCPServerAdapter(params) as tools:
            print("✅ Connected to Reasoning MCP server")
            print("🛠️  Available tools:", [t.name for t in tools])

            def get_tool(name: str):
                for t in tools:
                    if t.name == name:
                        return t
                raise RuntimeError(f"Tool {name!r} not found")

            reasoning_explain = get_tool("reasoning_explain")

            # ------------------------------------------------------------------
            # 1. Build a toy MILP-style solve request + response
            #    (in practice you’d pass the actual SolveRequest/SolveResponse
            #     you got from the MILP server or RL/heuristics).
            # ------------------------------------------------------------------
            solve_req = SolveRequest(
                battery=BatteryParams(
                    capacity_MWh=20.0,
                    soc_init=0.5,
                    soc_min=0.10,
                    soc_max=0.90,
                    cmax_MW=6.0,
                    dmax_MW=6.0,
                    eta_c=0.95,
                    eta_d=0.95,
                    soc_target=0.5,
                ),
                day=DayInputs(
                    prices_buy=[0.12] * 6 + [0.15] * 6 + [0.22] * 6 + [0.16] * 6,
                    demand_MW=[0.9] * 24,
                    allow_export=False,
                    dt_hours=1.0,
                ),
                solver=None,
                solver_opts=None,
            )

            # Dummy solve_response here; replace by actual output from MILP/RL/etc.
            # For demo purposes, we just build a simple "idle" solution.
            T = len(solve_req.day.prices_buy)
            solve_res = SolveResponse(
                status="success",
                message="Dummy solution for reasoning demo",
                objective_cost=sum(
                    solve_req.day.prices_buy[t]
                    * solve_req.day.demand_MW[t]
                    * solve_req.day.dt_hours
                    for t in range(T)
                ),
                charge_MW=[0.0] * T,
                discharge_MW=[0.0] * T,
                import_MW=solve_req.day.demand_MW,
                export_MW=[0.0] * T if solve_req.day.allow_export else None,
                soc=[solve_req.battery.soc_init] * (T + 1),
                decision=[0] * T,
                confidence=[1.0] * T,
            )

            # ------------------------------------------------------------------
            # 2. Build a ReasoningRequest for a specific timestep, say t = 5
            # ------------------------------------------------------------------
            t_index = 5
            r_req = ReasoningRequest(
                solve_request=solve_req,
                solve_response=solve_res,
                timestamp_index=t_index,
                context_window=6,  # e.g., 6 timesteps before/after
            )

            print(f"💬 Requesting explanation for timestep t={t_index}...")

            # Get the call function
            call_fn = (
                getattr(reasoning_explain, "call", None)
                or getattr(reasoning_explain, "run", None)
                or getattr(reasoning_explain, "__call__", None)
            )
            if call_fn is None:
                raise RuntimeError("Reasoning tool has no callable interface")

            # Call the reasoning tool
            raw = call_fn(args=r_req.model_dump())
            print(raw)
            # Parse response
            try:
                if isinstance(raw, dict):
                    res = ReasoningResponse(**raw)
                elif isinstance(raw, str):
                    parsed = json.loads(raw)
                    res = ReasoningResponse(**parsed)
                else:
                    res = ReasoningResponse.model_validate(raw)

                print("✅ Reasoning successful!")
                print(f"📝 Explanation (t={t_index}):\n{res.explanation}\n")
                print("📌 Key factors:", res.key_factors)
                print("🔎 Confidence:", res.confidence)

            except Exception as parse_error:
                print(f"❌ Error parsing reasoning response: {parse_error}")
                print(f"🔍 Raw response type: {type(raw)}")
                print(f"🔍 Raw response: {raw}")

    except Exception as e:
        print(f"💥 MCP reasoning client error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("\n🎉 Reasoning client completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n💥 Application error: {e}")
    finally:
        with suppress_stderr():
            import time

            time.sleep(0.2)
        print("👋 Goodbye!")
