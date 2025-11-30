# scripts/test_llm_solve_client.py

import os
import sys
import json
import asyncio

from mcp import StdioServerParameters
from crewai_tools.adapters.mcp_adapter import MCPServerAdapter

from agentic_energy.schemas import BatteryParams, DayInputs, SolveRequest, SolveResponse


def build_dummy_solve_request() -> SolveRequest:
    """Create a minimal but valid SolveRequest for testing llm_solve."""
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

    return SolveRequest(
        battery=battery,
        day=day,
        solver="llm",
        solver_opts=None,
    )


def call_llm_solve_via_mcp() -> SolveResponse:
    """
    Synchronous wrapper that:
      1. Starts the llm_mcp_server via stdio
      2. Calls the 'llm_solve' MCP tool
      3. Parses the JSON into a SolveResponse
    """
    # Command used for the MCP server (matches your __main__ at bottom)
    llm_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.language_models.basic_llm_amap"],
        env=os.environ,
    )

    solve_request = build_dummy_solve_request()

    # MCPServerAdapter will:
    #  - spawn the stdio MCP server
    #  - send listTools / callTool etc.
    with MCPServerAdapter(llm_params) as llm_tools:
        print("🔌 Connected to LLM MCP server, listing tools…")
        print("Available tools:", [t.name for t in llm_tools])

        print("🚀 Calling 'llm_solve'…")
        # Adapter expects a plain dict as arguments
        def get_tool(name: str):
            for t in llm_tools:
                if t.name == name:
                    return t
            raise RuntimeError(f"Tool {name!r} not found")

        # Grab both tools
        llm_solve = get_tool("llm_solve")

        call_fn = (
                getattr(llm_solve, "call", None)
                or getattr(llm_solve, "run", None)
                or getattr(llm_solve, "__call__", None)
            )
        if call_fn is None:
            raise RuntimeError("llm_solve tool has no callable interface")

        raw = call_fn(solverequest=solve_request.model_dump())
        print(raw)

        try:
            if isinstance(raw, dict):
                result = SolveResponse(**raw)
            elif isinstance(raw, str):
                parsed = json.loads(raw)
                result = SolveResponse(**parsed)
            else:
                result = SolveResponse.model_validate(raw)

            print("✅  Run successful!")
            print(f"📈 Status: {result.status}")
            print(f"💰 Objective cost: {result.objective_cost:.4f}")

            if result.charge_MW and result.discharge_MW:
                total_charge = sum(result.charge_MW)
                total_discharge = sum(result.discharge_MW)
                print(f"🔋 Total charging:   {total_charge:.2f} MWh")
                print(f"⚡ Total discharging: {total_discharge:.2f} MWh")
                print(f"SoC_length", len(result.soc) if result.soc else "N/A")

        except Exception as parse_error:
            print(f"❌ Error parsing response: {parse_error}")
            print(f"🔍 Raw response type: {type(raw)}")
            print(f"🔍 Raw response: {raw}")


if __name__ == "__main__":
    # Make sure project root is on sys.path so "agentic_energy" is importable
    ROOT = os.path.dirname(os.path.abspath(__file__))
    ROOT = os.path.dirname(ROOT)  # go one level up from scripts/
    if ROOT not in sys.path:
        sys.path.append(ROOT)

    try:
        response = call_llm_solve_via_mcp()
        # print(response)
        print("\n✅ Got SolveResponse from llm_solve MCP tool:")
        # print("  status        :", response.status)
        # print("  message       :", (response.message or "")[:200], "…")
        # print("  objective_cost:", response.objective_cost)
        # print("  len(charge_MW):", len(response.charge_MW))
        # print("  len(soc)      :", len(response.soc))
    except Exception as e:
        print("\n❌ Error while calling llm_solve:")
        print(type(e).__name__, ":", e)
