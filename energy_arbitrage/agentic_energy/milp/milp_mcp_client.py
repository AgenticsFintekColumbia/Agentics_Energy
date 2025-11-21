# milp_mcp_client_direct.py (adapter form without .call_tool)
import os, sys, asyncio
import warnings
import contextlib
import io
from dotenv import load_dotenv
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter
import json

from agentic_energy.schemas import (
    BatteryParams, DayInputs, SolveRequest, SolveResponse,
    EnergyDataRecord, SolveFromRecordsRequest,
)

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
    print("🔋 Starting MCP Battery Optimization Client...")
    
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

            # Create optimization request
            req = SolveRequest(
                battery=BatteryParams(
                    capacity_MWh=20.0, soc_init=0.5, soc_min=0.10, soc_max=0.90,
                    cmax_MW=6.0, dmax_MW=6.0, eta_c=0.95, eta_d=0.95, soc_target=0.5
                ),
                day=DayInputs(
                    prices_buy=[0.12]*6 + [0.15]*6 + [0.22]*6 + [0.16]*6,
                    demand_MW=[0.9]*24, allow_export=False, dt_hours=1.0
                ),
                solver=None, solver_opts=None
            )

            print("📊 Running battery optimization...")
            
            # Get the call function
            call_fn = getattr(milp_solve, "call", None) or getattr(milp_solve, "run", None) or getattr(milp_solve, "__call__", None)
            if call_fn is None:
                raise RuntimeError("Tool has no callable interface")

            # Call the optimization
            raw = call_fn(args=req.model_dump())
            
            # Parse response correctly (raw is already a dict, not JSON string)
            try:
                if isinstance(raw, dict):
                    res = SolveResponse(**raw)
                elif isinstance(raw, str):
                    # Only parse as JSON if it's actually a string
                    parsed = json.loads(raw)
                    res = SolveResponse(**parsed)
                else:
                    # Handle other types
                    res = SolveResponse.model_validate(raw)
                    
                print("✅ Optimization successful!")
                print(f"📈 Status: {res.status}")
                print(f"💰 Objective cost: ${res.objective_cost:.4f}")
                
                if res.charge_MW and res.discharge_MW:
                    total_charge = sum(res.charge_MW)
                    total_discharge = sum(res.discharge_MW)
                    print(f"🔋 Total charging: {total_charge:.2f} MWh")
                    print(f"⚡ Total discharging: {total_discharge:.2f} MWh")
                    
            except Exception as parse_error:
                print(f"❌ Error parsing response: {parse_error}")
                print(f"🔍 Raw response type: {type(raw)}")
                print(f"🔍 Raw response: {raw}")
                
    except Exception as e:
        print(f"💥 MCP client error: {e}")
        import traceback
        traceback.print_exc()

        # Records
        # records = [
        #     EnergyDataRecord(timestamps=f"2025-01-01T{h:02d}:00:00Z", prices=p, consumption=c)
        #     for h,(p,c) in enumerate([(0.12,0.9)]*6 + [(0.15,1.0)]*6 + [(0.22,1.4)]*6 + [(0.16,1.1)]*6)
        # ]
        # req2 = SolveFromRecordsRequest(
        #     battery=req.battery, records=records, dt_hours=1.0,
        #     allow_export=False, prices_sell=None, solver=None, solver_opts=None
        # )
        # call_fn2 = getattr(milp_solve_from_records, "call", None) or getattr(milp_solve_from_records, "run", None) or getattr(milp_solve_from_records, "__call__", None)
        # raw2 = call_fn2(args=req2.model_dump())
        # res2 = SolveResponse(**raw2)
        # print("\n— Records call —")
        # print("Status:", res2.status, "Objective ($):", res2.objective_cost)

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
