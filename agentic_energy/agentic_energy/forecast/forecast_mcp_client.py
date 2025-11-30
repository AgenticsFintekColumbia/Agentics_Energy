# forecast_mcp_client_direct.py
import os, sys, asyncio
import warnings
import contextlib
import io
from dotenv import load_dotenv
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter
import json

from agentic_energy.schemas import (
    ForecastRequest,
    ForecastResponse,
    ForecastFeatures,
)

# -------------------------------------------------------------------
# Error / warning handling (same style as milp_mcp_client_direct.py)
# -------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")

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

# -------------------------------------------------------------------
# MCP server parameters – forecast server
# -------------------------------------------------------------------
params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "agentic_energy.forecast.forecast_mcp_server"],  # <--- your forecast server module
    env=os.environ,
)

# -------------------------------------------------------------------
# Main client logic
# -------------------------------------------------------------------
async def main():
    print("📈 Starting MCP Forecast Client...")

    try:
        with MCPServerAdapter(params) as tools:
            print("✅ Connected to Forecast MCP server")
            print("🛠️  Available tools:", [t.name for t in tools])

            def get_tool(name: str):
                for t in tools:
                    if t.name == name:
                        return t
                raise RuntimeError(f"Tool {name!r} not found")

            # Get tools
            forecast_check_models = get_tool("forecast_check_models")
            forecast_for_date_tool = get_tool("forecast_for_date")
            forecast_predict_tool = get_tool("forecast_predict")

            # --------------------------------------------------------
            # 1) Check models
            # --------------------------------------------------------
            print("\n🔍 Checking loaded forecast models...")
            check_call = (
                getattr(forecast_check_models, "call", None)
                or getattr(forecast_check_models, "run", None)
                or getattr(forecast_check_models, "__call__", None)
            )
            if check_call is None:
                raise RuntimeError("forecast_check_models has no callable interface")

            raw_check = check_call()  # no args
            print("📦 forecast_check_models result:")
            print(raw_check)

            # --------------------------------------------------------
            # 2) Forecast for a specific date
            # --------------------------------------------------------
            print("\n📆 Calling forecast_for_date...")
            date_args = {
                "date": "2018-01-01",   # ⚠️ change to a date present in data_IT.csv
                "target": "prices",     # or "consumption"
                "model_type": "LSTM",   # or "RF"
            }

            ffd_call = (
                getattr(forecast_for_date_tool, "call", None)
                or getattr(forecast_for_date_tool, "run", None)
                or getattr(forecast_for_date_tool, "__call__", None)
            )
            if ffd_call is None:
                raise RuntimeError("forecast_for_date has no callable interface")

            raw_ffd = ffd_call(**date_args)
            print("📊 forecast_for_date result:")
            print(raw_ffd)

            # --------------------------------------------------------
            # 3) Direct forecast_predict with dummy features
            # --------------------------------------------------------
            print("\n🤖 Calling forecast_predict with dummy features...")

            # Build dummy features for 24 hours
            dummy_features = []
            for hour in range(1, 25):
                dummy_features.append(
                    ForecastFeatures(
                        temperature=15.0,
                        radiation_direct_horizontal=0.0,
                        radiation_diffuse_horizontal=0.0,
                        hour=hour,
                        month=1,
                        is_weekday=1,
                        is_holiday=0,
                    )
                )

            req = ForecastRequest(
                target="prices",
                model_type="LSTM",
                features=dummy_features,
                timestamps=[f"2018-01-10T{hour:02d}:00:00" for hour in range(24)],
            )

            fp_call = (
                getattr(forecast_predict_tool, "call", None)
                or getattr(forecast_predict_tool, "run", None)
                or getattr(forecast_predict_tool, "__call__", None)
            )
            if fp_call is None:
                raise RuntimeError("forecast_predict has no callable interface")

            # NOTE: parameter name in server is `args: ForecastRequest`
            raw_fp = fp_call(args=req.model_dump())

            # Try to parse into ForecastResponse similar to SolveResponse
            try:
                if isinstance(raw_fp, dict):
                    res = ForecastResponse(**raw_fp)
                elif isinstance(raw_fp, str):
                    parsed = json.loads(raw_fp)
                    res = ForecastResponse(**parsed)
                else:
                    res = ForecastResponse.model_validate(raw_fp)

                print("✅ forecast_predict successful!")
                # print(f"📌 Status: {res.status}")
                # print(f"🎯 Target: {res.target}, Model: {res.model_type}")
                # print(f"🔢 Num predictions: {res.num_predictions}")

                if res.predictions:
                    print("First 5 predictions:", res.predictions[:5])

            except Exception as parse_error:
                print(f"❌ Error parsing forecast_predict response: {parse_error}")
                print(f"🔍 Raw response type: {type(raw_fp)}")
                print(f"🔍 Raw response: {raw_fp}")

    except Exception as e:
        print(f"💥 MCP forecast client error: {e}")
        import traceback
        traceback.print_exc()

# -------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("\n🎉 Forecast client completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n💥 Application error: {e}")
    finally:
        with suppress_stderr():
            import time
            time.sleep(0.2)
        print("👋 Goodbye!")
