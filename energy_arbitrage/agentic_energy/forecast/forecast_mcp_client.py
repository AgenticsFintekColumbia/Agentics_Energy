"""
Forecast MCP Client - Test client for forecast MCP server
Tests price and consumption forecasting tools
"""
import os
import sys
import asyncio
import warnings
import contextlib
import io
from dotenv import load_dotenv
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

from agentic_energy.schemas import ForecastRequest, ForecastResponse, ForecastFeatures

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*I/O operation on closed file.*")

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output"""
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = old_stderr

load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")

# Configure MCP server parameters
params = StdioServerParameters(
    command=sys.executable,
    args=["-m", "agentic_energy.forecast.forecast_mcp_server"],
    env=os.environ,
)

async def main():
    print("🔮 Starting Forecast MCP Client...")
    
    try:
        with MCPServerAdapter(params) as tools:
            print("✅ Connected to MCP server")
            print("🛠️  Available tools:", [t.name for t in tools])
            
            def get_tool(name: str):
                for t in tools:
                    if t.name == name:
                        return t
                raise RuntimeError(f"Tool {name!r} not found")
            
            # Get tools
            forecast_predict = get_tool("forecast_predict")
            forecast_check = get_tool("forecast_check_models")
            
            # 1. Check models status
            print("\n" + "="*80)
            print("STEP 1: Checking model availability")
            print("="*80)
            
            call_fn = getattr(forecast_check, "call", None) or getattr(forecast_check, "run", None)
            status = call_fn(args={})
            
            print("\n📊 Model Status:")
            for model, state in status['models'].items():
                print(f"  {model}: {state}")
            
            # 2. Test RF forecast for prices
            print("\n" + "="*80)
            print("STEP 2: Testing RF price forecast (24 hours)")
            print("="*80)
            
            # Create sample features for 24 hours in January (winter)
            features_24h = []
            for hour in range(1, 25):
                features_24h.append(ForecastFeatures(
                    temperature=10.0 + 5.0 * (hour / 24),  # Temperature varies through day
                    radiation_direct_horizontal=max(0, 500 * (1 - abs(hour - 12) / 12)),  # Peak at noon
                    radiation_diffuse_horizontal=max(0, 200 * (1 - abs(hour - 12) / 12)),
                    hour=hour,
                    month=1,  # January
                    is_weekday=1,
                    is_holiday=0
                ))
            
            timestamps = [f"2025-01-15T{h-1:02d}:00:00Z" for h in range(1, 25)]
            
            req_rf_prices = ForecastRequest(
                target="prices",
                model_type="RF",
                features=features_24h,
                timestamps=timestamps
            )
            
            call_fn_predict = getattr(forecast_predict, "call", None) or getattr(forecast_predict, "run", None)
            response_rf = call_fn_predict(args=req_rf_prices.model_dump())
            
            if isinstance(response_rf, dict):
                res_rf = ForecastResponse(**response_rf)
            else:
                res_rf = ForecastResponse.model_validate(response_rf)
            
            print(f"\n✅ RF Price Forecast Status: {res_rf.status}")
            if res_rf.status == "success":
                print(f"📊 Generated {res_rf.num_predictions} predictions")
                print(f"💰 Price range: €{min(res_rf.predictions):.2f} - €{max(res_rf.predictions):.2f}")
                print(f"💰 Average price: €{sum(res_rf.predictions)/len(res_rf.predictions):.2f}")
                print("\n📈 Sample predictions (first 6 hours):")
                for i in range(min(6, len(res_rf.predictions))):
                    print(f"  Hour {i+1}: €{res_rf.predictions[i]:.4f}")
            else:
                print(f"❌ Error: {res_rf.message}")
            
            # 3. Test LSTM forecast for consumption
            print("\n" + "="*80)
            print("STEP 3: Testing LSTM consumption forecast (24 hours)")
            print("="*80)
            
            req_lstm_consumption = ForecastRequest(
                target="consumption",
                model_type="LSTM",
                features=features_24h,
                timestamps=timestamps
            )
            
            response_lstm = call_fn_predict(args=req_lstm_consumption.model_dump())
            
            if isinstance(response_lstm, dict):
                res_lstm = ForecastResponse(**response_lstm)
            else:
                res_lstm = ForecastResponse.model_validate(response_lstm)
            
            print(f"\n✅ LSTM Consumption Forecast Status: {res_lstm.status}")
            if res_lstm.status == "success":
                print(f"📊 Generated {res_lstm.num_predictions} predictions")
                print(f"⚡ Consumption range: {min(res_lstm.predictions):.2f} - {max(res_lstm.predictions):.2f} GWh")
                print(f"⚡ Average consumption: {sum(res_lstm.predictions)/len(res_lstm.predictions):.2f} GWh")
                print("\n📈 Sample predictions (first 6 hours):")
                for i in range(min(6, len(res_lstm.predictions))):
                    print(f"  Hour {i+1}: {res_lstm.predictions[i]:.4f} GWh")
            else:
                print(f"❌ Error: {res_lstm.message}")
            
            # 4. Test LSTM forecast for prices
            print("\n" + "="*80)
            print("STEP 4: Testing LSTM price forecast (24 hours)")
            print("="*80)
            
            req_lstm_prices = ForecastRequest(
                target="prices",
                model_type="LSTM",
                features=features_24h,
                timestamps=timestamps
            )
            
            response_lstm_p = call_fn_predict(args=req_lstm_prices.model_dump())
            
            if isinstance(response_lstm_p, dict):
                res_lstm_p = ForecastResponse(**response_lstm_p)
            else:
                res_lstm_p = ForecastResponse.model_validate(response_lstm_p)
            
            print(f"\n✅ LSTM Price Forecast Status: {res_lstm_p.status}")
            if res_lstm_p.status == "success":
                print(f"📊 Generated {res_lstm_p.num_predictions} predictions")
                print(f"💰 Price range: €{min(res_lstm_p.predictions):.2f} - €{max(res_lstm_p.predictions):.2f}")
                print(f"💰 Average price: €{sum(res_lstm_p.predictions)/len(res_lstm_p.predictions):.2f}")
                print("\n📈 Sample predictions (first 6 hours):")
                for i in range(min(6, len(res_lstm_p.predictions))):
                    print(f"  Hour {i+1}: €{res_lstm_p.predictions[i]:.4f}")
            else:
                print(f"❌ Error: {res_lstm_p.message}")
            
            # 5. Compare RF vs LSTM for same input
            if res_rf.status == "success" and res_lstm_p.status == "success":
                print("\n" + "="*80)
                print("COMPARISON: RF vs LSTM Price Forecasts")
                print("="*80)
                
                avg_diff = sum(abs(rf - lstm) for rf, lstm in zip(res_rf.predictions, res_lstm_p.predictions)) / len(res_rf.predictions)
                print(f"📊 Average absolute difference: €{avg_diff:.4f}")
                
                print("\n📈 Hour-by-hour comparison (first 6 hours):")
                print("  Hour | RF Price | LSTM Price | Difference")
                print("  " + "-"*50)
                for i in range(min(6, len(res_rf.predictions))):
                    diff = res_lstm_p.predictions[i] - res_rf.predictions[i]
                    print(f"  {i+1:4d} | €{res_rf.predictions[i]:7.4f} | €{res_lstm_p.predictions[i]:9.4f} | €{diff:+9.4f}")
            
    except Exception as e:
        print(f"💥 MCP client error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("\n" + "="*80)
        print("🎉 Client completed successfully!")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n💥 Application error: {e}")
    finally:
        with suppress_stderr():
            import time
            time.sleep(0.2)
        print("👋 Goodbye!")