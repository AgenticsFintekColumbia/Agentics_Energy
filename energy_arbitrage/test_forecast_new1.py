import os
import sys
import json
from pathlib import Path
from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv

load_dotenv()

# Use the FIXED server path
server_path = os.getenv("FORECAST_SERVER_PATH")

if not server_path or not Path(server_path).exists():
    print(f"❌ Server not found: {server_path}")
    sys.exit(1)

server_params = StdioServerParameters(
    command=sys.executable,
    args=[server_path],
    env={**os.environ},
)

print("🔮 Testing Forecast MCP Tool\n")

with MCPServerAdapter(server_params) as server_tools:
    print(f"✅ Connected!")


    print(f"🛠️  Tools: {[tool.name for tool in server_tools]}\n")
    
    # Get tool
    forecast_tool = [t for t in server_tools if t.name == "forecast_for_date"][0]
    call_fn = getattr(forecast_tool, "call", None) or getattr(forecast_tool, "run", None)
    
    # Test it - PASS ARGUMENTS DIRECTLY, NOT IN args={...}
    print("Testing LSTM price forecast for 2019-06-15...")
    result_raw = call_fn(
        date="2019-06-15",
        target="prices",
        model_type="LSTM"
    )
    
    # Parse JSON string
    if isinstance(result_raw, str):
        result = json.loads(result_raw)
    else:
        result = result_raw
    
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"MAE: {result['mae']:.2f} €/MWh")
        print(f"RMSE: {result['rmse']:.2f} €/MWh")
        print(f"Predictions (first 6): {result['predictions'][:6]}")
        print(f"Actual (first 6): {result['actual'][:6]}")
    else:
        print(f"Error: {result.get('message', 'Unknown')}")

print("\n✅ Done!")

#     print(f"🛠️  Tools: {[tool.name for tool in server_tools]}\n")
    
#     # ========================================
#     # TEST 1: Check models (simplest test)
#     # ========================================
#     print("="*60)
#     print("TEST 1: Checking model availability")
#     print("="*60)
    
#     check_tool = [t for t in server_tools if t.name == "forecast_check_models"][0]
#     call_fn_check = getattr(check_tool, "call", None) or getattr(check_tool, "run", None)
    
#     result_check = call_fn_check(args={})
    
#     print(f"Type: {type(result_check)}")
#     print(f"Content: {result_check}")
    
#     # Parse if needed
#     if isinstance(result_check, str):
#         result_check = json.loads(result_check)
    
#     print(f"\n✅ Model Status:")
#     for model, status in result_check['models'].items():
#         print(f"  {model}: {status}")
    
#     # ========================================
#     # TEST 2: Try forecast_for_date
#     # ========================================
#     print("\n" + "="*60)
#     print("TEST 2: Forecast for specific date")
#     print("="*60)
    
#     forecast_tool = [t for t in server_tools if t.name == "forecast_for_date"][0]
#     call_fn_forecast = getattr(forecast_tool, "call", None) or getattr(forecast_tool, "run", None)
    
#     print("Calling forecast_for_date...")
#     result_raw = call_fn_forecast(args={
#         "date": "2019-06-15",
#         "target": "prices",
#         "model_type": "LSTM"
#     })
    
#     print(f"\nType: {type(result_raw)}")
#     print(f"Length: {len(str(result_raw))}")
    
#     # Check if it's a Tool result object or direct value
#     if hasattr(result_raw, 'result'):
#         print(f"Has .result attribute: {result_raw.result}")
#         result = result_raw.result
#     elif hasattr(result_raw, 'content'):
#         print(f"Has .content attribute: {result_raw.content}")
#         result = result_raw.content
#     elif isinstance(result_raw, str):
#         print(f"String result (first 300 chars): {result_raw[:300]}")
#         if result_raw.strip():
#             result = json.loads(result_raw)
#         else:
#             print("❌ Empty string!")
#             result = {"status": "error", "message": "Empty result"}
#     elif isinstance(result_raw, dict):
#         print(f"Direct dict result")
#         result = result_raw
#     else:
#         print(f"Unknown type, trying to convert: {result_raw}")
#         result = {"status": "error", "message": f"Unknown return type: {type(result_raw)}"}
    
#     # Display result
#     print(f"\n📊 Final Result:")
#     print(f"  Status: {result.get('status', 'UNKNOWN')}")
    
#     if result.get('status') == 'success':
#         print(f"  ✅ MAE: {result.get('mae', 'N/A'):.2f} €/MWh")
#         print(f"  ✅ RMSE: {result.get('rmse', 'N/A'):.2f} €/MWh")
#         print(f"  📈 Predictions (first 6): {result.get('predictions', [])[:6]}")
#     else:
#         print(f"  ❌ Error: {result.get('message', 'Unknown error')}")

# print("\n✅ Done!")







    # print(f"🛠️  Tools: {[tool.name for tool in server_tools]}\n")
    
    # # Test forecast_for_date
    # forecast_tool = [t for t in server_tools if t.name == "forecast_for_date"][0]
    # call_fn = getattr(forecast_tool, "call", None) or getattr(forecast_tool, "run", None)
    

    # print("Testing LSTM price forecast for 2019-06-15...")
    # result_raw = call_fn(args={
    #     "date": "2019-06-15",
    #     "target": "prices",
    #     "model_type": "LSTM"
    # })
    
    # # Handle different return types from CrewAI MCP adapter
    # if hasattr(result_raw, 'result'):
    #     result = result_raw.result  # Wrapped in ToolResult object
    # elif hasattr(result_raw, 'content'):
    #     result = result_raw.content  # Alternative wrapper
    # elif isinstance(result_raw, str):
    #     result = json.loads(result_raw) if result_raw.strip() else {"status": "error", "message": "Empty response"}
    # else:
    #     result = result_raw  # Already a dict

    # print("Testing LSTM price forecast for 2019-06-15...")
    # result_raw = call_fn(args={
    #     "date": "2019-06-15",
    #     "target": "prices",
    #     "model_type": "LSTM"
    # })
    
    # # Parse JSON string to dict
    # result = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
    
    # print(f"Status: {result['status']}")
    # if result['status'] == 'success':
    #     print(f"MAE: {result['mae']:.2f} €/MWh")
    #     print(f"RMSE: {result['rmse']:.2f} €/MWh")
    #     print(f"Predictions: {result['predictions'][:6]}")
    # else:
    #     print(f"Error: {result.get('message', 'Unknown')}")
    # print("Testing LSTM price forecast for 2019-06-15...")
    # result = call_fn(args={
    #     "date": "2019-06-15",
    #     "target": "prices",
    #     "model_type": "LSTM"
    # })
    
    # print(f"Status: {result['status']}")
    # if result['status'] == 'success':
    #     print(f"MAE: {result['mae']:.2f} €/MWh")
    #     print(f"RMSE: {result['rmse']:.2f} €/MWh")
    #     print(f"Predictions: {result['predictions'][:6]}")
    # else:
    #     print(f"Error: {result.get('message', 'Unknown')}")

# print("\n✅ Done!")




# import os
# import sys
# from pathlib import Path
# from mcp import StdioServerParameters
# from crewai_tools import MCPServerAdapter
# from dotenv import load_dotenv

# load_dotenv()

# # Use the FIXED server path
# server_path = os.getenv("FORECAST_SERVER_PATH")

# if not server_path or not Path(server_path).exists():
#     print(f"❌ Server not found: {server_path}")
#     sys.exit(1)

# server_params = StdioServerParameters(
#     command=sys.executable,
#     args=[server_path],
#     env={**os.environ},
# )

# print("🔮 Testing Forecast MCP Tool\n")

# with MCPServerAdapter(server_params) as server_tools:
#     print(f"✅ Connected!")
#     print(f"🛠️  Tools: {[tool.name for tool in server_tools]}\n")
    
#     # Test forecast_for_date
#     forecast_tool = [t for t in server_tools if t.name == "forecast_for_date"][0]
#     call_fn = getattr(forecast_tool, "call", None) or getattr(forecast_tool, "run", None)
    
#     print("Testing LSTM price forecast for 2019-06-15...")
#     result = call_fn(args={
#         "date": "2019-06-15",
#         "target": "prices",
#         "model_type": "LSTM"
#     })
    
#     print(f"Status: {result['status']}")
#     if result['status'] == 'success':
#         print(f"MAE: {result['mae']:.2f} €/MWh")
#         print(f"RMSE: {result['rmse']:.2f} €/MWh")
#         print(f"Predictions: {result['predictions'][:6]}")
#     else:
#         print(f"Error: {result.get('message', 'Unknown')}")

# print("\n✅ Done!")



# import os
# from mcp import StdioServerParameters
# from crewai_tools import MCPServerAdapter
# from dotenv import load_dotenv

# load_dotenv()

# # Set the path to your standalone server
# os.environ["FORECAST_SERVER_PATH"] = os.path.join(
#     os.getcwd(), 
#     "agentic_energy", 
#     "forecast", 
#     "forecast_mcp_server.py"
# )

# # Use same pattern as your working example
# server_params = StdioServerParameters(
#     command="python",  # or sys.executable
#     args=[os.getenv("FORECAST_SERVER_PATH")],
#     env={**os.environ},
# )

# print("🔮 Testing Forecast MCP Tool\n")

# with MCPServerAdapter(server_params) as server_tools:
#     print(f"✅ Connected!")
#     print(f"🛠️  Tools: {[tool.name for tool in server_tools]}\n")
    
#     # Get tool
#     forecast_tool = [t for t in server_tools if t.name == "forecast_for_date"][0]
#     call_fn = getattr(forecast_tool, "call", None) or getattr(forecast_tool, "run", None)
    
#     # Test it
#     print("Testing LSTM price forecast for 2019-06-15...")
#     result = call_fn(args={
#         "date": "2019-06-15",
#         "target": "prices",
#         "model_type": "LSTM"
#     })
    
#     print(f"Status: {result['status']}")
#     if result['status'] == 'success':
#         print(f"MAE: {result['mae']:.2f} €/MWh")
#         print(f"RMSE: {result['rmse']:.2f} €/MWh")
#         print(f"Predictions: {result['predictions'][:6]}")
#     else:
#         print(f"Error: {result.get('message', 'Unknown')}")

# print("\n✅ Done!")


# """
# Test forecast MCP - matching milp_mcp_client.py pattern
# """
# import os
# import sys
# import asyncio
# from dotenv import load_dotenv
# from mcp import StdioServerParameters
# from crewai_tools import MCPServerAdapter

# load_dotenv()
# os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")

# # Use standalone server (no package import issues)
# server_path = os.path.join(os.getcwd(), "agentic_energy", "forecast", "forecast_mcp_server_standalone.py")
# params = StdioServerParameters(
#     command=sys.executable,
#     args=[server_path],
#     env=os.environ,
# )

# async def main():
#     print("🔮 Testing Forecast MCP Tool\n")
    
#     try:
#         with MCPServerAdapter(params) as tools:
#             print("✅ Connected to MCP server")
#             print(f"🛠️  Available tools: {[t.name for t in tools]}\n")
            
#             # Get tool
#             def get_tool(name: str):
#                 for t in tools:
#                     if t.name == name:
#                         return t
#                 raise RuntimeError(f"Tool {name!r} not found")
            
#             forecast_tool = get_tool("forecast_for_date")
            
#             # Get call function
#             call_fn = getattr(forecast_tool, "call", None) or getattr(forecast_tool, "run", None)
#             if call_fn is None:
#                 raise RuntimeError("Tool has no callable interface")
            
#             # Test: LSTM price forecast
#             print("Testing LSTM price forecast for 2019-06-15...")
#             result = call_fn(args={
#                 "date": "2019-06-15",
#                 "target": "prices",
#                 "model_type": "LSTM"
#             })
            
#             print(f"Status: {result['status']}")
#             if result['status'] == 'success':
#                 print(f"MAE: {result['mae']:.2f} €/MWh")
#                 print(f"RMSE: {result['rmse']:.2f} €/MWh")
#                 print(f"Predictions (first 6): {result['predictions'][:6]}")
#                 print(f"Actual (first 6): {result['actual'][:6]}")
#             else:
#                 print(f"Error: {result.get('message', 'Unknown error')}")
                
#     except Exception as e:
#         print(f"💥 Error: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     asyncio.run(main())
#     print("\n✅ Done!")




# """
# Diagnostic script - test if forecast server can start
# """
# import sys
# import subprocess

# print("Testing forecast MCP server startup...\n")

# # Test 1: Can we import the module?
# print("1. Testing module import...")
# try:
#     sys.path.insert(0, '.')
#     import agentic_energy.forecast.forecast_mcp_server as server
#     print("   ✅ Module imports successfully")
# except Exception as e:
#     print(f"   ❌ Import failed: {e}")
#     print("\n   Installing missing packages...")
#     subprocess.run([sys.executable, "-m", "pip", "install", "holidays", "--quiet"])
#     print("   Try running this script again")
#     sys.exit(1)

# # Test 2: Check if models exist
# print("\n2. Checking for trained models...")
# import os
# from pathlib import Path

# models_dir = Path("agentic_energy/trained_models")
# if not models_dir.exists():
#     print(f"   ❌ Models directory not found: {models_dir}")
#     print("   Please copy pickle files to agentic_energy/trained_models/")
#     sys.exit(1)

# required_models = ['rf_prices.pkl', 'rf_consumption.pkl', 'lstm_prices.pkl', 'lstm_consumption.pkl']
# missing = []
# for model in required_models:
#     if not (models_dir / model).exists():
#         missing.append(model)

# if missing:
#     print(f"   ❌ Missing models: {missing}")
#     print("   Please copy all 4 pickle files to agentic_energy/trained_models/")
#     sys.exit(1)

# print("   ✅ All 4 models found")

# # Test 3: Check data file
# print("\n3. Checking for data_IT.csv...")
# data_paths = [
#     "agentic_energy/data/data_IT.csv",
#     "data_IT.csv",
#     r"C:\Users\16467\OneDrive\Desktop\Columbia\Agentics\Another\Agentics_for_EnergyArbitrage_Battery\energy_arbitrage\agentic_energy\data\data_IT.csv"
# ]

# data_found = False
# for path in data_paths:
#     if Path(path).exists():
#         print(f"   ✅ Data found at: {path}")
#         data_found = True
#         break

# if not data_found:
#     print("   ⚠️  data_IT.csv not found in common locations")
#     print("   The server will search multiple paths at runtime")

# # Test 4: Try to start server manually
# print("\n4. Testing server startup...")
# print("   Starting server (press Ctrl+C to stop)...\n")

# try:
#     subprocess.run([
#         sys.executable,
#         "-m",
#         "agentic_energy.forecast.forecast_mcp_server"
#     ])
# except KeyboardInterrupt:
#     print("\n   ✅ Server started successfully (you stopped it)")
# except Exception as e:
#     print(f"\n   ❌ Server failed to start: {e}")
#     sys.exit(1)


# """
# Test forecast MCP - matching milp_mcp_client.py pattern
# """
# import os
# import sys
# import asyncio
# from dotenv import load_dotenv
# from mcp import StdioServerParameters
# from crewai_tools import MCPServerAdapter

# load_dotenv()
# os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")

# params = StdioServerParameters(
#     command=sys.executable,
#     args=["-m", "agentic_energy.forecast.forecast_mcp_server"],
#     env=os.environ,
# )

# async def main():
#     print("🔮 Testing Forecast MCP Tool\n")
    
#     try:
#         with MCPServerAdapter(params) as tools:
#             print("✅ Connected to MCP server")
#             print(f"🛠️  Available tools: {[t.name for t in tools]}\n")
            
#             # Get tool
#             def get_tool(name: str):
#                 for t in tools:
#                     if t.name == name:
#                         return t
#                 raise RuntimeError(f"Tool {name!r} not found")
            
#             forecast_tool = get_tool("forecast_for_date")
            
#             # Get call function
#             call_fn = getattr(forecast_tool, "call", None) or getattr(forecast_tool, "run", None)
#             if call_fn is None:
#                 raise RuntimeError("Tool has no callable interface")
            
#             # Test: LSTM price forecast
#             print("Testing LSTM price forecast for 2019-06-15...")
#             result = call_fn(args={
#                 "date": "2019-06-15",
#                 "target": "prices",
#                 "model_type": "LSTM"
#             })
            
#             print(f"Status: {result['status']}")
#             if result['status'] == 'success':
#                 print(f"MAE: {result['mae']:.2f} €/MWh")
#                 print(f"RMSE: {result['rmse']:.2f} €/MWh")
#                 print(f"Predictions (first 6): {result['predictions'][:6]}")
#                 print(f"Actual (first 6): {result['actual'][:6]}")
#             else:
#                 print(f"Error: {result.get('message', 'Unknown error')}")
                
#     except Exception as e:
#         print(f"💥 Error: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     asyncio.run(main())
#     print("\n✅ Done!")