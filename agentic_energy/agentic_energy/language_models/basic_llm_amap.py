from typing import Optional, Tuple, List, Dict, Any
from mcp.server.fastmcp import FastMCP


import os
import warnings
import numpy as np
from dotenv import load_dotenv

from agentics import AG
from agentics.core.llm_connections import get_llm_provider
from agentic_energy.schemas import (
    SolveResponse, SolveFromRecordsRequest, SolveRequest, 
    EnergyDataRecord, BatteryParams, DayInputs
)
from agentic_energy.data_loader import EnergyDataLoader

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")
mcp = FastMCP("Gemini")

def records_to_arrays(records: List[EnergyDataRecord]) -> Tuple[list, list]:
    rows = [r for r in records if r.prices is not None and r.consumption is not None]
    rows.sort(key=lambda r: r.timestamps)
    prices = [float(r.prices) for r in rows]
    demand = [float(r.consumption) for r in rows]
    return prices, demand

@mcp.tool()
async def llm_solve(solverequest: SolveRequest) -> SolveResponse:
    """Run day-ahead battery LLM and return schedules + cost."""
    return await solve_daily_llm(solverequest, llm_provider="gemini")

async def llm_solve_from_records(solverequest: SolveFromRecordsRequest) -> SolveResponse:
    """Run day-ahead LLM given a list of EnergyDataRecord rows."""
    prices, demand = records_to_arrays(solverequest.records)
    day = DayInputs(
        prices_buy=prices,
        demand_MW=demand,
        prices_sell=prices,
        allow_export=solverequest.allow_export,
        dt_hours=solverequest.dt_hours
    )
    return await solve_daily_llm(solverequest.battery, day, solverequest.solver, solverequest.solver_opts)

async def solve_daily_llm(
    request: SolveRequest,
    llm_provider: str = "gemini"
) -> SolveResponse:
    """EnergyDataRecords to DayInputs and call LLM optimization"""
    source = AG(
        atype=SolveRequest,
        states=[request],
    )

    # Build comprehensive instructions
    instructions = _build_optimization_instructions(
        battery=request.battery,
        day_inputs=request.day,
        metadata={}
    )
    
    # Create target AG object with LLM reasoning
    target = AG(
        atype=SolveResponse,
        llm = get_llm_provider(llm_provider),
        max_iter=1,  # Match working example
        verbose_agent=True,
        reasoning=True,
        instructions=instructions
    )
    
    # Execute optimization with error handling
    print(f"\n{'='*70}")
    print(f"{'='*70}\n")
    
    result = None
    try:
        result = await (target << source)
        print("IN MCP Server:",type(result))
        
        # Extract response and add metadata
        response = result.states[0] if result.states else None
        
        if response is None:
            print("Warning: LLM returned no states")
            raise ValueError("LLM returned no valid response")
        
        print(f"\n{'='*70}")
        print(f"✓ Optimization successful")
        print(result.pretty_print())
        print(f"{'='*70}\n")

        return result.states[0]
            
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ Error during LLM optimization:")
        print(f"  {type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        
        # Print more debug info
        if result is not None and hasattr(result, 'states'):
            print(f"Result has {len(result.states)} states")
        
        print("Returning fallback response with naive strategy...")
        
        # Calculate naive solution (just import everything)
        naive_cost = sum(request.day.prices_buy[t] * request.day.demand_MW[t] * request.day.dt_hours
                        for t in range(len(request.day.prices_buy)))

        # Return a fallback error response
        return SolveResponse(
            status="error",
            message=f"LLM optimization failed: {str(e)}. Returning naive solution (no battery usage). Try checking your API key or model configuration.",
            objective_cost=naive_cost,
            charge_MW=[0.0] * len(request.day.prices_buy),
            discharge_MW=[0.0] * len(request.day.prices_buy),
            import_MW=request.day.demand_MW,
            export_MW=[0.0] * len(request.day.prices_buy) if request.day.allow_export else None,
            soc=[request.battery.soc_init] * (len(request.day.prices_buy) + 1),
            decision=[0] * len(request.day.prices_buy),
        )
    
    # if response:
    #     # Add data source information
    #     response.data_source = metadata["data_source"]
    #     if metadata.get("forecast_models"):
    #         response.forecast_info = metadata["forecast_models"]
    
    # return response


def _build_optimization_instructions(
    battery: BatteryParams,
    day_inputs: DayInputs,
    metadata: dict
) -> str:
    """Build comprehensive optimization instructions for the LLM"""
    
    T = len(day_inputs.prices_buy)

    p_buy_actual = np.asarray(day_inputs.prices_buy, dtype=float)
    demand_actual  = np.asarray(day_inputs.demand_MW, dtype=float)
    if day_inputs.allow_export:
        p_sell_actual = np.asarray(day_inputs.prices_sell if day_inputs.prices_sell is not None else day_inputs.prices_buy, dtype=float)
    else:
        p_sell_actual = None

        # ---- Forecast arrays with safe fallbacks ----
    def _as_array_or_fallback(val, fallback_list):
        """Ensure we always get a 1D array, not a scalar or None."""
        if val is None:
            arr = np.asarray(fallback_list, dtype=float)
        else:
            arr = np.asarray(val, dtype=float)
        # If 0-D (scalar), broadcast to length T
        if arr.ndim == 0:
            arr = np.repeat(float(arr), T)
        return arr

    p_buy_forecast = _as_array_or_fallback(day_inputs.prices_buy_forecast, day_inputs.prices_buy)
    demand_forecast = _as_array_or_fallback(day_inputs.demand_MW_forecast, day_inputs.demand_MW)
    if day_inputs.allow_export:
        p_sell_forecast = _as_array_or_fallback(day_inputs.prices_sell_forecast if day_inputs.prices_sell_forecast is not None else day_inputs.prices_buy_forecast, day_inputs.prices_sell if day_inputs.prices_sell is not None else day_inputs.prices_buy)
    else:
        p_sell_forecast = None

    
    # Calculate statistics
    price_mean = sum(p_buy_actual) / len(p_buy_actual)
    price_min, price_max = min(p_buy_actual), max(p_buy_actual)
    instructions = f'''
        You are solving a daily battery scheduling optimization problem using forecast-based reasoning and constraint satisfaction.

        You are provided with both forecasted and actual market data:

        FORECAST INPUTS (for decision-making):
            - Forecasted buying prices: {p_buy_forecast}  (array of length T)
            - Forecasted selling prices: {p_sell_forecast}  (array of length T)
            - Forecasted demand: {demand_forecast}  (array of length T)

        ACTUAL INPUTS (for ex-post evaluation):
            - Realized buying prices: {p_buy_actual}  (array of length T)
            - Realized selling prices: {p_sell_actual}  (array of length T)
            - Realized demand: {demand_actual}  (array of length T)

        BATTERY PARAMETERS:
            - capacity_MWh: {battery.capacity_MWh}
            - charge/discharge limits: cmax_MW={battery.cmax_MW}, dmax_MW={battery.dmax_MW}
            - efficiencies: eta_c={battery.eta_c}, eta_d={battery.eta_d}
            - SoC bounds: {battery.soc_min} ≤ SoC ≤ {battery.soc_max}
            - initial SoC: soc_init={battery.soc_init}
            - target SoC: soc_target={battery.soc_target}

        HORIZON:
            - Number of timesteps: T = {len(p_buy_forecast)}
            - Duration per step: dt_hours = {day_inputs.dt_hours}
            - Export allowed: {day_inputs.allow_export}

        ------------------------------------------------------------
        STAGE 1: FORECAST-BASED DECISION OPTIMIZATION
        ------------------------------------------------------------
        Use forecasted information only (p_buy_forecast, p_sell_forecast, demand_forecast) to determine the following hourly decision variables:

            charge_MW[t], discharge_MW[t], import_MW[t], export_MW[t], soc[t]
        
        for every time t in {0} ≤ t < {T}

        Subject to constraints for all t:
            - SoC dynamics:
                SoC[t+1] = SoC[t] + ({battery.eta_c} * charge_MW[t] - discharge_MW[t] / {battery.eta_d}) * {day_inputs.dt_hours} / {battery.capacity_MWh}    
            - SoC bounds: {battery.soc_min} ≤ SoC[t] ≤ {battery.soc_max} for all t
            - Power limits: 
                0 <= charge_MW[t] <= {battery.cmax_MW}
                0 <= discharge_MW[t] <= {battery.dmax_MW}
            - Energy balance:
                import_MW[t] - export_MW[t] = demand_actual[t] + charge_MW[t] - discharge_MW[t]
            - Export constraint: export_MW[t] >= 0 only if allow_export = {day_inputs.allow_export}
            - Initial condition: SoC[0] = {battery.soc_init}
            - No simultaneous charge/discharge: The battery can either charge OR discharge OR stay idle in a given hour, not both.
            This means: NOT(charge_MW[t] > 0 AND discharge_MW[t] > 0)

        Forecast-based objective to minimize:
            forecast_cost = Σ_t [ (p_buy_forecast[t] * import_MW[t] - p_sell_forecast[t] * export_MW[t]) * {day_inputs.dt_hours} ]

        Decision logic:
            - Ensure SoC and power limits are respected
            - Price range: min={price_min:.2f}, max={price_max:.2f}, mean={price_mean:.2f}
            - Charge the battery when prices are LOW (below mean) for any time t, eg: Prefer charging when p_buy_forecast < {price_mean}
            - Discharge the battery when prices are HIGH (above mean) for any time t, eg: Prefer discharging when p_buy_forecast > {price_mean} 
            - Always meet {demand_actual} at every timestep t

        ------------------------------------------------------------
        STAGE 2: EX-POST EVALUATION (USING ACTUAL DATA)
        ------------------------------------------------------------
        Once the forecast-based decisions are determined (charge/discharge schedules fixed),
        apply them to actual data ({p_buy_actual, p_sell_actual, demand_actual}) to compute realized cost.

        Realized cost:
            realized_cost = Σ_t [ (p_buy_actual[t] * import_MW[t] - p_sell_actual[t] * export_MW[t]) * {day_inputs.dt_hours} ]

        ------------------------------------------------------------
        OUTPUT (SolveResponse)
        ------------------------------------------------------------
        Return the following fields:
            - status: "success" or "failure"
            - message:  Brief explanation of your optimization strategy (2-3 sentences)
            - objective_cost: realized_cost
            - charge_MW: list of {T} hourly charge power values, note that these values are capped by the battery's maximum charging power and at a time it can be either charging or discharging or idle.
            - discharge_MW: list of {T} hourly discharge power values, note that these values are capped by the battery's maximum discharging power and at a time it can be either charging or discharging or idle.
            - import_MW: list of {T} hourly grid import values and at a time it can be either importing from the grid or exporting to the grid and not both, but satisfying the demand and battery charge cum discharge power.
            - export_MW: list of {T} hourly grid export values and at a time it can be either importing from the grid or exporting to the grid and not both, but satisfying the demand and battery charge cum discharge power.
            - soc: list of {T+1} SoC fractions (0–1) which is a fraction value of the battery capacity.
            - decision: list of {T} values (+1=charge, -1=discharge, 0=idle)

        ------------------------------------------------------------
        GOAL
        ------------------------------------------------------------
        Generate physically feasible schedules that:
            1. Are optimized using forecasted data only,
            2. Are evaluated against actual realized data,
            3. Minimize realized total cost,
            4. Respect all technical and operational constraints.
            
        Make sure:
        - All lists have the correct length ({T} for hourly values, {T+1} for soc)
        - All constraints are satisfied at every timestep
        - The objective function (total cost) is minimized
        - The schedule is physically feasible following the entire battery physics and operational constraints with charging and discharging efficiency

        Think step by step:
        1. Identify low-price hours for charging
        2. Identify high-price hours for discharging  
        3. Calculate optimal charge/discharge amounts respecting battery limits
        4. Verify SoC stays within bounds
        5. Ensure demand is always met
        6. Calculate total cost

        Generate your complete SolveResponse now.
        '''
    
    return instructions

if __name__ == "__main__":
    mcp.run(transport="stdio")