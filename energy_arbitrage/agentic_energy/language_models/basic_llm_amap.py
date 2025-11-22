
"""
Day-Specific Battery Optimization using LLM

This module optimizes battery storage operations for a specific day using
LLM-based reasoning, with support for forecasted or actual data.
"""

import os
import warnings
import numpy as np
from dotenv import load_dotenv
# from agentics import agentics as AG
from agentics.core.agentics import AG

from typing import Optional, Tuple, List, Dict, Any
from agentic_energy.schemas import (
    SolveResponse, SolveFromRecordsRequest, SolveRequest, 
    EnergyDataRecord, BatteryParams, DayInputs
)
from agentic_energy.data_loader import EnergyDataLoader

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()
os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")

def records_to_arrays(records: List[EnergyDataRecord]) -> Tuple[list, list]:
    rows = [r for r in records if r.prices is not None and r.consumption is not None]
    rows.sort(key=lambda r: r.timestamps)
    prices = [float(r.prices) for r in rows]
    demand = [float(r.consumption) for r in rows]
    return prices, demand

async def llm_solve(args: SolveRequest) -> SolveResponse:
    """Run day-ahead battery LLM and return schedules + cost."""
    return await solve_daily_llm(args.battery, args.day, args.solver, args.solver_opts)

async def llm_solve_from_records(args: SolveFromRecordsRequest) -> SolveResponse:
    """Run day-ahead LLM given a list of EnergyDataRecord rows."""
    prices, demand = records_to_arrays(args.records)
    day = DayInputs(
        prices_buy=prices,
        demand_kw=demand,
        prices_sell=prices,
        allow_export=args.allow_export,
        dt_hours=args.dt_hours
    )
    return await solve_daily_llm(args.battery, day, args.solver, args.solver_opts)

async def solve_daily_llm(
    request: SolveRequest
) -> SolveResponse:
    """EnergyDataRecords to DayInputs and call LLM optimization"""
    source = AG(
        atype=SolveRequest, #you can put here as many states as you want to pass to the LLM to process in parallel
        states=[request], #over here many requests can be passed in parallel
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
        naive_cost = sum(request.day.prices_buy[t] * request.day.demand_kw[t] * request.day.dt_hours
                        for t in range(len(request.day.prices_buy)))

        # Return a fallback error response
        return SolveResponse(
            status="error",
            message=f"LLM optimization failed: {str(e)}. Returning naive solution (no battery usage). Try checking your API key or model configuration.",
            objective_cost=naive_cost,
            charge_kw=[0.0] * len(request.day.prices_buy),
            discharge_kw=[0.0] * len(request.day.prices_buy),
            import_kw=request.day.demand_kw,
            export_kw=[0.0] * len(request.day.prices_buy) if request.day.allow_export else None,
            soc=[request.battery.soc_init] * (len(request.day.prices_buy) + 1),
            decision=[0] * len(request.day.prices_buy),
        )
    
    # if response:
    #     # Add data source information
    #     response.data_source = metadata["data_source"]
    #     if metadata.get("forecast_models"):
    #         response.forecast_info = metadata["forecast_models"]
    
    # return response

# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM with explicit constraints"""
    
#     T = len(day_inputs.prices_buy)
#     dt = day_inputs.dt_hours
    
#     # Calculate storage duration (hours of operation at full power)
#     storage_duration_charge = battery.capacity_kwh / battery.cmax_kw if battery.cmax_kw > 0 else 0
#     storage_duration_discharge = battery.capacity_kwh / battery.dmax_kw if battery.dmax_kw > 0 else 0
    
#     # Prepare data arrays
#     p_buy_forecast = np.asarray(day_inputs.prices_buy_forecast, dtype=float)
#     demand_forecast = np.asarray(day_inputs.demand_kw_forecast, dtype=float)
#     p_sell_forecast = np.asarray(
#         day_inputs.prices_sell_forecast 
#         if day_inputs.allow_export and day_inputs.prices_sell_forecast 
#         else day_inputs.prices_buy_forecast, 
#         dtype=float
#     )
    
#     p_buy_actual = np.asarray(day_inputs.prices_buy, dtype=float)
#     demand_actual = np.asarray(day_inputs.demand_kw, dtype=float)
#     p_sell_actual = np.asarray(
#         day_inputs.prices_sell
#         if day_inputs.allow_export and day_inputs.prices_sell
#         else day_inputs.prices_buy,
#         dtype=float
#     )
    
#     # Calculate price statistics
#     forecast_mean = np.mean(p_buy_forecast)
#     forecast_std = np.std(p_buy_forecast)
#     forecast_min = np.min(p_buy_forecast)
#     forecast_max = np.max(p_buy_forecast)
    
#     instructions = f'''
# You are solving a battery scheduling optimization to minimize total operational cost.

# ══════════════════════════════════════════════════════════════════════
# BATTERY SPECIFICATIONS
# ══════════════════════════════════════════════════════════════════════
# - Capacity: {battery.capacity_kwh} kWh
# - Max Charge/Discharge Power: {battery.cmax_kw} kW / {battery.dmax_kw} kW
# - Efficiencies: η_c = {battery.eta_c}, η_d = {battery.eta_d}
# - Initial SoC: {battery.soc_init} ({battery.soc_init * battery.capacity_kwh:.2f} kWh)
# - SoC Range: [{battery.soc_min}, {battery.soc_max}]
# - **REQUIRED Final SoC: {battery.soc_target if battery.soc_target else battery.soc_init}** (HARD CONSTRAINT)

# This is a {storage_duration_discharge:.1f}-hour storage system. Starting from {battery.soc_init*100}% SoC:
# - Can add up to {(battery.soc_max - battery.soc_init) * battery.capacity_kwh:.2f} kWh (approx {(battery.soc_max - battery.soc_init) * storage_duration_charge:.1f}hrs at full power)
# - Can remove up to {(battery.soc_init - battery.soc_min) * battery.capacity_kwh:.2f} kWh (approx {(battery.soc_init - battery.soc_min) * storage_duration_discharge:.1f}hrs at full power)

# ══════════════════════════════════════════════════════════════════════
# DATA: {T} HOURS (Δt = {dt}h)
# ══════════════════════════════════════════════════════════════════════
# FORECAST (for planning):
# - Buy Prices: {p_buy_forecast.tolist()} €/MWh
#   [min={forecast_min:.1f}, max={forecast_max:.1f}, mean={forecast_mean:.1f}, std={forecast_std:.1f}]
# - Sell Prices: {p_sell_forecast.tolist()} €/MWh
# - Demand: {demand_forecast.tolist()} kW

# ACTUAL (for cost calculation):
# - Buy Prices: {p_buy_actual.tolist()} €/MWh
# - Sell Prices: {p_sell_actual.tolist()} €/MWh
# - Demand: {demand_actual.tolist()} kW

# Export allowed: {"YES" if day_inputs.allow_export else "NO"}

# ══════════════════════════════════════════════════════════════════════
# OBJECTIVE: MINIMIZE TOTAL COST (CALCULATED WITH ACTUAL PRICES)
# ══════════════════════════════════════════════════════════════════════

# At each hour t, power balance: import_kw[t] - export_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]

# If export NOT allowed: export_kw[t] = 0, so import_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]
# If export allowed: solve the balance (if net > 0: import that amount; if net < 0: export abs(net))

# Hourly cost: (price_buy_actual[t] × import_kw[t] - price_sell_actual[t] × export_kw[t]) × {dt}
# Total cost: Σ_(t=0 to {T-1}) hourly_cost[t]

# **You MUST use ACTUAL prices for final cost calculation, NOT forecasts.**

# ══════════════════════════════════════════════════════════════════════
# MANDATORY PHYSICAL CONSTRAINTS (ALL MUST BE SATISFIED)
# ══════════════════════════════════════════════════════════════════════

# 1. **SoC Dynamics:**
#    SoC[t+1] = SoC[t] + ({battery.eta_c} × charge_kw[t] × {dt} - discharge_kw[t] × {dt} / {battery.eta_d}) / {battery.capacity_kwh}

# 2. **SoC Limits (CRITICAL):**
#    - {battery.soc_min} ≤ SoC[t] ≤ {battery.soc_max} for ALL t ∈ [0, {T}]
#    - SoC[0] = {battery.soc_init}
#    - **SoC[{T}] ≥ {battery.soc_target if battery.soc_target else battery.soc_init}** ← THIS IS NON-NEGOTIABLE

# 3. **No Simultaneous Operations:**
#    At any hour t: charge_kw[t] = 0 OR discharge_kw[t] = 0 (or both = 0)

# 4. **Power Limits:**
#    0 ≤ charge_kw[t] ≤ {battery.cmax_kw}, 0 ≤ discharge_kw[t] ≤ {battery.dmax_kw}

# 5. **Power Balance:**
#    import_kw[t] - export_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]

# ⚠️ **CRITICAL: If your solution violates SoC[{T}] ≥ {battery.soc_target if battery.soc_target else battery.soc_init}, it is INVALID and must be rejected. You MUST find a feasible solution that meets this constraint, even if it increases cost.**

# ══════════════════════════════════════════════════════════════════════
# REQUIRED ANALYSIS (300+ words)
# ══════════════════════════════════════════════════════════════════════

# Your "message" field must include:

# **1. SCENARIO COMPARISON** (2-3 alternatives):
# Present 2-3 alternative strategies with estimated costs. Explain why your chosen strategy is best.
# Example: "Strategy A: charge [0-2], discharge [18-21], cost ~2500€. Strategy B: charge [3-5], discharge [16-19], cost ~2300€. I chose Strategy B because..."

# **2. PRICE ARBITRAGE ANALYSIS:**
# Identify lowest/highest price periods. Calculate arbitrage spread (max - min). Explain how your schedule exploits this. Discuss opportunities left unexploited due to constraints.

# **3. CONSTRAINT VERIFICATION:**
# Show key SoC calculations proving feasibility:
# - Start: SoC[0] = {battery.soc_init}
# - After charging: SoC[X] = ... (show calculation)
# - After discharging: SoC[Y] = ... (show calculation)
# - **Final: SoC[{T}] = ??? ≥ {battery.soc_target if battery.soc_target else battery.soc_init}? MUST BE YES**

# Verify: No simultaneous charge/discharge ✓, all SoC within [{battery.soc_min}, {battery.soc_max}] ✓

# **4. COST CALCULATION:**
# Show at least 3 example hours:
# - Hour X: import = demand + charge - discharge = ?, cost = price × import × {dt} = ?
# - Hour Y: ...
# - Total: Σ all hourly costs = objective_cost value

# **5. SENSITIVITY & TRADEOFFS:**
# Discuss forecast uncertainty, key assumptions, tradeoffs made (e.g., "sacrificed some arbitrage to meet final SoC constraint"). State confidence level and reasoning.

# ══════════════════════════════════════════════════════════════════════
# OUTPUT FORMAT
# ══════════════════════════════════════════════════════════════════════

# Return JSON SolveResponse with:
# - status: "success" (if feasible) or "failure" (if no feasible solution exists)
# - message: Your analysis (300+ words)
# - objective_cost: Total cost in € (using ACTUAL prices)
# - charge_kw: [{T} values]
# - discharge_kw: [{T} values]
# - import_kw: [{T} values]
# - export_kw: [{T} values]
# - soc: [{T+1} values from SoC[0] to SoC[{T}]]
# - decision: [{T} values: +1=charge, -1=discharge, 0=idle]
# - confidence: 0-1 (your confidence in solution quality)

# ══════════════════════════════════════════════════════════════════════
# FINAL CHECKLIST (verify before submitting)
# ══════════════════════════════════════════════════════════════════════

# MANDATORY CHECKS:
# □ All SoC values in [{battery.soc_min}, {battery.soc_max}]
# □ **SoC[{T}] ≥ {battery.soc_target if battery.soc_target else battery.soc_init}** ← VERIFY THIS TWICE
# □ No hour has both charge and discharge > 0
# □ objective_cost uses ACTUAL prices, not forecasts
# □ Power balance satisfied at every hour
# □ Message includes scenario comparison and constraint verification

# If ANY check fails, especially the final SoC constraint, your solution is INVALID. Find a feasible alternative that satisfies all constraints, even if the cost is higher.

# Strategy tips:
# - Use forecasted prices to identify cheap charging periods (prices < {forecast_mean:.1f}) and expensive discharging periods (prices > {forecast_mean:.1f})
# - Work backwards: ensure you can reach final SoC target from your last discharge period
# - If discharging too much would prevent meeting final SoC, either discharge less or add a final charging period

# Provide a thorough, feasible solution that minimizes cost while satisfying ALL constraints.
# '''
    
#     return instructions

### NEW TRY ###

# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM with explicit constraints"""
    
#     T = len(day_inputs.prices_buy)
#     dt = day_inputs.dt_hours
    
#     # Calculate storage duration (hours of operation at full power)
#     storage_duration_charge = battery.capacity_kwh / battery.cmax_kw if battery.cmax_kw > 0 else 0
#     storage_duration_discharge = battery.capacity_kwh / battery.dmax_kw if battery.dmax_kw > 0 else 0
    
#     # Prepare data arrays
#     p_buy_forecast = np.asarray(day_inputs.prices_buy_forecast, dtype=float)
#     demand_forecast = np.asarray(day_inputs.demand_kw_forecast, dtype=float)
#     p_sell_forecast = np.asarray(
#         day_inputs.prices_sell_forecast 
#         if day_inputs.allow_export and day_inputs.prices_sell_forecast 
#         else day_inputs.prices_buy_forecast, 
#         dtype=float
#     )
    
#     p_buy_actual = np.asarray(day_inputs.prices_buy, dtype=float)
#     demand_actual = np.asarray(day_inputs.demand_kw, dtype=float)
#     p_sell_actual = np.asarray(
#         day_inputs.prices_sell
#         if day_inputs.allow_export and day_inputs.prices_sell
#         else day_inputs.prices_buy,
#         dtype=float
#     )
    
#     # Calculate price statistics for strategy guidance
#     forecast_mean = np.mean(p_buy_forecast)
#     forecast_std = np.std(p_buy_forecast)
#     forecast_min = np.min(p_buy_forecast)
#     forecast_max = np.max(p_buy_forecast)
    
#     # Identify high/low price periods for guidance
#     high_price_threshold = forecast_mean + 0.5 * forecast_std
#     low_price_threshold = forecast_mean - 0.5 * forecast_std
    
#     instructions = f'''
# You are solving a daily battery scheduling optimization problem with CRITICAL ANALYSIS required.

# ═══════════════════════════════════════════════════════════════════════════════
# BATTERY TECHNICAL PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════
# - Capacity: {battery.capacity_kwh} kWh
# - Max Charge Power: {battery.cmax_kw} kW
# - Max Discharge Power: {battery.dmax_kw} kW
# - Charge Efficiency: η_c = {battery.eta_c} (meaning {battery.eta_c*100}% of charged power goes into storage)
# - Discharge Efficiency: η_d = {battery.eta_d} (meaning you get {battery.eta_d*100}% of stored energy as output)
# - Initial SoC: {battery.soc_init * 100}% ({battery.soc_init * battery.capacity_kwh:.2f} kWh stored)
# - SoC Bounds: [{battery.soc_min}, {battery.soc_max}] ({battery.soc_min * battery.capacity_kwh:.2f} to {battery.soc_max * battery.capacity_kwh:.2f} kWh)
# - Target End SoC: {battery.soc_target if battery.soc_target else battery.soc_init} ({((battery.soc_target if battery.soc_target else battery.soc_init) * battery.capacity_kwh):.2f} kWh)

# BATTERY STORAGE CHARACTERISTICS:
# This is a {storage_duration_discharge:.1f}-hour storage system, meaning:
# - At maximum discharge power ({battery.dmax_kw} kW), it would take {storage_duration_discharge:.1f} hours to discharge from 100% SoC to 0% SoC
# - At maximum charge power ({battery.cmax_kw} kW), it would take {storage_duration_charge:.1f} hours to charge from 0% SoC to 100% SoC

# CRITICAL UNDERSTANDING OF CONTINUOUS OPERATION LIMITS:
# Since your INITIAL SoC is {battery.soc_init * 100}%, the maximum continuous operation time depends on your starting point:

# For CHARGING:
# - You currently have {battery.soc_init * 100}% charge = {battery.soc_init * battery.capacity_kwh:.2f} kWh stored
# - Maximum capacity is {battery.soc_max * 100}% = {battery.soc_max * battery.capacity_kwh:.2f} kWh
# - You can add up to {(battery.soc_max - battery.soc_init) * battery.capacity_kwh:.2f} kWh more energy
# - At max charge power with efficiency, this takes roughly {(battery.soc_max - battery.soc_init) * storage_duration_charge:.1f} hours of continuous charging
# - BUT you can charge for longer at reduced power, or in multiple separate periods

# For DISCHARGING:
# - You currently have {battery.soc_init * battery.capacity_kwh:.2f} kWh stored
# - Minimum allowed is {battery.soc_min * 100}% = {battery.soc_min * battery.capacity_kwh:.2f} kWh  
# - You can discharge up to {(battery.soc_init - battery.soc_min) * battery.capacity_kwh:.2f} kWh
# - At max discharge power with efficiency, this takes roughly {(battery.soc_init - battery.soc_min) * storage_duration_discharge:.1f} hours of continuous discharging
# - BUT you can discharge for longer at reduced power, or in multiple separate periods

# IMPORTANT: These are theoretical limits assuming full power operation. In practice:
# - You can charge/discharge for MORE timesteps if using less than maximum power
# - You can split operations into multiple periods (charge, discharge, charge again, etc.)
# - The key constraint is the SoC must ALWAYS stay within [{battery.soc_min}, {battery.soc_max}]

# ═══════════════════════════════════════════════════════════════════════════════
# TIME HORIZON AND FORECASTED DATA
# ═══════════════════════════════════════════════════════════════════════════════
# Time horizon: {T} hours with Δt = {dt} hours per step

# FORECASTED DATA (for planning):
# - Buy Prices (€/MWh): {p_buy_forecast.tolist()}
#   Statistics: min={forecast_min:.2f}, max={forecast_max:.2f}, mean={forecast_mean:.2f}, std={forecast_std:.2f}
# - Sell Prices (€/MWh): {p_sell_forecast.tolist()}
# - Demand (kW): {demand_forecast.tolist()}

# ACTUAL DATA (for final cost calculation):
# - Buy Prices (€/MWh): {p_buy_actual.tolist()}
# - Sell Prices (€/MWh): {p_sell_actual.tolist()}
# - Demand (kW): {demand_actual.tolist()}

# Export allowed: {"YES" if day_inputs.allow_export else "NO"}

# ═══════════════════════════════════════════════════════════════════════════════
# OBJECTIVE: MINIMIZE TOTAL OPERATIONAL COST
# ═══════════════════════════════════════════════════════════════════════════════

# *** CRITICAL: HOW TO CALCULATE THE OBJECTIVE COST CORRECTLY ***

# The cost must be calculated using ACTUAL prices, not forecasted prices.

# Step 1: Calculate import and export power at each hour t:
#   Power balance equation:
#   import_kw[t] - export_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]
  
#   If export NOT allowed (export_kw[t] = 0 always):
#     import_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]
  
#   If export IS allowed:
#     Solve for import and export such that the balance holds
#     Typically: if (demand + charge - discharge) > 0, then import that amount, export = 0
#                if (demand + charge - discharge) < 0, then import = 0, export = -(demand + charge - discharge)

# Step 2: Calculate hourly cost at each hour t:
#   hourly_cost[t] = (price_buy_actual[t] * import_kw[t] - price_sell_actual[t] * export_kw[t]) * {dt}
  
#   Note: 
#   - When you import power, you PAY (positive cost)
#   - When you export power, you GET PAID (negative cost, reduces total cost)
#   - Multiply by dt_hours = {dt} to convert power (kW) to energy (kWh)

# Step 3: Sum all hourly costs:
#   total_cost = Σ_(t=0 to {T-1}) hourly_cost[t]

# *** EXAMPLE COST CALCULATION ***
# Suppose at hour t=5:
# - demand_actual[5] = 1000 kW
# - charge_kw[5] = 500 kW (you're charging battery)
# - discharge_kw[5] = 0 kW
# - price_buy_actual[5] = 50 €/MWh
# - dt = {dt} hours

# Then: import_kw[5] = 1000 + 500 - 0 = 1500 kW
#       export_kw[5] = 0 kW (nothing to export)
#       hourly_cost[5] = (50 * 1500 - 0) * {dt} = {50 * 1500 * dt} €

# Another example at hour t=15:
# - demand_actual[15] = 800 kW
# - charge_kw[15] = 0 kW
# - discharge_kw[15] = 1000 kW (you're discharging battery to help meet demand)
# - price_buy_actual[15] = 80 €/MWh
# - dt = {dt} hours

# Then: import_kw[15] = 800 + 0 - 1000 = -200 kW
#       Since this is negative, it means we're producing more than we need
#       If export allowed: import_kw[15] = 0, export_kw[15] = 200 kW
#       If export NOT allowed: This would be infeasible! We can't discharge more than demand if export isn't allowed.
      
#       Assuming export allowed and price_sell_actual[15] = 75 €/MWh:
#       hourly_cost[15] = (80 * 0 - 75 * 200) * {dt} = {-75 * 200 * dt} € (negative means we earned money!)

# *** REQUIREMENT: SHOW YOUR COST CALCULATION ***
# In your message/reasoning, you MUST include:
# 1. A table showing import_kw and export_kw for each hour
# 2. The hourly cost calculation for at least 3-5 representative hours
# 3. The final summation showing total_cost
# 4. Double-check that your total_cost in objective_cost field matches this calculation

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTRAINTS (MUST BE SATISFIED)
# ═══════════════════════════════════════════════════════════════════════════════

# 1. SoC Dynamics:
#    SoC[t+1] = SoC[t] + (η_c × charge_kw[t] × Δt - discharge_kw[t] × Δt / η_d) / Capacity
   
#    Specifically:
#    SoC[t+1] = SoC[t] + ({battery.eta_c} × charge_kw[t] × {dt} - discharge_kw[t] × {dt} / {battery.eta_d}) / {battery.capacity_kwh}

# 2. Power Balance:
#    import_kw[t] - export_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]
   
#    If export not allowed: export_kw[t] = 0 and import_kw[t] ≥ demand_kw[t] + charge_kw[t] - discharge_kw[t]

# 3. No Simultaneous Charge/Discharge:
#    At any time t: either charge_kw[t] = 0 OR discharge_kw[t] = 0 (or both are 0)

# 4. Power Limits:
#    - 0 ≤ charge_kw[t] ≤ {battery.cmax_kw} kW
#    - 0 ≤ discharge_kw[t] ≤ {battery.dmax_kw} kW

# 5. SoC Limits:
#    - {battery.soc_min} ≤ SoC[t] ≤ {battery.soc_max} for ALL t ∈ [0, {T}]
#    - SoC[0] = {battery.soc_init}
#    - SoC[{T}] ≥ {battery.soc_target if battery.soc_target else battery.soc_init}

# 6. Continuous Operation Limits (derived from battery physics):
#    - For a {storage_duration_discharge:.1f}-hour battery starting at {battery.soc_init*100}% SoC:
#      * At FULL charge power: can charge continuously for approximately {(battery.soc_max - battery.soc_init) * storage_duration_charge:.1f} hours before reaching max SoC
#      * At FULL discharge power: can discharge continuously for approximately {(battery.soc_init - battery.soc_min) * storage_duration_discharge:.1f} hours before reaching min SoC
#    - However, these limits are APPROXIMATE guides. The exact limit depends on:
#      * The actual charge/discharge power you choose (can be less than max)
#      * The efficiency losses (eta_c and eta_d)
#      * The current SoC at the start of the operation
#    - The REAL constraint is: SoC must never leave [{battery.soc_min}, {battery.soc_max}]
#    - You should verify feasibility by calculating the SoC trajectory, not by counting hours

# ═══════════════════════════════════════════════════════════════════════════════
# CRITICAL ANALYSIS REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════════════

# Your "message" field must include a COMPREHENSIVE CRITICAL ANALYSIS with:

# 1. **SCENARIO COMPARISON** (Required - compare at least 2-3 strategies):
#    - Present 2-3 alternative scheduling strategies you considered
#    - For each alternative, briefly describe the charge/discharge pattern
#    - Calculate or estimate the approximate cost for each alternative
#    - Explain why you chose your final strategy over the alternatives
   
#    Example:
#    "I considered three strategies:
#    Strategy A: Charge hours [0-2], discharge hours [18-21]
#      - Estimated cost: ~2500€
#      - Pros: Captures biggest price spread
#      - Cons: Doesn't fully utilize battery capacity
   
#    Strategy B: Charge hours [0-4], discharge hours [10-14], charge [22-23]
#      - Estimated cost: ~2200€
#      - Pros: More complete utilization
#      - Cons: Violates initial SoC constraint (can only charge 2hrs from 50% SoC)
   
#    Strategy C (CHOSEN): Charge hours [0-1], discharge hours [16-19], charge [22-23]
#      - Estimated cost: ~2100€
#      - Pros: Respects all constraints, good price arbitrage
#      - Cons: Leaves some capacity unused but constrained by initial SoC
   
#    I chose Strategy C because..."

# 2. **PRICE ANALYSIS AND ARBITRAGE OPPORTUNITIES**:
#    - Identify the lowest price periods: hours where price < {low_price_threshold:.2f}
#    - Identify the highest price periods: hours where price > {high_price_threshold:.2f}
#    - Calculate the potential arbitrage spread (max_price - min_price)
#    - Explain how your schedule exploits these opportunities
#    - Discuss any opportunities you had to leave unexploited and why

# 3. **CONSTRAINT VERIFICATION WITH CALCULATIONS**:
#    You must SHOW the calculations proving your schedule is feasible:
   
#    a) SoC trajectory calculation (show at least 5 key timesteps):
#       "Starting from SoC[0] = {battery.soc_init}:
#        Hour 0: charge 1000kW → SoC[1] = {battery.soc_init} + ({battery.eta_c}*1000*{dt})/{battery.capacity_kwh} = X
#        Hour 1: charge 1000kW → SoC[2] = X + ({battery.eta_c}*1000*{dt})/{battery.capacity_kwh} = Y
#        ...
#        Verify: {battery.soc_min} ≤ all SoC values ≤ {battery.soc_max}? YES ✓"
   
#    b) No simultaneous charge/discharge:
#       "At each hour, I verify charge_kw[t] * discharge_kw[t] = 0
#        All hours satisfy this ✓"
   
#    c) Final SoC target:
#       "SoC[{T}] = Z ≥ {battery.soc_target if battery.soc_target else battery.soc_init}? YES ✓"
   
#    d) Continuous operation check:
#       "I verify no continuous operation violates SoC limits by checking the SoC trajectory:
#        - Longest charging period: hours [X-Y] 
#          Starting SoC: A, Ending SoC: B
#          B ≤ {battery.soc_max}? YES ✓
#        - Longest discharging period: hours [M-N]
#          Starting SoC: C, Ending SoC: D  
#          D ≥ {battery.soc_min}? YES ✓"

# 4. **SENSITIVITY AND RISK ANALYSIS**:
#    - Discuss forecast uncertainty: "The forecast may be inaccurate. If actual prices differ by ±X%, my strategy would..."
#    - Identify critical assumptions: "My strategy assumes prices peak at hour Y. If they peak earlier..."
#    - Assess robustness: "This schedule is robust/sensitive to forecast errors because..."
#    - State your confidence and why: "Confidence: 0.X because..."

# 5. **TRADEOFF ANALYSIS**:
#    - Explain key tradeoffs you made
#    - "I chose to prioritize X over Y because..."
#    - "I sacrificed Z potential profit to ensure W constraint is met because..."
#    - "The efficiency losses are acceptable because the price spread is..."

# 6. **COST CALCULATION VERIFICATION**:
#    - Include a summary table or calculation as described in the objective section
#    - Show at least 3-5 example hours with detailed cost breakdown
#    - Verify the total matches your objective_cost value

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMAT
# ═══════════════════════════════════════════════════════════════════════════════

# You must output a JSON-compatible SolveResponse object with:

# - status: "success" or "failure"
# - message: Your comprehensive critical analysis (as described above, minimum 500 words)
# - objective_cost: The minimized total cost in € (calculated with ACTUAL prices using the method above)
# - charge_kw: List of {T} hourly charge values in kW
# - discharge_kw: List of {T} hourly discharge values in kW  
# - import_kw: List of {T} hourly grid import values in kW
# - export_kw: List of {T} hourly grid export values in kW
# - soc: List of {T+1} state of charge values (fraction 0-1)
# - decision: List of {T} decisions: +1=charge, -1=discharge, 0=idle
# - confidence: Float 0-1 indicating confidence in solution's feasibility and optimality

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL CHECKLIST (verify before submitting)
# ═══════════════════════════════════════════════════════════════════════════════

# Before finalizing your solution, verify:

# □ SoC trajectory never violates [{battery.soc_min}, {battery.soc_max}]
# □ No simultaneous charge and discharge at any hour
# □ No continuous operation exceeds physical limits
# □ Final SoC[{T}] ≥ {battery.soc_target if battery.soc_target else battery.soc_init}
# □ objective_cost calculated using ACTUAL prices (not forecast)
# □ import_kw and export_kw satisfy power balance at every hour
# □ Message includes scenario comparison (2-3 alternatives)
# □ Message includes constraint verification with calculations
# □ Message includes sensitivity/risk analysis
# □ Message includes detailed cost calculation verification
# □ confidence value reflects genuine assessment of solution quality

# Your solution will be evaluated on:
# 1. Feasibility (all constraints satisfied)
# 2. Optimality (lowest cost achievable given constraints)
# 3. Quality of critical analysis (depth of reasoning)
# 4. Correctness of cost calculation

# Provide a thorough, well-reasoned solution that demonstrates deep understanding of the optimization problem, tradeoffs, and constraints.
# '''
    
#     return instructions

def _build_optimization_instructions(
    battery: BatteryParams, #instruction should be adjusted to reflect battery param
    day_inputs: DayInputs,
    metadata: dict
) -> str:
    """Build comprehensive optimization instructions for the LLM"""
    
    T = len(day_inputs.prices_buy)

    p_buy_actual = np.asarray(day_inputs.prices_buy, dtype=float)
    demand_actual  = np.asarray(day_inputs.demand_kw, dtype=float)
    if day_inputs.allow_export:
        p_sell_actual = np.asarray(day_inputs.prices_sell if day_inputs.prices_sell is not None else day_inputs.prices_buy, dtype=float)
    else:
        p_sell_actual = None

    p_buy_forecast = np.asarray(day_inputs.prices_buy_forecast, dtype=float)
    demand_forecast = np.asarray(day_inputs.demand_kw_forecast, dtype=float)
    if day_inputs.allow_export:
        p_sell_forecast = np.asarray(day_inputs.prices_sell_forecast if day_inputs.prices_sell_forecast is not None else day_inputs.prices_buy_forecast, dtype=float)
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
            - capacity_kwh: {battery.capacity_kwh}
            - charge/discharge limits: cmax_kw={battery.cmax_kw}, dmax_kw={battery.dmax_kw}
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

            charge_kw[t], discharge_kw[t], import_kw[t], export_kw[t], soc[t]
        
        for every time t in {0} ≤ t < {T}

        Subject to constraints for all t:
            - SoC dynamics:
                SoC[t+1] = SoC[t] + ({battery.eta_c} * charge_kw[t] - discharge_kw[t] / {battery.eta_d}) * {day_inputs.dt_hours} / {battery.capacity_kwh}    
            - SoC bounds: {battery.soc_min} ≤ SoC[t] ≤ {battery.soc_max} for all t
            - Power limits: 
                0 ≤ charge_kw[t] ≤ {battery.cmax_kw}
                0 ≤ discharge_kw[t] ≤ {battery.dmax_kw}
            - Energy balance:
                import_kw[t] - export_kw[t] = demand_actual[t] + charge_kw[t] - discharge_kw[t]
            - Export constraint: export_kw[t] ≥ 0 only if allow_export = {day_inputs.allow_export}
            - Initial condition: SoC[0] = {battery.soc_init}
            - End condition: SoC[T] ≥ {battery.soc_target}
            - No simultaneous charge/discharge: The battery can either charge OR discharge OR stay idle in a given hour, not both.
            This means: NOT(charge_kw[t] > 0 AND discharge_kw[t] > 0)

        Forecast-based objective to minimize:
            forecast_cost = Σ_t [ (p_buy_forecast[t] * import_kw[t] - p_sell_forecast[t] * export_kw[t]) * {day_inputs.dt_hours} ]

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
            realized_cost = Σ_t [ (p_buy_actual[t] * import_kw[t] - p_sell_actual[t] * export_kw[t]) * {day_inputs.dt_hours} ]

        ------------------------------------------------------------
        OUTPUT (SolveResponse)
        ------------------------------------------------------------
        Return the following fields:
            - status: "success" or "failure"
            - message:  Brief explanation of your optimization strategy (2-3 sentences)
            - objective_cost: realized_cost
            - charge_kw: list of {T} hourly charge power values, note that these values are capped by the battery's maximum charging power and at a time it can be either charging or discharging or idle.
            - discharge_kw: list of {T} hourly discharge power values, note that these values are capped by the battery's maximum discharging power and at a time it can be either charging or discharging or idle.
            - import_kw: list of {T} hourly grid import values and at a time it can be either importing from the grid or exporting to the grid and not both, but satisfying the demand and battery charge cum discharge power.
            - export_kw: list of {T} hourly grid export values and at a time it can be either importing from the grid or exporting to the grid and not both, but satisfying the demand and battery charge cum discharge power.
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


### PROMPT BELOW WAS PRETTY GOOD ###

# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM with explicit constraints"""
    
#     T = len(day_inputs.prices_buy)
#     dt = day_inputs.dt_hours
    
#     # Calculate battery physical limits from current state
#     current_soc = battery.soc_init
#     max_charge_energy = (battery.soc_max - current_soc) * battery.capacity_kwh
#     max_discharge_energy = (current_soc - battery.soc_min) * battery.capacity_kwh
    
#     # Time limits for continuous operation
#     if battery.cmax_kw > 0 and battery.eta_c > 0:
#         max_continuous_charge_hours = max_charge_energy / (battery.cmax_kw * battery.eta_c)
#     else:
#         max_continuous_charge_hours = 0
        
#     if battery.dmax_kw > 0 and battery.eta_d > 0:
#         max_continuous_discharge_hours = (max_discharge_energy * battery.eta_d) / battery.dmax_kw
#     else:
#         max_continuous_discharge_hours = 0
    
#     # Prepare data arrays
#     p_buy_forecast = np.asarray(day_inputs.prices_buy_forecast, dtype=float)
#     demand_forecast = np.asarray(day_inputs.demand_kw_forecast, dtype=float)
#     p_sell_forecast = np.asarray(
#         day_inputs.prices_sell_forecast 
#         if day_inputs.allow_export and day_inputs.prices_sell_forecast 
#         else day_inputs.prices_buy_forecast, 
#         dtype=float
#     )
    
#     p_buy_actual = np.asarray(day_inputs.prices_buy, dtype=float)
#     demand_actual = np.asarray(day_inputs.demand_kw, dtype=float)
    
#     # Calculate price thresholds for strategy
#     forecast_mean = np.mean(p_buy_forecast)
#     forecast_std = np.std(p_buy_forecast)
    
#     instructions = f'''

# You are solving a daily battery scheduling optimization problem. 

# You have a battery with the following technical parameters:
# - Capacity: {battery.capacity_kwh} kWh
# - Max Charge Power: {battery.cmax_kw} kW
# - Max Discharge Power: {battery.dmax_kw} kW
# - Charge Efficiency: η_c = {battery.eta_c}
# - Discharge Efficiency: η_d = {battery.eta_d}
# - Initial SoC: {battery.soc_init}
# - SoC Bounds: [{battery.soc_min}, {battery.soc_max}]
# - Target End SoC: {battery.soc_target if battery.soc_target else battery.soc_init}
# You are scheduling for a time horizon of {T} hours with Δt = {dt} hours per step.
# You are given forecasted hourly data for the day:
# - Forecast Buy Prices (€/MWh): {p_buy_forecast.tolist()}
# - Forecast Sell Prices (€/MWh): {p_sell_forecast.tolist()}
# - Forecast Demand (kW): {demand_forecast.tolist()}

# You are minimizing the total operational cost:
# total_cost = Σ_t [ (price_buy[t] * import_kw[t] - price_sell[t] * export_kw[t]) * dt_hours ]
# where dt_hours = {dt}, export_kw[t] = 0 if export not allowed. 

# The imports and exports are constrained by the demand:
# demand_kw[t] = import_kw[t] - export_kw[t] + discharge_kw[t] - charge_kw[t]

# You must ensure the following critical physical constraints are satisfied:
#    SoC[t+1] = SoC[t] + (η_c × charge_kw[t] × Δt - discharge_kw[t] × Δt / η_d) / Capacity
   
#    Specifically:
#    SoC[t+1] = SoC[t] + ({battery.eta_c} × charge_kw[t] × {dt} - discharge_kw[t] × {dt} / {battery.eta_d}) / {battery.capacity_kwh}

# The import and export power balance:
#    import_kw[t] - export_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]

# You cannot charge and discharge simultaneously:
#     At any time t: either charge_kw[t] = 0 OR discharge_kw[t] = 0 (or both are 0)

# You must also respect the following limits:
# - 0 ≤ charge_kw[t] ≤ {battery.cmax_kw}
# - 0 ≤ discharge_kw[t] ≤ {battery.dmax_kw}
# - {battery.soc_min} ≤ SoC[t] ≤ {battery.soc_max} for all t (Note that your SOC cannot be negative or exceed max EVER)
# - SoC[0] = {battery.soc_init}
# - SoC[{T}] ≥ {battery.soc_target if battery.soc_target else battery.soc_init}

# You are given the ACTUAL hourly data for the day:
# - Actual Buy Prices (€/MWh): {p_buy_actual.tolist()}    
# - Actual Sell Prices (€/MWh): {p_buy_actual.tolist() if not day_inputs.allow_export else (day_inputs.prices_sell if day_inputs.prices_sell else day_inputs.prices_buy)}
# - Actual Demand (kW): {demand_actual.tolist()}
# You must calculate the final total cost using these ACTUAL values.
# You can use the forecasted data to plan your schedule, but the final cost must be computed with the actual data.

# You should leverage the forecasted prices and demand to identify low-price periods for charging and high-price periods for discharging, while ensuring the battery SoC remains within limits.

# You are outputing a JSON-compatible SolveResponse object with:
# - status: "success" or "failure"
# - message: Full reasoning steps you took to arrive at the solution. It should be something like this: "I observe that the initial SOC is 0.5. This means I can only charge for two more hours before discharging. I observe low forecated prices at night, hence I decide to charge at hours [0,1] to achieve a full state of charge. 
# After this, I observe high forecasted prices at hours [10,11,12,13], hence I decide to discharge during these hours to maximize profit and achieve a minimum SOC. After this, I observe low prices again at hours [14,15,16,17], hence I decide to charge again during these hours to achieve a full state of charge. Finally, I observe high prices at hours [18,19,20,21], hence I decide to discharge again during these hours to maximize profit. Finally, I also observe that the final SOC (target SOC) is 0.5. To achieve the target SOC, which is 0.5, I charge again at hours [22,23].
# This ensures that I never violate the SOC limits of [0,1], and I never charge or discharge for more than 4 hours continuously, at the same time I also ensure that I charged correcrly from the inital SOC for only 2 timestamps. I also ensured that I never charge and discharge simultaneously. I also ensured correct final SOC. The final schedule is feasible and respects all constraints."
# Another example of message could be: "I observe that the initial SOC is 0.5. This means I can only charge for two more hours before discharging. I observe low forecated prices at night, hence I decide to charge at hours [0,1] to achieve a full state of charge. After this I only observe high forecasted prices at hours [16,17,18,19], hence I decide to discharge during these hours to maximize profit and achieve a minimum SOC. After this, I also observe that the final SOC (target SOC) is 0.5. To achieve the target SOC, which is 0.5, I charge again at hours [21,22], since the prices are the lowest at those hours according to the forecast.
# I also ensured that I never violate the SOC limits of [0,1], and I never charge or discharge for more than 4 hours continuously, at the same time I also ensure that I charged correcrly from the inital SOC for only 2 timestamps. I also ensured that I never charge and discharge simultaneously. I also ensured correct final SOC. The final schedule is feasible and respects all constraints."
# Please ensure your reasoning is detailed and explains how you respected all constraints. At the same time, you need to critically analyze the forecasted prices to come up with a profitable schedule and to reflect on the opportunities of arbitrage. You need to show that you have thought through the problem carefully and arrived at a feasible and optimal solution. You can show that you potentially considered different scenarios and scheduling options. 
# - objective_cost: the minimized total cost using ACTUAL prices
# - charge_kw: list of hourly charge values (kW)
# - discharge_kw: list of hourly discharge values (kW)
# - import_kw: list of hourly grid import values (kW)
# - export_kw: list of hourly grid export values (kW)
# - soc: list of hourly state of charge values (fraction of capacity between 0 and 1)
# - decision: list of hourly decisions where +1 means charging, -1 means discharging, 0 means idle
# - confidence: a number between 0 and 1 indicating your confidence in the solution's feasibility and optimality.

# Again, your decision reflects charging and discharging. Our input storage is a 4-hour battery. This means you can charge at max power for 4 hours continuously before hitting max SoC (if starting from 0 SOC), and discharge at max power for 4 hours continuously before hitting min SoC (if discharging from max SOC). You should never schedule discharging for more than 4 hours continuously, as that would violate the physical constraints of the battery.

# This means, for instance, that if your decision vector is like [1, 1, 1, 1, -1, -1, -1, -1, 0, 0, ...], this is feasible as you charge for 4 hours then discharge for 4 hours. But a decision like [1, 1, 1, 1, -1, -1, -1, -1, -1, 0, ...] is NOT feasible as you are discharging for 5 hours continuously which exceeds the battery's capability.

# But keep in mind that the actual feasible charge/discharge duration also depends on the starting SoC. For example, if you start at 50% SoC, you can only charge for 2 more hours at max power before hitting max SoC, and you can discharge for 6 hours at max power before hitting min SoC. So your scheduling should respect these dynamic limits based on the initial SoC.

# Before finalizing your schedule, please double-check that:
# 1. The SoC trajectory never goes below {battery.soc_min} or above {battery.soc_max}.
# 2. The battery is never scheduled to charge and discharge simultaneously.   
# 3. The battery is never scheduled to charge or discharge for more than 4 hours continuously (or a lesser amount in case the SOC is not max or min).
# 4. The final SoC at hour {T} meets or exceeds the target SoC of {battery.soc_target if battery.soc_target else battery.soc_init}.
# 5. The total cost is calculated using the ACTUAL prices and demand, not the forecasted values.
# Make sure the final schedule satisfies all physical constraints and the objective function is minimized.

# Here is an example of a feasible schedule:
# Suppose the observed forecasted prices are something like this: [40 40 40 45 50 55 60 70 80 90 100 90 60 60 60 50 40 70 70 70 70 40 50 60]
# Then you might decide to initially charge during hours 0-3 when prices are low, then discharge during hours 10-13 when prices are high.
# Then you would charge again during hours 15-18 when prices drop, and discharge again during hours 19-22 when prices rise again.
# Here is an example of an output schedule:
# - decision: [1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0]
# However, we note that our initial SoC is 0.5, so we can only charge for 2 hours at max power before hitting max SoC, and we can discharge for 4 hours (because it is a 4-hr storage) at max power before hitting min SoC.
# After that, our SOC is empty again, so we can only charge for 4 hours at max power before hitting max SoC again. And then we can discharge for 4 hours at max power before hitting min SoC again.
# Hence, the final feasible decision should be:
# - decision: [1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0]

# Please note that decisions like [1, 1, 1, 1, 1, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0] or [1, 1, 1, 1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1] are NOT feasible as they violate the battery's physical constraints.
# At the same time, decisoins like [1, 1, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, 0] are also NOT feasible as they schedule charging for more than 4 hours continuously (i.e., hours [0,1] and hours [3,4,5,6]).

# The reasoning for your decision should reflect this understanding of the battery's physical constraints and the price patterns observed.

# Please ensure your final output strictly adheres to these constraints, and reason your decisions accordingly.

# '''
    
#     return instructions



### PROMPT ABOVE WORKED PRETTY WELL ###

# You are solving a battery scheduling optimization problem with physical constraints.

# ============================================================
# PROBLEM PARAMETERS
# ============================================================
# Time Horizon: {T} hours (Δt = {dt} hours per step)
# Battery Specifications:
# - Capacity: {battery.capacity_kwh} kWh
# - Max Charge Power: {battery.cmax_kw} kW
# - Max Discharge Power: {battery.dmax_kw} kW
# - Charge Efficiency: η_c = {battery.eta_c}
# - Discharge Efficiency: η_d = {battery.eta_d}
# - Initial SoC: {battery.soc_init}
# - SoC Bounds: [{battery.soc_min}, {battery.soc_max}]
# - Target End SoC: {battery.soc_target if battery.soc_target else battery.soc_init}

# ============================================================
# CRITICAL PHYSICAL CONSTRAINTS
# ============================================================

# 1. STATE OF CHARGE EVOLUTION (for each hour t):
#    SoC[t+1] = SoC[t] + (η_c × charge_kw[t] × Δt - discharge_kw[t] × Δt / η_d) / Capacity
   
#    Specifically:
#    SoC[t+1] = SoC[t] + ({battery.eta_c} × charge_kw[t] × {dt} - discharge_kw[t] × {dt} / {battery.eta_d}) / {battery.capacity_kwh}

# 2. ENERGY STORAGE LIMITS:
#    Starting from SoC = {battery.soc_init}:
#    - Maximum energy that CAN be charged: {max_charge_energy:.2f} kWh
#    - Maximum energy that CAN be discharged: {max_discharge_energy:.2f} kWh
   
#    This means at maximum power:
#    - Can charge continuously for at most {max_continuous_charge_hours:.2f} hours
#    - Can discharge continuously for at most {max_continuous_discharge_hours:.2f} hours
   
#    VIOLATION CHECK: If you discharge at {battery.dmax_kw} kW for more than {max_continuous_discharge_hours:.1f} hours,
#    the battery would go below minimum SoC - THIS IS PHYSICALLY IMPOSSIBLE!

# 3. NO SIMULTANEOUS CHARGE/DISCHARGE:
#    At any time t: either charge_kw[t] = 0 OR discharge_kw[t] = 0 (or both are 0)

# 4. POWER BALANCE:
#    import_kw[t] - export_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]
#    (Note: export_kw[t] = 0 if export not allowed)

# 5. BOUND CONSTRAINTS:
#    - 0 ≤ charge_kw[t] ≤ {battery.cmax_kw}
#    - 0 ≤ discharge_kw[t] ≤ {battery.dmax_kw}
#    - {battery.soc_min} ≤ SoC[t] ≤ {battery.soc_max} for all t
#    - SoC[0] = {battery.soc_init}
#    - SoC[{T}] ≥ {battery.soc_target if battery.soc_target else battery.soc_init}

# ============================================================
# TWO-STAGE OPTIMIZATION
# ============================================================

# STAGE 1: DECISION MAKING (Use FORECAST data)
# ----------------------------------------
# Forecast Buy Prices (€/MWh): {p_buy_forecast.tolist()}
# Forecast Demand (kW): {demand_forecast.tolist()}

# Strategy:
# - Identify LOW price hours (< {forecast_mean - 0.5*forecast_std:.2f}): Good for charging
# - Identify HIGH price hours (> {forecast_mean + 0.5*forecast_std:.2f}): Good for discharging
# - Respect energy limits: Don't plan to charge/discharge more than physically possible!

# STAGE 2: EVALUATION (Apply to ACTUAL data)
# ----------------------------------------
# Once charge_kw and discharge_kw are determined from Stage 1:
# - Calculate import_kw and export_kw using actual demand: {demand_actual.tolist()}
# - Calculate total cost using actual prices: {p_buy_actual.tolist()}

# ============================================================
# REQUIRED OUTPUT
# ============================================================

# You must return ALL of the following arrays:

# 1. charge_kw: List of {T} values (charging power in kW for each hour)
# 2. discharge_kw: List of {T} values (discharging power in kW for each hour)
# 3. import_kw: List of {T} values (grid import in kW for each hour)
# 4. export_kw: List of {T} values (grid export in kW, 0 if not allowed)
# 5. soc: List of {T+1} values (SoC trajectory, starting with {battery.soc_init})
# 6. decision: List of {T} values where:
#    - +1 means charging in that hour
#    - -1 means discharging in that hour
#    - 0 means idle (neither charging nor discharging)
# 7. objective_cost: Total cost using ACTUAL prices (single number)

# You are also returning a message field:

# In that field please provide a REASONING summary of decision making. It can something like this:
# "Multi-cycle continuous arbitrage strategy executed across 11 active hours. Strategy: (1) Early discharge at €130.9 to create capacity, (2) Strategic intermediate cycles hours 3-9 to capture medium spreads, (3) Aggressive charging hours 9-15 during valley period (€79-93), (4) Maximum discharge hours 18-21 during peak (€125-165). Battery fully utilized with SoC cycling respecting the target SOC of 0.5 at the last time stamp. Achieved 5 charge cycles and 6 discharge cycles. Buying at average €96.79/MWh, selling at €139.77/MWh generates €98.11 objective cost.""

# Please note that charging/discharging from SOC = 0.5 cannot be larger than two consecutive entries and charging from SOC = 0 or discharging from SOC = 1 cannot be larger than four entry (you are operating a 4-hour storage).

# Additionally, you have a confidence output number between 0 and 1 indicating how confident you are that the solution is feasible and optimal.
# ============================================================
# VALIDATION BEFORE RETURNING
# ============================================================

# Check your solution:
# ✓ SoC stays within [{battery.soc_min}, {battery.soc_max}] at ALL times
# ✓ SoC evolution equation is correct for each hour
# ✓ No hour has both charge_kw > 0 AND discharge_kw > 0
# ✓ Power balance holds for each hour
# ✓ No consecutive charging exceeds {max_continuous_charge_hours:.1f} hours at full power
# ✓ No consecutive discharging exceeds {max_continuous_discharge_hours:.1f} hours at full power
# ✓ decision[t] = +1 when charge_kw[t] > 0, -1 when discharge_kw[t] > 0, 0 otherwise
# ✓ Length of arrays: charge_kw, discharge_kw, import_kw, export_kw, decision = {T}; soc = {T+1}

# Return SolveResponse with status="success" and respective outputs.
# '''

# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM with explicit MILP formulation"""
    
#     T = len(day_inputs.prices_buy)
#     dt = day_inputs.dt_hours
    
#     # Calculate battery physical limits from current state
#     current_soc = battery.soc_init
#     max_charge_energy = (battery.soc_max - current_soc) * battery.capacity_kwh
#     max_discharge_energy = (current_soc - battery.soc_min) * battery.capacity_kwh
    
#     # Time limits for continuous operation
#     if battery.cmax_kw > 0:
#         max_continuous_charge_hours = max_charge_energy / (battery.cmax_kw * battery.eta_c)
#     else:
#         max_continuous_charge_hours = 0
        
#     if battery.dmax_kw > 0:
#         max_continuous_discharge_hours = (max_discharge_energy * battery.eta_d) / battery.dmax_kw
#     else:
#         max_continuous_discharge_hours = 0
    
#     # Convert to time steps
#     max_feasible_charge_steps = int(max_continuous_charge_hours / dt)
#     max_feasible_discharge_steps = int(max_continuous_discharge_hours / dt)
    
#     # Prepare data arrays
#     p_buy_forecast = day_inputs.prices_buy_forecast
#     demand_forecast = day_inputs.demand_kw_forecast
#     p_sell_forecast = (day_inputs.prices_sell_forecast 
#                       if day_inputs.allow_export and day_inputs.prices_sell_forecast 
#                       else day_inputs.prices_buy_forecast)
    
#     p_buy_actual = day_inputs.prices_buy
#     demand_actual = day_inputs.demand_kw
#     p_sell_actual = (day_inputs.prices_sell 
#                     if day_inputs.allow_export and day_inputs.prices_sell 
#                     else day_inputs.prices_buy)
    
#     # Calculate price thresholds for strategy
#     forecast_mean = np.mean(p_buy_forecast)
#     forecast_std = np.std(p_buy_forecast)
#     price_threshold_low = forecast_mean - 0.5 * forecast_std
#     price_threshold_high = forecast_mean + 0.5 * forecast_std
    
#     instructions = f"""
# You are solving a Mixed Integer Linear Programming (MILP) problem for battery energy storage scheduling.

# ## Problem Formulation

# ### Parameters
# - **Time horizon**: T = {T} time steps, each Δt = {dt} hours
# - **Battery capacity**: C = {battery.capacity_kwh} kWh
# - **Maximum charge power**: P_c^max = {battery.cmax_kw} kW
# - **Maximum discharge power**: P_d^max = {battery.dmax_kw} kW
# - **Charge efficiency**: η_c = {battery.eta_c}
# - **Discharge efficiency**: η_d = {battery.eta_d}
# - **Initial SoC**: SoC₀ = {battery.soc_init}
# - **SoC bounds**: [{battery.soc_min}, {battery.soc_max}]
# - **Target end SoC**: SoC_target = {battery.soc_target if battery.soc_target is not None else battery.soc_init}
# - **Export allowed**: {day_inputs.allow_export}

# ### Decision Variables (for each time t ∈ [0, {T-1}])
# - **charge_kw[t]**: Charge power (kW), 0 ≤ charge_kw[t] ≤ {battery.cmax_kw}
# - **discharge_kw[t]**: Discharge power (kW), 0 ≤ discharge_kw[t] ≤ {battery.dmax_kw}
# - **import_kw[t]**: Import from grid (kW), import_kw[t] ≥ 0
# - **export_kw[t]**: Export to grid (kW), export_kw[t] ≥ 0 (only if export allowed)
# - **soc[t]**: State of charge at time t (fraction), {battery.soc_min} ≤ soc[t] ≤ {battery.soc_max}

# ### CRITICAL CONSTRAINT 1: State of Charge Evolution
# For each time step t:
# ```
# soc[t+1] = soc[t] + ({battery.eta_c} × charge_kw[t] × {dt} - discharge_kw[t] × {dt} / {battery.eta_d}) / {battery.capacity_kwh}
# ```

# ### CRITICAL CONSTRAINT 2: Energy Storage Limits
# **The battery has FINITE energy storage capacity!**

# Starting from SoC = {battery.soc_init}:
# - **Maximum energy available for charging**: {max_charge_energy:.2f} kWh
# - **Maximum energy available for discharging**: {max_discharge_energy:.2f} kWh

# This means:
# - **Maximum consecutive charge time steps**: {max_feasible_charge_steps} steps (at full power)
# - **Maximum consecutive discharge time steps**: {max_feasible_discharge_steps} steps (at full power)

# **IMPORTANT**: If you discharge for {max_feasible_discharge_steps + 3} consecutive steps at full power, 
# you would need {((max_feasible_discharge_steps + 3) * battery.dmax_kw * dt / battery.eta_d):.1f} kWh 
# but only have {max_discharge_energy:.2f} kWh available - THIS IS PHYSICALLY IMPOSSIBLE!

# ### CONSTRAINT 3: No Simultaneous Charge/Discharge
# For each t: charge_kw[t] × discharge_kw[t] = 0 (cannot both be positive)

# ### CONSTRAINT 4: Power Balance
# For each t:
# ```
# import_kw[t] - export_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]
# ```
# Note: export_kw[t] = 0 for all t if export is not allowed.

# ### Optimization Process

# **STAGE 1: Make decisions using FORECAST data**
# Forecast prices (€/MWh): {[f'{p:.2f}' for p in p_buy_forecast]}
# Forecast demand (kW): {[f'{d:.1f}' for d in demand_forecast]}

# Identify opportunities:
# - Charge when price < {price_threshold_low:.2f} €/MWh
# - Discharge when price > {price_threshold_high:.2f} €/MWh
# - Always respect SoC bounds and energy limits!

# **STAGE 2: Calculate actual cost using ACTUAL data**
# After determining charge_kw[t] and discharge_kw[t], calculate:
# - import_kw and export_kw using power balance with actual demand
# - Total cost using actual prices: {[f'{p:.2f}' for p in p_buy_actual[:10]]}... (showing first 10)

# ### Solution Requirements

# Provide arrays of length {T} for:
# 1. **charge_kw**: Charging power at each hour
# 2. **discharge_kw**: Discharging power at each hour  
# 3. **import_kw**: Grid import at each hour
# 4. **export_kw**: Grid export at each hour
# 5. **soc**: Array of length {T+1} showing SoC trajectory

# Also provide:
# 6. **objective_cost**: Total cost calculated with ACTUAL prices

# ### Pre-Solution Validation

# Before finalizing your solution, verify:
# 1. Count consecutive discharge periods - none should exceed {max_feasible_discharge_steps} steps
# 2. Count consecutive charge periods - none should exceed {max_feasible_charge_steps} steps
# 3. Verify SoC stays in [{battery.soc_min}, {battery.soc_max}] at all times
# 4. Check soc[{T}] ≥ {battery.soc_target if battery.soc_target is not None else battery.soc_init}

# Return status="success" only if ALL constraints are satisfied.
# """
    
#     return instructions


# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM"""
    
#     T = len(day_inputs.prices_buy)
    
#     # Calculate battery physical limits
#     max_charge_hours = battery.capacity_kwh / (battery.cmax_kw * battery.eta_c * day_inputs.dt_hours)
#     max_discharge_hours = (battery.capacity_kwh * battery.eta_d) / (battery.dmax_kw * day_inputs.dt_hours)
    
#     p_buy_forecast = np.asarray(day_inputs.prices_buy_forecast, dtype=float)
#     demand_forecast = np.asarray(day_inputs.demand_kw_forecast, dtype=float)
#     p_sell_forecast = np.asarray(day_inputs.prices_sell_forecast if day_inputs.allow_export and day_inputs.prices_sell_forecast else day_inputs.prices_buy_forecast, dtype=float)
    
#     p_buy_actual = np.asarray(day_inputs.prices_buy, dtype=float)
#     demand_actual = np.asarray(day_inputs.demand_kw, dtype=float)
    
#     # Calculate forecast statistics for decision-making
#     forecast_mean = np.mean(p_buy_forecast)
#     forecast_std = np.std(p_buy_forecast)
    
#     instructions = f'''
# You are solving a battery scheduling optimization problem with forecast-based decision making and ex-post evaluation.

# ============================================================
# CRITICAL PHYSICAL CONSTRAINTS - MUST BE SATISFIED
# ============================================================
# Battery Capacity: {battery.capacity_kwh} kWh
# Maximum Power: {battery.cmax_kw} kW (both charge and discharge)

# PHYSICAL LIMIT: The battery can charge from empty to full in {max_charge_hours:.1f} hours minimum.
# - Starting from SoC={battery.soc_init}, you can charge at most {(battery.soc_max - battery.soc_init) * max_charge_hours:.1f} consecutive hours
# - Starting from SoC={battery.soc_init}, you can discharge at most {(battery.soc_init - battery.soc_min) * max_discharge_hours:.1f} consecutive hours
# - You CANNOT charge or discharge indefinitely - respect the battery's finite capacity!

# ============================================================
# TWO-STAGE PROCESS
# ============================================================

# STAGE 1: DECISION MAKING (Use FORECAST data only)
# --------------------------------------------------
# Use ONLY forecasted data to decide when to charge/discharge:
# - Forecast prices: mean={forecast_mean:.2f}, std={forecast_std:.2f}
# - Forecast demand: {demand_forecast.tolist()}

# Decision strategy:
# 1. Identify LOW price hours (< {forecast_mean - 0.5*forecast_std:.2f}): Good for charging
# 2. Identify HIGH price hours (> {forecast_mean + 0.5*forecast_std:.2f}): Good for discharging
# 3. Plan charge/discharge cycles respecting physical limits

# STAGE 2: EVALUATION (Apply decisions to ACTUAL data)
# --------------------------------------------------
# Once decisions are made, calculate the realized cost using actual data:
# - The schedule (charge_kw, discharge_kw) remains FIXED from Stage 1
# - Calculate actual cost = Σ(actual_price × import_kw - actual_price × export_kw)

# ============================================================
# EXAMPLE OPTIMIZATION (Illustrative - NOT actual data)
# ============================================================
# Consider a simplified 8-hour example:
# - Battery: 10 kWh capacity, 5 kW max power, η=0.9, initial SoC=0.5
# - Physical limit: Can charge/discharge for max 2.2 hours from 50% SoC
# - Forecast prices: [45, 40, 35, 30, 50, 55, 60, 65] €/MWh
# - Forecast demand: [10, 10, 10, 10, 10, 10, 10, 10] kW

# DECISION PROCESS (Stage 1):
# Hours 0-1: Prices above average (45, 40) → Stay idle, observe
# Hour 2-3: Prices dropping (35, 30) → CHARGE at 5 kW for 2 hours
#   - SoC: 0.5 → 0.5 + (0.9×5×2)/10 = 0.95 (near max, must stop charging)
# Hours 4-7: Prices rising (50, 55, 60, 65) → DISCHARGE at 5 kW
#   - But can only discharge for ~2 hours from 95% SoC
#   - Hours 4-5: Discharge at 5 kW
#   - SoC: 0.95 → 0.95 - (5×2)/(0.9×10) = 0.84
#   - Hours 6-7: Continue discharging would violate minimum SoC, so idle

# This respects physical limits while capturing price arbitrage.

# ============================================================
# YOUR TASK
# ============================================================
# Time horizon: {T} hours
# Battery parameters:
# - Capacity: {battery.capacity_kwh} kWh
# - Power: {battery.cmax_kw} kW
# - Efficiency: charge={battery.eta_c}, discharge={battery.eta_d}
# - Initial SoC: {battery.soc_init}
# - SoC limits: [{battery.soc_min}, {battery.soc_max}]
# - Target end SoC: {battery.soc_target}

# Forecast data for decisions:
# - Prices: {p_buy_forecast.tolist()}
# - Demand: {demand_forecast.tolist()}

# Generate feasible schedule with:
# 1. charge_kw[t]: list of {T} values (0 to {battery.cmax_kw})
# 2. discharge_kw[t]: list of {T} values (0 to {battery.dmax_kw})
# 3. import_kw[t]: list of {T} values (>= 0)
# 4. export_kw[t]: list of {T} values (>= 0 if allowed)
# 5. soc[t]: list of {T+1} values (including initial)
# 6. decision[t]: list of {T} values (+1=charge, -1=discharge, 0=idle)
# 7. objective_cost: Total cost using ACTUAL prices

# CRITICAL CHECKS:
# ✓ Battery cannot charge/discharge beyond capacity limits
# ✓ SoC[t+1] = SoC[t] + (η_c × charge[t] - discharge[t]/η_d) × dt / capacity
# ✓ {battery.soc_min} ≤ SoC[t] ≤ {battery.soc_max} for all t
# ✓ Cannot charge AND discharge simultaneously
# ✓ Demand must always be met: import[t] + discharge[t] = demand[t] + charge[t] + export[t]
# ✓ Maximum consecutive charge hours: ~{max_charge_hours:.1f} (from empty to full)
# ✓ Maximum consecutive discharge hours: ~{max_discharge_hours:.1f} (from full to empty)

# Return SolveResponse with status="success" and a brief message explaining your strategy.
# '''
    
#     return instructions

# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM"""
    
#     T = len(day_inputs.prices_buy)

#     p_buy_actual = np.asarray(day_inputs.prices_buy, dtype=float)
#     demand_actual  = np.asarray(day_inputs.demand_kw, dtype=float)
#     if day_inputs.allow_export:
#         p_sell_actual = np.asarray(day_inputs.prices_sell if day_inputs.prices_sell is not None else day_inputs.prices_buy, dtype=float)
#     else:
#         p_sell_actual = None

#     p_buy_forecast = np.asarray(day_inputs.prices_buy_forecast, dtype=float)
#     demand_forecast = np.asarray(day_inputs.demand_kw_forecast, dtype=float)
#     if day_inputs.allow_export:
#         p_sell_forecast = np.asarray(day_inputs.prices_sell_forecast if day_inputs.prices_sell_forecast is not None else day_inputs.prices_buy_forecast, dtype=float)
#     else:
#         p_sell_forecast = None

    
#     # Calculate statistics
#     price_mean = sum(p_buy_actual) / len(p_buy_actual)
#     price_min, price_max = min(p_buy_actual), max(p_buy_actual)
#     instructions = f'''
#         You are solving a daily battery scheduling optimization problem using forecast-based reasoning and constraint satisfaction.

#         You are provided with both forecasted and actual market data:

#         FORECAST INPUTS (for decision-making):
#             - Forecasted buying prices: {p_buy_forecast}  (array of length T)
#             - Forecasted selling prices: {p_sell_forecast}  (array of length T)
#             - Forecasted demand: {demand_forecast}  (array of length T)

#         ACTUAL INPUTS (for ex-post evaluation):
#             - Realized buying prices: {p_buy_actual}  (array of length T)
#             - Realized selling prices: {p_sell_actual}  (array of length T)
#             - Realized demand: {demand_actual}  (array of length T)

#         BATTERY PARAMETERS:
#             - capacity_kwh: {battery.capacity_kwh}
#             - charge/discharge limits: cmax_kw={battery.cmax_kw}, dmax_kw={battery.dmax_kw}
#             - efficiencies: eta_c={battery.eta_c}, eta_d={battery.eta_d}
#             - SoC bounds: {battery.soc_min} ≤ SoC ≤ {battery.soc_max}
#             - initial SoC: soc_init={battery.soc_init}
#             - target SoC: soc_target={battery.soc_target}

#         HORIZON:
#             - Number of timesteps: T = {len(p_buy_forecast)}
#             - Duration per step: dt_hours = {day_inputs.dt_hours}
#             - Export allowed: {day_inputs.allow_export}

#         ------------------------------------------------------------
#         STAGE 1: FORECAST-BASED DECISION OPTIMIZATION
#         ------------------------------------------------------------
#         Use forecasted information only (p_buy_forecast, p_sell_forecast, demand_forecast) to determine the following hourly decision variables:

#             charge_kw[t], discharge_kw[t], import_kw[t], export_kw[t], soc[t]
        
#         for every time t in {0} ≤ t < {T}

#         Subject to constraints for all t:
#             - SoC dynamics:
#                 SoC[t+1] = SoC[t] + ({battery.eta_c} * charge_kw[t] - discharge_kw[t] / {battery.eta_d}) * {day_inputs.dt_hours} / {battery.capacity_kwh}    
#             - SoC bounds: {battery.soc_min} ≤ SoC[t] ≤ {battery.soc_max} for all t
#             - Power limits: 
#                 0 ≤ charge_kw[t] ≤ {battery.cmax_kw}
#                 0 ≤ discharge_kw[t] ≤ {battery.dmax_kw}
#             - Energy balance:
#                 import_kw[t] - export_kw[t] = demand_actual[t] + charge_kw[t] - discharge_kw[t]
#             - Export constraint: export_kw[t] ≥ 0 only if allow_export = {day_inputs.allow_export}
#             - Initial condition: SoC[0] = {battery.soc_init}
#             - End condition: SoC[T] ≥ {battery.soc_target}
#             - No simultaneous charge/discharge: The battery can either charge OR discharge OR stay idle in a given hour, not both.
#             This means: NOT(charge_kw[t] > 0 AND discharge_kw[t] > 0)

#         Forecast-based objective to minimize:
#             forecast_cost = Σ_t [ (p_buy_forecast[t] * import_kw[t] - p_sell_forecast[t] * export_kw[t]) * {day_inputs.dt_hours} ]

#         Decision logic:
#             - Ensure SoC and power limits are respected
#             - Price range: min={price_min:.2f}, max={price_max:.2f}, mean={price_mean:.2f}
#             - Charge the battery when prices are LOW (below mean) for any time t, eg: Prefer charging when p_buy_forecast < {price_mean}
#             - Discharge the battery when prices are HIGH (above mean) for any time t, eg: Prefer discharging when p_buy_forecast > {price_mean} 
#             - Always meet {demand_actual} at every timestep t

#         ------------------------------------------------------------
#         STAGE 2: EX-POST EVALUATION (USING ACTUAL DATA)
#         ------------------------------------------------------------
#         Once the forecast-based decisions are determined (charge/discharge schedules fixed),
#         apply them to actual data ({p_buy_actual, p_sell_actual, demand_actual}) to compute realized cost.

#         Realized cost:
#             realized_cost = Σ_t [ (p_buy_actual[t] * import_kw[t] - p_sell_actual[t] * export_kw[t]) * {day_inputs.dt_hours} ]

#         ------------------------------------------------------------
#         OUTPUT (SolveResponse)
#         ------------------------------------------------------------
#         Return the following fields:
#             - status: "success" or "failure"
#             - message:  Brief explanation of your optimization strategy (2-3 sentences)
#             - objective_cost: realized_cost
#             - charge_kw: list of {T} hourly charge power values, note that these values are capped by the battery's maximum charging power and at a time it can be either charging or discharging or idle.
#             - discharge_kw: list of {T} hourly discharge power values, note that these values are capped by the battery's maximum discharging power and at a time it can be either charging or discharging or idle.
#             - import_kw: list of {T} hourly grid import values and at a time it can be either importing from the grid or exporting to the grid and not both, but satisfying the demand and battery charge cum discharge power.
#             - export_kw: list of {T} hourly grid export values and at a time it can be either importing from the grid or exporting to the grid and not both, but satisfying the demand and battery charge cum discharge power.
#             - soc: list of {T+1} SoC fractions (0–1) which is a fraction value of the battery capacity.
#             - decision: list of {T} values (+1=charge, -1=discharge, 0=idle)

#         ------------------------------------------------------------
#         GOAL
#         ------------------------------------------------------------
#         Generate physically feasible schedules that:
#             1. Are optimized using forecasted data only,
#             2. Are evaluated against actual realized data,
#             3. Minimize realized total cost,
#             4. Respect all technical and operational constraints.
            
#         Make sure:
#         - All lists have the correct length ({T} for hourly values, {T+1} for soc)
#         - All constraints are satisfied at every timestep
#         - The objective function (total cost) is minimized
#         - The schedule is physically feasible following the entire battery physics and operational constraints with charging and discharging efficiency

#         Think step by step:
#         1. Identify low-price hours for charging
#         2. Identify high-price hours for discharging  
#         3. Calculate optimal charge/discharge amounts respecting battery limits
#         4. Verify SoC stays within bounds
#         5. Ensure demand is always met
#         6. Calculate total cost

#         Generate your complete SolveResponse now.
#         '''
    
#     return instructions


# """
# Day-Specific Battery Optimization using LLM

# This module optimizes battery storage operations for a specific day using
# LLM-based reasoning, with support for forecasted or actual data.
# """

# import os
# import warnings
# from dotenv import load_dotenv
# from agentics.core.agentics import AG

# from .schemas import (
#     DayOptimizationRequest, SolveResponse, SolveRequest, 
#     BatteryParams, DayInputs
# )
# from .day_data_loader import DayDataLoader

# warnings.filterwarnings("ignore", category=UserWarning)
# load_dotenv()
# os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


# async def optimize_day(
#     request: DayOptimizationRequest,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Optimize battery operations for a specific day
    
#     Args:
#         request: Day optimization request with date, battery params, and forecast options
#         data_dir: Optional path to data directory
        
#     Returns:
#         SolveResponse with optimized schedule and comprehensive explanation
#     """
    
#     # Load data for the specific day
#     loader = DayDataLoader(data_dir=data_dir)
#     day_inputs, metadata = loader.load_day_data(
#         date=request.date,
#         use_forecast=request.use_forecast,
#         forecast_models=request.forecast_models,
#         allow_export=request.allow_export,
#         dt_hours=request.dt_hours
#     )
    
#     # Create solve request
#     solve_req = SolveRequest(
#         battery=request.battery,
#         day=day_inputs,
#         solver=request.solver,
#         solver_opts=request.solver_opts
#     )
    
#     # Create source AG object
#     source = AG(
#         atype=SolveRequest,
#         states=[solve_req]
#     )
    
#     # Build comprehensive instructions
#     instructions = _build_optimization_instructions(
#         battery=request.battery,
#         day_inputs=day_inputs,
#         metadata=metadata
#     )
    
#     # Create target AG object with LLM reasoning
#     target = AG(
#         atype=SolveResponse,
#         max_iter=3,  # Increased iterations for better convergence
#         verbose_agent=True,
#         reasoning=True,
#         instructions=instructions
#     )
    
#     # Execute optimization with error handling
#     try:
#         result = await (target << source)
        
#         # Extract response and add metadata
#         response = result.states[0] if result.states else None
        
#         if response is None:
#             raise ValueError("LLM returned no valid response")
            
#     except Exception as e:
#         print(f"Error during LLM optimization: {e}")
#         print("Returning fallback response...")
        
#         # Return a fallback error response
#         return SolveResponse(
#             status="error",
#             message=f"LLM optimization failed: {str(e)}. Please try again or adjust parameters.",
#             objective_cost=0.0,
#             charge_kw=[0.0] * len(day_inputs.prices_buy),
#             discharge_kw=[0.0] * len(day_inputs.prices_buy),
#             import_kw=day_inputs.demand_kw,
#             export_kw=[0.0] * len(day_inputs.prices_buy) if day_inputs.allow_export else None,
#             soc=[request.battery.soc_init] * (len(day_inputs.prices_buy) + 1),
#             data_source=metadata["data_source"]
#         )
    
#     if response:
#         # Add data source information
#         response.data_source = metadata["data_source"]
#         if metadata.get("forecast_models"):
#             response.forecast_info = metadata["forecast_models"]
    
#     return response


# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM"""
    
#     T = len(day_inputs.prices_buy)
#     prices = day_inputs.prices_buy
#     demand = day_inputs.demand_kw
    
#     # Calculate statistics
#     price_mean = sum(prices) / len(prices)
#     price_min, price_max = min(prices), max(prices)
    
#     # Generate simple heuristic solution as example/baseline
#     charge_kw = []
#     discharge_kw = []
#     import_kw = []
#     export_kw = []
#     soc = [battery.soc_init]
#     decision = []
    
#     for t in range(T):
#         # Simple heuristic: charge when price < mean, discharge when price > mean
#         if prices[t] < price_mean * 0.85 and soc[-1] < battery.soc_max - 0.1:
#             # Charge
#             charge = min(battery.cmax_kw, (battery.soc_max - soc[-1]) * battery.capacity_kwh / day_inputs.dt_hours)
#             charge_kw.append(charge)
#             discharge_kw.append(0.0)
#             import_kw.append(demand[t] + charge)
#             export_kw.append(0.0)
#             decision.append(1)
#             new_soc = soc[-1] + (battery.eta_c * charge * day_inputs.dt_hours) / battery.capacity_kwh
#         elif prices[t] > price_mean * 1.15 and soc[-1] > battery.soc_min + 0.1:
#             # Discharge
#             discharge = min(battery.dmax_kw, (soc[-1] - battery.soc_min) * battery.capacity_kwh / day_inputs.dt_hours)
#             discharge = min(discharge, demand[t])  # Don't discharge more than needed
#             charge_kw.append(0.0)
#             discharge_kw.append(discharge)
#             import_kw.append(max(0, demand[t] - discharge))
#             export_kw.append(0.0)
#             decision.append(-1)
#             new_soc = soc[-1] - (discharge * day_inputs.dt_hours) / (battery.eta_d * battery.capacity_kwh)
#         else:
#             # Idle
#             charge_kw.append(0.0)
#             discharge_kw.append(0.0)
#             import_kw.append(demand[t])
#             export_kw.append(0.0)
#             decision.append(0)
#             new_soc = soc[-1]
        
#         soc.append(max(battery.soc_min, min(battery.soc_max, new_soc)))
    
#     # Calculate baseline cost
#     baseline_cost = sum(prices[t] * import_kw[t] * day_inputs.dt_hours for t in range(T))
    
#     instructions = f"""
# You must generate a valid SolveResponse for battery optimization.

# **PROBLEM:**
# Date: {metadata['date']}
# Timesteps: {T} hours
# Battery: {battery.capacity_kwh} kWh capacity, {battery.cmax_kw} kW power
# Prices: min={price_min:.2f}, max={price_max:.2f}, mean={price_mean:.2f} €/kWh

# **STRATEGY:**
# 1. Charge battery when prices are LOW (< {price_mean * 0.85:.2f})
# 2. Discharge battery when prices are HIGH (> {price_mean * 1.15:.2f})
# 3. Stay idle otherwise
# 4. Always meet demand

# **BASELINE SOLUTION (Simple Heuristic):**
# Total Cost: €{baseline_cost:.2f}
# Charges: {sum(1 for d in decision if d > 0)} hours
# Discharges: {sum(1 for d in decision if d < 0)} hours

# **YOUR TASK:** 
# Return a VALID SolveResponse JSON object with these EXACT fields:

# {{
#   "status": "success",
#   "message": "I am an energy optimization AI. For {metadata['date']}, I analyzed the price pattern (range €{price_min:.2f}-{price_max:.2f}). I charged the battery during low-price hours and discharged during high-price hours to minimize costs while meeting all demand and battery constraints.",
#   "objective_cost": {baseline_cost:.2f},
#   "charge_kw": {charge_kw},
#   "discharge_kw": {discharge_kw},
#   "import_kw": {import_kw},
#   "export_kw": {'null' if not day_inputs.allow_export else export_kw},
#   "soc": {soc},
#   "decision": {decision},
#   "confidence": null
# }}

# You can IMPROVE on this baseline by optimizing better, but you MUST return a valid JSON matching this structure with all {T} timesteps.

# Output the JSON now:"""
    
#     return instructions


# # Convenience function for notebook use
# async def optimize_day_simple(
#     date: str,
#     capacity_kwh: float = None,
#     power_kw: float = None,
#     battery_sizing: str = "manual",
#     use_forecast: bool = False,
#     prices_model: str = None,
#     consumption_model: str = None,
#     allow_export: bool = False,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Simplified interface for day optimization
    
#     Args:
#         date: Date string (YYYY-MM-DD)
#         capacity_kwh: Battery capacity in kWh (if battery_sizing="manual")
#         power_kw: Max charge/discharge power in kW (if battery_sizing="manual")
#         battery_sizing: "manual" or "interquartile"
#             - "manual": use provided capacity_kwh and power_kw
#             - "interquartile": auto-calculate based on load IQR * 4 hours
#         use_forecast: Whether to use forecast data
#         prices_model: Forecast model for prices (RF_pred, LSTM_pred, etc.)
#         consumption_model: Forecast model for consumption
#         allow_export: Allow grid export
#         data_dir: Data directory path
        
#     Returns:
#         SolveResponse with optimization results
#     """
#     from .schemas import ForecastModel
    
#     # Determine battery capacity based on sizing method
#     if battery_sizing == "interquartile":
#         # Calculate battery size based on load statistics
#         loader = DayDataLoader(data_dir=data_dir)
#         stats = loader.get_load_statistics()
        
#         # IQR * 4 hours rule (convert from MW to kW)
#         capacity_kwh = stats['recommended_capacity_mwh'] * 1000  # MWh to kWh
#         power_kw = stats['recommended_power_mw'] * 1000  # MW to kW
        
#         print(f"📊 Automatic Battery Sizing (IQR Method):")
#         print(f"   Load Mean: {stats['mean']:.2f} MW")
#         print(f"   Load IQR: {stats['iqr']:.2f} MW (P25: {stats['p25']:.2f}, P75: {stats['p75']:.2f})")
#         print(f"   → Battery Capacity: {capacity_kwh:.2f} kWh ({capacity_kwh/1000:.2f} MWh)")
#         print(f"   → Charge/Discharge Power: {power_kw:.2f} kW ({power_kw/1000:.2f} MW)")
#         print()
#     else:
#         # Manual sizing
#         if capacity_kwh is None or power_kw is None:
#             raise ValueError("capacity_kwh and power_kw must be provided when battery_sizing='manual'")
    
#     battery = BatteryParams(
#         capacity_kwh=capacity_kwh,
#         soc_init=0.5,
#         soc_min=0.1,
#         soc_max=0.9,
#         cmax_kw=power_kw,
#         dmax_kw=power_kw,
#         eta_c=0.95,
#         eta_d=0.95,
#         soc_target=0.5
#     )
    
#     forecast_models = None
#     if use_forecast:
#         forecast_models = ForecastModel(
#             prices_model=prices_model,
#             consumption_model=consumption_model
#         )
    
#     request = DayOptimizationRequest(
#         date=date,
#         battery=battery,
#         use_forecast=use_forecast,
#         forecast_models=forecast_models,
#         allow_export=allow_export,
#         dt_hours=1.0
#     )
    
#     return await optimize_day(request, data_dir=data_dir)

# """
# Day-Specific Battery Optimization using LLM

# This module optimizes battery storage operations for a specific day using
# LLM-based reasoning, with support for forecasted or actual data.
# """

# import os
# import warnings
# from dotenv import load_dotenv
# from agentics.core.agentics import AG

# from .schemas import (
#     DayOptimizationRequest, SolveResponse, SolveRequest, 
#     BatteryParams, DayInputs
# )
# from .day_data_loader import DayDataLoader

# warnings.filterwarnings("ignore", category=UserWarning)
# load_dotenv()
# os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


# async def optimize_day(
#     request: DayOptimizationRequest,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Optimize battery operations for a specific day
    
#     Args:
#         request: Day optimization request with date, battery params, and forecast options
#         data_dir: Optional path to data directory
        
#     Returns:
#         SolveResponse with optimized schedule and comprehensive explanation
#     """
    
#     # Load data for the specific day
#     loader = DayDataLoader(data_dir=data_dir)
#     day_inputs, metadata = loader.load_day_data(
#         date=request.date,
#         use_forecast=request.use_forecast,
#         forecast_models=request.forecast_models,
#         allow_export=request.allow_export,
#         dt_hours=request.dt_hours
#     )
    
#     # Create solve request
#     solve_req = SolveRequest(
#         battery=request.battery,
#         day=day_inputs,
#         solver=request.solver,
#         solver_opts=request.solver_opts
#     )
    
#     # Create source AG object
#     source = AG(
#         atype=SolveRequest,
#         states=[solve_req]
#     )
    
#     # Build comprehensive instructions
#     instructions = _build_optimization_instructions(
#         battery=request.battery,
#         day_inputs=day_inputs,
#         metadata=metadata
#     )
    
#     # Create target AG object with LLM reasoning
#     target = AG(
#         atype=SolveResponse,
#         max_iter=3,  # Increased iterations for better convergence
#         verbose_agent=True,
#         reasoning=True,
#         instructions=instructions
#     )
    
#     # Execute optimization with error handling
#     try:
#         result = await (target << source)
        
#         # Extract response and add metadata
#         response = result.states[0] if result.states else None
        
#         if response is None:
#             raise ValueError("LLM returned no valid response")
            
#     except Exception as e:
#         print(f"Error during LLM optimization: {e}")
#         print("Returning fallback response...")
        
#         # Return a fallback error response
#         return SolveResponse(
#             status="error",
#             message=f"LLM optimization failed: {str(e)}. Please try again or adjust parameters.",
#             objective_cost=0.0,
#             charge_kw=[0.0] * len(day_inputs.prices_buy),
#             discharge_kw=[0.0] * len(day_inputs.prices_buy),
#             import_kw=day_inputs.demand_kw,
#             export_kw=[0.0] * len(day_inputs.prices_buy) if day_inputs.allow_export else None,
#             soc=[request.battery.soc_init] * (len(day_inputs.prices_buy) + 1),
#             data_source=metadata["data_source"]
#         )
    
#     if response:
#         # Add data source information
#         response.data_source = metadata["data_source"]
#         if metadata.get("forecast_models"):
#             response.forecast_info = metadata["forecast_models"]
    
#     return response


# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM"""
    
#     T = len(day_inputs.prices_buy)
#     prices = day_inputs.prices_buy
#     demand = day_inputs.demand_kw
    
#     # Calculate statistics
#     price_mean = sum(prices) / len(prices)
#     price_min, price_max = min(prices), max(prices)
    
#     instructions = f"""
# You are an expert energy storage optimization AI solving a battery scheduling problem.

# **OBJECTIVE:** Minimize total cost = Σ [(price_buy[t] × import_kw[t] - price_sell[t] × export_kw[t]) × {day_inputs.dt_hours}]

# **GIVEN DATA FOR {metadata['date']}:**
# - {T} hourly timesteps
# - Prices (€/kWh): range [{price_min:.2f}, {price_max:.2f}], mean {price_mean:.2f}
# - Demand (kW): {demand[:3]}...
# - Data source: {metadata['data_source']}

# **BATTERY CONSTRAINTS:**
# - Capacity: {battery.capacity_kwh} kWh
# - Power: charge ≤ {battery.cmax_kw} kW, discharge ≤ {battery.dmax_kw} kW
# - Efficiency: charge {battery.eta_c}, discharge {battery.eta_d}
# - SoC limits: [{battery.soc_min}, {battery.soc_max}]
# - Initial SoC: {battery.soc_init}, Target end SoC: {battery.soc_target or battery.soc_init}

# **REQUIRED CONSTRAINTS:**
# 1. Energy balance: import - export = demand + charge - discharge
# 2. SoC update: soc[t+1] = soc[t] + ({battery.eta_c}*charge[t] - discharge[t]/{battery.eta_d})*{day_inputs.dt_hours}/{battery.capacity_kwh}
# 3. Cannot charge AND discharge simultaneously
# 4. Meet demand: demand[t] ≤ import[t] + discharge[t] - charge[t]

# **YOUR TASK:**
# Generate optimal {T}-hour schedule that:
# - Charges when prices are LOW
# - Discharges when prices are HIGH  
# - Always meets demand
# - Respects all battery constraints

# **OUTPUT AS SolveResponse with:**
# - status: "success"
# - message: Brief explanation (3-5 sentences) of your strategy
# - objective_cost: total cost in €
# - charge_kw: list of {T} charge values
# - discharge_kw: list of {T} discharge values
# - import_kw: list of {T} import values
# - export_kw: list of {T} export values {'(or null)' if not day_inputs.allow_export else ''}
# - soc: list of {T+1} SoC fractions [including initial]
# - decision: list of {T} values: +1=charge, -1=discharge, 0=idle

# Generate the response now.
# """
    
#     return instructions


# # Convenience function for notebook use
# async def optimize_day_simple(
#     date: str,
#     capacity_kwh: float = 100.0,
#     power_kw: float = 50.0,
#     use_forecast: bool = False,
#     prices_model: str = None,
#     consumption_model: str = None,
#     allow_export: bool = False,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Simplified interface for day optimization
    
#     Args:
#         date: Date string (YYYY-MM-DD)
#         capacity_kwh: Battery capacity in kWh
#         power_kw: Max charge/discharge power in kW
#         use_forecast: Whether to use forecast data
#         prices_model: Forecast model for prices (RF_pred, LSTM_pred, etc.)
#         consumption_model: Forecast model for consumption
#         allow_export: Allow grid export
#         data_dir: Data directory path
        
#     Returns:
#         SolveResponse with optimization results
#     """
#     from .schemas import ForecastModel
    
#     battery = BatteryParams(
#         capacity_kwh=capacity_kwh,
#         soc_init=0.5,
#         soc_min=0.1,
#         soc_max=0.9,
#         cmax_kw=power_kw,
#         dmax_kw=power_kw,
#         eta_c=0.95,
#         eta_d=0.95,
#         soc_target=0.5
#     )
    
#     forecast_models = None
#     if use_forecast:
#         forecast_models = ForecastModel(
#             prices_model=prices_model,
#             consumption_model=consumption_model
#         )
    
#     request = DayOptimizationRequest(
#         date=date,
#         battery=battery,
#         use_forecast=use_forecast,
#         forecast_models=forecast_models,
#         allow_export=allow_export,
#         dt_hours=1.0
#     )
    
#     return await optimize_day(request, data_dir=data_dir)


# """
# Day-Specific Battery Optimization using LLM

# This module optimizes battery storage operations for a specific day using
# LLM-based reasoning, with support for forecasted or actual data.
# """

# import os
# import warnings
# from dotenv import load_dotenv
# from agentics.core.agentics import AG

# from .schemas import (
#     DayOptimizationRequest, SolveResponse, SolveRequest, 
#     BatteryParams, DayInputs
# )
# from .day_data_loader import DayDataLoader

# warnings.filterwarnings("ignore", category=UserWarning)
# load_dotenv()
# os.environ.setdefault("CREWAI_TOOLS_DISABLE_AUTO_INSTALL", "1")


# async def optimize_day(
#     request: DayOptimizationRequest,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Optimize battery operations for a specific day
    
#     Args:
#         request: Day optimization request with date, battery params, and forecast options
#         data_dir: Optional path to data directory
        
#     Returns:
#         SolveResponse with optimized schedule and comprehensive explanation
#     """
    
#     # Load data for the specific day
#     loader = DayDataLoader(data_dir=data_dir)
#     day_inputs, metadata = loader.load_day_data(
#         date=request.date,
#         use_forecast=request.use_forecast,
#         forecast_models=request.forecast_models,
#         allow_export=request.allow_export,
#         dt_hours=request.dt_hours
#     )
    
#     # Create solve request
#     solve_req = SolveRequest(
#         battery=request.battery,
#         day=day_inputs,
#         solver=request.solver,
#         solver_opts=request.solver_opts
#     )
    
#     # Create source AG object
#     source = AG(
#         atype=SolveRequest,
#         states=[solve_req]
#     )
    
#     # Build comprehensive instructions
#     instructions = _build_optimization_instructions(
#         battery=request.battery,
#         day_inputs=day_inputs,
#         metadata=metadata
#     )
    
#     # Create target AG object with LLM reasoning
#     target = AG(
#         atype=SolveResponse,
#         max_iter=1,
#         verbose_agent=True,
#         reasoning=True,
#         instructions=instructions
#     )
    
#     # Execute optimization
#     result = await (target << source)
    
#     # Extract response and add metadata
#     response = result.states[0] if result.states else None
    
#     if response:
#         # Add data source information
#         response.data_source = metadata["data_source"]
#         if metadata.get("forecast_models"):
#             response.forecast_info = metadata["forecast_models"]
    
#     return response


# def _build_optimization_instructions(
#     battery: BatteryParams,
#     day_inputs: DayInputs,
#     metadata: dict
# ) -> str:
#     """Build comprehensive optimization instructions for the LLM"""
    
#     T = len(day_inputs.prices_buy)
#     prices = day_inputs.prices_buy
#     demand = day_inputs.demand_kw
    
#     # Calculate statistics
#     price_mean = sum(prices) / len(prices)
#     price_min, price_max = min(prices), max(prices)
#     demand_mean = sum(demand) / len(demand)
    
#     # Identify price patterns
#     low_price_hours = [i for i, p in enumerate(prices) if p < price_mean * 0.9]
#     high_price_hours = [i for i, p in enumerate(prices) if p > price_mean * 1.1]
    
#     instructions = f"""
# You are an AI energy storage optimization agent tasked with minimizing operational costs for a battery energy storage system.

# **YOUR IDENTITY:**
# You are an expert energy arbitrage optimizer with deep knowledge of:
# - Mixed Integer Linear Programming (MILP) for energy storage
# - Electricity market dynamics and price patterns
# - Battery physics and operational constraints
# - Multi-stage decision-making under uncertainty

# **PROBLEM CONTEXT:**
# Date: {metadata['date']}
# Data Source: {metadata['data_source'].upper()}
# {f"Forecast Models: Prices={metadata.get('forecast_models', {}).get('prices', 'actual')}, Consumption={metadata.get('forecast_models', {}).get('consumption', 'actual')}" if metadata['data_source'] == 'forecast' else ''}
# Time Steps: {T} hours
# Time Resolution: {day_inputs.dt_hours} hours per step

# **GIVEN DATA:**
# Prices (€/kWh): {prices[:5]}... (range: {price_min:.2f} - {price_max:.2f}, mean: {price_mean:.2f})
# Demand (kW): {demand[:5]}... (mean: {demand_mean:.2f})
# Allow Export: {day_inputs.allow_export}

# **BATTERY SPECIFICATIONS:**
# - Capacity: {battery.capacity_kwh} kWh
# - Max Charge Rate: {battery.cmax_kw} kW
# - Max Discharge Rate: {battery.dmax_kw} kW
# - Charge Efficiency: {battery.eta_c * 100}%
# - Discharge Efficiency: {battery.eta_d * 100}%
# - Initial SoC: {battery.soc_init * 100}%
# - SoC Limits: [{battery.soc_min * 100}%, {battery.soc_max * 100}%]
# - Target End SoC: {(battery.soc_target or battery.soc_init) * 100}%

# **OPTIMIZATION OBJECTIVE:**
# Minimize total cost = Σ_t [(price_buy[t] × import_kw[t] - price_sell[t] × export_kw[t]) × dt_hours]

# **HARD CONSTRAINTS (MUST SATISFY):**
# 1. **Energy Balance at each timestep t:**
#    import_kw[t] - export_kw[t] = demand_kw[t] + charge_kw[t] - discharge_kw[t]
   
# 2. **State of Charge Dynamics:**
#    soc[t+1] = soc[t] + (eta_c × charge_kw[t] × dt - discharge_kw[t] × dt / eta_d) / capacity_kwh
#    - soc[0] = {battery.soc_init}
#    - soc[T] ≥ {battery.soc_target or battery.soc_init} (end target)
   
# 3. **SoC Limits:**
#    {battery.soc_min} ≤ soc[t] ≤ {battery.soc_max} for all t

# 4. **Power Limits:**
#    - 0 ≤ charge_kw[t] ≤ {battery.cmax_kw}
#    - 0 ≤ discharge_kw[t] ≤ {battery.dmax_kw}

# 5. **No Simultaneous Charge/Discharge:**
#    Battery can either charge OR discharge OR idle at each timestep, but NOT both charge and discharge simultaneously.

# 6. **Non-negative Grid Operations:**
#    - import_kw[t] ≥ 0 (always)
#    - export_kw[t] ≥ 0 (only if allow_export = {day_inputs.allow_export})

# **OPTIMIZATION STRATEGY:**

# **Phase 1: Price Pattern Analysis**
# Analyze the price profile to identify:
# - Low price periods (below {price_mean * 0.9:.2f}): hours {low_price_hours[:5]}...
# - High price periods (above {price_mean * 1.1:.2f}): hours {high_price_hours[:5]}...
# - Price trends and volatility

# **Phase 2: Multi-Cycle Planning**
# The battery should cycle MULTIPLE times throughout the day if profitable:
# - Round-trip efficiency loss: {(1 - battery.eta_c * battery.eta_d) * 100:.1f}%
# - Breakeven spread: price_sell > price_buy × {1/(battery.eta_c * battery.eta_d):.3f}
# - Identify ALL profitable charge-discharge opportunities

# **Phase 3: SoC Trajectory Planning**
# Plan the SoC trajectory to:
# - Start at {battery.soc_init * 100}%
# - Never violate [{battery.soc_min * 100}%, {battery.soc_max * 100}%] bounds
# - End at {(battery.soc_target or battery.soc_init) * 100}%
# - Maximize utilization without hitting limits prematurely

# **Phase 4: Demand Coverage**
# At each hour, ensure:
# demand_kw[t] ≤ import_kw[t] + discharge_kw[t] - charge_kw[t] - export_kw[t]

# **DECISION-MAKING EXAMPLE:**

# Consider a simple case with prices = [100, 50, 150] €/kWh, demand = [10, 10, 10] kW:

# Hour 0 (€100, medium price):
# - Price neither high nor low → Consider battery state
# - If SoC allows, might discharge to avoid high import cost
# - Decision: discharge 5 kW, import 5 kW
# - Cost: 100 × 5 × 1 = €500

# Hour 1 (€50, LOW price):
# - Excellent charging opportunity
# - Demand + charge from grid, possibly prepare for Hour 2
# - Decision: charge 10 kW, import 20 kW total (10 demand + 10 charge)
# - Cost: 50 × 20 × 1 = €1000

# Hour 2 (€150, HIGH price):
# - Most expensive hour → maximize battery discharge
# - Can discharge what we charged in Hour 1
# - Decision: discharge 15 kW (covers 10 kW demand + 5 kW export)
# - Revenue: 150 × 5 × 1 = €750 (export)
# - Cost: 0 (no import needed)
# - Net Hour 2: -€750 (profit)

# Total cost: €500 + €1000 - €750 = €750

# **YOUR TASK:**

# 1. **Analyze** the full price and demand profiles
# 2. **Plan** a multi-cycle charging strategy
# 3. **Generate** hour-by-hour decisions for:
#    - charge_kw[t]: Battery charging power
#    - discharge_kw[t]: Battery discharging power
#    - import_kw[t]: Grid import power
#    - export_kw[t]: Grid export power (if allowed)
#    - soc[t]: State of charge trajectory
#    - decision[t]: +1 (charge), -1 (discharge), 0 (idle)

# 4. **Verify** all constraints are satisfied
# 5. **Calculate** the total objective cost
# 6. **Explain** your strategy comprehensively

# **OUTPUT FORMAT:**

# Return a SolveResponse object with:
# - status: "success" (if feasible) or "failure"
# - message: YOUR COMPREHENSIVE EXPLANATION (see below)
# - objective_cost: Total cost in €
# - charge_kw: List of {T} charging values
# - discharge_kw: List of {T} discharging values
# - import_kw: List of {T} import values
# - export_kw: List of {T} export values (or None if not allowed)
# - soc: List of {T+1} SoC values (including initial state)
# - decision: List of {T} decisions (+1, -1, or 0)

# **MESSAGE FIELD REQUIREMENTS:**

# Your message field must include a COMPREHENSIVE explanation with:

# 1. **Introduction (2-3 sentences):**
#    - Introduce yourself as an energy optimization AI
#    - State the problem you're solving
#    - Mention the date and data source

# 2. **Price Analysis (3-4 sentences):**
#    - Describe the price pattern (volatility, trends)
#    - Identify cheapest and most expensive periods
#    - Calculate price spreads and arbitrage opportunities

# 3. **Strategy Overview (4-5 sentences):**
#    - Explain your high-level approach
#    - Number of charge-discharge cycles planned
#    - How you're balancing demand coverage and arbitrage
#    - Key decision-making principles

# 4. **Hour-by-Hour Decisions (1 paragraph):**
#    - Summarize key decision points
#    - Explain WHY you charge/discharge at specific hours
#    - Mention any constraints that limited your options

# 5. **Battery Utilization (2-3 sentences):**
#    - SoC trajectory summary
#    - How you managed SoC limits
#    - Whether you achieved target end SoC

# 6. **Cost Breakdown (3-4 sentences):**
#    - Total import costs
#    - Total export revenues (if any)
#    - Net cost and potential savings
#    - Compare to naive strategy (e.g., no battery)

# 7. **Validation (2-3 sentences):**
#    - Confirm all constraints satisfied
#    - Mention any challenges or trade-offs
#    - Confidence in solution optimality

# Make your explanation clear, technical, and insightful. Use specific numbers and hours. This will help users understand your decision-making process.

# **CRITICAL REMINDERS:**
# - NEVER charge and discharge simultaneously in the same hour
# - ALWAYS satisfy demand: demand_kw[t] ≤ total supply at time t
# - Battery SoC MUST stay within [{battery.soc_min}, {battery.soc_max}]
# - End SoC MUST be ≥ {battery.soc_target or battery.soc_init}
# - Import costs money, export generates revenue (if allowed)
# - Minimize TOTAL COST across all hours

# Begin your optimization now.
# """
    
#     return instructions


# # Convenience function for notebook use
# async def optimize_day_simple(
#     date: str,
#     capacity_kwh: float = 100.0,
#     power_kw: float = 50.0,
#     use_forecast: bool = False,
#     prices_model: str = None,
#     consumption_model: str = None,
#     allow_export: bool = False,
#     data_dir: str = None
# ) -> SolveResponse:
#     """
#     Simplified interface for day optimization
    
#     Args:
#         date: Date string (YYYY-MM-DD)
#         capacity_kwh: Battery capacity in kWh
#         power_kw: Max charge/discharge power in kW
#         use_forecast: Whether to use forecast data
#         prices_model: Forecast model for prices (RF_pred, LSTM_pred, etc.)
#         consumption_model: Forecast model for consumption
#         allow_export: Allow grid export
#         data_dir: Data directory path
        
#     Returns:
#         SolveResponse with optimization results
#     """
#     from .schemas import ForecastModel
    
#     battery = BatteryParams(
#         capacity_kwh=capacity_kwh,
#         soc_init=0.5,
#         soc_min=0.1,
#         soc_max=0.9,
#         cmax_kw=power_kw,
#         dmax_kw=power_kw,
#         eta_c=0.95,
#         eta_d=0.95,
#         soc_target=0.5
#     )
    
#     forecast_models = None
#     if use_forecast:
#         forecast_models = ForecastModel(
#             prices_model=prices_model,
#             consumption_model=consumption_model
#         )
    
#     request = DayOptimizationRequest(
#         date=date,
#         battery=battery,
#         use_forecast=use_forecast,
#         forecast_models=forecast_models,
#         allow_export=allow_export,
#         dt_hours=1.0
#     )
    
#     return await optimize_day(request, data_dir=data_dir)