import os
import sys
import json
from typing import List,Sequence, Optional

from mcp import StdioServerParameters
from crewai_tools import MCPServerAdapter

from agentic_energy.schemas import (
    BatteryParams,
    DayInputs,
    SolveRequest,
    SolveResponse,
    PlotRequest,
    PlotResponse,
    PriceForecastPlotRequest,
    ReasoningRequest,
    ReasoningResponse,
)

import pandas as pd
import numpy as np

# Path to the root of your project (two levels above mcp_clients.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# Folder containing the static RL schedules
OUTPUT_FILES_DIR = os.path.join(PROJECT_ROOT, "output_files")

def get_tool(tools, name: str):
    """Return MCP tool object by name."""
    for t in tools:
        if t.name == name:
            return t
    raise RuntimeError(f"Tool {name!r} not found. Available: {[t.name for t in tools]}")

def cost_from_soc(
    soc: Sequence[float],
    prices_buy: Sequence[float],
    demand_MW: Sequence[float],
    *,
    battery: BatteryParams,
    prices_sell: Optional[Sequence[float]] = None,
    allow_export: bool = False,
    dt_hours: float = 1.0,
):
    soc = np.asarray(soc, dtype=float)
    assert len(soc) >= 2, "SOC must include at least t=0 and t=1"
    T = len(soc) - 1

    prices_buy  = np.asarray(prices_buy, dtype=float)
    demand_MW   = np.asarray(demand_MW, dtype=float)
    assert len(prices_buy) == T and len(demand_MW) == T

    if prices_sell is None:
        prices_sell = prices_buy
    prices_sell = np.asarray(prices_sell, dtype=float)
    assert len(prices_sell) == T

    # Per-step energy change in MWh
    dE = (soc[1:] - soc[:-1]) * battery.capacity_MWh

    # Recover charge/discharge MW from SOC deltas and efficiencies
    charge_MW    = np.maximum(dE, 0.0) / (battery.eta_c * dt_hours)
    discharge_MW = np.maximum(-dE, 0.0) * (battery.eta_d / dt_hours)

    # Enforce hardware limits
    charge_MW    = np.minimum(charge_MW,    battery.cmax_MW)
    discharge_MW = np.minimum(discharge_MW, battery.dmax_MW)

    # Grid net load
    net = demand_MW + charge_MW - discharge_MW
    imp = np.maximum(net, 0.0)
    exp = np.maximum(-net, 0.0) if allow_export else np.zeros_like(net)

    # Cost (buy imports, optionally credit exports)
    cost = float(np.sum(prices_buy * imp * dt_hours) - np.sum(prices_sell * exp * dt_hours))

    out = {
        "charge_MW": charge_MW,
        "discharge_MW": discharge_MW,
        "import_MW": imp,
        "export_MW": exp,
        "net_MW": net,
        "objective_cost": cost,
    }
    return out


# ---------- Optimizers MCP Client ----------

def run_milp_solver(solve_request: SolveRequest) -> SolveResponse:
    """Call the milp_solve MCP tool and return a SolveResponse."""
    milp_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.milp.milp_mcp_server"],
        env=os.environ,
    )

    with MCPServerAdapter(milp_params) as milp_tools:
        milp_tool = get_tool(milp_tools, "milp_solve")
        call_fn = (
            getattr(milp_tool, "call", None)
            or getattr(milp_tool, "run", None)
            or getattr(milp_tool, "__call__", None)
        )
        if call_fn is None:
            raise RuntimeError("Tool milp_solve has no callable interface")

        raw = call_fn(solverequest=solve_request.model_dump(exclude_none=True))
        data = json.loads(raw)
        return SolveResponse.model_validate(data)

def run_heuristic(solve_request: SolveRequest, mode: str) -> SolveResponse:
    """Call the heuristic MCP tool and return a SolveResponse."""
    heuristic_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.heuristics.heuristic_mcp_server"],
        env=os.environ,
    )

    with MCPServerAdapter(heuristic_params) as heuristic_tools:
        if mode == "time":
            tool_name = "heuristic_time_solve"
        elif mode == "quantile":
            tool_name = "heuristic_quantile_solve"
        heuristic_tool = get_tool(heuristic_tools, tool_name)
        call_fn = (
            getattr(heuristic_tool, "call", None)
            or getattr(heuristic_tool, "run", None)
            or getattr(heuristic_tool, "__call__", None)
        )
        if call_fn is None:
            raise RuntimeError(f"Tool {tool_name} has no callable interface")

        raw = call_fn(solverequest=solve_request.model_dump())
        if isinstance(raw, dict):
            result = SolveResponse(**raw)
        elif isinstance(raw, str):
            parsed = json.loads(raw)
            result = SolveResponse(**parsed)
        else:
            result = SolveResponse.model_validate(raw)
        
        # data = json.loads(raw)
        return result
    
def run_rl_agent(solve_request: SolveRequest, date: str) -> SolveResponse:
    """
    Load a pre-computed RL schedule from a static CSV and wrap it in a SolveResponse.

    Parameters
    ----------
    solve_request : SolveRequest
        Contains battery params, day inputs (prices, dt_hours, etc.).
    date : str
        Date string (e.g. "2018-01-01"). Currently unused except for logging;
        you can use it to choose different CSVs per day if you want.

    Returns
    -------
    SolveResponse
        A schedule that looks like the MILP output, but comes from the RL CSV.
    """

    # ----- 1. Load the static RL schedule CSV -----
    filename = "rlPPO_output.csv"
    csv_path = os.path.join(OUTPUT_FILES_DIR, filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"RL schedule CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Expected columns:
    # prices_actual,prices_forecast,actual_demand,forecast_demand,
    # soc,charge_kw,discharge_kw,import_kw,export_kw

    # ----- 2. Convert kW → MW and extract sequences -----
    charge_MW = (df["charge_kw"].astype(float)).tolist()
    discharge_MW = (df["discharge_kw"].astype(float)).tolist()
    import_MW = (df["import_kw"].astype(float)).tolist()
    export_MW = (df["export_kw"].astype(float)).tolist()

    # SoC from CSV is length T; our plotting code expects length T+1.
    soc = df["soc"].astype(float).tolist()
    if len(soc) == len(charge_MW):
        # Just append the last value once so we get T+1.
        soc.append(soc[-1])

    T = len(charge_MW)

    # ----- 3. Build a simple decision signal for visualization -----
    decisions = []
    eps = 1e-6
    for ch, dis in zip(charge_MW, discharge_MW):
        if ch > eps and dis < eps:
            decisions.append(1)   # charge
        elif dis > eps and ch < eps:
            decisions.append(-1)  # discharge
        else:
            decisions.append(0)   # idle

    # ----- 4. Compute an objective value: realized P&L on the given day prices -----
    day = solve_request.day
    prices = np.asarray(day.prices_buy, dtype=float)

    if len(prices) != T:
        raise ValueError(
            f"Length mismatch between prices ({len(prices)}) "
            f"and RL schedule ({T})."
        )

    dt = day.dt_hours

    # Profit per timestep: (revenue from discharge - cost of charge)*dt
    cashflows = (np.asarray(discharge_MW) * prices
                 - np.asarray(charge_MW) * prices) * dt
    objective_cost = -1 * float(cashflows.sum())

    # ----- 5. Wrap everything into a SolveResponse -----
    rl_response = SolveResponse(
        status="rl_schedule_loaded",
        objective_cost=objective_cost,
        soc=soc,
        charge_MW=charge_MW,
        discharge_MW=discharge_MW,
        import_MW=import_MW,
        export_MW=export_MW,
        decision=decisions,
        # add any other required fields your schema has, if needed
    )

    return rl_response  

def run_llm_agent(model_name: str, solve_request: SolveRequest) -> SolveResponse:
    """Call the llm_agent MCP tool and return a SolveResponse."""
    if model_name.lower() == "gemini":
        llm_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agentic_energy.language_models.basic_llm_amap"],
            env=os.environ,
        )
        with MCPServerAdapter(llm_params) as llm_tools:
            llm_tool = get_tool(llm_tools, "llm_solve")
            call_fn = (
                getattr(llm_tool, "call", None)
                or getattr(llm_tool, "run", None)
                or getattr(llm_tool, "__call__", None)
            )
            if call_fn is None:
                raise RuntimeError(f"Tool llm_agent_{model_name} has no callable interface")

            raw = call_fn(solverequest=solve_request.model_dump(exclude_none=True))
            print(raw)
            data = json.loads(raw)
            solverespon = SolveResponse.model_validate(data)

            out = cost_from_soc(
                soc = solverespon.soc,
                prices_buy=solve_request.day.prices_buy,
                demand_MW=solve_request.day.demand_MW,
                battery=solve_request.battery,
                prices_sell=solve_request.day.prices_sell,
                allow_export=True,
                dt_hours=1
            )

            solverespon.objective_cost = out['objective_cost']
            return solverespon

    else:
         # ----- 1. Load the static RL schedule CSV -----
        filename = "Ollama_output_(trainonactuals).csv"
        csv_path = os.path.join(OUTPUT_FILES_DIR, filename)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"RL schedule CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)

        # Expected columns:
        # prices_actual,prices_forecast,actual_demand,forecast_demand,
        # soc,charge_kw,discharge_kw,import_kw,export_kw

        # ----- 2. Convert kW → MW and extract sequences -----
        charge_MW = (df["charge_kw"].astype(float) / 1000.0).tolist()
        discharge_MW = (df["discharge_kw"].astype(float) / 1000.0).tolist()
        import_MW = (df["import_kw"].astype(float) / 1000.0).tolist()
        export_MW = (df["export_kw"].astype(float) / 1000.0).tolist()

        # SoC from CSV is length T; our plotting code expects length T+1.
        soc = df["soc"].astype(float).tolist()
        if len(soc) == len(charge_MW):
            # Just append the last value once so we get T+1.
            soc.append(soc[-1])

        T = len(charge_MW)

        # ----- 3. Build a simple decision signal for visualization -----
        decisions = []
        eps = 1e-6
        for ch, dis in zip(charge_MW, discharge_MW):
            if ch > eps and dis < eps:
                decisions.append(1)   # charge
            elif dis > eps and ch < eps:
                decisions.append(-1)  # discharge
            else:
                decisions.append(0)   # idle

        # ----- 4. Compute an objective value: realized P&L on the given day prices -----
        day = solve_request.day
        prices = np.asarray(day.prices_buy, dtype=float)

        if len(prices) != T:
            raise ValueError(
                f"Length mismatch between prices ({len(prices)}) "
                f"and RL schedule ({T})."
            )

        dt = day.dt_hours

        # Profit per timestep: (revenue from discharge - cost of charge)*dt
        cashflows = (np.asarray(discharge_MW) * prices
                    - np.asarray(charge_MW) * prices) * dt
        objective_cost = -1 * float(cashflows.sum())

        # ----- 5. Wrap everything into a SolveResponse -----
        ollama_response = SolveResponse(
            status="ollama_schedule_loaded",
            objective_cost=objective_cost,
            soc=soc,
            charge_MW=charge_MW,
            discharge_MW=discharge_MW,
            import_MW=import_MW,
            export_MW=export_MW,
            decision=decisions,
            # add any other required fields your schema has, if needed
        )

        return ollama_response  


# ---------- Visualization MCP client ----------

def _viz_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.visualization.visualization_mcp_server"],
        env=os.environ,
    )


def run_price_forecast_plot(
    prices: List[float],
    dt_hours: float,
    out_path: str,
    title: str,
) -> PlotResponse:
    """Use visualization MCP tool plot_price_forecast."""
    viz_params = _viz_params()

    req = PriceForecastPlotRequest(
        prices=prices,
        dt_hours=dt_hours,
        title=title,
        out_path=out_path,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with MCPServerAdapter(viz_params) as viz_tools:
        viz_tool = get_tool(viz_tools, "plot_price_forecast")
        call_fn = (
            getattr(viz_tool, "call", None)
            or getattr(viz_tool, "run", None)
            or getattr(viz_tool, "__call__", None)
        )
        if call_fn is None:
            raise RuntimeError("Tool plot_price_forecast has no callable interface")

        raw = call_fn(plotrequest=req.model_dump(exclude_none=True))
        data = json.loads(raw)
        return PlotResponse.model_validate(data)


def run_schedule_animation(
    solve_request: SolveRequest,
    solve_response: SolveResponse,
    out_path: str,
) -> PlotResponse:
    """Use visualization MCP tool plot_price_soc (animated schedule)."""
    viz_params = _viz_params()

    plot_request = PlotRequest(
        solve_request=solve_request,
        solve_response=solve_response,
        out_path=out_path,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with MCPServerAdapter(viz_params) as viz_tools:
        viz_tool = get_tool(viz_tools, "plot_price_soc")
        call_fn = (
            getattr(viz_tool, "call", None)
            or getattr(viz_tool, "run", None)
            or getattr(viz_tool, "__call__", None)
        )
        if call_fn is None:
            raise RuntimeError("Tool plot_price_soc has no callable interface")

        raw = call_fn(plotrequest=plot_request.model_dump(exclude_none=True))
        print("RAW TOOL OUTPUT:", repr(raw))
        data = json.loads(raw)
        return PlotResponse.model_validate(data)


def run_explanation_plot(
    solve_request: SolveRequest,
    solve_response: SolveResponse,
    out_path: str,
) -> PlotResponse:
    """Use visualization MCP tool plot_arbitrage_explanation."""
    viz_params = _viz_params()

    plot_request = PlotRequest(
        solve_request=solve_request,
        solve_response=solve_response,
        out_path=out_path,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with MCPServerAdapter(viz_params) as viz_tools:
        viz_tool = get_tool(viz_tools, "plot_arbitrage_explanation")
        call_fn = (
            getattr(viz_tool, "call", None)
            or getattr(viz_tool, "run", None)
            or getattr(viz_tool, "__call__", None)
        )
        if call_fn is None:
            raise RuntimeError("Tool plot_arbitrage_explanation has no callable interface")

        raw = call_fn(plotrequest=plot_request.model_dump(exclude_none=True))
        data = json.loads(raw)
        return PlotResponse.model_validate(data)


# ---------- Reasoning MCP client ----------

def run_reasoning_tool(
    solve_request: SolveRequest,
    solve_response: SolveResponse,
    timestamp_index: int,
) -> str:
    """Use reasoning_explain MCP tool to get textual explanation."""
    reasoning_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agentic_energy.reasoning.reasoning_server"],
        env=os.environ,
    )

    req = ReasoningRequest(
        solve_request=solve_request,
        solve_response=solve_response,
        timestamp_index=timestamp_index,
    )

    with MCPServerAdapter(reasoning_params) as tools:
        reasoning_tool = get_tool(tools, "reasoning_explain")
        call_fn = (
            getattr(reasoning_tool, "call", None)
            or getattr(reasoning_tool, "run", None)
            or getattr(reasoning_tool, "__call__", None)
        )

        raw = call_fn(reasoningrequest=req.model_dump(exclude_none=True))

        if isinstance(raw, str):
            data = json.loads(raw)
        elif isinstance(raw, dict):
            data = raw
        elif hasattr(raw, "model_dump"):
            data = raw.model_dump()
        else:
            raise TypeError(f"Unexpected reasoning tool return type: {type(raw)}")

        resp = ReasoningResponse.model_validate(data)
        return resp.explanation
