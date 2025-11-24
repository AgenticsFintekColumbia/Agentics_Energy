# agentic_energy/heuristics/heuristic_mcp_server.py

from __future__ import annotations
from typing import Optional

from mcp.server.fastmcp import FastMCP

from agentic_energy.schemas import (
    SolveRequest,
    SolveResponse,
    SolveFromRecordsRequest,
)
from agentic_energy.heuristics.heuristic_trader import HeuristicTrader

# Create MCP server
mcp = FastMCP("HEURISTIC")

# Instantiate two heuristic traders: time-based and quantile-based
_time_trader = HeuristicTrader(mode="time")
_quantile_trader = HeuristicTrader(mode="quantile")


@mcp.tool()
def heuristic_time_solve(args: SolveRequest) -> SolveResponse:
    """
    Run the time-window heuristic controller for a single day.

    Notes:
      - Uses HeuristicTrader(mode="time").
      - You can override charge/discharge windows via args.solver_opts, e.g.:
          solver_opts = {
              "mode": "time",
              "charge_windows": [(1, 5), (11, 15)],
              "discharge_windows": [(5, 11), (15, 24)]
          }
    """
    # Ensure solver_opts.mode is set to "time" if user passes solver string
    if args.solver and (args.solver_opts is None or "mode" not in args.solver_opts):
        solver_opts = dict(args.solver_opts or {})
        solver_opts["mode"] = "time"
        args = SolveRequest(
            battery=args.battery,
            day=args.day,
            solver=args.solver,
            solver_opts=solver_opts,
        )
    return _time_trader.solve(args)


@mcp.tool()
def heuristic_time_solve_from_records(args: SolveFromRecordsRequest) -> SolveResponse:
    """
    Run the time-window heuristic given a list of EnergyDataRecord rows.

    Notes:
      - Uses HeuristicTrader(mode="time").
      - Records are converted to prices/demand internally by the trader.
    """
    if args.solver and (args.solver_opts is None or "mode" not in args.solver_opts):
        solver_opts = dict(args.solver_opts or {})
        solver_opts["mode"] = "time"
        args = SolveFromRecordsRequest(
            battery=args.battery,
            records=args.records,
            dt_hours=args.dt_hours,
            allow_export=args.allow_export,
            prices_sell=args.prices_sell,
            solver=args.solver,
            solver_opts=solver_opts,
        )
    return _time_trader.solve_from_records(args)


@mcp.tool()
def heuristic_quantile_solve(args: SolveRequest) -> SolveResponse:
    """
    Run the quantile-based heuristic controller for a single day.

    Notes:
      - Uses HeuristicTrader(mode="quantile").
      - You can override quantiles via args.solver_opts, e.g.:
          solver_opts = {
              "mode": "quantile",
              "low_q": 0.25,
              "high_q": 0.75,
          }
    """
    if args.solver and (args.solver_opts is None or "mode" not in args.solver_opts):
        solver_opts = dict(args.solver_opts or {})
        solver_opts["mode"] = "quantile"
        args = SolveRequest(
            battery=args.battery,
            day=args.day,
            solver=args.solver,
            solver_opts=solver_opts,
        )
    return _quantile_trader.solve(args)


@mcp.tool()
def heuristic_quantile_solve_from_records(args: SolveFromRecordsRequest) -> SolveResponse:
    """
    Run the quantile-based heuristic given a list of EnergyDataRecord rows.

    Notes:
      - Uses HeuristicTrader(mode="quantile").
      - Records are converted to prices/demand internally by the trader.
    """
    if args.solver and (args.solver_opts is None or "mode" not in args.solver_opts):
        solver_opts = dict(args.solver_opts or {})
        solver_opts["mode"] = "quantile"
        args = SolveFromRecordsRequest(
            battery=args.battery,
            records=args.records,
            dt_hours=args.dt_hours,
            allow_export=args.allow_export,
            prices_sell=args.prices_sell,
            solver=args.solver,
            solver_opts=solver_opts,
        )
    return _quantile_trader.solve_from_records(args)


if __name__ == "__main__":
    mcp.run(transport="stdio")
