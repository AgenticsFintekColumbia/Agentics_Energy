from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from agentic_energy.schemas import BatteryParams, DayInputs, SolveRequest, EnergyDataRecord, SolveResponse, SolveFromRecordsRequest
from agentics import AG
import cvxpy as cp
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MILP")

def records_to_arrays(records: List[EnergyDataRecord]) -> Tuple[list, list]:
    rows = [r for r in records if r.prices is not None and r.consumption is not None]
    rows.sort(key=lambda r: r.timestamps)
    prices = [float(r.prices) for r in rows]
    demand = [float(r.consumption) for r in rows]
    return prices, demand


def solve_daily_milp(
    batt: BatteryParams,
    day: DayInputs,
    solver: Optional[str] = None,
    solver_opts: Optional[Dict[str, Any]] = None,
) -> SolveResponse:

    T = len(day.prices_buy)
    if len(day.demand_MW) != T:
        return SolveResponse(status="error", message="prices_buy and demand_MW lengths differ")

    dt   = float(day.dt_hours)
    C    = float(batt.capacity_MWh)
    eta_c, eta_d = float(batt.eta_c), float(batt.eta_d)
    soc_lo, soc_hi = float(batt.soc_min), float(batt.soc_max)
    soc0  = float(batt.soc_init)
    soc_tgt = soc0 if batt.soc_target is None else float(batt.soc_target)

    p_buy = np.asarray(day.prices_buy, dtype=float)
    load  = np.asarray(day.demand_MW, dtype=float)
    if day.allow_export:
        p_sell = np.asarray(day.prices_sell if day.prices_sell is not None else day.prices_buy, dtype=float)
    else:
        p_sell = None

    # Variables
    c   = cp.Variable(T, nonneg=True, name="charge_MW")
    d   = cp.Variable(T, nonneg=True, name="discharge_MW")
    imp = cp.Variable(T, nonneg=True, name="import_MW")
    exp = cp.Variable(T, nonneg=True, name="export_MW") if day.allow_export else None

    y_c = cp.Variable(T, boolean=True, name="y_charge")
    y_d = cp.Variable(T, boolean=True, name="y_discharge")

    soc = cp.Variable(T+1, name="soc")

    cons = [
        soc >= soc_lo, soc <= soc_hi,
        soc[0] == soc0, 
        # soc[T] >= soc_tgt
    ]
    for t in range(T):
        cons += [
            c[t] <= batt.cmax_MW * y_c[t],
            d[t] <= batt.dmax_MW * y_d[t],
            y_c[t] + y_d[t] <= 1,
            soc[t+1] == soc[t] + (eta_c*c[t]*dt - (d[t]*dt)/eta_d)/C,
        ]
        net = load[t] + c[t] - d[t]
        if day.allow_export:
            cons += [imp[t] - exp[t] == net]
        else:
            cons += [imp[t] >= net]

    if day.allow_export:
        objective = cp.sum(p_buy * imp * dt) - cp.sum(p_sell * exp * dt)
    else:
        objective = cp.sum(p_buy * imp * dt)

    prob = cp.Problem(cp.Minimize(objective), cons)

    if solver is None:
        for cand in ["GUROBI", "CPLEX", "SCIPY", "CBC", "GLPK_MI", "ECOS_BB"]:
            if cand in cp.installed_solvers():
                solver = cand
                break

    if solver_opts is None:
        solver_opts = {}
    else:
        solver_opts = dict(solver_opts)  # ensure it's a dict

    try:
        if solver:
            prob.solve(solver=cp.GUROBI, **solver_opts)
        else:
            prob.solve(**solver_opts)  # may fail if default solver isn't MILP-capable
    except Exception as e:
        return SolveResponse(status="error", message=str(e))
    # can you write the decision list as well here in a simple for loop no need to do comprehension

    decision_list = []
    for t in range(T):
        if y_c.value[t] == 1:
            decision_list.append(y_c.value[t])
        elif y_d.value[t] == 1:
            decision_list.append(-1*y_d.value[t])
        else:
            decision_list.append(0)

    return SolveResponse(
        status=prob.status,
        objective_cost=float(prob.value) if prob.value is not None else None,
        charge_MW=c.value.tolist() if c.value is not None else None,
        discharge_MW=d.value.tolist() if d.value is not None else None,
        import_MW=imp.value.tolist() if imp.value is not None else None,
        export_MW=(exp.value.tolist() if (day.allow_export and exp is not None and exp.value is not None) else None),
        soc=soc.value.tolist() if soc.value is not None else None,
        decision=decision_list,
        confidence=[ max(y_c.value[t], y_d.value[t], 1 - y_c.value[t] - y_d.value[t]) for t in range(T)] if y_c.value is not None and y_d.value is not None else None,
    )



@mcp.tool()
def milp_solve(solverequest: SolveRequest) -> SolveResponse:
    """Run day-ahead battery MILP and return schedules + cost."""
    return solve_daily_milp(solverequest.battery, solverequest.day, solverequest.solver, solverequest.solver_opts)

@mcp.tool()
def milp_solve_from_records(solverecordrequest: SolveFromRecordsRequest) -> SolveResponse:
    """Run day-ahead MILP given a list of EnergyDataRecord rows."""
    prices, demand = records_to_arrays(solverecordrequest.records)
    day = DayInputs(
        prices_buy=prices,
        demand_MW=demand,
        prices_sell=prices,
        allow_export=solverecordrequest.allow_export,
        dt_hours=solverecordrequest.dt_hours
    )
    return solve_daily_milp(solverecordrequest.battery, day, solverecordrequest.solver, solverecordrequest.solver_opts)

if __name__ == "__main__":
    mcp.run(transport="stdio")