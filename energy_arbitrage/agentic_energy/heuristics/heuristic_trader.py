from __future__ import annotations
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel, Field
import numpy as np
import math

from agentic_energy.schemas import SolveRequest, SolveResponse, BatteryParams, DayInputs, SolveFromRecordsRequest, EnergyDataRecord
class HeuristicTrader:
    """
    Two heuristic modes:
      - mode='time': fixed time-of-day windows
      - mode='quantile': buy-price quantiles (charge on low_q, discharge on high_q)

    Override defaults via constructor or req.solver_opts.
    """

    def __init__(
        self,
        mode: str = "time",                                 # "time" or "quantile"
        charge_windows: Optional[List[Tuple[int, int]]] = None,
        discharge_windows: Optional[List[Tuple[int, int]]] = None,
        low_q: float = 0.30,
        high_q: float = 0.70,
    ):
        self.mode = mode
        self.charge_windows = charge_windows or [(2, 6), (10, 16), (20, 22)]
        self.discharge_windows = discharge_windows or [(0, 2), (6, 10), (16, 20), (22, 24)]
        self.low_q = low_q
        self.high_q = high_q

    # -------- Public API --------

    def solve(self, req: SolveRequest) -> SolveResponse:
        try:
            opts = req.solver_opts or {}
            mode = str(opts.get("mode", self.mode)).lower()

            if mode == "time":
                cw = opts.get("charge_windows", self.charge_windows)
                dw = opts.get("discharge_windows", self.discharge_windows)
                return self._run_day_time(req.battery, req.day, cw, dw)

            elif mode == "quantile":
                lq = float(opts.get("low_q", self.low_q))
                hq = float(opts.get("high_q", self.high_q))
                return self._run_day_quantile(req.battery, req.day, lq, hq)

            else:
                return SolveResponse(
                    status="error",
                    message=f"Unknown heuristic mode: {mode}",
                    objective_cost=math.nan,
                )
        except Exception as e:
            return SolveResponse(
                status="error",
                message=f"Heuristic solve failed: {e}",
                objective_cost=math.nan,
            )

    def solve_from_records(self, req: SolveFromRecordsRequest) -> SolveResponse:
        try:
            prices, demand = records_to_arrays(req.records)
            day = DayInputs(
                prices_buy=prices,
                demand_kw=demand,
                prices_sell=prices,
                allow_export=req.allow_export,
                dt_hours=req.dt_hours,
            )
            sr = SolveRequest(battery=req.battery, day=day, solver=req.solver, solver_opts=req.solver_opts)
            return self.solve(sr)
        except Exception as e:
            return SolveResponse(
                status="error",
                message=f"Heuristic solve_from_records failed: {e}",
                objective_cost=math.nan,
            )

    # -------- Core: Time-window mode --------

    def _run_day_time(
        self,
        bat: BatteryParams,
        day: DayInputs,
        charge_windows: List[Tuple[int, int]],
        discharge_windows: List[Tuple[int, int]],
    ) -> SolveResponse:

        n = len(day.prices_buy)
        if len(day.demand_kw) != n:
            raise ValueError("prices_buy and demand_kw must have the same length")
        if day.prices_sell is not None and len(day.prices_sell) != n:
            raise ValueError("prices_sell length must match prices_buy when provided")
        if day.dt_hours <= 0:
            raise ValueError("dt_hours must be > 0")

        # Inputs
        p_buy = np.asarray(day.prices_buy, dtype=np.float64)
        p_sell = (
            np.asarray(day.prices_sell, dtype=np.float64)
            if (day.allow_export and day.prices_sell is not None)
            else p_buy if day.allow_export
            else np.zeros_like(p_buy)
        )
        load = np.asarray(day.demand_kw, dtype=np.float64)

        # Battery params
        dt = float(day.dt_hours)
        C = float(bat.capacity_kwh)              # kWh
        soc_min = float(bat.soc_min)
        soc_max = float(bat.soc_max)
        soc = float(np.clip(bat.soc_init, soc_min, soc_max))  # FRACTION in [0,1]

        cmax = float(bat.cmax_kw)                # kW
        dmax = float(bat.dmax_kw)                # kW
        eta_c = float(bat.eta_c)
        eta_d = float(bat.eta_d)

        # Outputs
        charge_kw = np.zeros(n, dtype=np.float64)
        discharge_kw = np.zeros(n, dtype=np.float64)
        import_kw = np.zeros(n, dtype=np.float64)
        export_kw = np.zeros(n, dtype=np.float64)  # will remain zeros if !allow_export
        soc_series = np.zeros(n + 1, dtype=np.float64)
        soc_series[0] = soc
        decision = np.zeros(n, dtype=np.float64)

        objective = 0.0

        for t in range(n):
            hour = int((t * dt) % 24)
            action = self._get_action(hour, charge_windows, discharge_windows)

            c_t = 0.0  # charge power (kW)
            d_t = 0.0  # discharge power (kW)

            if action == "charge":
                # Headroom energy in kWh (fractional SoC space * C)
                headroom_kwh = (soc_max - soc) * C
                # SoC gain per kW over dt is (eta_c * dt) / C
                # But we must limit input energy from grid (kWh): E_in <= headroom_kwh / eta_c
                max_in_kwh = max(0.0, headroom_kwh / max(eta_c, 1e-12))
                c_t = min(cmax, max_in_kwh / dt)
                decision[t] = +1.0

            elif action == "discharge":
                # Available SoC energy (kWh) above soc_min
                avail_kwh = (soc - soc_min) * C
                # Energy to grid per kW over dt is (dt); grid receives eta_d * E_soc
                # So max energy to grid limited by avail_kwh * eta_d
                max_out_kwh_to_grid = max(0.0, avail_kwh * eta_d)
                d_t = min(dmax, max_out_kwh_to_grid / dt)
                decision[t] = -1.0

            else:
                decision[t] = 0.0

            # Grid balance constraints (heuristic implementation)
            # net = load + c_t - d_t
            net = float(load[t] + c_t - d_t)

            if day.allow_export:
                # imp - exp = net, with imp,exp >= 0  -> canonical decomposition
                if net >= 0:
                    import_kw[t] = net
                    export_kw[t] = 0.0
                else:
                    import_kw[t] = 0.0
                    export_kw[t] = -net
            else:
                # imp >= net, exp = 0; if net<0, set imp=0 (can't export)
                import_kw[t] = max(0.0, net)
                export_kw[t] = 0.0

            # Advance SoC (fractional) with the canonical equation:
            # soc_{t+1} = soc_t + (eta_c*c_t*dt - d_t*dt/eta_d)/C
            soc_next = soc + (eta_c * c_t * dt - (d_t * dt) / max(eta_d, 1e-12)) / C
            soc = float(np.clip(soc_next, soc_min, soc_max))
            soc_series[t + 1] = soc

            # Cost accumulation (import cost - export revenue) * dt
            objective += (p_buy[t] * import_kw[t] - p_sell[t] * export_kw[t]) * dt

            # Record powers
            charge_kw[t] = c_t
            discharge_kw[t] = d_t

        return SolveResponse(
            status="ok-time",
            message=None,
            objective_cost=float(objective),
            charge_kw=charge_kw.tolist(),
            discharge_kw=discharge_kw.tolist(),
            import_kw=import_kw.tolist(),
            export_kw=(export_kw.tolist() if day.allow_export else None),
            soc=soc_series[1:].tolist(),   # SoC per step, fractional
            decision=decision.tolist(),
            confidence=None,
        )

    # -------- Core: Quantile mode  --------

    def _run_day_quantile(
        self,
        bat: BatteryParams,
        day: DayInputs,
        low_q: float,
        high_q: float,
    ) -> SolveResponse:

        if not (0.0 <= low_q < high_q <= 1.0):
            raise ValueError("Require 0 <= low_q < high_q <= 1")

        n = len(day.prices_buy)
        if len(day.demand_kw) != n:
            raise ValueError("prices_buy and demand_kw must have the same length")
        if day.prices_sell is not None and len(day.prices_sell) != n:
            raise ValueError("prices_sell length must match prices_buy when provided")
        if day.dt_hours <= 0:
            raise ValueError("dt_hours must be > 0")

        # Inputs
        dt = float(day.dt_hours)
        prices = np.asarray(day.prices_buy, dtype=np.float64)
        load = np.asarray(day.demand_kw, dtype=np.float64)
        allow_export = bool(day.allow_export)
        p_sell = (
            np.asarray(day.prices_sell, dtype=np.float64)
            if (allow_export and day.prices_sell is not None)
            else prices if allow_export
            else np.zeros_like(prices)
        )

        # Quantile thresholds
        thr_low = float(np.quantile(prices, low_q))
        thr_high = float(np.quantile(prices, high_q))

        # Battery params
        C = float(bat.capacity_kwh)       # kWh
        soc_min = float(bat.soc_min)
        soc_max = float(bat.soc_max)
        soc = float(np.clip(bat.soc_init, soc_min, soc_max))  # FRACTION

        eta_c = float(bat.eta_c)
        eta_d = float(bat.eta_d)
        cmax = float(bat.cmax_kw)
        dmax = float(bat.dmax_kw)

        # Outputs
        charge = np.zeros(n, dtype=np.float64)
        disch = np.zeros(n, dtype=np.float64)
        imp = np.zeros(n, dtype=np.float64)
        exp = np.zeros(n, dtype=np.float64)
        soc_series = np.zeros(n + 1, dtype=np.float64)
        soc_series[0] = soc
        decision = np.zeros(n, dtype=np.float64)

        objective = 0.0

        for t in range(n):
            price = prices[t]
            c_t, d_t = 0.0, 0.0

            # Mode decision (mutually exclusive):
            if price <= thr_low and soc < soc_max - 1e-12:
                # Charge as much as feasible under SoC headroom and power
                headroom_kwh = (soc_max - soc) * C
                max_in_kwh = max(0.0, headroom_kwh / max(eta_c, 1e-12))
                c_t = min(cmax, max_in_kwh / dt)
                decision[t] = +1.0

            elif price >= thr_high and soc > soc_min + 1e-12:
                # Discharge as much as feasible under SoC availability and power
                avail_kwh = (soc - soc_min) * C
                max_out_kwh_to_grid = max(0.0, avail_kwh * eta_d)
                d_t = min(dmax, max_out_kwh_to_grid / dt)
                decision[t] = -1.0

            else:
                decision[t] = 0.0

            # Grid balance
            net = float(load[t] + c_t - d_t)
            if allow_export:
                if net >= 0:
                    imp[t] = net
                    exp[t] = 0.0
                else:
                    imp[t] = 0.0
                    exp[t] = -net
            else:
                imp[t] = max(0.0, net)
                exp[t] = 0.0

            # SoC update in FRACTION (exact MILP-style equation)
            soc_next = soc + (eta_c * c_t * dt - (d_t * dt) / max(eta_d, 1e-12)) / C
            soc = float(np.clip(soc_next, soc_min, soc_max))
            soc_series[t + 1] = soc

            # Cost
            objective += (prices[t] * imp[t] - p_sell[t] * exp[t]) * dt

            # Store powers
            charge[t] = c_t
            disch[t] = d_t

        total_cost = float(objective)

        return SolveResponse(
            status="ok-quantile",
            message=None,
            objective_cost=total_cost,
            charge_kw=charge.tolist(),
            discharge_kw=disch.tolist(),
            import_kw=imp.tolist(),
            export_kw=(exp.tolist() if allow_export else None),
            soc=soc_series[1:].tolist(),  # FRACTION per step
            decision=decision.tolist(),
            confidence=None,
        )


    # -------- Helpers --------

    @staticmethod
    def _get_action(hour: int, charge_windows: List[Tuple[int, int]], discharge_windows: List[Tuple[int, int]]) -> str:
        for s, e in charge_windows:
            if s <= hour < e:
                return "charge"
        for s, e in discharge_windows:
            if s <= hour < e:
                return "discharge"
        return "idle"

def records_to_arrays(records: List[EnergyDataRecord]) -> Tuple[List[float], List[float]]:
    """Extract price and demand arrays from records (None -> 0.0)."""
    prices = [float(r.prices) if r.prices is not None else 0.0 for r in records]
    demand = [float(r.consumption) if r.consumption is not None else 0.0 for r in records]
    return prices, demand


def run_heuristic_day(
    trader: HeuristicTrader,
    battery: BatteryParams,
    records: List[EnergyDataRecord],
    dt_hours: float = 1.0,
    allow_export: bool = True,
    solver_opts: dict | None = None,
) -> SolveResponse:
    """Wrap trader.solve_from_records with convenient defaults."""
    req = SolveFromRecordsRequest(
        battery=battery,
        records=records,
        dt_hours=dt_hours,
        allow_export=allow_export,
        solver=None,
        solver_opts=solver_opts or {},
    )
    return trader.solve_from_records(req)