from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from agentic_energy.schemas import BatteryParams, DayInputs, EnergyDataRecord, SolveResponse


def _records_to_arrays(records: List[EnergyDataRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """Turn EnergyDataRecord rows into aligned numpy arrays (prices, demand)."""
    rows = [r for r in records if r.prices is not None and r.consumption is not None]
    rows.sort(key=lambda r: r.timestamps)
    prices = np.array([float(r.prices) for r in rows], dtype=np.float32)
    demand = np.array([float(r.consumption) for r in rows], dtype=np.float32)
    return prices, demand


def group_records_by_day(records: List[EnergyDataRecord], dt_hours: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Group rows by UTC day and return { 'YYYY-MM-DD': (prices[expected], demand[expected]) }.
    """
    from datetime import datetime
    from collections import defaultdict

    def parse(ts: str) -> datetime:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)

    expected = int(round(24 / dt_hours))
    buckets: Dict[str, List[EnergyDataRecord]] = defaultdict(list)
    for r in records:
        if r.prices is None or r.consumption is None:
            continue
        buckets[parse(r.timestamps).date().isoformat()].append(r)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for day, rows in buckets.items():
        rows.sort(key=lambda r: r.timestamps)
        if len(rows) >= expected:
            p, d = _records_to_arrays(rows[:expected])
            out[day] = (p, d)
    return out


# agentic_energy/milp/rl/env_rllib.py (replace class with this upgraded one)
class BatteryArbRLEnv(gym.Env):
    """
    RLlib-compatible env for battery arbitrage with receding-horizon observations.

    env_config:
      battery: BatteryParams (dict or instance)
      day OR days: DayInputs or list[DayInputs]
      obs_mode: "compact" | "forecast"          # determines whether obs uses actuals or forecasts
      obs_window: int = 24                      # length of receding window
    """
    metadata = {"render_modes": []}

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        # ---- parse config (battery, day(s)) ----
        batt = env_config.get("battery")
        if isinstance(batt, dict): batt = BatteryParams(**batt)
        self.batt: BatteryParams = batt

        day = env_config.get("day")
        if day is not None and isinstance(day, dict): day = DayInputs(**day)
        days = env_config.get("days")
        if days is not None and len(days) > 0 and isinstance(days[0], dict):
            days = [DayInputs(**d) for d in days]
        assert (day is not None) ^ (days is not None), "Provide exactly one of day or days"
        self.day: Optional[DayInputs] = day
        self.days: Optional[List[DayInputs]] = days

        # ---- obs settings ----
        self.obs_mode: str = str(env_config.get("obs_mode", "compact")).lower()
        if self.obs_mode not in ("compact", "forecast"):
            raise ValueError("obs_mode must be 'compact' or 'forecast'")
        self.obs_window: int = int(env_config.get("obs_window", 24))

        # ---- battery constants ----
        self.C = float(batt.capacity_MWh)
        self.eta_c = float(batt.eta_c)
        self.eta_d = float(batt.eta_d)
        self.cmax = float(batt.cmax_MW)
        self.dmax = float(batt.dmax_MW)
        self.soc_min = float(batt.soc_min)
        self.soc_max = float(batt.soc_max)
        self.soc0 = float(batt.soc_init)
        self.soc_target = self.soc0 if batt.soc_target is None else float(batt.soc_target)
        # inside __init__ after reading env_config
        # self.lambda_smooth: float = float(env_config.get("lambda_smooth", 0.05))  # e.g., 0.01
        self.lambda_smooth: float = 0.1
        self._prev_action: float = 0.0  # will be reset each episode

        self.price_scale = 1.0
        self.demand_scale = 1.0
        self.obs_clip = 10.0  # simple hard clip for stability

        self.r_eps = 1e-6        # to avoid divide-by-zero
        self.r_clip = 5.0        # clamp scaled reward into [-5, 5]
        self.r_scale = 1.0       # per-episode reward scale (set in _start_day)


        # ---- spaces ----
        # Dict observation with fixed-size windowed vectors for prices & demand, plus [t_norm, soc]
        self.observation_space = spaces.Dict({
            "soc": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "prices_buy": spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(self.obs_window,), dtype=np.float32), # window of prices
            "prices_sell": spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(self.obs_window,), dtype=np.float32), # window of prices
            "demand": spaces.Box(low=-np.inf, high=np.inf,
                                 shape=(self.obs_window,), dtype=np.float32), # window of demand
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # ---- episode vars ----
        self.dt = day.dt_hours if day is not None else 1.0
        self.allow_export = bool(day.allow_export) if day is not None else False

        # Actuals used for physics and billing (reward)
        # arrays allocated on reset/_start_day
        self.prices_buy_actual  = np.zeros(1, dtype=np.float32)
        self.prices_sell_actual = np.zeros(1, dtype=np.float32)
        self.demand_actual      = np.zeros(1, dtype=np.float32)

        self.prices_buy_forecast  = self.prices_buy_actual.copy()
        self.prices_sell_forecast = self.prices_sell_actual.copy()
        self.demand_MW_forecast   = self.demand_actual.copy()

        self.prices_buy_obs  = np.zeros(1, dtype=np.float32)
        self.prices_sell_obs = np.zeros(1, dtype=np.float32)
        self.demand_obs      = np.zeros(1, dtype=np.float32)

        self.T = 0
        self.t = 0
        self.soc = self.soc0
        self.term_penalty = float(env_config.get("term_penalty", 10))


        # logs
        self._import = None
        self._export = None
        self._charge = None
        self._disch = None
        self._soc_series = None

        # forecast error metrics (computed at terminal if forecasts provided)
        self._mae_price = None
        self._mape_price = None
        self._mae_demand = None
        self._mape_demand = None

    # ---------- helpers ----------
    def _hybrid_series(self, actual: np.ndarray, forecast: np.ndarray, t: int) -> np.ndarray:
        """Hybrid: actual[:t] + forecast[t:]. Length == len(actual)."""
        if t <= 0:  return forecast.copy()
        if t >= len(actual): return actual.copy()
        return np.concatenate([actual[:t], forecast[t:]], 0).astype(np.float32, copy=False)

    def _horizon_slice_long(self, long_series: np.ndarray, t: int, H: int) -> np.ndarray:
        """Slice long_series[t:t+H] (expects long_series length >= T+H). Pad if needed."""
        end = t + H
        if end <= len(long_series):
            return long_series[t:end].astype(np.float32, copy=False)
        # pad with last value (should rarely happen if long_series is long enough)
        out = np.empty(H, dtype=np.float32)
        n = len(long_series) - t
        out[:max(0, n)] = long_series[t:] if n > 0 else long_series[-1]
        out[max(0, n):] = long_series[-1]
        return out


    # def _obs(self) -> Dict[str, np.ndarray]:
    #     t_norm = (self.t / max(1, self.T - 1)) if self.T > 1 else 0.0
    #     feat = np.array([t_norm, np.float32(self.soc)], dtype=np.float32)

    #     self.prices_buy_obs = self._hybrid_series(self.prices_buy_actual, self.prices_buy_forecast, self.t)
    #     self.prices_sell_obs = self._hybrid_series(self.prices_sell_actual, self.prices_sell_forecast, self.t)
    #     self.demand_obs = self._hybrid_series(self.demand_actual, self.demand_MW_forecast, self.t)
    #     # --- Simple normalization ---
    #     pb_scaled = np.clip(self.prices_buy_obs / self.price_scale,  0.0, self.obs_clip)
    #     ps_scaled = np.clip(self.prices_sell_obs / self.price_scale,  0.0, self.obs_clip)  # same scale as buy
    #     d_scaled  = np.clip(self.demand_obs / self.demand_scale, 0.0, self.obs_clip)
    #     return {
    #         "feat":   feat,
    #         "prices_buy": pb_scaled,
    #         "prices_sell": ps_scaled,
    #         "demand": d_scaled,
        # }

    def compute_stats(self, values:np.ndarray):
        mu = float(np.mean(values)) if len(values) else 0.0
        sd = float(np.std(values)) if len(values) else 1.0
        sd = sd if sd > 1e-6 else 1.0
        return mu, sd

    def _z(self, x: np.ndarray, mu: float, sd: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - mu) / (sd + 1e-6)
    
    def _obs(self) -> Dict[str, np.ndarray]:
        t_norm = (self.t / max(1, self.T - 1)) if self.T > 1 else 0.0
        feat = np.array([t_norm, np.float32(self.soc)], dtype=np.float32)
        soc = np.array([np.float32(self.soc)], dtype=np.float32)


        # Always observe next 24 hours from current time (forecast), crossing midnight
        pb = self._horizon_slice_long(self._pb_forecast_long, self.t, self.obs_window)
        ps = self._horizon_slice_long(self._ps_forecast_long, self.t, self.obs_window)
        dd = self._horizon_slice_long(self._d_forecast_long,  self.t, self.obs_window)

        pb_mu, pb_sd = self.compute_stats(self.prices_buy_actual)
        ps_mu, ps_sd = self.compute_stats(self.prices_sell_actual)
        dd_mu, dd_sd = self.compute_stats(self.demand_actual)

        # only the future values are relevant for normalization in the state space
        pb_scaled = np.clip(self._z(pb, pb_mu, pb_sd), -self.obs_clip, self.obs_clip)
        ps_scaled = np.clip(self._z(ps, ps_mu, ps_sd), -self.obs_clip, self.obs_clip)
        d_scaled  = np.clip(self._z(dd,  dd_mu,  dd_sd), -self.obs_clip, self.obs_clip)


        return {
            "soc": soc,
            "prices_buy": pb_scaled,
            "prices_sell": ps_scaled,
            "demand": d_scaled,
        }


    # def _start_day(self, day: DayInputs):
    #     self.dt = float(day.dt_hours)
    #     self.allow_export = bool(day.allow_export)

    #     # Actuals used for physics and billing (reward)
    #     self.prices_buy_actual = np.asarray(day.prices_buy, dtype=np.float32)      # BUY PRICES (full day)
    #     self.demand_actual = np.asarray(day.demand_MW,  dtype=np.float32)

    #     self.prices_buy_forecast = np.asarray(day.prices_buy_forecast, dtype=np.float32) if (day is not None and day.prices_buy_forecast is not None) else None
    #     self.demand_MW_forecast  = np.asarray(day.demand_MW_forecast,  dtype=np.float32) if (day is not None and day.demand_MW_forecast is not None) else None

    #     # Sell prices (used only if export is allowed)
    #     self.prices_sell_actual = np.asarray(
    #         day.prices_sell if (self.allow_export and day.prices_sell is not None) else day.prices_buy,
    #         dtype=np.float32
    #     )

    #     self.prices_sell_forecast = np.asarray(
    #         day.prices_sell_forecast if (self.allow_export and day.prices_sell_forecast is not None) else day.prices_buy_forecast,
    #         dtype=np.float32
    #     )

    #     # What the policy observes (forecasts or actuals)
    #     if self.obs_mode == "forecast" and day.prices_buy_forecast and day.demand_MW_forecast:
    #         self.prices_buy_obs = np.asarray(day.prices_buy_forecast, dtype=np.float32)
    #         self.prices_sell_obs = np.asarray(day.prices_sell_forecast, dtype=np.float32)
    #         self.demand_obs = np.asarray(day.demand_MW_forecast,  dtype=np.float32)
    #     else:
    #         self.prices_buy_obs = self.prices_buy_actual.copy()
    #         self.prices_sell_obs = self.prices_sell_actual.copy()
    #         self.demand_obs = self.demand_actual.copy()

    #     # Simple per-episode scales: use the means of forecasts (fallback actuals)
    #     pb_src = self.prices_buy_forecast if self.prices_buy_forecast is not None else self.prices_buy_actual
    #     dd_src = self.demand_MW_forecast     if self.demand_MW_forecast     is not None else self.demand_actual
    #     pb_mu = float(np.mean(pb_src)) if len(pb_src) else 1.0
    #     dd_mu = float(np.mean(dd_src)) if len(dd_src) else 1.0
    #     # Avoid divide-by-zero; keep small floor
    #     self.price_scale  = pb_mu if pb_mu > 1e-6 else 1.0
    #     self.demand_scale = dd_mu if dd_mu > 1e-6 else 1.0

    #     #basecondition without battery
    #     step_cost_forecast = pb_src * dd_src * self.dt
    #     mu_reward = float(np.mean(step_cost_forecast)) if len(step_cost_forecast) else 1.0
    #     self.r_scale = mu_reward if mu_reward > 1e-6 else 1

    #     self.T = len(self.prices_buy_actual)
    #     self.t = 0
    #     self.soc = float(self.soc0)
    #     self._prev_action = 0.0

    #     self._import = np.zeros(self.T, dtype=np.float32)
    #     self._export = np.zeros(self.T, dtype=np.float32)
    #     self._charge = np.zeros(self.T, dtype=np.float32)
    #     self._disch = np.zeros(self.T, dtype=np.float32)
    #     self._soc_series = np.zeros(self.T + 1, dtype=np.float32); self._soc_series[0] = self.soc

    #     # reset metrics
    #     self._mae_price = self._mape_price = self._mae_demand = self._mape_demand = None

    def _start_day(self, day: DayInputs, day_next: Optional[DayInputs] = None):
        """
        Initialize one episode (one day for dynamics/reward), and build a 24-hour lookahead
        forecast horizon that can cross midnight by concatenating today's and tomorrow's forecasts.

        - Physics/reward use *today actuals* at self.t.
        - Observations will use *forecast horizon* from current self.t to self.t + obs_window.
        """

        # -------------------------
        # Episode configuration
        # -------------------------
        self.dt = float(day.dt_hours)
        self.allow_export = bool(day.allow_export)

        # -------------------------
        # ACTUALS (used for reward and battery physics)
        # -------------------------
        self.prices_buy_actual = np.asarray(day.prices_buy, dtype=np.float32)
        self.demand_actual     = np.asarray(day.demand_MW, dtype=np.float32)

        # Sell side actuals (only meaningful if allow_export=True)
        if self.allow_export and (day.prices_sell is not None):
            self.prices_sell_actual = np.asarray(day.prices_sell, dtype=np.float32)
        else:
            # if export not allowed, we can still store something
            self.prices_sell_actual = self.prices_buy_actual.copy()

        # -------------------------
        # FORECASTS (used for observation, not reward)
        # -------------------------
        # Today's forecasts (fallback to actuals if not provided)
        pb0 = np.asarray(day.prices_buy_forecast, dtype=np.float32) if day.prices_buy_forecast is not None else self.prices_buy_actual
        d0  = np.asarray(day.demand_MW_forecast,  dtype=np.float32) if day.demand_MW_forecast  is not None else self.demand_actual

        if self.allow_export:
            if day.prices_sell_forecast is not None:
                ps0 = np.asarray(day.prices_sell_forecast, dtype=np.float32)
            elif day.prices_sell is not None:
                # if no forecast, fallback to sell actuals
                ps0 = np.asarray(day.prices_sell, dtype=np.float32)
            else:
                ps0 = pb0
        else:
            ps0 = pb0

        # Tomorrow forecasts if provided (needed to cross midnight for next-24 window)
        if day_next is not None:
            pb1 = np.asarray(day_next.prices_buy_forecast, dtype=np.float32) if day_next.prices_buy_forecast is not None else np.asarray(day_next.prices_buy, dtype=np.float32)
            d1  = np.asarray(day_next.demand_MW_forecast,  dtype=np.float32) if day_next.demand_MW_forecast  is not None else np.asarray(day_next.demand_MW, dtype=np.float32)

            if self.allow_export:
                if day_next.prices_sell_forecast is not None:
                    ps1 = np.asarray(day_next.prices_sell_forecast, dtype=np.float32)
                elif day_next.prices_sell is not None:
                    ps1 = np.asarray(day_next.prices_sell, dtype=np.float32)
                else:
                    ps1 = pb1
            else:
                ps1 = pb1
        else:
            # If tomorrow isn't available, repeat today so horizon slicing still works
            pb1, ps1, d1 = pb0, ps0, d0

        # Store the "long" forecast series (length ~48 for hourly data)
        self._pb_forecast_long = np.concatenate([pb0, pb1], axis=0).astype(np.float32, copy=False)
        self._ps_forecast_long = np.concatenate([ps0, ps1], axis=0).astype(np.float32, copy=False)
        self._d_forecast_long  = np.concatenate([d0,  d1], axis=0).astype(np.float32, copy=False)

        # Also keep pointers to “today forecast” if you want them for scaling/debugging
        self.prices_buy_forecast = pb0
        self.prices_sell_forecast = ps0
        self.demand_MW_forecast = d0

        # -------------------------
        # Normalization scales (use today's forecast if available)
        # -------------------------
        pb_mu = float(np.mean(pb0)) if len(pb0) else 1.0
        d_mu  = float(np.mean(d0))  if len(d0)  else 1.0

        self.price_scale  = pb_mu if pb_mu > 1e-6 else 1.0
        self.demand_scale = d_mu  if d_mu  > 1e-6 else 1.0

        # Reward scaling baseline (typical cost per step)
        step_cost_forecast = pb0 * d0 * self.dt
        mu_reward = float(np.mean(step_cost_forecast)) if len(step_cost_forecast) else 1.0
        self.r_scale = mu_reward if mu_reward > 1e-6 else 1.0

        # -------------------------
        # Episode counters / state
        # -------------------------
        self.T = len(self.prices_buy_actual)
        self.t = 0
        self.soc = float(self.soc0)
        self._prev_action = 0.0

        # -------------------------
        # Logs
        # -------------------------
        self._import = np.zeros(self.T, dtype=np.float32)
        self._export = np.zeros(self.T, dtype=np.float32)
        self._charge = np.zeros(self.T, dtype=np.float32)
        self._disch  = np.zeros(self.T, dtype=np.float32)
        self._soc_series = np.zeros(self.T + 1, dtype=np.float32)
        self._soc_series[0] = self.soc

        # Reset forecast error metrics
        self._mae_price = self._mape_price = None
        self._mae_demand = self._mape_demand = None


    # ---------- gym API ----------
    # reset modified check below
    # def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    #     super().reset(seed=seed)
    #     print(len(self.days) if self.days is not None else "single day")
    #     day = self.days[np.random.randint(0, len(self.days))] if self.days is not None else self.day
    #     self._start_day(day)
    #     return self._obs(), {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.days is not None:
            i = np.random.randint(0, len(self.days))
            day = self.days[i]
            day_next = self.days[(i + 1) % len(self.days)]   # wrap around
            self._start_day(day, day_next=day_next)
        else:
            self._start_day(self.day, day_next=None)

        # standardize zscore for the reward the current day's actual price/demand at time t (since reward is based on actual at t)
        pb_mu, pb_sd = self.compute_stats(self.prices_buy_actual)
        ps_mu, ps_sd = self.compute_stats(self.prices_sell_actual)
        dd_mu, dd_sd = self.compute_stats(self.demand_actual)
        self.scaled_pb_actual = self._z(self.prices_buy_actual, pb_mu, pb_sd)
        self.scaled_ps_actual = self._z(self.prices_sell_actual, ps_mu, ps_sd)
        self.scaled_d_actual  = self._z(self.demand_actual,  dd_mu,  dd_sd)

        return self._obs(), {}


    def step(self, action: np.ndarray):
        a = float(np.clip(action[0], -1.0, 1.0))
        charge_MW    = max(0.0,  a) * self.cmax
        discharge_MW = max(0.0, -a) * self.dmax

        # SoC with bound enforcement
        delta_soc = (self.eta_c*charge_MW*self.dt - discharge_MW*self.dt/self.eta_d)/self.C
        soc_next = self.soc + delta_soc
        if soc_next > self.soc_max + 1e-9:
            excess = soc_next - self.soc_max
            reduce_charge = excess * self.C / (self.eta_c*self.dt + 1e-12)
            charge_MW = max(0.0, charge_MW - reduce_charge)
            soc_next = self.soc + (self.eta_c*charge_MW*self.dt - discharge_MW*self.dt/self.eta_d)/self.C
        elif soc_next < self.soc_min - 1e-9:
            deficit = self.soc_min - soc_next
            reduce_discharge = deficit * self.C * self.eta_d / (self.dt + 1e-12)
            discharge_MW = max(0.0, discharge_MW - reduce_discharge)
            soc_next = self.soc + (self.eta_c*charge_MW*self.dt - discharge_MW*self.dt/self.eta_d)/self.C

        # Balance and reward use ACTUALS
        net = self.demand_actual[self.t] + charge_MW - discharge_MW
        imp = max(0.0, net)
        exp = max(0.0, -net) if self.allow_export else 0.0

        cost = self.scaled_pb_actual[self.t] * charge_MW * self.dt
        revenue = self.scaled_ps_actual[self.t] * discharge_MW * self.dt if self.allow_export else 0.0
        # print("allow expert", self.allow_export)
        raw_reward = -(cost - revenue)
        # reward = raw_reward / (self.r_scale + self.r_eps)  # scale to around +/- 1.0
        # reward = float(np.clip(raw_reward, -self.r_clip, self.r_clip))
        reward =raw_reward

        # # action smoothness penalty
        # if self.lambda_smooth > 0.0:
        #     reward -= self.lambda_smooth * abs(a - self._prev_action)
        # self._prev_action = a


        # logs & advance
        self._import[self.t] = imp
        self._export[self.t] = exp
        self._charge[self.t] = charge_MW
        self._disch[self.t]  = discharge_MW
        self.soc = float(np.clip(soc_next, self.soc_min, self.soc_max))
        self._soc_series[self.t + 1] = self.soc

        self.t += 1
        terminated = (self.t >= self.T)
        truncated = False

        info = {}
        if terminated:
            # terminal SoC target penalty
                # energy shortfall in MWh
            energy_gap_MWh = abs(self.soc - self.soc_target) * self.C

            # reference price scale (use today's mean forecast/actual)
            p_ref = self.price_scale  # already ~ mean price

            # penalty in "money" units
            penalty_cost = self.term_penalty * energy_gap_MWh * p_ref

            # convert to reward scale (same as your reward normalization)
            # reward -= penalty_cost / (self.r_scale + self.r_eps)
            
            # reward -= 0.5 * abs(self.soc - self.soc_target)/ self.r_scale

            # # forecast vs actual error metrics (if forecasts were actually used)
            # if self.obs_mode == "forecast" and not np.allclose(self.prices_buy_obs, self.prices_buy_actual):
            #     # compute across the whole day
            #     pa, pf = self.prices_buy_actual, self.prices_buy_obs
            #     da, df = self.demand_actual, self.demand_obs
            #     mae_p = float(np.mean(np.abs(pf - pa)))
            #     mae_d = float(np.mean(np.abs(df - da)))
            #     # avoid divide-by-zero for MAPE
            #     mape_p = float(np.mean(np.abs((pf - pa) / (pa + 1e-8))))
            #     mape_d = float(np.mean(np.abs((df - da) / (da + 1e-8))))
            #     self._mae_price, self._mape_price = mae_p, mape_p
            #     self._mae_demand, self._mape_demand = mae_d, mape_d
            #     info.update({
            #         "mae_price": mae_p, "mape_price": mape_p,
            #         "mae_demand": mae_d, "mape_demand": mape_d,
            #     })

            # return zero obs at terminal
            obs = {
                "soc": np.zeros(1, np.float32),
                "prices_buy": np.zeros(self.obs_window, np.float32),
                "prices_sell": np.zeros(self.obs_window, np.float32),
                "demand": np.zeros(self.obs_window, np.float32),
            }
        else:
            obs = self._obs()

        return obs, float(reward), terminated, truncated, info

    def export_solve_response(self, day: DayInputs) -> SolveResponse:
        prices = self.prices_buy_actual
        dt = float(day.dt_hours)
        imp = self._import
        exp = self._export
        p_sell = self.prices_sell_actual
        total_cost = float(np.sum(prices * imp * dt) - (np.sum(p_sell * exp * dt) if self.allow_export else 0.0))
        return SolveResponse(
            status="rllib_policy",
            objective_cost=total_cost,
            charge_MW=self._charge.tolist(),
            discharge_MW=self._disch.tolist(),
            import_MW=self._import.tolist(),
            export_MW=(self._export.tolist() if self.allow_export else None),
            soc=self._soc_series.tolist(),
        )
