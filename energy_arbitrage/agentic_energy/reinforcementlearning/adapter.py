from __future__ import annotations
from typing import Dict, Any, List, Optional

from agentic_energy.schemas import SolveRequest, DayInputs

_VALID_OBS_MODES = {"compact", "forecast"}

def _validate_obs_mode(val: str) -> str:
    mode = (val or "compact").lower()
    if mode not in _VALID_OBS_MODES:
        raise ValueError(f"obs_mode must be one of {_VALID_OBS_MODES}, got {val!r}")
    return mode

def _resolve_obs_settings(
    explicit_obs_mode: Optional[str],
    explicit_obs_window: Optional[int],
    solver_opts: Optional[Dict] = None,
) -> tuple[str, int]:
    """
    Merge explicit args with request.solver_opts, with explicit taking precedence.
    Accept legacy 'Tmax' key as alias for obs_window.
    """
    # defaults
    mode = explicit_obs_mode
    win  = explicit_obs_window

    # pull from solver_opts if present (only if not explicitly provided)
    if solver_opts:
        if mode is None:
            mode = solver_opts.get("obs_mode")
        if win is None:
            win = solver_opts.get("obs_window", solver_opts.get("Tmax"))

    # final defaults
    mode = _validate_obs_mode(mode or "compact")
    win = int(win or 24)
    if win <= 0:
        raise ValueError(f"obs_window must be positive; got {win}")
    return mode, win


def request_to_env_config(
    req: SolveRequest,
    *,
    obs_mode: Optional[str] = None,
    obs_window: Optional[int] = None,
    allow_solver_opts_overrides: bool = True,
) -> Dict[str, Any]:
    """
    Build a single-day env_config for RLlib from SolveRequest.
    If allow_solver_opts_overrides=True, will read 'obs_mode' and 'obs_window' (or 'Tmax') from req.solver_opts.
    """
    mode, win = _resolve_obs_settings(
        obs_mode,
        obs_window,
        solver_opts=(req.solver_opts if allow_solver_opts_overrides else None),
    )
    return {
        "battery": req.battery.model_dump(),
        "day": req.day.model_dump(),
        "obs_mode": mode,
        "obs_window": win,
        "lambda_smooth": float(req.solver_opts.get("lambda_smooth", 0.0)) if req.solver_opts else 0.0,
    }


def request_to_train_env_config(
    req: SolveRequest,
    days: List[DayInputs],
    *,
    obs_mode: Optional[str] = None,
    obs_window: Optional[int] = None,
    allow_solver_opts_overrides: bool = True,
) -> Dict[str, Any]:
    """
    Build a training env_config that samples randomly from the provided list of DayInputs each reset.
    If allow_solver_opts_overrides=True, will read 'obs_mode' and 'obs_window' (or 'Tmax') from req.solver_opts.
    """
    mode, win = _resolve_obs_settings(
        obs_mode,
        obs_window,
        solver_opts=(req.solver_opts if allow_solver_opts_overrides else None),
    )
    return {
        "battery": req.battery.model_dump(),
        "days": [d.model_dump() for d in days],
        "obs_mode": mode,
        "obs_window": win,
        "lambda_smooth": float(req.solver_opts.get("lambda_smooth", 0.0)) if req.solver_opts else 0.0,
    }
