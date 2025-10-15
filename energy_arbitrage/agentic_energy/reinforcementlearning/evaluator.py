from __future__ import annotations
from pathlib import Path
import ray
from typing import Union
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from .env import BatteryArbRLEnv
from .adapter import request_to_env_config
from .config import apply_process_env
from .logging import setup_python_logging

def _resolve_ckpt_dir(p: Union[str, Path]) -> Path:
    """Return a directory that contains rllib_checkpoint.json (walk up if needed)."""
    p = Path(p).resolve()
    if p.is_file():
        p = p.parent
    # Walk up at most 3 levels to find the marker file.
    for up in [p, p.parent, p.parent.parent]:
        if (up / "rllib_checkpoint.json").exists():
            return up
    raise FileNotFoundError(
        f"Could not find rllib_checkpoint.json starting from {p}. "
        f"Pass the directory that contains it (your screenshot shows runs/rllib_battery)."
    )

def _env_creator(env_config):
    return BatteryArbRLEnv(env_config)

def build_eval_config(env_config):
    return (
        PPOConfig()
        .environment(env="battery-arb", env_config=env_config)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
        .debugging(log_level="ERROR")
    )

def rollout_day(
    checkpoint: str | Path,
    req,
    *,
    obs_mode: str = "compact",
    obs_window: int = 24,
):
    """
    Load a saved PPO policy and roll out one DayInputs in a fresh env.
    Returns SolveResponse.
    """
    apply_process_env()
    setup_python_logging()

    ray.init(ignore_reinit_error=True, include_dashboard=False,
             logging_level="ERROR", local_mode=True, log_to_driver=False)

    register_env("battery-arb", _env_creator)

    env_config = request_to_env_config(
        req, obs_mode=obs_mode, obs_window=obs_window
    )

    config = build_eval_config(env_config)
    ckpt_dir = _resolve_ckpt_dir(checkpoint) 
    algo = config.build_algo()
    algo.restore(str(ckpt_dir))   # <— load your checkpoint

    env = BatteryArbRLEnv(env_config)
    obs, _ = env.reset()
    done = False
    ep_ret = 0.0
    while not done:
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        ep_ret += float(reward)

    out = env.export_solve_response(req.day)
    print(f"Rollout done. Episode return (scaled): {ep_ret:.3f} | Cost: {out.objective_cost:.3f}")

    ray.shutdown()
    return out
