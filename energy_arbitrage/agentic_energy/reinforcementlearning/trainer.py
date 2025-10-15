# agentic_energy/reinforcementlearning/trainer.py
from __future__ import annotations
from pathlib import Path
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from .env import BatteryArbRLEnv
from .adapter import request_to_train_env_config
from .logging import PrintCallbacks, setup_python_logging, make_logger_creator
from .config import DEFAULT_SAVE_DIR, apply_process_env, ensure_dir

def _env_creator(env_config):
    return BatteryArbRLEnv(env_config)

def build_config(env_config, *, num_workers: int):
    return (
        PPOConfig()
        .environment(env="battery-arb", env_config=env_config)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=num_workers)
        .resources(num_gpus=0)
        .debugging(log_level="ERROR", seed=0)
        .callbacks(PrintCallbacks)
        .evaluation(
            evaluation_interval=5,
            evaluation_duration=5,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False},
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=10,
            clip_param=0.2,
            vf_clip_param=10.0,
        )
    )

def train_rllib(
    req,
    train_days,
    *,
    num_iterations: int = 50,
    num_workers: int = 0,
    save_dir: str = DEFAULT_SAVE_DIR,
    obs_mode: str = "compact",
    obs_window: int = 24,
) -> Path:
    """
    Train PPO on battery arbitrage using RLlib.
    Returns a filesystem path to the final checkpoint.
    """
    apply_process_env()
    setup_python_logging()
    ensure_dir(save_dir)

    ray.init(ignore_reinit_error=True, include_dashboard=False,
             logging_level="ERROR", local_mode=True, log_to_driver=False)

    register_env("battery-arb", _env_creator)

    env_config = request_to_train_env_config(
        req, train_days, obs_mode=obs_mode, obs_window=obs_window
    )

    config = build_config(env_config, num_workers=num_workers)
    logger_creator = make_logger_creator(save_dir, trial_dir_name="PPO_battery")

    algo = config.build(logger_creator=logger_creator)

    last_result = None
    for i in range(1, num_iterations + 1):
        last_result = algo.train()
        print("TB logdir:", last_result.get("log_dir") or last_result.get("logdir"))

    # Save a checkpoint in save_dir
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out = algo.save(checkpoint_dir=str(save_path))
    if isinstance(out, str):
        ckpt_dir =  Path(out)
    if hasattr(out, "checkpoint") and hasattr(out.checkpoint, "path"):
        ckpt_dir =  Path(out.checkpoint.path)
    if hasattr(out, "path"):
        ckpt_dir =  Path(out.path)
    ray.shutdown()
    return ckpt_dir
