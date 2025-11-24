# agentic_energy/reinforcementlearning/trainer.py
from __future__ import annotations
from pathlib import Path
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from .env import BatteryArbRLEnv
from .adapter import request_to_train_env_config
from .logging import PrintCallbacks, setup_python_logging, make_logger_creator, MetricTracker
from .config import DEFAULT_SAVE_DIR, apply_process_env, ensure_dir, PPOTrainSettings

def _env_creator(env_config):
    return BatteryArbRLEnv(env_config)

def _validate(settings: PPOTrainSettings):
    if settings.rollout_fragment_length <= 0:
        raise ValueError("rollout_fragment_length must be > 0")
    if settings.train_batch_size % settings.rollout_fragment_length != 0:
        raise ValueError(
            f"train_batch_size ({settings.train_batch_size}) should be a multiple "
            f"of rollout_fragment_length ({settings.rollout_fragment_length})"
        )
    if settings.minibatch_size > settings.train_batch_size:
        raise ValueError("minibatch_size cannot exceed train_batch_size")

def build_config(env_config, settings:PPOTrainSettings):
    _validate(settings=settings)
    return (
        PPOConfig()
        .environment(env="battery-arb", env_config=env_config)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False,
                   enable_env_runner_and_connector_v2=False)
        .env_runners(
            num_env_runners=settings.num_env_runners,
            rollout_fragment_length=settings.rollout_fragment_length
        )
        .resources(num_gpus=0)
        .debugging(log_level="ERROR", seed=0)
        .callbacks(PrintCallbacks)
        .evaluation(
            evaluation_interval=settings.evaluation_interval,
            evaluation_duration=settings.evaluation_episodes,
            evaluation_duration_unit="episodes",
            evaluation_config={
                "explore": False,
                "num_env_runners":1,
            },
        )
        .training(
            gamma=settings.gamma,
            lr=settings.lr,
            train_batch_size=settings.train_batch_size,
            minibatch_size=settings.minibatch_size,
            num_epochs=settings.num_epochs,
            clip_param=settings.clip_param,
            vf_clip_param=settings.vf_clip_param,
        )
    )

def train_rllib(
    req,
    train_days,
    *,
    settings: PPOTrainSettings | None = None,
    num_iterations: int = 50,
    save_dir: str = DEFAULT_SAVE_DIR,
    obs_mode: str = "compact",
    obs_window: int = 24,
) -> Path:
    """
    Train PPO on battery arbitrage using RLlib.
    Returns a filesystem path to the final checkpoint.
    """
    settings = settings or PPOTrainSettings()
    apply_process_env()
    setup_python_logging()
    ensure_dir(save_dir)

    ray.init(ignore_reinit_error=True, include_dashboard=False,
             logging_level="ERROR", local_mode=True, log_to_driver=False)

    register_env("battery-arb", _env_creator)

    env_config = request_to_train_env_config(
        req, train_days, obs_mode=obs_mode, obs_window=obs_window
    )

    config = build_config(env_config, settings=settings)
    logger_creator = make_logger_creator(save_dir, trial_dir_name="PPO_battery_Italy")

    algo = config.build(logger_creator=logger_creator)
    
    # One-time: print where TB events are going
    tb_dir = getattr(getattr(algo, "logger", None), "logdir", None) or getattr(algo, "logdir", None)
    print("TensorBoard logdir:", tb_dir)

    tracker = MetricTracker(ema_alpha=0.1)
    last_result = None
    for i in range(1, num_iterations + 1):
        last_result = algo.train()
        tracker.update_and_print(i, last_result)
        # print("TB logdir:", last_result.get("log_dir") or last_result.get("logdir"))

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
