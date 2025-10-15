from __future__ import annotations
import numpy as np
from collections import deque
import logging, os
from pathlib import Path
from typing import Callable
from ray.tune.logger import UnifiedLogger, TBXLogger, JsonLogger, CSVLogger
from ray.rllib.algorithms.callbacks import DefaultCallbacks


def setup_python_logging(level=logging.WARNING):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s: %(message)s",
    )
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("ray.tune").setLevel(logging.WARNING)
    logging.getLogger("ray.rllib").setLevel(logging.WARNING)
    logging.getLogger("ray._private").setLevel(logging.ERROR)
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    logging.getLogger("ray.serve").setLevel(logging.ERROR)

def make_logger_creator(log_root: str, trial_dir_name: str = "PPO_battery") -> Callable:
    """
    Returns a function RLlib will call to create the logger.
    Ensures TensorBoard, JSON, and CSV logs go to a stable place.
    """
    def _logger_creator(cfg):
        logdir = Path(log_root) / trial_dir_name
        logdir.mkdir(parents=True, exist_ok=True)
        return UnifiedLogger(
            cfg, str(logdir),
            loggers=[TBXLogger, JsonLogger, CSVLogger]
        )
    return _logger_creator



def _safe(d, *keys, default=None):
    x = d
    for k in keys:
        if not isinstance(x, dict) or k not in x:
            return default
        x = x[k]
    return x

def _is_eval_episode(worker) -> bool:
    try:
        if getattr(worker, "in_evaluation", False):
            return True
        er = getattr(worker, "env_runner", None)
        if er is not None and (getattr(er, "in_evaluation", False) or getattr(er, "is_evaluation", False)):
            return True
    except Exception:
        pass
    return False

class PrintCallbacks(DefaultCallbacks):
    """Compact, EpisodeV2-safe console logging each iter."""
    def __init__(self):
        super().__init__()
        self._train_rewards = []
        self._train_lengths = []
        self._recent_train_means = deque(maxlen=20)
        self._recent_eval_means  = deque(maxlen=20)
        self._train_ep_count = 0
        self._eval_ep_count = 0
        self._ep_print_every = 25

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        R = float(episode.total_reward)
        L = int(episode.length)
        if _is_eval_episode(worker):
            self._eval_ep_count += 1
            print(f"[EVAL ] ep#{self._eval_ep_count:>5}  R={R:8.3f} len={L}")
        else:
            self._train_ep_count += 1
            self._train_rewards.append(R)
            self._train_lengths.append(L)
            if self._train_ep_count % self._ep_print_every == 0:
                print(f"[TRAIN] ep#{self._train_ep_count:>5}  R={R:8.3f} len={L}")

    def on_train_result(self, *, algorithm=None, result: dict, **kwargs):
        iter_id = int(result.get("training_iteration", -1))

        def slope(vals):
            if len(vals) < 3:
                return None
            x = np.arange(len(vals), dtype=float)
            y = np.array(vals, dtype=float)
            m, _ = np.polyfit(x, y, 1)
            return float(m)

        tr = np.array(self._train_rewards, dtype=float) if self._train_rewards else np.array([])
        tl = np.array(self._train_lengths, dtype=float) if self._train_lengths else np.array([])
        train_mean = float(tr.mean()) if tr.size else None
        train_min  = float(tr.min())  if tr.size else None
        train_max  = float(tr.max())  if tr.size else None
        len_mean   = float(tl.mean()) if tl.size else None
        if train_mean is not None:
            self._recent_train_means.append(train_mean)

        ev = result.get("evaluation") or {}
        eval_mean = ev.get("episode_reward_mean", None)
        eval_len  = ev.get("episode_len_mean", None)
        if eval_mean is not None:
            eval_mean = float(eval_mean)
            self._recent_eval_means.append(eval_mean)

        learner = _safe(result, "info", "learner", default={})
        pol_stats = None
        if isinstance(learner, dict):
            for v in learner.values():
                if isinstance(v, dict):
                    pol_stats = v.get("learner_stats", v)
                    break
        def fget(k):
            return float(pol_stats[k]) if (pol_stats and k in pol_stats) else None

        kl = fget("kl"); vf_loss = fget("vf_loss"); pol_loss = fget("policy_loss")
        ent = fget("entropy"); evv = fget("vf_explained_var"); cur_lr = fget("cur_lr")

        train_slope = slope(self._recent_train_means)
        eval_slope  = slope(self._recent_eval_means)

        line = [f"[ITER {iter_id}]"]
        if train_mean is not None:
            line.append(f"trainR μ={train_mean:7.3f} [{train_min:7.3f},{train_max:7.3f}] len={len_mean:.1f} n={tr.size}")
        if eval_mean is not None:
            line.append(f"| evalR μ={eval_mean:7.3f} len={eval_len:.1f}")
        if kl is not None:       line.append(f"| KL={kl:.4f}")
        if ent is not None:      line.append(f"entropy={ent:.3f}")
        if vf_loss is not None:  line.append(f"Vloss={vf_loss:.3f}")
        if evv is not None:      line.append(f"VexpVar={evv:.3f}")
        if pol_loss is not None: line.append(f"Ploss={pol_loss:.4f}")
        if cur_lr is not None:   line.append(f"lr={cur_lr:.1e}")
        if train_slope is not None: line.append(f"| trend(train)={train_slope:+.3f}/iter")
        if eval_slope  is not None: line.append(f"trend(eval)={eval_slope:+.3f}/iter")
        print("  ".join(line))

        self._train_rewards.clear()
        self._train_lengths.clear()
