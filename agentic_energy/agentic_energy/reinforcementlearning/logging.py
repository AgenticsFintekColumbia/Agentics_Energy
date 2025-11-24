from __future__ import annotations
import numpy as np
from collections import deque
import logging, os
from pathlib import Path
from typing import Callable
from ray.tune.logger import UnifiedLogger, TBXLogger, JsonLogger, CSVLogger
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from collections import deque
import math

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
        # print("  ".join(line))

        self._train_rewards.clear()
        self._train_lengths.clear()



class MetricTracker:
    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self.train_return_ema = None
        self.history = deque(maxlen=100)

    def _safe_get(self, d, path, default=None):
        cur = d
        for k in path:
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(k, None)
            else:
                cur = getattr(cur, k, None)
        return default if cur is None else cur

    def _extract(self, result: dict):
        train_mean = (
            self._safe_get(result, ["episode_reward_mean"]) or
            self._safe_get(result, ["env_runners", "episode_reward_mean"]) or
            self._safe_get(result, ["env_runners", "policy_reward_mean", "default_policy"])
        )
        eval_mean = self._safe_get(result, ["evaluation", "episode_reward_mean"])

        learner = self._safe_get(result, ["info", "learner"], {}) or {}
        pol_keys = [k for k, v in getattr(learner, "items", lambda: learner.items())() if isinstance(v, dict)]
        pol = learner[pol_keys[0]] if pol_keys else {}
        ls  = self._safe_get(pol, ["learner_stats"], {}) or pol

        kl        = self._safe_get(ls, ["kl"])
        ent       = self._safe_get(ls, ["entropy"])
        vexp      = self._safe_get(ls, ["vf_explained_var"])
        return train_mean, eval_mean, kl, ent, vexp

    def _fmt(self, x, nd=3, default="NA"):
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return default
        return f"{float(x):.{nd}f}"

    def _get_eval_return_mean(self, result: dict):
        # Newer style: nested dict
        if "evaluation" in result and isinstance(result["evaluation"], dict):
            return result["evaluation"].get("episode_reward_mean")
        # Flattened keys (some versions)
        for k in ("evaluation/episode_reward_mean", "evaluation_episode_reward_mean"):
            if k in result:
                return result[k]
        return None

    def update_and_print(self, it: int, result: dict):
        train_mean, eval_mean, kl, ent, vexp = self._extract(result)
        eval_mean = self._get_eval_return_mean(result)
        if train_mean is not None:
            if self.train_return_ema is None:
                self.train_return_ema = float(train_mean)
            else:
                a = self.ema_alpha
                self.train_return_ema = a * float(train_mean) + (1 - a) * self.train_return_ema

        kl_ok    = (kl is not None) and (0.005 <= float(kl) <= 0.02)
        vexp_ok  = (vexp is not None) and (float(vexp) >= 0.80)

        line = (
            f"[ITER {it:>3}] "
            f"trainR={self._fmt(train_mean, nd=2)} "
            f"(EMA={self._fmt(self.train_return_ema, nd=2)})  "
            f"evalR={self._fmt(eval_mean, nd=2)}  "
            f"KL={self._fmt(kl)}{'✅' if kl_ok else '⚠'}  "
            f"Entropy={self._fmt(ent, nd=3)}  "
            f"Vexp={self._fmt(vexp, nd=3)}{'✅' if vexp_ok else '⚠'}"
        )
        print(line)

        tips = []
        if kl is not None and float(kl) > 0.03:
            tips.append("KL high → lower lr or fewer epochs")
        if kl is not None and float(kl) < 0.003:
            tips.append("KL low → higher lr or more epochs")
        if vexp is not None and float(vexp) < 0.7:
            tips.append("Value net weak → adjust value loss coeff or lr")
        if tips:
            print("        notes:", " | ".join(tips))
