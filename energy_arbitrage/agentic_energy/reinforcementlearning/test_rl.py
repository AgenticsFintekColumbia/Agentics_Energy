from __future__ import annotations
from pathlib import Path
import numpy as np

from agentic_energy.schemas import BatteryParams, DayInputs, SolveRequest
from .trainer import train_rllib
from .evaluator import rollout_day

def demo():
    '''
        - Activate energy-rl conda environment to work with
        - collect samples until it has at least train_batch_size (4000) steps (each step is 1 hour here, so 4000 steps = ~166 days of data)
        - compute advantages and returns,
        - optimize policy for num_epochs (10) epochs over the collected data, using minibatches of size minibatch_size (256) 
        epochs = how many passes you make over that same dataset. Re-using the same batch multiple times improves sample efficiency.
        minibatch_size = how many samples you feed to the optimizer per gradient step. The big batch is split into smaller chunks for stable SGD.
        - evaluate every 5 iters on 5 episodes and print eval stats
        repeat for num_iterations (1 here for demo)

        With train_batch_size=4000, minibatch_size=256, num_epochs=10:

        - RLlib collects 4000 transitions from the env and computes advantages/returns + stores old log-probs.
        - For each epoch (10 times):
            1.Shuffle the 4000 samples.
            2.Split into ~ceil(4000/256)=16 mini-batches.
            3.For each mini-batch:
            Forward pass: compute policy ratio r = exp(new_logp - old_logp), clipped objective, value loss, entropy bonus.
            Backward pass: take one optimizer step updating actor+critic (subject to PPO clipping).
        - After ~16 * 10 = 160 gradient updates, discard the 4000 samples and go collect a fresh batch from the env.
    '''

    batt = BatteryParams(
        capacity_kwh=20.0, soc_init=0.5, soc_min=0.10, soc_max=0.90,
        cmax_kw=6.0, dmax_kw=6.0, eta_c=0.95, eta_d=0.95, soc_target=0.5
    )
    base_prices = np.array([0.12]*6 + [0.15]*6 + [0.22]*6 + [0.16]*6, dtype=np.float32)
    base_demand = np.array([0.9]*24, dtype=np.float32)

    rng = np.random.default_rng(0)
    days = []
    for _ in range(20):
        prices = (base_prices * (1.0 + rng.normal(0, 0.05, size=24))).clip(0.05, 0.6)
        demand = (base_demand * (1.0 + rng.normal(0, 0.10, size=24))).clip(0.2, 2.0)
        days.append(DayInputs(
            prices_buy=prices.tolist(),
            demand_kw=demand.tolist(),
            prices_sell=prices.tolist(),
            allow_export=False,
            dt_hours=1.0,
            prices_buy_forecast=prices.tolist(),
            demand_kw_forecast=demand.tolist(),
            prices_sell_forecast=prices.tolist(),
        ))

    eval_req = SolveRequest(battery=batt, day=days[0], solver="RL", solver_opts={"lambda_smooth": 0.01})

    ckpt = train_rllib(
        eval_req, days,
        num_iterations=10, num_workers=0,
        obs_mode="compact", obs_window=24
    )
    res = rollout_day(ckpt, eval_req, obs_mode="compact", obs_window=24)
    print("RLlib rollout (compact) day cost:", res.objective_cost)

if __name__ == "__main__":
    demo()
