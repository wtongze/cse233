# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import random

import ray
import numpy as np
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from Wrappers.GymWrapper import GymWrapper

MAX_STEPS = 30
MAX_EPISODES = 20_000
random.seed(153)
np.random.seed(153)

if __name__ == "__main__":
    ray.init()

    config = (
        PPOConfig()
        .training(
            gamma=0.995,
            lr=5e-04,
            model={
                "fcnet_hiddens": [1024, 1024],
                "use_lstm": True
            }
        )
        .framework('torch')
        .evaluation(evaluation_interval=None)
        .rollouts(num_rollout_workers=10, horizon=MAX_STEPS)
        .resources(num_gpus=1)
        .environment(env=GymWrapper)
    )

    tuner = tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            stop={
                "episodes_total": MAX_EPISODES
            },
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",
                num_to_keep=1,
            ),
            local_dir="checkpoints/",
            name="train"
        ),
        param_space=config
    )

    '''
    CSE233 Project: Here you should call red agent training function
    '''
    results = tuner.fit()
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    best_checkpoint = best_result.checkpoint

    best_checkpoint.to_directory("checkpoints/final/")
