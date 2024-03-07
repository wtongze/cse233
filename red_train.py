# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import random

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from Wrappers.GymWrapper import GymWrapper

MAX_STEPS = 30
MAX_EPS = 20_000
random.seed(153)

if __name__ == "__main__":
    ray.init()

    config = (
        PPOConfig()
        .training(
            gamma=0.995,
            lr=5e-04,
            model={
                "fcnet_hiddens": tune.grid_search([
                    [1024, 1024],
                    [2048, 2048]
                ]),
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
                "episodes_total": MAX_EPS
            },
            checkpoint_config=air.CheckpointConfig(
                checkpoint_score_attribute="max-episode_reward_mean",
                num_to_keep=1,
            ),
            local_dir="checkpoints/",
            name="tune"
        ),
        param_space=config
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    best_checkpoint = best_result.checkpoint

    best_checkpoint.to_directory("checkpoints/best/")

    # max_episodes = 1
    # max_time_steps = 1
    # for i_episode in range(1, max_episodes + 1):
    #     observation = env.reset()
    #
    #     time_step = 0
    #     action_space: int = env.get_action_space('Red')  # type: ignore
    #
    #     for t in range(max_time_steps):
    #         time_step += 1
    #         action = red_agent.get_action(observation, action_space)
    #
    #         observation, reward, done, info = env.step(3)
    #
    #         '''
    #         CSE233 Project: Here you should call red agent training function
    #         '''
    #         # red_agent.train(...) # CSE233 Project: uncomment when you implement red agent training
