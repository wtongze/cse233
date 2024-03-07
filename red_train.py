# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import random

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from Wrappers.GymWrapper import GymWrapper

MAX_STEPS = 30
MAX_EPS = 50
random.seed(153)

if __name__ == "__main__":
    ray.init()

    algo = (
        PPOConfig()
        .training(
            gamma=0.9
        )
        .framework('torch')
        .evaluation(evaluation_interval=None)
        .rollouts(num_rollout_workers=10, horizon=MAX_STEPS)
        .resources(num_gpus=1)
        .environment(env=GymWrapper)
        .build()
    )

    for i in range(1, MAX_EPS + 1):
        print(f"====== Step: {i} ======")
        result = algo.train()
        print(f"episodes_total: {result['episodes_total']}")
        print(f"max: {result['episode_reward_max']:.5f}")
        print(f"mean: {result['episode_reward_mean']:.5f}")

        if i % 5 == 0:
            policy = algo.get_policy(policy_id="default_policy")
            policy.export_checkpoint(f"policy/{int(i / 5)}")
            print(">>> Policy saved")

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
