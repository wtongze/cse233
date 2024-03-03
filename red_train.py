# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import random

import ray
from ray.rllib.algorithms.dqn import DQN

from Wrappers.GymWrapper import GymWrapper

MAX_EPS = 100
random.seed(153)

if __name__ == "__main__":
    ray.init()

    trainer = DQN(env=GymWrapper, config={  # type: ignore
        "env_config": {
            "max_steps": 100
        },
        "framework": "torch",
        "num_gpus": 1,
    })

    for i in range(50):
        result = trainer.train()
        print(result['episode_reward_mean'])

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
