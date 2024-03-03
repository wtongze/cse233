# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import inspect
import random

from Agents.RedAgent import RedAgent
from Agents.WrappedAgent import WrappedBlueAgent
from CybORG import CybORG, CYBORG_VERSION
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

MAX_EPS = 100
random.seed(153)

if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'

    # Load scenario
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

    # Load blue agent
    blue_agent = WrappedBlueAgent
    red_agent = RedAgent()
    # Set up environment with blue agent running in the background and 
    # red agent as the main agent
    cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})
    env = ChallengeWrapper2(env=cyborg, agent_name="Red")

    max_episodes = 1
    max_time_steps = 1
    for i_episode in range(1, max_episodes + 1):
        observation = env.reset()

        time_step = 0
        action_space: int = env.get_action_space('Red')  # type: ignore

        for t in range(max_time_steps):
            time_step += 1
            action = red_agent.get_action(observation, action_space)

            observation, reward, done, info = env.step(3)

            '''
            CSE233 Project: Here you should call red agent training function 
            '''
            # red_agent.train(...) # CSE233 Project: uncomment when you implement red agent training
