# Implementation of the red evaluation script
# This script is used to evaluate the performance of a red agent against a blue agent
# The blue agent is https://github.com/john-cardiff/-cyborg-cage-2.git
# Modified by Prof. H. Sasahara
import inspect
import random
import numpy as np

from Agents.RedAgent import RedAgent
from Agents.WrappedAgent import WrappedBlueAgent
from CybORG import CybORG, CYBORG_VERSION
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

MAX_EPS = 100
agent_name = 'Red'
random.seed(153)


# changed to ChallengeWrapper2
def wrap(env):
    return ChallengeWrapper2(env=env, agent_name=agent_name)


if __name__ == "__main__":
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    # commit_hash = get_git_revision_hash()

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

    num_steps = 30
    observation = env.reset()

    # action_space = wrapped_cyborg.get_action_space(agent_name)
    action_space = env.get_action_space(agent_name)
    total_reward = []
    actions = []
    for i in range(MAX_EPS):
        r = []
        a = []
        for j in range(num_steps):
            # This line is changed to connect our trained red agent.
            action = red_agent.get_action(observation, action_space)

            observation, rew, done, info = env.step(action)
            r.append(rew)
            a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
        total_reward.append(sum(r))
        actions.append(a)
        observation = env.reset()

        # This line is added to reset our red agent after each episode
        # due to our use of LSTM layer.
        red_agent.reset()

    # The following lines are added to print out the reward,
    # and calculate min, max, and mean award.
    result = np.array(total_reward)
    print(result)
    print()
    print(f"mean: {result.mean():.2f} min: {result.min():.2f} max: {result.max():.2f}")
