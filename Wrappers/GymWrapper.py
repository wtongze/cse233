import inspect

import gym
from ray.rllib.env.env_context import EnvContext

from Agents.RedAgent import RedAgent
from Agents.WrappedAgent import WrappedBlueAgent
from CybORG import CybORG
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2

'''
Reference: https://github.com/alan-turing-institute/cage-challenge-2-public/blob/submission-final/agents/baseline_sub_agents/CybORGAgent.py
'''


class GymWrapper(gym.Env):
    observation, reward, done, info = (None, None, None, None)

    def __init__(self, config: EnvContext):
        scenario = 'Scenario2'

        # Load scenario
        path = str(inspect.getfile(CybORG))
        path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'

        # Load blue agent
        blue_agent = WrappedBlueAgent

        # Set up environment with blue agent running in the background and
        # red agent as the main agent
        cyborg = CybORG(path, 'sim', agents={'Blue': blue_agent})

        self.env = ChallengeWrapper2(env=cyborg, agent_name='Red')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action=None):
        try:
            self.observation, self.reward, self.done, self.info = self.env.step(action)
        except:
            print(">>> Suppressing CybORG exception")
            init_obs = self.env.reset()
            return init_obs, -100.0, True, {}
        return self.observation, self.reward, self.done, self.info

    def seed(self, seed=None):
        self.env.seed(seed)
