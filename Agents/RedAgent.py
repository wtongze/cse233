from ray.rllib.algorithms.algorithm import Policy

from CybORG.Agents import BaseAgent


class RedAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()
        # CSE233 Project: you should load your red agent model here
        self.policy = Policy.from_checkpoint('/home/wtongze/Downloads/best-ppo-lstm/policies/default_policy')
        self.state = self.policy.get_initial_state()

    def get_action(self, observation, action_space):
        """
        gets an action from the agent that should be performed based on the agent's internal state and provided
        observation and action space
        """
        # CSE233 Project: you should modify this line to get action from red agent
        action, state, _ = self.policy.compute_single_action(obs=observation, state=self.state)
        self.state = state
        # action = random.randint(0, action_space - 1)
        return action

    def train(self):  # CSE233 Project: you should modify this line to implement red agent training
        """
        allows an agent to learn a policy
        """
        # CSE233 Project: you should modify this line to implement red agent training
        raise NotImplementedError

    def reset(self):
        self.state = self.policy.get_initial_state()
