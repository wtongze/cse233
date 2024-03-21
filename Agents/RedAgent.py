from ray.rllib.algorithms.algorithm import Policy

from CybORG.Agents import BaseAgent


class RedAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__()
        # CSE233 Project: you should load your red agent model here
        self.policy = Policy.from_checkpoint('policies/default_policy')
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

    def train(self):
        """
        allows an agent to learn a policy
        """
        # This method is not implemented due to the use of ray / rllib framework.
        # We will directly ask the framework to train the agent and store its policy.
        print("Please use our red_train.py to directly train the agent.")
        raise NotImplementedError

    def reset(self):
        # This method is added to reset our red agent after each episode
        # due to our use of LSTM layer.
        self.state = self.policy.get_initial_state()
