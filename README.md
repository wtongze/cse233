# Repository for the advanced computer security class - Winter 2023

The project uses the simulator [CybORG](https://github.com/cage-challenge/cage-challenge-2/tree/main). Please follow the [instructions](https://github.com/cage-challenge/cage-challenge-2/tree/main/CybORG) to install the simulator.

## Requirements

The Cyborg Simulator and pytorch

## 

The objective is each group gets a red agent that maximizes the reward using reinforcement learning.

In the current version, the red agent produces a random action sampled from the action space. 
```
action = random.randint(0, action_space - 1) # TODO: get action from red agent
```

Each group should modify the ```red_train.py``` file to implement the RL-agent training. Similarly, they should modify the ```red_evaluation.py``` file to evaluate their trained agent.