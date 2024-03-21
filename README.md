# CSE 233 Project

By Adi Shamir / Alan Turing Group

## Note

- Given that we are using PPO+LSTM, we have modified the
  `RedAgent.py` to add a custom method `reset()` that needs to be called to reset our agent after each episode.
- The `red_evaluation.py` has been modified to connect our trained agent. We also added few lines
  to print out all the awards that our agent gets and the min, max and mean award.
- Please check out the comments in the aforementioned files.
- The trained policy is stored under the `policies/default_policy/` folder.
- We directly called our reinforcement learning framework to train the agent instead of
  implementing the `train()` method in `RedAgent.py`
- `CybORG` is unchanged.

## Evaluation

Here are the steps to evaluate the performance of our policy.

### Method 1 - Provided Docker Image

We have prepared a Docker image for you. You can get it from Docker Hub.

```bash
# Pull image from Docker Hub
docker pull wtongze/cse233

# Run `red_evaluation.py` to evaluate the performance of our agent
docker run --rm wtongze/cse233
```

### Method 2 - Build Docker Image from this Repo

```bash
# Build docker image
docker build https://github.com/wtongze/cse233.git -t wtongze/cse233

# Run `red_evaluation.py` to evaluate the performance of our agent
docker run --rm wtongze/cse233
```

## Installation

### From Source

**Our project requires Python 3.10**

```bash
# !!! Python 3.10 Only !!!

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Run our customized installation script
bash install.sh
```

The project uses the simulator [CybORG](https://github.com/cage-challenge/cage-challenge-2/tree/main).
We also uses [rllib](https://docs.ray.io/en/releases-2.2.0/rllib/index.html) and [PyTorch](https://pytorch.org/) as the
foundation to train our agent.

## Instructions

### Evaluate Policy Performance

```bash
python3 red_evaluation.py
```

### Train our agent

**An Nvidia GPU is strongly recommended to have in order to train our model**

```bash
python3 red_train.py
```

### Contribution

Please check out the Git commits for contribution details.

- `wtongze` - Tongze Wang
- `ArberSephirotheca` - Zheyuan Chen
