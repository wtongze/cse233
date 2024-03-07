import ray
from ray.rllib.algorithms import Algorithm

ray.init()

algo = Algorithm.from_checkpoint("checkpoints/tune/PPO_GymWrapper_71cbd_00000_0_fcnet_hiddens=1024_1024_2024-03-07_03-25-47/checkpoint_000150")

algo.train()
