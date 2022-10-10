
from mappo.utils import FlatImageGrid
import minigrid
import gymnasium as gym_ # the gym environment has moved to gymnasium for minigrid
import numpy as np

# create a simple empty partially observable 8x8 grid
env = gym_.make('MiniGrid-Empty-8x8-v0')
env = FlatImageGrid(env)
observation, info = env.reset()

from mappo.networks.network import ACModel
import torch

print("The observation size of the network", env.observation_space.shape[0])

actor = ACModel(
    env.action_space.n, 
    input_dims=env.observation_space.shape[0],
    alpha = 0.001
)

# check that the actor knows what to do with a state
state = torch.tensor(observation, dtype=torch.float).to(actor.device)
state = state.unsqueeze(0).unsqueeze(0)
hxs = torch.zeros(1, 1, 256, dtype=torch.float32).to(actor.device)
cxs = torch.zeros(1, 1, 256, dtype=torch.float32).to(actor.device)

# input the state into the network
dist, (hxs, cxs) = actor(state, (hxs, cxs))

# Check that the distribution smapling is meaningful
print("Actor output distribution", dist) 
action = dist.sample()
print("Action", torch.squeeze(action).item())
