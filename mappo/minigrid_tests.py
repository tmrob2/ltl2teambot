# minigrid works with gym 0.26
# rware does not support gym 0.26 so this is something to be aware of up front

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from utils import FlatImageGrid
import minigrid
import gymnasium as gym_ # the gym environment has moved to gymnasium for minigrid
import numpy as np

# create a simple empty partially observable 8x8 grid
env = gym_.make('MiniGrid-Empty-8x8-v0')

print("observation space: ", env.observation_space.shape)
observation, info = env.reset()

# this is the deafault partially observable environment. 
#print("state: ", observation) 

# Flat observation wrapper will give us a single tensor with the direction 
# and all objects encoded without the NLP mission string.
from minigrid.wrappers import FlatObsWrapper

print("image shape", observation["image"].shape)

from functools import reduce
import operator

print(reduce(operator.mul, observation["image"].shape, 1))

print(observation["direction"])

print("flat obs: ", np.append(observation["image"].flatten(), observation["direction"]))

# use a flat image created from utils

env = FlatImageGrid(env)


obs_flat, info = env.reset()

print("flat observation", obs_flat)
obs_shape = env.observation_space.shape[0]
print("observation space", obs_shape)

# Now with the flat observation we want to input it into a test LSTM layer

import torch
import torch.nn as nn
import torch.nn.functional as F
lstm = nn.LSTM(obs_shape, 256, batch_first=True)

# convert the observation to a tensor

obs_ = torch.tensor(obs_flat, dtype=torch.float32)
hxs = torch.zeros(1, 256, dtype=torch.float32, device= "cpu")
cxs = torch.zeros(1, 256, dtype=torch.float32, device="cpu")
x, (hxs, cxs) = lstm(obs_.unsqueeze(0), (hxs, cxs))

print(x)










