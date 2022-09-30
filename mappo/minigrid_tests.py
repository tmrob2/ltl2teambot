# minigrid works with gym 0.26
# rware does not support gym 0.26 so this is something to be aware of up front


import minigrid
import gym

# create a simple empty partially observable 8x8 grid
env = gym.make('MiniGrid-Empty-8x8-v0')

print("observation space: ", env.observation_space.shape)
observation, info = env.reset()

# this is the deafault partially observable environment. 
print("state: ", observation) 

# Flat observation wrapper will give us a single tensor with the direction 
# and all objects encoded without the NLP mission string.
from minigrid.wrappers import FlatObsWrapper

print("image shape", observation["image"].shape)

from functools import reduce
import operator

print(reduce(operator.mul, observation["image"].shape, 1))





