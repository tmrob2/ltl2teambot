# minigrid works with gym 0.26
# rware does not support gym 0.26 so this is something to be aware of up front

#import os

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

from cgitb import text
from mappo.tests.utils import FlatImageGrid
import minigrid
import gymnasium as gym_ # the gym environment has moved to gymnasium for minigrid
import numpy as np
import re
import numpy

# create a simple empty partially observable 8x8 grid
env = gym_.make('MiniGrid-Empty-8x8-v0', render_mode="human")

#envs = gym_.vector.make('MiniGrid-Empty-8x8-v0', num_envs=3, shared_memory=False)


print("observation space: ", env.observation_space.shape)
observation, info = env.reset()

# this is the deafault partially observable environment. 
#print("state: ", observation) 

# Flat observation wrapper will give us a single tensor with the direction 
# and all objects encoded without the NLP mission string.
from minigrid.wrappers import FlatObsWrapper

print("image shape", observation["image"].shape)
mission = observation["mission"]

print("observation mission", observation["mission"])

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

#print(x)

#for _ in range(100):
#    obs, reward, done, trunc, info = env.step(env.action_space.sample())
#    env.render()

def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]


vocab = Vocabulary(100)

input_text_tensor = preprocess_texts(mission, vocab)

print("input shape", input_text_tensor.shape)


text_embedding=nn.Embedding(input_text_tensor.shape[0], 32)
x = text_embedding(input_text_tensor)
print("text embedding", x.shape)










