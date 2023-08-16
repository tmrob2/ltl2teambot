import teamgrid
from gym import register
import gym
import numpy as np
import torch

register(
    'MA-LTL-Empty-v0', 
    entry_point='mappo.envs:LTLSimpleMAEnv',
)


print(
"""
---------------------------------------\n
TEST: Testing initial environment setup\n
---------------------------------------\n
"""
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mu = torch.tensor(np.array([[0.1, 0.9], [0.9, 0.1]]), device=device, dtype=torch.float)
env = gym.make('MA-LTL-Empty-v0')
env.update(mu.cpu().numpy())
obs, _ = env.reset()

print("obs\n", obs)

num_frames_per_proc = 128
num_tasks = 2
num_agents = 2
num_procs = 5

arr = np.array(obs)
print("np arr shape", arr.shape)
t = torch.tensor(arr, dtype=torch.float, device=device)
print("tensor shape", t.shape)

def process_obs(obs, device=None):
    t = torch.tensor(np.array(obs), device=device, dtype=torch.float)
    t = t.reshape(-1, *t.shape[2:])
    return t

from mappo.utils.ma_penv import make_env, ParallelEnv

envs = []
for i in range(num_procs):
    envs.append(make_env("MA-LTL-Empty-v0", 1234 + 10000))

env = ParallelEnv(envs, mu.cpu().numpy())

shape = (num_frames_per_proc, num_procs)
mo_shape = (num_frames_per_proc, num_procs, num_tasks + 1)

obs = env.reset()

print(len(obs))

# construct a series of models
from mappo.networks.mo_ma_ltlnet import AC_MA_MO_LTL_Model


models = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for i in range(num_agents):
    models.append(AC_MA_MO_LTL_Model(envs[0].action_space, 2, use_memory=True))
    models[i].to(device)


def process_obs(obs, device=None):
    t = torch.tensor(np.array(obs), device=device, dtype=torch.float)
    tup = torch.split(t.transpose(0, 1), 1)
    tup = tuple([tup[t].squeeze() for t in range(num_agents)])
    return tup


preprocessed_obs = process_obs(obs, device)

memory = torch.zeros(num_agents, shape[1], models[0].memory_size, device=device)

dist, value, memory[0] = models[0](preprocessed_obs[0], memory[0])


print(
"""
---------------------------------------\n
TEST: Testing decentralised experience collection\n
---------------------------------------\n
"""
)

from mappo.algorithms.dec_ma_mo_base import BaseAlgorithm

base = BaseAlgorithm(envs, models, device, num_agents, num_tasks + 1,
    128, 0.99, 0.001, 0.95, 0.01, 0.5, 0.5, 4, None, mu.cpu().numpy()
)

exps, logs1, ini_values = base.collect_experiences(mu, 0.95, 0.95)

from mappo.algorithms.dec_ma_mo_ppo import PPO

ppo = PPO(envs, models, num_agents, num_tasks + 1, device, mu=mu.cpu().numpy())

logs2 = ppo.update_parameters(exps)


logs = {**logs1, **logs2}

print(logs)






