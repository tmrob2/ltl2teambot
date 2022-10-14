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

#mu = torch.tensor(
#    np.array([1. / num_agents] * (num_agents * num_tasks)).reshape(num_agents, num_agents), 
#    dtype=torch.float,
#    device=device
#)


print(
"""
---------------------------------------\n
TEST: Testing team-grid environment\n
---------------------------------------\n
"""
)

env.update(mu.cpu().numpy())
for _ in range(200):
    obs, rewards, done, trunc, info = \
        env.step([env.action_space.sample() for _ in range(env.num_agents)])
    print(rewards)
    env.render("human")
    if all(done) or all(trunc):
        break


print(
"""
---------------------------------------\n
TEST: Testing multi-objective multiagent network\n
---------------------------------------\n
"""
)

from mappo.networks.mo_ma_ltlnet import AC_MA_MO_LTL_Model

model = AC_MA_MO_LTL_Model(env.action_space, 2, use_memory=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)



shape = (num_frames_per_proc,  num_agents)
mo_shape = (num_frames_per_proc, num_agents, num_tasks + 1)


memory = torch.zeros(shape[1], model.memory_size, device=device)

obs_ = torch.tensor(np.array(obs), device=device, dtype=torch.float)
dist, value, memory = model(obs_, memory)


print(
"""
---------------------------------------\n
TEST: Testing parallel environment setup\n
---------------------------------------\n
"""
)

def process_obs(obs, device=None):
    t = torch.tensor(np.array(obs), device=device, dtype=torch.float)
    t = t.reshape(-1, *t.shape[2:])
    return t

from mappo.utils.ma_penv import make_env, ParallelEnv

envs = []
for i in range(num_procs):
    envs.append(make_env("MA-LTL-Empty-v0", 1234 + 10000))

env = ParallelEnv(envs, mu.cpu().numpy())

shape = (num_frames_per_proc, num_procs * num_agents)
mo_shape = (num_frames_per_proc, num_procs * num_agents, num_tasks + 1)

obs = env.reset()
obss = [None] * (shape[0])


memory = torch.zeros(shape[1], model.memory_size, device=device)
memories = torch.zeros(*shape, model.memory_size, device=device)

mask = torch.ones(shape[1], device=device)
actions = torch.zeros(*shape, device=device, dtype=torch.int)
masks = torch.zeros(*shape, device=device)
values = torch.zeros(*mo_shape, device=device)
rewards = torch.zeros(*mo_shape, device=device)
advantages = torch.zeros(*shape, device=device)
log_probs = torch.zeros(*shape, device=device)

preprocessed_obs = process_obs(obs, device)

with torch.no_grad():
    dist, value, memory = model(preprocessed_obs, memory)

action = dist.sample()
action_ = action.reshape(num_procs, num_agents).cpu().numpy()
obs, reward, done, trunc, _ = env.step(action_)

# convert the observation, rewards back into its expected form 
# A x P x D -> (A * P) x D -> D x (A * P)

obs = np.array(obs).reshape(-1, obs[0][0].shape[0]).tolist()
reward = np.array(reward).reshape(-1, num_tasks + 1)

obss[0] = obs
memories[0] = memory
masks[0] = mask
done = torch.tensor(done, device=device, dtype=torch.float)
trunc = torch.tensor(trunc, device=device, dtype=torch.float)
max_done_or_trunc = torch.max(done, trunc)
mask = 1 - max_done_or_trunc
actions[0] = action
values[0] = value

rewards[0] = torch.tensor(reward, device=device)
log_probs[0] = dist.log_prob(action)

print(
"""
---------------------------------------\n
TEST: Testing update scheduler\n
---------------------------------------\n
"""
)

env.update(mu.cpu().numpy())

print(
"""
---------------------------------------\n
TEST: Testing PPO batch collection\n
---------------------------------------\n
"""
)



from mappo.algorithms.ma_mo_base import BaseAlgorithm

base = BaseAlgorithm(envs, model, device, num_agents, num_tasks + 1,
    128, 0.99, 0.001, 0.95, 0.01, 0.5, 0.5, 4, None, mu.cpu().numpy()
)

exps, logs1, ini_values = base.collect_experiences(mu, 0.95, 0.95)
print(logs1)

# The next thing to do is to test PPO now that the experiences
# have been correctly collected.


from mappo.algorithms.ma_mo_ppo import PPO

ppo = PPO(envs, model, num_agents, num_tasks + 1, device, mu=mu.cpu().numpy())

logs2 = ppo.update_parameters(exps)

logs = {**logs1, **logs2}
print(logs)