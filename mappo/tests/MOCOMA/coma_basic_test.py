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
mu = torch.tensor(np.array([[0., 1.], [1., 0.]]), device=device, dtype=torch.float)

env = gym.make('MA-LTL-Empty-v0')
env.update(mu.cpu().numpy())
obs, _ = env.reset()


print(
"""
---------------------------------------\n
TEST: Constructing COMA network\n
---------------------------------------\n
"""
)

num_agents = 2
num_tasks = 2
num_actions = env.action_space.n
c = 0.95
e = 0.95

# we are particularly intested in the Critic network because

from mappo.networks.coma_network import COMA

print(
"""
---------------------------------------\n
TEST: Test the Actor(s) network\n
---------------------------------------\n
"""
)

coma = COMA(num_agents, env.action_space, num_tasks, env, device)
print("device", coma.device)

print("observation shape", np.array(obs).shape)

# sample some actions
actions = [env.action_space.sample() for _ in range(num_agents)]
actions = torch.tensor(np.array(actions), device=device, dtype=torch.float)
#agent_id = torch.tensor(np.array([1]), device=device, dtype=torch.float)
coma.episode_memory.observations.append(obs)
pi, actions, memory = coma.act(obs)
next_obs, reward, done, trunc, _ = coma.env.step(actions.cpu().numpy())
coma.episode_memory.pi.append(pi)
coma.episode_memory.actions.append(actions)
coma.episode_memory.reward.append(reward)
coma.episode_memory.memories.append(memory)

obs, actions, pi, rewards, done, memory = coma.episode_memory.get_memory()

print(
"""
---------------------------------------\n
TEST: Test the Critic network\n
---------------------------------------\n
"""
)

epsisode_lenth = obs.shape[0]

t = 0

ini_values = coma.get_ini_values()

for i in range(num_agents):
    agent_id = torch.tensor(np.array([i]), device=coma.device, dtype=torch.float)
    H = coma.computeH(ini_values, mu, i, c, e).detach()
    qt_target, coma.cmem = coma.critic(obs[t], actions[t], agent_id, coma.cmem)
    qt_target = qt_target.detach()

    action_taken = actions.type(torch.long)[t][i] # should be a scalar  
    baseline = torch.sum(pi[t][i].unsqueeze(1) * qt_target).detach()

    qt_taken_target = qt_target[action_taken]
    advantage = qt_taken_target - baseline  # should all be multi-objective

    mod_advantage = torch.matmul(advantage, H)

    # multiply advantage by H

    log_pi = torch.log(pi[t][i] * mod_advantage)

# when the episode is computed then calculate actor loss and critic loss
        

print(
"""
---------------------------------------\n
TEST: Collect a COMA episode\n
---------------------------------------\n
"""
)

exps = coma.collect_episode_trajectory(seed=1234)

print("agent 0 observations shape", exps[0].observations.shape)


print(
"""
---------------------------------------\n
TEST: Test one COMA training step\n
---------------------------------------\n
"""
)

coma.train(exps, mu.detach(), c, e)