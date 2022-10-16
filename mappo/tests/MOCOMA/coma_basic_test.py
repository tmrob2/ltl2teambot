import teamgrid
from gym import register
import gym
import numpy as np
import torch

register(
    'MA-LTL-Empty-v0', 
    entry_point='mappo.envs:LTLSimpleMAEnv',
)

torch.autograd.set_detect_anomaly(True)

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

## sample some actions
#actions = [env.action_space.sample() for _ in range(num_agents)]
#actions = torch.tensor(np.array(actions), device=device, dtype=torch.float)
##agent_id = torch.tensor(np.array([1]), device=device, dtype=torch.float)
#coma.episode_memory.observations.append(obs)
##pi, actions, memory = coma.act(obs)
#coma.act(obs)
#next_obs, reward, done, trunc, _ = coma.env.step(actions.cpu().numpy())
#coma.episode_memory.reward.append(reward)
##coma.episode_memory.memories.append(memory)
#
##obs, actions, pi, rewards, done, memory = coma.episode_memory.get_memory()
#obs, actions, pi, rewards, done, _ = coma.episode_memory.get_memory()


print(
"""
---------------------------------------\n
TEST: Test the Critic network\n
---------------------------------------\n
"""
)


t = 0

#ini_values = coma.get_ini_values()
#obs = obs[t].reshape(1, obs.shape[2] * num_agents)
#
#for i in range(num_agents):
#    agent_id = (torch.ones(1, device=device) * i).view(-1, 1)
#    H = coma.computeH(ini_values, mu, i, c, e).detach()
#    #qt_target, coma.cmem = coma.critic(obs[t], actions[t], agent_id, coma.cmem)
#    qt_target = coma.critic(obs, actions[t].unsqueeze(0), agent_id)
#    qt_target = qt_target.detach()
#
#    action_taken = actions.type(torch.long)[t][i] # should be a scalar  
#    baseline = torch.sum(pi[t][i].unsqueeze(1) * qt_target).detach()
#
#    qt_taken_target = qt_target[:, action_taken, :].squeeze()
#    advantage = qt_taken_target - baseline  # should all be multi-objective
#
#    mod_advantage = torch.matmul(advantage, H)
#
#    # multiply advantage by H
#
#    log_pi = torch.log(pi[t][i] * mod_advantage)
#
## when the episode is computed then calculate actor loss and critic loss
#        
#coma.episode_memory.clear()

print(
"""
---------------------------------------\n
TEST: Collect a COMA episode\n
---------------------------------------\n
"""
)

coma.collect_episode_trajectory(seed=1234)


print(
"""
---------------------------------------\n
TEST: Test one COMA training step\n
---------------------------------------\n
"""
)

coma.train(mu.detach(), c, e)