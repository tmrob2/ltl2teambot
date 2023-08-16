from mappo.tests.team_grid.ma_mo_agent import Agent
import teamgrid
import gym # the gym environment has moved to gymnasium for minigrid
from gym import register
import numpy as np
import torch
import time
import os


#register(
#    'MA-LTL-Empty-v0', 
#    entry_point='mappo.envs:LTLSimpleMAEnv',
#)

num_agents = 2
num_tasks = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#register(
#    'LTLA2T4-v0', 
#    entry_point='mappo.envs:LTL4TA2',
#)

register(
    'LTLA2T4-v0', 
    entry_point='mappo.envs:LTLSimpleMAEnv',
)

kappa = torch.ones(num_agents, num_tasks, device=device, dtype=torch.float)
#mu = torch.tensor(np.array([[1., 0.], [0., 1.]]), device=device, dtype=torch.float)

#alloc_optim = torch.optim.SGD([kappa], lr=0.1)

alloc_layer = torch.nn.Softmax(dim=0)
mu = alloc_layer(kappa)

# create a simple empty partially observable 8x8 grid
env = gym.make("LTLA2T4-v0")
seed = 1234

# load the ppo model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env.seed(seed)
obs, info = env.reset()
#dir = '/home/thomas/ai_projects/MAS_MT_RL/mappo/tmp/ppo'
#agent = Agent(action_space=env.action_space, num_agents=num_agents, num_tasks=num_tasks,
#    device=device, preprocess_obs=None, use_memory=True, argmax=False, deep_model=True)

#mu = torch.tensor(np.array([[0., 1.], [1., 0.]]), device=device, dtype=torch.float)
env.update(mu.cpu().numpy())
for _ in range(20):
    env.seed(seed)
    observation, info = env.reset()
    while True:
        #action = agent.get_action(obs)

        obs, reward, done, trunc, _ = env.step([6, 6])
        env.render("human")
        time.sleep(0.3)
        if all(done) or all(trunc):
            break