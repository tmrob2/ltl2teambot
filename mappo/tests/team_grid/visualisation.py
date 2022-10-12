from mappo.tests.team_grid.ma_mo_agent import Agent
import teamgrid
import gym # the gym environment has moved to gymnasium for minigrid
from gym import register
import numpy as np
import torch
import time

register(
    'MA-LTL-Empty-v0', 
    entry_point='mappo.envs:LTLSimpleMAEnv',
)

# create a simple empty partially observable 8x8 grid
env = gym.make('MA-LTL-Empty-v0')

# load the ppo model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

obs, info = env.reset()
dir = '/home/thomas/ai_projects/MAS_MT_RL/mappo/tmp/ppo'
agent = Agent(action_space=env.action_space, num_agents=2, num_tasks=2,
    device=device, preprocess_obs=None, use_memory=True, argmax=False)

mu = torch.tensor(np.array([[0., 1.], [1., 0.]]), device=device, dtype=torch.float)
env.update(mu.cpu().numpy())
for _ in range(20):
    observation, info = env.reset()
    while True:
        action = agent.get_action(obs)

        obs, reward, done, trunc, _ = env.step(action)
        env.render("human")
        time.sleep(0.3)
        if all(done) or all(trunc):
            break