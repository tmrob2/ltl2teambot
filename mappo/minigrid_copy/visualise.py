from mappo.utils import FlatImageGrid
from mappo.minigrid_copy.utils.agent import Agent
import minigrid
from minigrid.wrappers import ImgObsWrapper
import gymnasium as gym_ # the gym environment has moved to gymnasium for minigrid
import numpy as np
import torch


# create a simple empty partially observable 8x8 grid
env = gym_.make('MiniGrid-Empty-8x8-v0', render_mode='human')

env = ImgObsWrapper(env)

# load the ppo model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

observation, info = env.reset()
dir = '/home/thomas/ai_projects/MAS_MT_RL/mappo/minigrid_copy/tmp/ppo'
agent = Agent(obs_space=env.observation_space, action_space=env.action_space, file=dir, device=device)

while True:
    action = agent.get_action(observation)

    obs, reward, done, trunc, _ = env.step(action)

    if done or trunc or env.window.closed:
        break

    

