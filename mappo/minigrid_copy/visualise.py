from mappo.utils import FlatImageGrid
from mappo.minigrid_copy.utils.agent import Agent
from mappo.minigrid_copy.utils.penv import get_obss_preprocessor
import minigrid
from minigrid.wrappers import ImgObsWrapper
import gymnasium as gym_ # the gym environment has moved to gymnasium for minigrid
import numpy as np
import torch


# create a simple empty partially observable 8x8 grid
env = gym_.make('MiniGrid-DoorKey-5x5-v0', render_mode='human')

env = ImgObsWrapper(env)

obs_space, preprocess_obs = get_obss_preprocessor(env.observation_space)

# load the ppo model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

obs, info = env.reset()
dir = '/home/thomas/ai_projects/MAS_MT_RL/mappo/minigrid_copy/tmp/ppo'
agent = Agent(obs_space=obs_space, action_space=env.action_space, 
    device=device, preprocess_obs=preprocess_obs, use_memory=False, argmax=False)

for _ in range(20):
    observation, info = env.reset()
    while True:
        action = agent.get_action(obs)

        obs, reward, done, trunc, _ = env.step(action)

        if done or trunc or env.window.closed:
            break

    

