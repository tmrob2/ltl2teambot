from mappo.tests.multi_objective.mo_agent import Agent
import minigrid
import gymnasium as gym_ # the gym environment has moved to gymnasium for minigrid
from gymnasium import register
import numpy as np
import torch

register(
    'MOLTLTest-6x6-v0', 
    entry_point='mappo.tests.multi_objective:MOLTLTestEnv'
)


# create a simple empty partially observable 8x8 grid
env = gym_.make('MOLTLTest-6x6-v0', render_mode='human')

# load the ppo model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

obs, info = env.reset()
dir = '/home/thomas/ai_projects/MAS_MT_RL/mappo/tmp/ppo'
agent = Agent(action_space=env.action_space, num_agents=1, num_tasks=2,
    device=device, preprocess_obs=None, use_memory=True, argmax=False)

for _ in range(20):
    observation, info = env.reset()
    while True:
        action = agent.get_action(obs)

        obs, reward, done, trunc, _ = env.step(action)

        if done or trunc or env.window.closed:
            break