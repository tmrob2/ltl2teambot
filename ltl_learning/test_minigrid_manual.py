import os
import time
import torch

from gym_env import AdversarialEnv9x9

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


MANUAL_CONTROL = False
MYOPIC = False
MODEL = "ga"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def control(letter):
    ''' Helper function to manually control the agent. '''
    if letter == 'a':   return 0 # sx
    elif letter == 'd': return 1 # dx
    elif letter == 'w': return 2 # forward
    else:               return 3 # no move


env = AdversarialEnv9x9()

obs = env.reset()
env.render('human')
for i in range(10000):
    action = control(input())
    obs, rew, done, _ = env.step(action)
    env.render('human')
    time.sleep(.25)
    if done:
        env.reset()