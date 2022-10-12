import numpy as np
from pettingzoo.mpe import simple_adversary_v2
from pettingzoo.test import api_test

env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=True)
api_test(env, num_cycles=1000, verbose_progress=False)

print("number of agents", env.num_agents)
print("observation space", env.observation_spaces)
print("action space", env.action_spaces)

env.step(env.action_space("agent_0").sample())
print(f"fagent 0 obs: {env.observe('agent_0')}, reward: {env.rewards}, terminate: {env.terminations}, info: {env.infos}")