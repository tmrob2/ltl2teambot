import teamgrid
from gym import register
import gym
import numpy as np
import torch
from mappo.utils.storage import truncate, get_txt_logger
from mappo.networks.coma_network import COMA
from collections import deque

register(
    'MA-LTL-Empty-v0', 
    entry_point='mappo.envs:LTLSimpleMAEnv',
)

txt_logger = get_txt_logger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mu = torch.tensor(np.array([[0., 1.], [1., 0.]]), device=device, dtype=torch.float)

env = gym.make('MA-LTL-Empty-v0')
env.update(mu.cpu().numpy())
episode = 0
n_episodes = 10000

num_agents = 2
num_tasks = 2
num_actions = env.action_space.n
c = 0.95
e = 0.95

total_rewards = deque(maxlen=100)

coma = COMA(num_agents, env.action_space, num_tasks, env, device)

while episode < n_episodes:

    coma.collect_episode_trajectory(seed=1234)

    rewards, aloss, closs = coma.train(mu.detach(), c, e)

    total_rewards.append([y for x in rewards for y in x])
    ave_rewards = np.mean(total_rewards, 0)

    headers = ["episode", "duration"]
    data = [episode, coma.frames]
    headers += ["rewards"]
    data += [list(map(lambda n: truncate(n, 2), ave_rewards))]
    data += [[list(map(lambda n: truncate(n, 3), x)) for x in rewards]]
    headers += ["actor loss"]
    data += [list(map(lambda n: truncate(n, 3), aloss))]
    headers += ["critic loss"]
    data += [closs]

    txt_logger.info("U {} | F {:06} | R:μ {} |r:μ {} | pL {} | C {:.3F}".format(*data))
    episode += 1

