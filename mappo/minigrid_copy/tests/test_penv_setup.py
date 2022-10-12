from mappo.minigrid_copy.ppo import PPO
import minigrid
import gymnasium as gym_
from mappo.utils import FlatImageGrid
from mappo.minigrid_copy.utils.penv import ParallelEnv, make_env, get_obss_preprocessor

#from mappo.minigrid_copy.base_algorithm import *
from mappo.minigrid_copy.algorithm import BaseAlgorithm
from mappo.minigrid_copy.network import ACNetwork

# construct a set of environments
num_procs = 16

envs = []
for i in range(num_procs):
    envs.append(make_env('MiniGrid-Empty-8x8-v0', 1234 + 10000 * i))

lr = 0.001

obs_space, preprocess_obs = get_obss_preprocessor(envs[0].observation_space)

model = ACNetwork(obs_space, envs[0].action_space, use_memory=True)

base = BaseAlgorithm(envs, model, lr=0.001, gae_lambda=0.95, entropy_coef=0.01, 
    value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4, preprocess_obss=preprocess_obs,
    num_frames_per_proc=128, discount=0.99)

exps, logs1 = base.collect_experiences()

ppo = PPO(envs=envs, acmodel=model, device=model.device)

logs2 = ppo.update_parameters(exps)

logs = {**logs1, **logs2}
print(logs)