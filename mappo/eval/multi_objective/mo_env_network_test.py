from gymnasium import register
import gymnasium as gym
from mappo.utils.penv import ParallelEnv, make_env
from mappo.utils.dictlist import DictList
from mappo.utils.storage import get_txt_logger, synthesize

print(
"""
---------------------------------------\n
TEST: Testing the environment registration and setup\n
---------------------------------------\n
"""
)
register(
    'MOLTLTest-6x6-v0', 
    entry_point='mappo.tests.multi_objective:MOLTLTestEnv'
)

env = gym.make("MOLTLTest-6x6-v0")
obs, _ = env.reset()

print("complete\n")

print("Running the environment for 10 step...")

for _ in range(10):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())

print("\ncomplete")
# input some data into the network to check that it is correct.

# make a new network
from mappo.networks.moltlnet import AC_MO_LTL_Model
import torch
torch.manual_seed(0)
import numpy
numpy.random.seed(1234)
print(
"""
---------------------------------------\n
TEST: Constructing MO AC Arch\n
---------------------------------------\n
"""
)

model = AC_MO_LTL_Model(env.action_space, 2, use_memory=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


print("complete\n")

print(("testing one feed-forward call to the recurrent architecture"))

# initialise the memory cell for the LSTM
memory = torch.zeros(1, model.memory_size, device=device)
obs_ = torch.tensor(numpy.array([obs]), device=device, dtype=torch.float)
# test the forward propogation through the network
dist, value, memory = model(obs_, memory)
print("sampled action", dist.sample(),"\n")
print("value output shape", value.shape,"\n")


# Now, using penv, with only one env input, compute the multi-objective feed forward
# for PPO

print(
"""
---------------------------------------\n
TEST: Running one iteration of the batch collection\n
---------------------------------------\n
"""
)
num_procs = 5
num_frames_per_proc = 128
num_frames = num_frames_per_proc * num_procs
num_tasks = 2
num_agents = 1

def process_obs(obs, device=None):
    return torch.tensor(numpy.array(obs), device=device, dtype=torch.float)

envs = []
for i in range(num_agents * num_procs):
    envs.append(make_env("MOLTLTest-6x6-v0", 1234 + 10000, img_only=False))

env = ParallelEnv(envs)

shape = (num_frames_per_proc,  num_procs * num_agents)
mo_shape = (num_frames_per_proc, num_procs * num_agents, num_tasks + 1)

obs = env.reset()
obss = [None]*(shape[0])
if model.recurrent:
    memory = torch.zeros(shape[1], model.memory_size, device=device)
    memories = torch.zeros(*shape, model.memory_size, device=device)
mask = torch.ones(shape[1], device=device)
actions = torch.zeros(*shape, device=device, dtype=torch.int)
masks = torch.zeros(*shape, device=device)
values = torch.zeros(*mo_shape, device=device)
rewards = torch.zeros(*mo_shape, device=device)
advantages = torch.zeros(*mo_shape, device=device)
log_probs = torch.zeros(*shape, device=device)

processed_obs = process_obs(obs, device)

with torch.no_grad():
    dist, value, memory = model(processed_obs, memory)

action = dist.sample()

obs, reward, done, trunc, _ = env.step(action.cpu().numpy())

obss[0] = obs
memories[0] = memory
masks[0] = mask
done = torch.tensor(done, device=device, dtype=torch.float)
trunc = torch.tensor(trunc, device=device, dtype=torch.float)
max_done_or_trunc = torch.max(done, trunc)
mask = 1 - max_done_or_trunc
actions[0] = action
values[0] = value

rewards[0] = torch.tensor(reward, device=device)
log_probs[0] = dist.log_prob(action)


print(
"""
---------------------------------------\n
TEST: Testing one batch collection\n
---------------------------------------\n
"""
)

print("initialising the log values\n")

mo_log_shape = (num_procs * num_agents, num_tasks + 1)

log_episode_return = torch.zeros(*mo_shape, device=device)
log_episode_reshaped_return = torch.zeros(*mo_shape, device=device)
log_episode_num_frames = torch.zeros(num_procs * num_agents, device=device)
log_done_counter = 0
log_return = [0] * num_procs
log_reshaped_return = [0] * num_procs
log_num_frames = [0] * num_procs
discount = 0.99
gae_lambda = 0.95

for i in range(num_frames_per_proc):

    preprocessed_obs = process_obs(obs, device=device)

    with torch.no_grad():
        dist, value, memory_ = model(preprocessed_obs, memory)
    
    obs_, reward, done, trunc, _ = env.step(action.cpu().numpy())

    obss[i] = obs
    obs = obs_

    memories[i] = memory
    memory = memory_
    masks[i] = mask
    done = torch.tensor(done, device=device, dtype=torch.float)
    trunc = torch.tensor(trunc, device=device, dtype=torch.float)
    max_done_or_trunc = torch.max(done, trunc)
    mask = 1 - max_done_or_trunc

    actions[i] = action
    values[i] = value

    rewards[i] = torch.tensor(reward, device=device)
    log_probs[i] = dist.log_prob(action)

    log_episode_return += torch.tensor(reward, device=device, dtype=torch.float)
    log_episode_reshaped_return += rewards[i]
    log_episode_num_frames += torch.ones(num_procs * num_agents, device=device)

    for i, done_ in enumerate(done):
        if done_:
            log_done_counter += 1
            log_return.append(log_episode_return[i].item())
            log_reshaped_return.append(log_episode_reshaped_return[i].item())
            log_num_frames.append(log_episode_num_frames[i].item())

    log_episode_return *= mask.unsqueeze(1)
    log_episode_reshaped_return *= mask.unsqueeze(1)
    log_episode_num_frames *= mask

print(
"""
---------------------------------------\n
TEST: Testing advantage function\n
---------------------------------------\n
"""
)

print("computing the next value..\n")
with torch.no_grad():
    _, next_value, _ = model(preprocessed_obs, memory * mask.unsqueeze(1))

print("computing the advantages...\n")
for i in reversed(range(num_frames_per_proc)):
    next_mask = masks[i + 1] if i < num_frames_per_proc - 1 else mask
    next_value = values[i + 1] if i < num_frames_per_proc - 1 else next_value
    next_advantage = advantages[i + 1] if i < num_frames_per_proc - 1 else torch.zeros(num_tasks + 1, dtype=float, device=device)

    delta = rewards[i] + discount * next_value * next_mask.unsqueeze(1) - values[i]
    advantages[i] = delta + discount * gae_lambda * next_advantage * next_mask.unsqueeze(1)

print("complete")

# we need to multiply the advantages by the H values
c = 0.95
e = 0.95
mu = torch.tensor(numpy.array([[0.5, 0.5]]), 
    device=device, dtype=torch.float)
def df(x):
    #if x <= c: 
    #    return 2*(c - x)
    #else:
    #    return 0.0
    return torch.where(x <= c, 2*(c - x), 0.0)


def dh(x):
    #if x <= e:
    #    return 2 * (e - x)
    #else:
    #    return 0.0
    return torch.where(x <= e, 2 * (e - x), 0.0)


def computeH(X, mu, agent):
    # todo need to loop over each environment
    _, _, y = X.shape
    H_ = [] 
    H_.append(df(X[agent, :, 0]))
    for j in range(1, y):
        #print(X[:, k, j - 1])
        H_.append(
            dh(torch.sum(mu[:, j - 1].unsqueeze(1) * X[:, :, j], dim=0)) * mu[agent, j - 1]
        )

    H = torch.stack(H_, 1)
    return H

# TODO this needs to be done before the reshape so that we get the ini values for each of the
# environment trajectories.
ini_values = values[0, :] # the 0th value for all procs
ini_values = torch.reshape(values.transpose(0, 1), 
    (num_agents, num_procs, num_frames_per_proc, num_tasks + 1))[:, :, 0, :]

print("values shape", ini_values.shape)
print("ini_values", ini_values)
print()

H = []
for i in range(num_agents):
    Hi = computeH(ini_values, mu, i)
    H.append(Hi)

H = torch.stack(H)

H_ = H.reshape(num_agents * num_procs, 3)
#advantages = advantages.transpose(0, 1)
stack = []
for k in range(num_agents * num_procs):
    stack.append(torch.matmul(advantages[:, k, :], H_[k, :]))
AH = torch.stack(stack)

# The shape is (A . P ) x T -> (A . P . T): then update the model after all agents? or on each agent that is the question

# then reshape back into agents, procs, batch length

# TODO multiply the H values with the advantages


#computeH(ini_values, ini_values, mu, num_agents, num_procs)
print(
"""
---------------------------------------\n
TEST: Testing multi-objective output shapes\n
---------------------------------------\n
"""
)
exps = DictList()
exps.obs = [obss[i][j] for j in range(num_procs) for i in range(num_frames_per_proc)]
# T x P x D -> P x T x D -> (P.T) x D remains the same as single objective.
exps.memory = memories.transpose(0, 1).reshape(-1, *memories.shape[2:])
exps.mask = masks.transpose(0, 1).reshape(-1).unsqueeze(1)
exps.action = actions.transpose(0, 1).reshape(-1)

# values T x P x O where O is the objective dim
print("testing values reshape...\n")
print("values shape", values.shape)
print("values reshaped shape", values.transpose(0, 1).reshape(-1, *values.shape[2:]).shape)
exps.value = values.transpose(0, 1).reshape(-1, *values.shape[2:])
exps.reward = rewards.transpose(0, 1).reshape(-1, *rewards.shape[2:])

print(
"""
---------------------------------------\n
TEST: Testing base actor-critic data collection\n
---------------------------------------\n
"""
)
from mappo.algorithms.mo_base import BaseAlgorithm

base = BaseAlgorithm(
    envs, model, device, num_agents, num_tasks + 1, 
    128, 0.99, 0.001, 0.95, 0.01, 0.5, 0.5, 4, None
)

import time

update = 0

start_time = time.time()
update_start_time = time.time()
exps, logs1, ini_values = base.collect_experiences(mu, 0.95, 0.95)

from mappo.algorithms.mo_ppo import PPO

ppo = PPO(envs, model, num_agents, num_tasks + 1, device)

logs2 = ppo.update_parameters(exps)

logs = {**logs1, **logs2}
print(logs)

update_end_time = time.time()

num_frames += logs["num_frames"]
update += 1

print(
"""
---------------------------------------\n
TEST: Testing outputs\n
---------------------------------------\n
"""
)

fps = logs["num_frames"] / (update_end_time - update_start_time)
print(fps)
duration = int(time.time() - start_time)

return_per_episode = synthesize(logs["return_per_episode"], multi_obj=True)
rreturn_per_episode = synthesize(logs["reshaped_return_per_episode"], multi_obj=True)
num_frames_per_episode = synthesize(logs["num_frames_per_episode"])

print("return per episode", return_per_episode)
print("rreturn per episode", rreturn_per_episode)
print("num frames per episode", num_frames_per_episode)
print("mean episode score", return_per_episode["mean"])

txt_logger = get_txt_logger()

header = ["update", "frames", "FPS", "duration"]
data = [update, num_frames, fps, duration]
header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
data += rreturn_per_episode.values()
header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
data += num_frames_per_episode.values()
header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
numpy.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
txt_logger.info(
    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {} {} {} {} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
    .format(*data))
