from mappo.algorithms.ma_mo_ppo import PPO
import teamgrid
import gym
from gym import register
from mappo.utils.ma_penv import make_env
import time
from mappo.utils.storage import get_txt_logger, synthesize
from mappo.networks.mo_ma_ltlnet import AC_MA_MO_LTL_Model
from torch.utils.tensorboard import SummaryWriter
import torch
torch.manual_seed(0)
from collections import deque
import numpy as np
np.random.seed(1234)

register(
    'MA-LTL-Empty-v0', 
    entry_point='mappo.envs:LTLSimpleMAEnv',
)

# construct a set of environments

writer = SummaryWriter()

# TODO introduce some args to continue training a model

num_procs = 10
num_frames_per_proc = 128
num_frames = num_frames_per_proc * num_procs
num_tasks = 2
num_agents = 2
env = ""
load_model = False

envs = []
for i in range(num_procs):
    envs.append(make_env("MA-LTL-Empty-v0", 1234 + 10000 * i))


lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#obs_space, preprocess_obs = get_obss_preprocessor(envs[0].observation_space)

model = AC_MA_MO_LTL_Model(envs[0].action_space, 2, use_memory=True)
if load_model:
    model.load_models()
model.to(device)

# we have to instantiate mu quite early because it will be used to compute
# the task weighted rewards in the environment
#mu = torch.tensor(
#    np.array([1 / num_tasks] * num_tasks * num_agents).reshape(num_agents, num_tasks), 
#    device=device, dtype=torch.float
#)
mu = torch.tensor(np.array([[0., 1.], [1., 0.]]), device=device, dtype=torch.float)
ppo = PPO(envs, model, num_agents, num_tasks + 1, device, mu=mu.cpu().numpy())

txt_logger = get_txt_logger()
model_dir = '/home/thomas/ai_projects/MAS_MT_RL/mappo/tmp/ppo'

frames = 10000000  #status["num_frames"]
update = 0 # status["update"]
num_frames = 0
start_time = time.time()
#best_score = 0
best_scores = [[0.] * (num_tasks + 1)] * num_agents
score_history = deque(maxlen=100)
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


while num_frames < frames and any(i < 0.95 for a in best_scores for i in a):
    # Update model parameters

    update_start_time = time.time()
    exps, logs1, ini_values = ppo.collect_experiences(mu, 0.95, 0.95)
    logs2 = ppo.update_parameters(exps)
    logs = {**logs1, **logs2}
    #print(logs)
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % 1 == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = synthesize(logs["return_per_episode"], multi_obj=True, n_agents=num_agents)
        #rreturn_per_episode = synthesize(logs["reshaped_return_per_episode"], multi_obj=True)
        num_frames_per_episode = synthesize(logs["num_frames_per_episode"])
        
        # for each agent check the progress against the recorded best score
        for i in range(num_agents):
            if any(x > best_scores[i][k]  for k, x in enumerate(return_per_episode['mean'][i])):
                best_scores[i] = return_per_episode['mean'][i]
                model.save_models()  

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μ {} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        for field, value in zip(header, data):
            if "rreturn_" in field or "return" in field:
                # split the agents
                # split the cost and probabilities
                for i in range(num_agents):
                    for k in range(num_tasks + 1):
                        if k == 0:
                            writer.add_scalar(field+f"_agent_{i}_cost", value[i][k], num_frames)
                        else:
                            writer.add_scalar(field+f"_agent_{i}_task_{k}", value[i][k], num_frames)
            else:
                writer.add_scalar(field, value, num_frames)