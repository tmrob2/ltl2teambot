from mappo.algorithms.ma_mo_ppo import PPO
from mappo.algorithms.ma_mo_base import compute_alloc_loss
import teamgrid
import gym
from gym import register
from mappo.utils.ma_penv import make_env
import time
from mappo.utils.storage import get_txt_logger, synthesize, truncate
from mappo.networks.mo_ma_ltlnet_deep import AC_MA_MO_LTL_Model
from torch.utils.tensorboard import SummaryWriter
import torch
torch.manual_seed(0)
from collections import deque
import numpy as np
np.random.seed(1234)

register(
    'LTL10TA5-v0', 
    entry_point='mappo.envs:LTL10TA5',
)

# construct a set of environments

writer = SummaryWriter("runs/T10A5")

# TODO introduce some args to continue training a model

num_procs = 4
num_frames_per_proc = 128
num_frames = num_frames_per_proc * num_procs
num_tasks = 10
num_agents = 5
env = ""
load_model = False
seed = 1234

envs = []
for i in range(num_procs):
    envs.append(make_env("LTL10TA5-v0", seed))


lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#obs_space, preprocess_obs = get_obss_preprocessor(envs[0].observation_space)

model = AC_MA_MO_LTL_Model(envs[0].action_space, num_tasks, use_memory=True)
if load_model:
    model.load_models()
model.to(device)

# we have to instantiate mu quite early because it will be used to compute
# the task weighted rewards in the environment
#mu = torch.tensor(
#    np.array([1 / num_tasks] * num_tasks * num_agents).reshape(num_agents, num_tasks), 
#    device=device, dtype=torch.float
#)
# construct the original parmas tensor for kappa

# we also need the loss function for updating kappa
lr2 = 0.01
kappa = torch.randn(num_agents, num_tasks, device=device, dtype=torch.float).requires_grad_()
#mu = torch.tensor(np.array([[1., 0.], [0., 1.]]), device=device, dtype=torch.float)

alloc_optim = torch.optim.Adam([kappa], lr=0.01)

alloc_layer = torch.nn.Softmax(dim=0)
mu = alloc_layer(kappa)
ppo = PPO(envs, model, num_agents, num_tasks + 1, device, mu=mu.detach().cpu().numpy(), 
    seed=seed, recurrence=16)
ppo.update_environments(mu.detach().cpu().numpy())

txt_logger = get_txt_logger()
model_dir = '/home/thomas/ai_projects/MAS_MT_RL/mappo/tmp/ppo'

frames = 10000000  #status["num_frames"]
update = 0 # status["update"]
num_frames = 0
start_time = time.time()
best_score = 0.
#best_scores = [[0.] * (num_tasks + 1)] * num_agents
score_history = deque(maxlen=100)
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

task_target = torch.full((num_procs, num_tasks), 0.95, device=device)
cost_target = torch.full((num_procs, 1), 0.95, device=device)
loss = torch.nn.MSELoss(reduction='mean')
determ_alloc = np.mean(np.max(mu.detach().cpu().numpy(), 0))
mean_agent_costs = 0.
while num_frames < frames and (best_score < 0.95 or mean_agent_costs < 0.95 or determ_alloc < 0.95):#any(i < 0.95 for a in best_scores for i in a):
    
    # Update model parameters

    update_start_time = time.time()
    exps, logs1, ini_values = ppo.collect_experiences(mu.detach(), 0.95, 0.95)
    logs2 = ppo.update_parameters(exps)

    task_input = [torch.sum(torch.stack([mu[i, j] * ini_values[:, i, j + 1] for i in range(2)]), 0) for j in range(num_tasks)]
    task_input = torch.stack(task_input).transpose(0, 1)

    cost_input = torch.min(ini_values[:, :, 0], dim=1)[0]

    # compute the task allocation loss
    alloc_optim.zero_grad()
    #alloc_loss = compute_alloc_loss(ini_values, mu, 0.95)
    alloc_loss = loss(task_input, task_target) + 10. * loss(cost_input, cost_target)
    alloc_loss.backward()
    alloc_optim.step()
    
    ## Get the new mu
    mu = alloc_layer(kappa)
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
        #batch_score = np.mean(return_per_episode['mean'])
        mean_return = np.array(return_per_episode['mean'])
        output = np.max(mean_return[:, 1:], 0)
        mean_agent_costs = np.min(mean_return[:, 0])
        #average_cost = np.mean(output[:, 0])
        #average_task_score = np.mean(mu.detach().cpu().numpy() * output[:, 1:])
        #batch_score = np.mean([average_cost, average_task_score])
        batch_score = np.mean(np.concatenate([[np.min(mean_return[:, 0])], np.max(mean_return[:, 1:], 0)]))
        if batch_score > best_score:
            best_score = batch_score
            model.save_models()
        #for i in range(num_agents):
        ##if any(x > best_scores[i][k]  for k, x in enumerate(return_per_episode['mean'][i])):
        ##    best_scores[i] = return_per_episode['mean'][i]
        #model.save_models()  
        

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        print_data = [update, num_frames, fps, duration]
        header += ["return_" + key for key in return_per_episode.keys()]
        data += [np.concatenate([[np.min(mean_return[:, 0])], np.max(mean_return[:, 1:], 0)]).tolist()]
        print_data += [np.concatenate([[np.min(mean_return[:, 0])], np.max(mean_return[:, 1:], 0)]).tolist()]
        #header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        #data += num_frames_per_episode.values()
        header += ["mu"]
        allocation = list(map(lambda n: truncate(n, 2), mu.reshape(-1).detach().cpu().numpy().tolist()))
        data += [allocation]
        print_data += [np.argmax(mu.detach().cpu().numpy(), 0)]
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
        print_data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]


        #txt_logger.info(
        #    "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μ {} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
        #    .format(*data))

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μ {} | μ: {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*print_data))
        #txt_logger.info(
        #    f"allocation {allocation}"
        #)

        #header += ["return_" + key for key in return_per_episode.keys()]
        #data += return_per_episode.values()

        for field, value in zip(header, data):
            if "rreturn_" in field or "return" in field:
                # split the agents
                # split the cost and probabilities
                for k in range(num_tasks + 1):
                    if k == 0:
                        writer.add_scalar(field+f"_worst_agent_cost", value[k], num_frames)
                    else:
                        writer.add_scalar(field+f"_max_task_{k}", value[k], num_frames)
            elif "mu" in field:
                for k in range(num_tasks):
                    for i in range(num_agents):
                        writer.add_scalar(field + f"_agent_{i}_task_{k}", value[num_tasks * i + k], num_frames)

            else:
                writer.add_scalar(field, value, num_frames)
    determ_alloc = np.mean(np.max(mu.detach().cpu().numpy(), 0))