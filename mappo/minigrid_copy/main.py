from mappo.minigrid_copy.ppo import PPO
import minigrid
import gymnasium as gym_
from mappo.minigrid_copy.utils.penv import make_env, get_obss_preprocessor
import time
from mappo.minigrid_copy.utils.storage import get_txt_logger, synthesize
from mappo.minigrid_copy.network import ACModel
from torch.utils.tensorboard import SummaryWriter
import torch
from collections import deque
import numpy as np

# construct a set of environments

writer = SummaryWriter()

num_procs = 10
env = ""

envs = []
for i in range(num_procs):
    envs.append(make_env('MiniGrid-Fetch-8x8-N3-v0', 1234 + 10000 * i))


lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

obs_space, preprocess_obs = get_obss_preprocessor(envs[0].observation_space)

model = ACModel(obs_space, envs[0].action_space, use_memory=True)
model.to(device)

ppo = PPO(envs=envs, acmodel=model, device=device, preprocess_obss=preprocess_obs, 
    lr=lr, entropy_coef=0.01, recurrence=4)

txt_logger = get_txt_logger()
model_dir = '/home/thomas/ai_projects/MAS_MT_RL/mappo/minigrid_copy/tmp/ppo'

frames = 100000  #status["num_frames"]
update = 0 # status["update"]
num_frames = 0
start_time = time.time()
best_score = 0
score_history = deque(maxlen=100)

while num_frames < frames and best_score < 0.97:
    # Update model parameters

    update_start_time = time.time()
    exps, logs1 = ppo.collect_experiences()
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
        return_per_episode = synthesize(logs["return_per_episode"])
        rreturn_per_episode = synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = synthesize(logs["num_frames_per_episode"])
        

        score_history.append(return_per_episode["mean"])
        if np.mean(score_history) > best_score:
            best_score = np.mean(score_history)
            print("best score: ", best_score)
            # save the models
            model.save_models()  

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        for field, value in zip(header, data):
            writer.add_scalar(field, value, num_frames)

