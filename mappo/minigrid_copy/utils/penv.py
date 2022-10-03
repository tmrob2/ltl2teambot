from multiprocessing import Process, Pipe
import gymnasium as gym
from mappo.minigrid_copy.utils.dictlist import DictList
import torch
import numpy as np
import minigrid
from minigrid.wrappers import ImgObsWrapper

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            if terminated or truncated:
                obs, info = env.reset()
            conn.send((obs, reward, terminated, truncated, info))
        elif cmd == "reset":
            obs, info = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        local_obs, _ = self.envs[0].reset()
        results = [local_obs] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, trunc,  info = self.envs[0].step(actions[0])
        if done or trunc:
            obs, info = self.envs[0].reset()
        results = zip(*[(obs, reward, done, trunc, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError


def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return DictList({
                "image": preprocess_images(obss, device=device)
            })

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss

def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = np.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env = ImgObsWrapper(env)
    env.reset(seed=seed)
    return env