from multiprocessing import Process, Pipe
import gym
from mappo.minigrid_copy.utils.dictlist import DictList
import torch
import numpy as np
import teamgrid

def worker(conn, env):
    while True:
        cmd, data, seed = conn.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            if all(terminated) or all(truncated):
                if seed:
                    env.seed(seed)
                obs, info = env.reset()
            conn.send((obs, reward, terminated, truncated, info))
        elif cmd == "reset":
            if seed:
                env.seed(seed)
            obs, info = env.reset()
            conn.send(obs)
        elif cmd == "update":
            env.update(data)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs, mu, seed=None):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.mu = mu
        self.seed = seed

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def update(self, mu):
        for local in self.locals:
            local.send(("update", mu, None))
        self.envs[0].update(mu)

    def reset(self):
        for i, local in enumerate(self.locals):
            local.send(("reset", None, self.seed))
        if self.seed:
            self.envs[0].seed(self.seed)
        local_obs, _ = self.envs[0].reset()
        results = [local_obs] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action, self.seed))
        obs, reward, done, trunc,  info = self.envs[0].step(actions[0])
        # let done be a vector in the size of the number of agents, if
        #  all agents in the vector record done then reset the environment
        # The same goes for truncated, except truncated is just cloned for 
        #  each agent in the environment
        if all(done) or any(trunc):
            if self.seed:
                self.envs[0].seed(self.seed)
            obs, info = self.envs[0].reset()
        results = zip(*[(obs, reward, done, trunc, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError


def make_env(env_key, seed=None):
    env = gym.make(env_key, disable_env_checker=True)
    if seed:
        env.seed(seed)
    env.reset()
    return env