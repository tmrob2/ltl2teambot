import random, math, os
import numpy as np

import gym
from gym import spaces

# rough hack
import sys
# sys.path.insert(0, '../')
from ltl_operators import progress, is_accomplished
from utils import build_relations

from random import randint


class LTLBootcamp(gym.Env):
    """
    An environment to pre-train the LTL module for the adversarial env.
    """

    def __init__(
        self,
        fixed_task=None,    # set an LTL instruction to be kept at every env reset
        timeout=25,         # max steps that the agent can do
    ):

        self.timeout = timeout
        self.time = 0
        self.fixed_task = fixed_task
        self.task = None

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(' rgb'))

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(110,), # Mission needs to be padded
            dtype='uint8'
        )

        # Initialize the state
        self.reset()


    def draw_task(self):
        ''' Helper function to randomly draw a new LTL task from the task distribution. '''

        if self.fixed_task is not None:
            return self.fixed_task

        tasks = [
            ['A', ['G', ['N', 'b']], ['E', 'r']],
            ['A', ['G', ['N', 'b']], ['E', 'g']],
            ['A', ['G', ['N', 'r']], ['E', 'b']],
            ['A', ['G', ['N', 'g']], ['E', 'b']],
            ['A', ['E', 'b'], ['E', 'g']],
            ['O', ['E', 'b'], ['E', 'g']],
            ['A', ['E', 'b'], ['E', 'r']],
            ['O', ['E', 'b'], ['E', 'r']],
            ['E', ['A', 'r', ['E', 'b']]],
            ['E', ['A', 'b', ['E', 'r']]],
            ['E', ['A', 'g', ['E', 'b']]],
            ['E', ['A', 'b', ['E', 'g']]],
            ['E', ['A', 'r', ['E', ['A', 'b', ['E', 'r']]]]],
            ['E', ['A', 'g', ['E', ['A', 'b', ['E', 'g']]]]],
            ['E', ['A', 'b', ['E', ['A', 'r', ['E', 'b']]]]],
            ['E', ['A', 'b', ['E', ['A', 'g', ['E', 'b']]]]],
            ['E', 'r'],
            ['E', 'b'],
            ['E', 'g'],
        ]
        return tasks[randint(0, len(tasks) - 1)]



    def reset(self):
        ''' Env reset, must be called every time 'done' becomes True. '''

        self.task = self.draw_task()
        self.mission = str(self.task)

        return self.gen_obs()


    def reward(self):
        '''
            Helper function to establish the reward and the done signals.
            Returns the (reward, done) tuple.
        '''

        if self.task == "True" or is_accomplished(self.task):   return (1, True)
        elif self.task == "False":  return (-1, True)
        return (0, False)

    
    def gen_obs(self):

        def encode_mission(mission):
            assert mission == str(self.task), "Task and mission are not equivalent!"

            syms = "AONGUXErgb"
            V = {k: v+1 for v, k in enumerate(syms)}
            enc = np.array([V[e] for e in mission if e not in "\', []"])
            # TODO: parameterize
            rels = build_relations(self.task, 10)
            return np.concatenate([enc, np.zeros(10 - len(enc))]), np.array(rels)

        obs = np.zeros(110) # max mission length
        if self.mission == 'True' or self.mission == 'False':
            return obs
        mission, rels = encode_mission(self.mission)
        return np.concatenate((mission, rels.reshape(-1))) # 10 + 10x10 = 110


    def step(self, action):

        # event detector
        if action == 0:
            event_objs = []
        elif action == 1:
            event_objs = ['r']
        elif action == 2:
            event_objs = ['g']
        elif action == 3:
            event_objs = ['b']

        # prog function call
        self.task = progress(self.task, event_objs)
        self.mission = str(self.task)

        reward, done = self.reward()

        # max steps elapsed
        self.time += 1
        if self.time > self.timeout:
            reward, done = -1, True
            self.time = 0

        return self.gen_obs(), reward, done, {}