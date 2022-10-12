from random import randint
import random as rnd
from ltl_learning.ltl_operators import is_accomplished, progress
from minigrid.core.constants import COLOR_NAMES, COLORS
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.minigrid_env import Goal
import numpy as np
from gymnasium import spaces

class MOLTLTestEnv(RoomGrid):
    def __init__(self,**kwargs):
        room_size=8
        m_tasks = 2
        mission_space = MissionSpace(
            mission_func=lambda color: f"pick up the {color} box", 
            ordered_placeholders=[COLOR_NAMES],
        )
        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            max_steps=8 * room_size**2,
            **kwargs,
        )
        self.m_tasks = m_tasks
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(258,), # Mission needs to be padded TODO: parameterize
            dtype='uint8'
        )

    #def reset(self):
    #    obs, info = super().reset()
    #    self.time = 0
    #    return obs, info 

    def draw_tasks(self):
        tasks = [
            #['A', ['E', 'b'], ['E', 'a']],
            #['A', ['G', ['N', 'c']], ['E', 'd']]
            ['E', 'a'],
            ['E', 'b']
        ]
        #return tasks[randint(0, len(tasks) - 1)]
        return rnd.sample(tasks, 2)

    def encode_mission(self, mission):
        try:
            syms = "AONGUXE[]abcd"
            V = {k: v+1 for v, k in enumerate(syms)}
            return [V[e] for e in mission if e not in ["\'", ",", " "]]    
        except Exception as e:
            if mission in ['True', 'False']:
                return []
            else:
                print(f"Unhandled encoding error: {e}")


    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        #
        # Add a box to the room on the right
        #obj, _ = self.add_object(1, 0, kind="box")
        #door, _ = self.add_door(0, 0, 0, locked=True)
        #self.connect_all()
        # Add a key to unlock the door
        obj_key1, pos_key1 = self.add_object(0, 0, "key", "red")
        #obj_key2, pos_key2 = self.add_object(0, 0, "key", "green")
        obj_ball, pos_ball = self.add_object(0, 0, "ball", "blue")
        #obj_ball2, pos_ball2 = self.add_object(0, 0, "ball", "red")
        self.place_agent(0, 0)
        #self.obj = obj
        # LTL objects 
        self.event_objects = [
            (obj_key1, pos_key1, "a"), 
            (obj_ball, pos_ball, "b"),
            #(obj_key2, pos_key2, "c"),
            #(obj_ball2, pos_ball2, "d")
        ]
        #self.mission = "Do the LTL mission"
        #
        self.tasks = self.draw_tasks()
        self.finished_tasks = [False] * len(self.tasks)
        self.mission = [str(task) for task in self.tasks]

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        obs = super().gen_obs()
        img = np.array(obs["image"]).reshape(-1) #147
        direction = np.array([obs["direction"]]) # 1
        #missions = []
        #mission = np.array(self.encode_mission(obs["mission"]))
        mission = np.array([])
        for i in range(self.m_tasks):
            # encode the task
            input_ = np.array(self.encode_mission(obs["mission"][i]))
            # specify a block
            arr = np.zeros(13)
            # input the task into the block
            arr[:input_.shape[0]] = input_
            # append the task to the mission
            mission = np.append(mission, arr)
        # The mission needs to be flat. 
        mission.flatten()
        
        
        new_obs = np.concatenate((img, direction, mission))

        obs = np.zeros(180 + (self.m_tasks - 1) * 13)
        obs[:new_obs.shape[0]] = new_obs
        
        return obs

    def rewards(self):
        rewards_ = [self.individual_task_reward(j) for j in range(self.m_tasks)]
        rewards, terminated = zip(*rewards_)
        return rewards, terminated

    def mission_terminated(self, terminated):
        return all(terminated)

    def individual_task_reward(self, j):
        """Reward function"""
        if self.tasks[j] == "True" or is_accomplished(self.tasks[j]): 
            if not self.finished_tasks[j]:
                self.finished_tasks[j] = True
                return (1, True)
            else:
                return (0, True)
        elif self.tasks[j] == "False": 
            if not self.finished_tasks[j]:
                self.finished_tasks[j] = True
                return (-1, True)
            else:
                return (0, True)
        else: return (0, False)

    def step(self, action):
        obs, reward, _, truncated, info = super().step(action)

        # progress each task in the mission
        self.tasks = [
            progress(t, self.get_events()) 
            if not self.finished_tasks[i] else t 
            for i, t in enumerate(self.tasks)
        ]
        # the mission is a vector of strings
        self.mission = [str(t) for t in self.tasks]

        #print("agent pos: ", self.agent_pos, "mission", self.mission, "task", self.task, "obs", obs)
        # rewards, and terminated_ is also a vector
        reward, terminated_ = self.rewards()
        terminated = self.mission_terminated(terminated_)

        if terminated:
            success = 1 - 0.9 * (self.step_count / self.max_steps)
            reward = [success] + list(reward)
        else:
            reward = [0] + list(reward)
        
        return obs, list(reward), terminated, truncated, info

    def get_events(self):
        '''Event detector => emits a proposition for which we need a truth assignment'''
        events = []
        for (obj, pos, label) in self.event_objects:
            #print(obj.__dict__, pos, label)
            #if self.carrying is not None: 
            #    print("agent carrying", self.carrying.__dict__)
            if self.carrying == obj:
                #print("carrying ", obj.__dict__)
                events.append(label)
            elif tuple(self.agent_pos) == pos:
                events.append(label)
        return events

