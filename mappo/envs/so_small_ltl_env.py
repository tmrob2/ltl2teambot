from random import randint
from ltl_learning.ltl_operators import is_accomplished, progress
from minigrid.core.constants import COLOR_NAMES, COLORS
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.minigrid_env import Goal
import numpy as np
from gymnasium import spaces

class LTLTestEnv1(RoomGrid):
    def __init__(self,**kwargs):
        room_size=6
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
            ['A', ['E', 'b'], ['E', 'r']]
        ]
        return tasks[randint(0, len(tasks) - 1)]

    def encode_mission(self, mission):
        syms = "AONGUXE[]rgb"
        V = {k: v+1 for v, k in enumerate(syms)}
        return [V[e] for e in mission if e not in ["\'", ",", " "]]    

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        #obj, _ = self.add_object(1, 0, kind="box")
        #door, _ = self.add_door(0, 0, 0, locked=True)
        #self.connect_all()
        # Add a key to unlock the door
        obj_key, pos_key = self.add_object(0, 0, "key", "red")
        obj_ball, pos_ball = self.add_object(0, 0, "ball", "blue")
        self.place_agent(0, 0)
        #self.obj = obj
        # LTL objects 
        self.event_objects = [(obj_key, pos_key, "r"), (obj_ball, pos_ball, "b")]
        #self.mission = "Do the LTL mission"

        self.task = self.draw_tasks()
        self.mission = str(self.task)

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        obs = super().gen_obs()
        
        # obs = {
        #     'image': image, (7,7,3)
        #     'direction': self.agent_dir,
        #     'mission': self.mission
        # }
        
        img = np.array(obs["image"]).reshape(-1) #147
        direction = np.array([obs["direction"]]) # 1
        mission = np.array(self.encode_mission(obs["mission"]))

        new_obs = np.concatenate((img, direction, mission))

        obs = np.zeros(180)
        obs[:new_obs.shape[0]] = new_obs
        
        return obs

    def reward(self):
        """Reward function"""
        success = 1 - 0.9 * (self.step_count / self.max_steps)
        if self.task == "True" or is_accomplished(self.task): return (success, True)
        elif self.task == "False": return (-1, True)
        else: return (0, False)

    def step(self, action):
        obs, reward, _, truncated, info = super().step(action)

        self.task = progress(self.task, self.get_events())
        self.mission = str(self.task)

        #print("agent pos: ", self.agent_pos, "mission", self.mission, "task", self.task, "obs", obs)

        reward, terminated = self.reward()

        return obs, reward, terminated, truncated, info

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