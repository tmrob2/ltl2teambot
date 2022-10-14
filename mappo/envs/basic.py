from random import randint
from symbol import term
#from MiniGrid.minigrid.minigrid_env import Goal
from ltl_learning.ltl_operators import is_accomplished, progress
from teamgrid.minigrid import COLOR_NAMES, COLORS, MiniGridEnv, Grid, Goal, Key, Ball, Lava
import numpy as np
from gym import spaces
import random as rnd

class LTLSimpleMAEnv(MiniGridEnv):
    def __init__(self,**kwargs):
        room_size=6
        self.num_agents = 2
        self.m_tasks = 2
        self.mu = np.array([[0., 1.0], [1., 0.]])
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(258,), # Mission needs to be padded TODO: parameterize
            dtype='uint8'
        )
        super().__init__(
            grid_size=room_size,
            max_steps=8 * room_size**2,
            **kwargs,
        )

    #def reset(self):
    #    obs, info = super().reset()
    #    #self.time = 0
    #    for i in range(self.num_agents):
    #        self.agent_finished[i] = False
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
            syms = "AONGUXE[]abxy"
            V = {k: v+1 for v, k in enumerate(syms)}
            return [V[e] for e in mission if e not in ["\'", ",", " "]]    
        except Exception as e:
            if mission in ['True', 'False']:
                return []
            else:
                print(f"Unhandled encoding error: {e}") 

    def _gen_grid(self, width, height):
        #super()._gen_grid(width, height)

        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        # Place the goal objects randomly
        obj_key = Key('red')
        pos_key = self.place_obj(obj_key)

        obj_ball = Ball("blue")
        pos_ball = self.place_obj(obj_ball)

        #goal1 = Goal()
        #pos_goal1 = self.place_obj(goal1)

        #goal2 = Goal()
        #pos_goal2 = self.place_obj(goal2)

        for _ in range(self.num_agents):
            self.place_agent()
        #self.obj = obj
        # LTL objects 
        self.event_objects = [
            (obj_key, pos_key, "a"), 
            (obj_ball, pos_ball, "b"),
            #(goal1, pos_goal1, "x"),
            #(goal2, pos_goal2, "y")    
        ]
        #self.mission = "Do the LTL mission"

        tasks = self.draw_tasks()
        self.task = {i: tasks for i in range(self.num_agents)}
        self.mission = {i: str(self.task[i]) for i in range(self.num_agents)}
        self.finished_tasks = {i: [False] * len(tasks) for i in range(self.num_agents)}
        self.finished_mission = [False] * self.m_tasks
        #self.agent_finished = {i: False for i in range(self.num_agents)}
        self.agent_task_costs = {i: [0] * len(tasks) for i in range(self.num_agents)}

    def gen_obss(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        obs = super().gen_obss()
        
        obs_ = []
        for i in range(self.num_agents):
            img = np.array(obs[i]).reshape(-1) #147
            #direction = np.array([obs[i]]) # 1

            mission = np.array([])
            for j in range(self.m_tasks):
                # encode the task
                input_ = np.array(self.encode_mission(self.mission[i][j]))
                # specify a block
                arr = np.zeros(20)
                # input the task into the block
                arr[:input_.shape[0]] = input_
                # append the task to the mission
                mission = np.append(mission, arr)
            # The mission needs to be flat. 
            mission.flatten()

            new_obs = np.concatenate((img, self.mu[i, :], mission))

            obsi = np.zeros(180 + self.m_tasks + self.m_tasks * 20)
            obsi[:new_obs.shape[0]] = new_obs
            obs_.append(obsi)
        return obs_

    def rewards(self):
        agent_rewards = []
        agent_terminated = []
        for i in range(self.num_agents):
            rewards_ = [self.individual_task_reward(j, i) for j in range(self.m_tasks)]
            rewards, terminated = zip(*rewards_)
            # compute the weighted cost f the agent completing this task if the agent did
            # indeed complete the task
            #sum_costs = np.sum([self.agent_task_costs[i][j] for j in range(self.num_agents)])
            #rewards = [sum_costs] + list(rewards)
            agent_rewards.append(rewards)
            agent_terminated.append(terminated)
            # return sum costs to zero
            self.reset_agent_task_cost(i)
        return agent_rewards, agent_terminated

    def update(self, mu):
        self.mu = mu
        #print("mu updated")

    def reset_agent_task_cost(self, agent):
        # this makes sure only a one off cost is given to the agent when it has finished
        # its task
        for j in range(self.m_tasks):
            if self.finished_tasks[agent][j]:
                self.agent_task_costs[agent][j] = 0.

    def individual_task_reward(self, task, agent):
        """Reward function"""
        if self.task[agent][task] == "True" or is_accomplished(self.task[agent][task]): 
            if not self.finished_tasks[agent][task]:
                self.finished_tasks[agent][task] = True
                self.finished_mission[task] = True
                #self.agent_task_costs[agent][task] = \
                #    self.mu[agent, task] * (1 - 0.9 * (self.agent_step_count[agent] / self.max_steps))
                return (1, True)
            else:
                return (0, True)
        elif self.task[task] == "False": 
            if not self.finished_tasks[agent][task]:
                self.finished_tasks[agent][task] = True
                self.finished_mission[task] = True
                #self.agent_task_costs[agent][task] = \
                #    self.mu(agent, task) * (1 - 0.9 * (self.agent_step_count[agent] / self.max_steps))
                return (-1, True)
            else:
                return (0, True)
        else: return (0, False)

    def task_progress(self, agent):
        return [
            progress(t, self.get_events(agent)) 
            if not self.finished_tasks[agent][i] else t 
            for i, t in enumerate(self.task[agent])
        ]

    #def mission_terminate(self):
    #    terminated = True
    #    for i in range(self.num_agents):
    #        if not self.agent_finished[i]:
    #            terminated = False
    #            break
    #    return terminated

    #def check_agent_finished(self, agent_termination):
    #    #full_agent_rewards = []
    #    for i in range(self.num_agents):
    #        if not self.agent_finished[i] and all(agent_termination[i]):
    #            self.agent_finished[i] = True
    #            #success = 1 - 0.9 * (self.step_count / self.max_steps)
    #            #full_agent_rewards.append([success] + list(rewards[i]))
    #        #else: 
    #        #    full_agent_rewards.append([0.] + list(rewards[i]))
    #    #return full_agent_rewards

    def mission_reward(self, reward, terminated):
        if terminated:
            return [[1-0.9 * self.agent_step_count[i] / self.max_steps] + list(r) for i, r in enumerate(reward)]
        else:
            return [[0.] + list(r) for r in reward]

    def step(self, action):
        obs, reward, _, truncated, info = super().step(action)

        for agent in range(self.num_agents):
            self.task[agent] = self.task_progress(agent)
            self.mission[agent] = [str(t) for t in self.task[agent]]

        #print("agent pos: ", [x.cur_pos for x in self.agents], "mission", self.mission, "task", self.task)

        reward, _ = self.rewards()

        terminated_ = True if all(self.finished_mission) else False
        reward = self.mission_reward(reward, terminated_)
        terminated = [terminated_] * self.num_agents
        truncated_ = [truncated] * self.num_agents

        return obs, reward, terminated, truncated_, info

    def get_events(self, agent):
        '''Event detector => emits a proposition for which we need a truth assignment'''
        events = []
        for (obj, pos, label) in self.event_objects:
            #print(obj.__dict__, pos, label)
            #if self.carrying is not None: 
            #    print("agent carrying", self.carrying.__dict__)
            if self.agents[agent].carrying == obj:
                #print("carrying ", obj.__dict__)
                events.append(label)
            elif all(self.agents[agent].cur_pos == pos):
                events.append(label)
        return events