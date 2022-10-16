from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from mappo.utils.dictlist import DictList  

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class EpisodeMemory:
    def __init__(self, num_agents, num_actions, device=None):
        self.num_agents = num_agents
        self.num_agents = num_actions
        self.device = device

        self.observations = []
        self.actions = []
        self.pi = []
        self.reward = []
        self.done = []
        self.memories = []

    def get_memory(self):

        actions = torch.stack(self.actions)
        observations = torch.tensor(np.array(self.observations), device=self.device, dtype=torch.float)
        pi = torch.stack(self.pi)
        rewards = torch.tensor(np.array(self.reward), device=self.device)
        done = torch.tensor(np.array(self.done), device=self.device)
        memory = torch.stack(self.memories)

        return observations, actions, pi, rewards, done, memory


class Actor(nn.Module):
    def __init__(
        self,
        action_space,
        num_tasks,
        partial_obs_dims = (7, 7),
        name="",
        device=None
        ) -> None:
        super(Actor, self).__init__()

        self.checkpoint_file = \
            f"/home/thomas/ai_projects/MAS_MT_RL/mappo/tmp/ppo/coma_actor_torch_{name}"

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        self.n, self.m = partial_obs_dims
        self.image_embedding_size = ((self.n-1)//2-2)*((self.m-1)//2-2)*64
        self.num_tasks = num_tasks

        # Define memory
        self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        self.ltl_embedder = nn.Embedding(20, 8, padding_idx=0)
        self.ltl_rnn = nn.GRU(8, 32, num_layers=2, bidirectional=True, batch_first=True)

        self.embedding_size = self.semi_memory_size + self.num_tasks + 32 * 2

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Initialize parameters correctly
        self.apply(init_params)
        self.to(device)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory=None):
        img = obs[:, :147]  # Observation of the env
        task = obs[:, 147:147 + self.num_tasks] # Task allocation
        ltl = obs[: , 147 + self.num_tasks:] # Progression of the task

        #x = img.transpose(1, 3).transpose(2, 3)
        x = img.reshape(obs.shape[0], self.n, self.m, 3).permute(0, 3, 1, 2)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        #if self.use_memory:
        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)
        #else:
        #    embedding = x

        embedded_formula = self.ltl_embedder(ltl.to(torch.long))
        _, h = self.ltl_rnn(embedded_formula)
        embedded_formula = h[:-2, :, :].transpose(0, 1).reshape(obs.shape[0], -1)

        composed_x = torch.cat([embedding, task, embedded_formula], dim=1)

        x = self.actor(composed_x)
        dist = F.log_softmax(x, dim=1)

        return dist, memory 

class Critic(nn.Module):
    def __init__(
        self,
        num_agents,
        num_tasks,
        n_actions,
        partial_obs_dims = (7, 7),
        name=""
        ) -> None:
        super(Critic, self).__init__()

        self.checkpoint_file = \
            f"/home/thomas/ai_projects/MAS_MT_RL/mappo/tmp/ppo/coma_actor_torch_{name}"
        
        self.n, self.m = partial_obs_dims
        self.image_embedding_size = ((self.n-1)//2-2)*((self.m-1)//2-2)*64
        self.num_tasks = num_tasks
        self.embedding_size = (self.semi_memory_size + self.num_tasks + 32 * 2) * num_agents + num_agents + 1
        self.n_actions = n_actions
        self.objectives = num_tasks + 1

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        # Define memory
        self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        self.ltl_embedder = nn.Embedding(20, 8, padding_idx=0)
        self.ltl_rnn = nn.GRU(8, 32, num_layers=2, bidirectional=True, batch_first=True)

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, self.n_actions * self.objectives)
        )

        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, input, actions, agent_id, memory=None):
        
        img = input[:, :147]  # Observation of the env
        task = input[:, 147:147 + self.num_tasks] # Task allocation
        ltl = input[: , 147 + self.num_tasks:] # Progression of the task

        num_agents = input.shape[0]

        ##### ENV RECOGNITION LAYERS #####
        #x = img.transpose(1, 3).transpose(2, 3)
        # The img shape is A x (7, 7)
        x = img.reshape(-1, self.n, self.m, 3).permute(0, 3, 1, 2)
        x = self.image_conv(x)
        x = x.reshape(num_agents, -1)
        
        ##### MEMORY LAYERS #####
        #if self.use_memory:
        hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)
        
        ##### TASK PROGRESS LAYERS #####
        embedded_formula = self.ltl_embedder(ltl.to(torch.long))
        _, h = self.ltl_rnn(embedded_formula)
        embedded_formula = h[:-2, :, :].transpose(0, 1).reshape(input.shape[0], -1)

        # In the composed layer we want to flatten everything, so that
        # we have
        
        composed_x = torch.cat([embedding, task, embedded_formula], dim=1).reshape(-1)
        joint_state_action = torch.cat([agent_id, composed_x, actions])


        x = self.critic(joint_state_action)
        value = x.reshape(self.n_actions, self.objectives)

        return value, memory

    def save_models(self):
        print('... saving models ...')
        torch.save(deepcopy(self.state_dict()), self.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class COMA:
    def __init__(self, num_agents, action_space, num_tasks, env, device=None):
        self.num_agents = num_agents
        self.device = device
        self.env = env

        self.agents = [Actor(action_space, num_tasks, name=f"agent{i}", device=self.device) for i in range(num_agents)] 
        self.amem = [torch.zeros(1, self.agents[i].memory_size, device=self.device) for i in range(num_agents)]

        self.critic = Critic(num_agents, num_tasks, action_space.n)
        self.critic.to(device)
        self.cmem = torch.zeros(num_agents, self.critic.memory_size, device=device, dtype=torch.float)

        self.episode_memory = EpisodeMemory(num_agents, action_space.n, self.device)

        self.frames = 0

    def reset(self, seed=None):
        if seed:
            self.env.seed(seed)
        return self.env.reset()


    def act(self, observation):
        # observation is already a list
        pi = []; actions = []
        observation = np.array(observation)
        for i in range(self.num_agents):
            obs = self.preprocess_obs(observation[i])
            dist, self.amem[i] = self.agents[i](obs, self.amem[i])
            action = Categorical(dist).sample()
            pi.append(dist); actions.append(action)
        pi = torch.cat(pi)
        actions = torch.cat(actions)
        memory = torch.cat(self.amem)
        return pi, actions, memory

    def collect_episode_trajectory(self, seed=None):
        episode_done = False
        obs, _ = self.reset(seed)
        while not episode_done:
            pi, actions, memory = self.act(obs)

            next_obs, reward, done, trunc, _ = self.env.step(actions.cpu().numpy())
            self.episode_memory.observations.append(next_obs)
            self.episode_memory.actions.append(actions)
            self.episode_memory.reward.append(reward)
            self.episode_memory.done.append(done)
            self.episode_memory.pi.append(pi)
            self.episode_memory.memories.append(memory)
            self.frames += 1
            obs = next_obs

            if all(done) or all(trunc):
                # the episode is finished
                break
        return self.create_experiences()

    def create_experiences(self):
        observations, actions, pi, rewards, done, memory = \
            self.episode_memory.get_memory()
        exps = DictList()
        # Tranpose the outputs so that the agent is 0th index
        exps.observations = observations.transpose(0, 1)
        exps.actions = actions.transpose(0, 1)
        exps.pi = pi.transpose(0, 1)
        exps.rewards = rewards.transpose(0, 1)
        exps.done = done.transpose(0, 1)
        exps.memory = memory.transpose(0, 1)
        return exps

    def preprocess_obs(self, obs):
        t = torch.tensor(obs, device=self.device, dtype=torch.float).unsqueeze(0)
        return t

    def get_ini_values(self):
        # get the critic value for the initial observations
        obs, actions, pi, _, _, _ = self.episode_memory.get_memory()
        ini_values = []
        with torch.no_grad():
            for i in range(self.num_agents):
                agent_id = torch.tensor(np.array([i]), device=self.device, dtype=torch.float)
                qtarget, self.cmem = self.critic(obs[0], actions[0], agent_id, self.cmem)
                v_ini_i = torch.sum(pi[0][i].unsqueeze(1) * qtarget, dim=0)
                ini_values.append(v_ini_i)
        return torch.stack(ini_values).detach() # A x O

    def df(self, x, c):
        return torch.where(x <= c, 2 * (c - x), 0.0)

    def dh(self, x, e):
        return torch.where(x<= e, 2 * (e - x), 0.0)

    def computeH(self, X, mu, agent, c, e):
        _, y = X.shape

        H_ = []
        H_.append(self.df(X[agent, 0], c))
        for j in range(1, y):
            H_.append(
                self.dh(torch.sum(mu[:, j - 1] * X[:, j]), e) * mu[agent, j - 1]
            )
        H = torch.stack(H_)
        return H

    def train(self, exps, mu, c, e):
        # TODO try batch first, if  this works use this otherwise use 
        # a for loop 
        ini_values = self.get_ini_values()
        for agent in range(self.num_agents):
            H = self.computeH(ini_values, mu, agent, c, e)
            batch_size = len(exps[agent].observations)
            ids = (torch.ones(batch_size) * agent).view(-1, 1)
            Q_target, self.cmem = self.critic(
                exps[agent].observations, exps[agent].actions, ids, self.cmem)
            Q_target = Q_target.detach()

            action_taken = exps[agent].type(torch.long)[:, agent].reshape(-1, 1)

            #baseline = torch.sum(exps[agent].pi * )
            


            



        
        