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
        self.num_actions = num_actions
        self.device = device

        self.observations = []
        self.actions = []
        self.pi = [[] for _ in range(self.num_agents)]
        self.reward = []
        self.done = []
        self.trunc = []
        #self.log_probs = []
        #self.memories = []

    def get_memory(self):

        actions = torch.tensor(self.actions, device=self.device, dtype=torch.float)
        pi = []
        for i in range(self.num_agents):
            pi.append(torch.cat(self.pi[i]).view(len(self.pi[i]), self.num_actions))
        rewards = torch.tensor(np.array(self.reward), device=self.device)
        done = torch.tensor(np.array(self.done), device=self.device)
        #memory = torch.stack(self.memories)
        trunc = torch.tensor(np.array(self.trunc), device=self.device)
        observations = self.observations
        return observations, actions, pi, rewards, done, trunc #, memory

    def clear(self):
        self.observations = []
        self.actions = []
        self.pi = []
        self.reward = []
        self.done = []
        self.trunc = []
        #self.log_probs = []


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
        #self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

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

    def forward(self, obs):#, memory=None):
        img = obs[:, :147]  # Observation of the env
        task = obs[:, 147:147 + self.num_tasks] # Task allocation
        ltl = obs[: , 147 + self.num_tasks:] # Progression of the task

        #x = img.transpose(1, 3).transpose(2, 3)
        x = img.reshape(obs.shape[0], self.n, self.m, 3).permute(0, 3, 1, 2)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        #if self.use_memory:
        #hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        #hidden = self.memory_rnn(x, hidden)
        #embedding = hidden[0]
        #memory = torch.cat(hidden, dim=1)
        #else:
        embedding = x

        embedded_formula = self.ltl_embedder(ltl.to(torch.long))
        _, h = self.ltl_rnn(embedded_formula)
        embedded_formula = h[:-2, :, :].transpose(0, 1).reshape(obs.shape[0], -1)

        composed_x = torch.cat([embedding, task, embedded_formula], dim=1)

        x = self.actor(composed_x)
        dist = F.softmax(x, dim=1)

        return dist #, memory 

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
        self.num_agents = num_agents
        self.embedding_size = (self.semi_memory_size + self.num_tasks + 32 * 2) + num_agents + 1
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
        #self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
        

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

    def forward(self, input, actions, agent_id): #, memory=None):
        
        img = input[:, :147]  # Observation of the env
        task = input[:, 147:147 + self.num_tasks] # Task allocation
        ltl = input[: , 147 + self.num_tasks:] # Progression of the task

        batch_size = input.shape[0]

        ##### ENV RECOGNITION LAYERS #####
        #x = img.transpose(1, 3).transpose(2, 3)
        # The img shape is A x (7, 7)
        x = img.reshape(-1, self.n, self.m, 3).permute(0, 3, 1, 2)
        x = self.image_conv(x)
        x = x.reshape(batch_size, -1)
        
        ##### MEMORY LAYERS #####
        #if self.use_memory:
        #hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
        #hidden = self.memory_rnn(x, hidden)
        #embedding = hidden[0]
        #memory = torch.cat(hidden, dim=1)
        embedding = x
        
        ##### TASK PROGRESS LAYERS #####
        embedded_formula = self.ltl_embedder(ltl.to(torch.long))
        _, h = self.ltl_rnn(embedded_formula)
        embedded_formula = h[:-2, :, :].transpose(0, 1).reshape(input.shape[0], -1)

        # In the composed layer we want to flatten everything, so that
        # we have
        
        composed_x = torch.cat([embedding, task, embedded_formula], dim=1)
        joint_state_action = torch.cat([agent_id, composed_x, actions], dim=1) 


        x = self.critic(joint_state_action)
        value = x.reshape(batch_size, self.n_actions, self.objectives)

        return value #, memory

    def save_models(self):
        print('... saving models ...')
        torch.save(deepcopy(self.state_dict()), self.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class COMA:
    def __init__(
            self, 
            num_agents, 
            action_space, 
            num_tasks, 
            env, 
            device=None,
            lr_a = 0.001, 
            lr_c = 0.001,
            gamma = 0.95,
            target_update_steps=10
        ):
        self.num_agents = num_agents
        self.device = device
        self.env = env
        self.target_update_steps = target_update_steps

        self.agents = [Actor(action_space, num_tasks, name=f"agent{i}", device=self.device) for i in range(num_agents)] 
        #self.amem =  [torch.zeros(1, self.agents[i].memory_size, device=self.device) for i in range(num_agents)]
        self.actor_optimisers = [torch.optim.Adam(self.agents[i].parameters(), lr=lr_a) for i in range(num_agents)]

        self.critic = Critic(num_agents, num_tasks, action_space.n)
        self.critic.to(device)
        self.critic_optimiser = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        self.critic_target = Critic(num_agents, num_tasks, action_space.n)
        self.critic_target.to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        #self.cmem = torch.zeros(num_agents, self.critic.memory_size, device=device, dtype=torch.float)

        self.episode_memory = EpisodeMemory(num_agents, action_space.n, self.device)

        self.frames = 0
        self.count_steps = 0
        self.gamma = gamma
        self.objectives = num_tasks + 1

    def reset(self, seed=None):
        if seed:
            self.env.seed(seed)
        return self.env.reset()


    def act(self, observation):
        # observation is already a list
        observation = np.array(observation)
        actions = []
        for i in range(self.num_agents):
            obs = self.preprocess_obs(observation[i])
            #dist, self.amem[i] = self.agents[i](obs) #, self.amem[i])
            dist = self.agents[i](obs) #, self.amem[i])
            action = Categorical(dist).sample()
            self.episode_memory.pi[i].append(dist)
            actions.append(action.item())
        self.episode_memory.actions.append(actions)
        self.episode_memory.observations.append(
            torch.tensor(observation, device=self.device, dtype=torch.float)
        )
        return actions

    def collect_episode_trajectory(self, seed=None):
        episode_done = False
        obs, _ = self.reset(seed)
        while not episode_done:
            #pi, actions, memory = self.act(obs)
            actions = self.act(obs)

            next_obs, reward, done, trunc, _ = self.env.step(np.array(actions))

            self.episode_memory.reward.append(reward)
            self.episode_memory.done.append(done)
            self.episode_memory.trunc.append(trunc)
            #self.episode_memory.memories.append(memory)
            self.frames += 1
            obs = next_obs

            if all(done) or all(trunc):
                # the episode is finished
                break

    def preprocess_obs(self, obs):
        t = torch.tensor(obs, device=self.device, dtype=torch.float).unsqueeze(0)
        return t

    def process_observations(self, observations, batch_size, state_size):
        observations = torch.stack(observations).view(batch_size, state_size * self.num_agents)
        return observations

    def get_ini_values(self, obs, actions, pi):
        # get the critic value for the initial observations

        obs = obs[0].unsqueeze(0)
        ini_values = []
        with torch.no_grad():
            for i in range(self.num_agents):
                agent_id = (torch.ones(1, device=self.device) * i).view(-1, 1)
                #qtarget, self.cmem = self.critic(obs[0], actions[0], agent_id, self.cmem)
                qtarget = self.critic(obs, actions[0].unsqueeze(0), agent_id)
                v_ini_i = torch.sum(pi[0][i].unsqueeze(1) * qtarget, dim=1)
                ini_values.append(v_ini_i)
        return torch.stack(ini_values).squeeze() # A x O

    def df(self, x, c):
        return torch.where(x <= c, 2 * (c - x), 0.0)

    def dh(self, x, e):
        return torch.where(x <= e, 2 * (e - x), 0.0)

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

    def train(self, mu, c, e):

        observations, actions, pi, rewards, done, trunc  = self.episode_memory.get_memory()
        batch_size = len(observations)
        observations = self.process_observations(
            observations, len(observations), observations[0].shape[1]
        )
        ini_values = self.get_ini_values(observations, actions, pi).detach()
        for agent in range(self.num_agents):
            H = self.computeH(ini_values, mu, agent, c, e)
            ids = (torch.ones(batch_size, device=self.device) * agent).view(-1, 1)
            #obs = observations.transpose(0, 1).reshape(batch_size, -1)
            Q_target = self.critic_target(observations, actions, ids).detach()

            action_taken = actions.type(torch.long)[:, agent].reshape(-1, 1)

            baseline = torch.sum(pi[agent].unsqueeze(2) * Q_target, dim=1).detach()
            Q_taken_target = torch.stack(
                [torch.gather(Q_target[:, :, 0], dim=1, index=action_taken).squeeze() 
                for _ in range(Q_target.shape[2])]
            ).transpose(0, 1)
            advantage = Q_taken_target - baseline
            mod_advantage = torch.matmul(advantage, H)

            log_pi = torch.log(torch.gather(pi[agent], dim=1, index=action_taken).squeeze())

            actor_loss = -torch.mean(log_pi * mod_advantage)

            self.actor_optimisers[agent].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[agent].parameters(), 5)
            self.actor_optimisers[agent].step()

            ########################
            # Train the critic
            ########################

            Q = self.critic(observations, actions, ids)

            #action_taken = exps.actions.type(torch.long)[agent, :].reshape(-1, 1)
            Q_taken = torch.stack(
                [torch.gather(Q[:, :, 0], dim=1, index=action_taken).squeeze() 
                for _ in range(Q.shape[2])]
            ).transpose(0, 1)

            # TD(0)
            r = torch.zeros(batch_size, self.objectives, device=self.device)
            for t in range(batch_size):
                if done[t][agent] or trunc[t][agent]:
                    r[t] = rewards[t][agent]
                else:
                    r[t] = rewards[t][agent] + self.gamma * Q_taken_target[t + 1]

            critic_loss = torch.mean((r - Q_taken) ** 2)

            self.critic_optimiser.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            self.critic_optimiser.step()

        if self.count_steps == self.target_update_steps:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.count_steps = 0
        else:
            self.count_steps += 1
        
        self.episode_memory.clear()




            


            



        
        