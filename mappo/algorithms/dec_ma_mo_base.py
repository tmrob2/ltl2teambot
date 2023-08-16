import numpy as np
import torch
from mappo.utils.ma_penv import ParallelEnv
from mappo.utils.dictlist import DictList

    
class BaseAlgorithm:
    def __init__(
        self, 
        envs, 
        acmodels,
        device,
        num_agents,
        num_objectives,
        num_frames_per_proc, 
        discount, 
        lr, 
        gae_lambda, 
        entropy_coef,
        value_loss_coef, 
        max_grad_norm, 
        recurrence, 
        preprocess_obss, 
        mu = None,
        seed=None
        #reshape_reward
        ):
        """
        Initializes a `BaseAlgo` instance.
        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        assert mu is not None

        self.env = ParallelEnv(envs, mu, seed)
        self.acmodels = acmodels
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.num_agents = num_agents
        self.num_objectives = num_objectives
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss if preprocess_obss is not None else self.default_preprocess_obss
        self.reshape_reward = None

        assert self.num_frames_per_proc % self.recurrence == 0

        [self.acmodels[i].train() for i in range(num_agents)]

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        shape = (self.num_frames_per_proc, self.num_agents, self.num_procs)
        mo_shape = (self.num_frames_per_proc, self.num_agents, self.num_procs, self.num_objectives)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        if self.acmodels[0].recurrent:
            self.memory = torch.zeros(num_agents, shape[2], self.acmodels[0].memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodels[0].memory_size, device=self.device)
        self.mask = torch.ones(num_agents, shape[2], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*mo_shape, device=self.device)
        self.rewards = torch.zeros(*mo_shape, device=self.device)
        self.advantages = torch.zeros(*mo_shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(
            self.num_agents, self.num_procs, self.num_objectives, device=self.device
        )
        self.log_episode_reshaped_return = torch.zeros(
            self.num_agents, self.num_procs, self.num_objectives, device=self.device
        )
        self.log_episode_num_frames = torch.zeros(self.num_agents, self.num_procs, device=self.device)

        self.log_done_counter = [0] * self.num_agents 
        #self.log_total_done_counter = 0
        self.log_return = [[np.zeros(3)] * self.num_procs for _ in range(self.num_agents)]
        #self.log_reshaped_return = [[] for _ in range(self.num_agents)]
        self.log_num_frames = [0] * self.num_procs * self.num_agents

    def update_environments(self, mu):
        self.env.update(mu)

    def collect_experiences(self, mu, c, e):
        """Collects rollouts and computes advantages.
        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.
        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        self.env.update(mu.detach().cpu().numpy())

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            actions = []
            values = []
            memories = []

            # for each of the models get the action distribution  and the critic value
            with torch.no_grad():
                for agent in range(self.num_agents):
                    if self.acmodels[agent].recurrent:
                        dist, value, memory = self.acmodels[agent](preprocessed_obs[agent], 
                            self.memory[agent] * self.mask[agent].unsqueeze(1))
                    else:
                        dist, value = self.acmodels[agent](preprocessed_obs)
                    actions.append(dist.sample())
                    values.append(value)
                    memories.append(memory)
            
            action_ = torch.cat(actions).reshape(self.num_procs, self.num_agents).cpu().numpy()

            obs, reward, done, trunc, _ = self.env.step(action_)

            # Update experiences values
            # convert the observation into the correct format for storage and processing
            obs_ = np.array(self.obs).transpose(1, 0, 2)
            # In the decentralised experience collection, we need to keep the shape A x P so that 
            # we can split everything on agents
            reward = np.array(reward).transpose(1, 0, 2)
            self.obss[i] = obs_ # the observation will then be a tensor of shape A x P x T
            self.obs = obs
            if self.acmodels[0].recurrent:
                self.memories[i] = self.memory
                self.memory = torch.cat(memories).reshape(self.num_agents, self.num_procs, self.num_frames_per_proc)
            self.masks[i] = self.mask # TODO check the shape of this
            done = torch.tensor(done, device=self.device, dtype=torch.float).transpose(0, 1) # make sure that this shape is correct
            trunc = torch.tensor(trunc, device=self.device, dtype=torch.float).transpose(0, 1)
            #done0 = done.reshape(-1)
            #trunc0 = trunc.reshape(-1)
            max_done_or_trunc = torch.max(done, trunc)
            self.mask = 1 - max_done_or_trunc
            self.actions[i] = torch.cat(actions).reshape(self.num_agents, self.num_procs)
            self.values[i] = torch.cat(values).reshape(self.num_agents, self.num_procs, self.num_objectives)
            self.rewards[i] = torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_probs[i] = dist.log_prob(self.actions[i])

            # Update log values
            self.log_episode_return += \
                torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += \
                torch.ones(self.num_agents, self.num_procs, device=self.device)

            reshaped_log_epsisode_frames = \
                self.log_episode_num_frames.reshape(self.num_procs, self.num_agents)
            for agent, done_ in enumerate(done):
                for i, done__ in enumerate(done_):
                    if done__:
                        # TODO: the issue is that one agent finishes and the other continues
                        # diluting the value of the agent which finished earlier
                        #self.log_total_done_counter += 1
                        if torch.count_nonzero(self.log_episode_return[agent, i]) > 0:
                            self.log_done_counter[agent] += 1
                            self.log_return[agent].append(self.log_episode_return[agent, i].cpu().numpy())
                            #self.log_return.append(self.log_episode_return[i].cpu().numpy())
                            #self.log_reshaped_return.append(self.log_episode_reshaped_return[i].cpu().numpy())
                            self.log_num_frames.append(reshaped_log_epsisode_frames[i][0].item())
                self.log_episode_return[agent] *= self.mask[agent].unsqueeze(1)
            #self.log_episode_reshaped_return *= self.mask.unsqueeze(1)
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        next_values = []
        with torch.no_grad():
            for agent in range(self.num_agents):
                if self.acmodels[agent].recurrent:
                    _, next_value, _ = self.acmodels[agent](preprocessed_obs[agent], 
                        self.memory[agent] * self.mask[agent].unsqueeze(1))
                else:
                    _, next_value = self.acmodel(preprocessed_obs)
                next_values.append(next_value)

        next_values = torch.cat(next_values)

        for i in reversed(range(self.num_frames_per_proc)):
            
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 \
                else torch.zeros(self.num_agents, self.num_procs, self.num_objectives, device=self.device, dtype=torch.float)
            advantages = []
            for agent in range(self.num_agents):
                delta = self.rewards[i][agent] + self.discount * \
                    next_value[agent] * next_mask[agent].unsqueeze(1) - self.values[i][agent]
                advantage = delta + self.discount * self.gae_lambda * next_advantage[agent] * next_mask[agent].unsqueeze(1)
                advantages.append(advantage)
            self.advantages[i] = torch.stack(advantages)


        # Modify the advantage with the task allocation parameters

        ini_values = self.values[0]
        # ini values has shape A x P x O

        # Compute H
        H = []
        for i in range(self.num_agents):
            Hi = self.computeH(ini_values, mu, i, c, e)
            H.append(Hi)

        H = torch.stack(H)
        # The output of H has shape A x P x O -> P x A x O -> (P . A ) x O
        #H_ = H.transpose(0, 1).reshape(self.num_agents * self.num_procs, self.num_objectives)

        mod_advantage = []
        for agent in range(self.num_agents):
            stack = []
            for k in range(self.num_procs):
                stack.append(torch.matmul(self.advantages[:, agent, k, :], H[agent, k, :]))
            mod_advantage.append(torch.stack(stack))
        mod_advantage = torch.stack(mod_advantage)
        mod_advantage = mod_advantage.reshape(self.num_agents, -1)

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs P = Procs x Agents in that order,
        #   - D is the dimensionality.

        # The dimension of the experiences should be A x (P * T)
        
        exps = DictList()
        exps.obs = torch.tensor(np.array(self.obss), device=self.device, dtype=torch.float).permute(1, 2, 0, 3). \
            reshape(self.num_agents, -1, self.obss[0][0].shape[1])
        if self.acmodels[0].recurrent:
            # T x A x P x D -> A x P x T x D -> A x (P * T) x D
            exps.memory = self.memories.permute(1, 2, 0, 3).reshape(self.num_agents, -1, *self.memories.shape[3:])
            # T x A x P -> A x P x T -> A x (P * T) x 1
            exps.mask = self.masks.permute(1, 2, 0).reshape(self.num_agents, -1).unsqueeze(2)
        
        # for all tensors below, T x A x P -> A x P x T -> A x P * T
        exps.action = self.actions.permute(1, 2, 0).reshape(self.num_agents, -1)
        #exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        # mod advantage already has shape A x P * T
        exps.advantage = mod_advantage
        #exps.ini_values = ini_values
        exps.log_prob = self.log_probs.permute(1, 2, 0).reshape(self.num_agents, -1)
        
        # T x A x P x O -> A x P x T x O -> A x P * T x O
        exps.value = self.values.permute(1, 2, 0, 3).reshape(self.num_agents, -1, *self.values.shape[3:])
        exps.reward = self.rewards.permute(1, 2, 0, 3).reshape(self.num_agents, -1, *self.rewards.shape[3:])
        exps.returnn = exps.value + \
            self.advantages.permute(1, 2, 0, 3).reshape(self.num_agents, -1, *self.advantages.shape[3:])
        # Preprocess experiences

        #exps.obs = torch.tensor(np.array(exps.obs), device=self.device, dtype=torch.float)

        # Log some values

        keeps = [
            max(self.log_done_counter[agent], self.num_procs) 
            for agent in range(self.num_agents)
        ]

        #keep = max(self.log_total_done_counter, self.num_procs * self.num_agents)
        keep = max(max(self.log_done_counter), self.num_procs * self.num_agents)

        try:
            logs = {
                "return_per_episode": [
                    self.log_return[i][-keeps[i]:] for i in range(self.num_agents)
                ],
                #"reshaped_return_per_episode": np.array(self.log_reshaped_return[-keep:]).reshape(self.num_agents, self.num_procs, self.num_objectives),
                "num_frames_per_episode": self.log_num_frames[-keep:],
                "num_frames": self.num_frames
            }
        except Exception as e:
            print(e)

        self.log_done_counter = [0] * self.num_agents
        #self.log_total_done_counter = 0
        self.log_return = [
            self.log_return[i][-self.num_procs:] 
            for i in range(self.num_agents)
        ]
       # self.log_reshaped_return = self.log_reshaped_return[-self.num_procs * self.num_agents:]
        self.log_num_frames = self.log_num_frames[-self.num_procs * self.num_agents:]

        return exps, logs, ini_values
    
    def default_preprocess_obss(self, obss, device=None):
        t = torch.tensor(np.array(obss), device=device, dtype=torch.float)
        tup = torch.split(t.transpose(0, 1), 1)
        tup = tuple([tup[t].squeeze() for t in range(self.num_agents)])
        return tup

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()


    def df(self, x, c):
        #if x <= c: 
        #    return 2*(c - x)
        #else:
        #    return 0.0
        return torch.where(x <= c, 2*(c - x), 0.0)


    def dh(self, x, e):
        #if x <= e:
        #    return 2 * (e - x)
        #else:
        #    return 0.0
        return torch.where(x <= e, 2 * (e - x), 0.0)


    def computeH(self, X, mu, agent, c, e):
        # todo need to loop over each environment
        _, _, y = X.shape
        H_ = [] 
        H_.append(2.0 * self.df(X[agent, :, 0], c))
        for j in range(1, y):
            #print(X[:, k, j - 1])
            H_.append(
                self.dh(torch.sum(mu[:, j - 1].unsqueeze(1) * X[:, :, j], dim=0), e) \
                    * mu[agent, j - 1]
            )

        H = torch.stack(H_, 1)
        return H

def compute_alloc_loss(X, mu, e):
    _, _, y = X.shape
    loss = 0.
    for j in range(1, y):
        with torch.no_grad():
            h = dh(torch.sum(mu[:, j - 1].unsqueeze(1) * X.transpose(1, 0)[:, :, j], dim=0), e)
        kappa_component = torch.sum(mu[:, j - 1].unsqueeze(1) * X.transpose(1, 0)[:, :, j], dim=0)
        loss += torch.matmul(h, kappa_component)
    return loss

def dh(x, e):
    return torch.where(x <= e, 2 * (e - x), 0.0)
    
