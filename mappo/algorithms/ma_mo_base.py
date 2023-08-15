import numpy as np
import torch
from mappo.utils.ma_penv import ParallelEnv
from mappo.utils.dictlist import DictList

    
class BaseAlgorithm:
    def __init__(
        self, 
        envs, 
        acmodel,
        device,
        num_agents,
        num_objectives,
        num_frames_per_proc, 
        chi,
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
        self.acmodel = acmodel
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
        self.chi = chi

        assert self.num_frames_per_proc % self.recurrence == 0

        self.acmodel.train()

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs * self.num_agents

        shape = (self.num_frames_per_proc, self.num_procs * self.num_agents)
        mo_shape = (self.num_frames_per_proc, self.num_procs * self.num_agents, self.num_objectives)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*mo_shape, device=self.device)
        self.rewards = torch.zeros(*mo_shape, device=self.device)
        self.advantages = torch.zeros(*mo_shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(
            self.num_agents * self.num_procs, self.num_objectives, device=self.device
        )
        self.log_episode_reshaped_return = torch.zeros(
            self.num_agents * self.num_procs, self.num_objectives, device=self.device
        )
        self.log_episode_num_frames = torch.zeros(self.num_procs * self.num_agents, device=self.device)

        self.log_done_counter = [0] * self.num_agents 
        #self.log_total_done_counter = 0
        self.log_return = [[np.zeros(self.num_objectives)] * self.num_procs for _ in range(self.num_agents)]
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
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()
            action_ = action.reshape(self.num_procs, self.num_agents).cpu().numpy()

            obs, reward, done, trunc, _ = self.env.step(action_)

            # Update experiences values
            # convert the observation into the correct format for storage and processing
            obs_ = np.array(self.obs).reshape(-1, obs[0][0].shape[0]).tolist()
            # it is tricky now not to mix data so I am identifying exactly what transformations
            # I am doing to the tensor
            # the shape of rewards is P x A x O -> (P . A) x O therefore A is the second dim
            reward = np.array(reward).reshape(-1, self.num_objectives)
            self.obss[i] = obs_
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            done = torch.tensor(done, device=self.device, dtype=torch.float)
            trunc = torch.tensor(trunc, device=self.device, dtype=torch.float)
            done0 = done.reshape(-1)
            trunc0 = trunc.reshape(-1)
            max_done_or_trunc = torch.max(done0, trunc0)
            self.mask = 1 - max_done_or_trunc
            self.actions[i] = action
            self.values[i] = value
            self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values
            self.log_episode_return += \
                torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += \
                torch.ones(self.num_procs * self.num_agents, device=self.device)

            reshaped_log_episode_return = \
                self.log_episode_return.reshape(self.num_procs, self.num_agents, self.num_objectives)
            reshaped_log_epsisode_frames = \
                self.log_episode_num_frames.reshape(self.num_procs, self.num_agents)
            for i, done_ in enumerate(done):
                for agent, done__ in enumerate(done_):
                    if done__:
                        # TODO: the issue is that one agent finishes and the other continues
                        # diluting the value of the agent which finished earlier
                        #self.log_total_done_counter += 1
                        if torch.count_nonzero(reshaped_log_episode_return[i, agent]) > 0:
                            self.log_done_counter[agent] += 1
                            self.log_return[agent].append(reshaped_log_episode_return[i, agent].cpu().numpy())
                            #self.log_return.append(self.log_episode_return[i].cpu().numpy())
                            #self.log_reshaped_return.append(self.log_episode_reshaped_return[i].cpu().numpy())
                            self.log_num_frames.append(reshaped_log_epsisode_frames[i][0].item())
            self.log_episode_return *= self.mask.unsqueeze(1)
            #self.log_episode_reshaped_return *= self.mask.unsqueeze(1)
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask.unsqueeze(1) - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask.unsqueeze(1)

        # Modify the advantage with the task allocation parameters

        ini_values = torch.reshape(self.values.transpose(0, 1), 
            (self.num_procs, self.num_agents, self.num_frames_per_proc, self.num_objectives))[:, :, 0, :]
        

        # Compute H
        H = []
        for i in range(self.num_agents):
            Hi = self.computeH(ini_values, mu, i, c, e)
            H.append(Hi)

        H = torch.stack(H)
        # The output of H has shape A x P x O -> P x A x O -> (P . A ) x O
        H_ = H.transpose(0, 1).reshape(self.num_agents * self.num_procs, self.num_objectives)

        stack = []
        for k in range(self.num_agents *self.num_procs):
            stack.append(torch.matmul(self.advantages[:, k, :], H_[k, :]))

        mod_advantage = torch.stack(stack)

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs P = Procs x Agents in that order,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs * self.num_agents)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        #exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        # mod advantage already has shape P x T -> P * T
        exps.advantage = mod_advantage.reshape(-1)
        #exps.ini_values = ini_values
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        
        # T x P x O -> P x T x O -> P * T x O
        exps.value = self.values.transpose(0, 1).reshape(-1, *self.values.shape[2:])
        exps.reward = self.rewards.transpose(0, 1).reshape(-1, *self.rewards.shape[2:])
        exps.returnn = exps.value + \
            self.advantages.transpose(0, 1).reshape(-1, *self.advantages.shape[2:])
        # Preprocess experiences

        exps.obs = torch.tensor(np.array(exps.obs), device=self.device, dtype=torch.float)

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
        #return torch.tensor(np.array(obss), device=device, dtype=torch.float)
        t = torch.tensor(np.array(obss), device=device, dtype=torch.float)
        t = t.reshape(-1, *t.shape[2:])
        return t

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
        #H_2 =[] 
        H_.append(self.chi * self.df(X[:, agent, 0], c))
        #H_2.append(self.df(X[:, agent, 0], c))
        for j in range(1, y):
            #print(X[:, k, j - 1])
            acc = []
            for i in range(self.num_agents):
                acc.append(self.dh(mu[i, j - 1] * X[:, i, j], e))
            H_.append(torch.sum(torch.stack(acc, 1), 1) * mu[agent, j - 1])
            #H_2.append(
            #    self.dh(torch.sum(mu[:, j - 1].unsqueeze(1) * X.transpose(1, 0)[:, :, j], dim=0), e) \
            #        * mu[agent, j - 1]
            #)

        H = torch.stack(H_, 1)
        #H2 = torch.stack(H_2, 1)
        return H

def compute_alloc_loss(X, mu, e):
    _, num_agents, y = X.shape
    loss = 0.
    for j in range(1, y):
        with torch.no_grad():
            acc = []
            for i in range(num_agents):
                acc.append(dh(mu[i, j - 1] * X[:, i, j], e))
            h = torch.sum(torch.stack(acc, 1), 1)
            #h = dh(torch.sum(mu[:, j - 1].unsqueeze(1) * X.transpose(1, 0)[:, :, j], dim=0), e)
        acc = []
        for i in range(num_agents):
            mu_ = mu[i, j - 1] * X[:, i, j]
            acc.append(mu_)
        kappa_component = torch.sum(torch.stack(acc, 1), 1)
        #kappa_component = torch.sum(mu[:, j - 1].unsqueeze(1) * X.transpose(1, 0)[:, :, j], dim=0)
        loss += torch.mean(h * kappa_component)
    return loss

def dh(x, e):
    return torch.where(x <= e, 2 * (e - x), 0.0)
    
