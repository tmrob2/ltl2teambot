import torch
from network import ActorNetwork, CriticNetwork
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class Agent:
    def __init__(
            self, 
            actor_dims, 
            critic_dims, 
            n_actions, 
            n_agents, 
            agent_idx, 
            chkpt_dir,
            alpha=0.01, 
            beta=0.01, 
            fc1=64, 
            fc2 = 64,
            gamma=0.95, 
            tau=0.01
        ):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = f"agent_{agent_idx}"
        
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
            chkpt_dir=chkpt_dir, name=self.agent_name+'_actor')
        
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, 
            n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_critic')

        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
            chkpt_dir=chkpt_dir, name=self.agent_name+'_target_actor')

        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, 
            n_actions, chkpt_dir=chkpt_dir, name=self.agent_name+'_target_critic')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        observation = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        mu = self.actor.forward(observation)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                    (1- tau) * target_actor_state_dict[name].clone()
            self.target_actor.load_state_dict(actor_state_dict)

            target_critic_params = self.target_critic.named_parameters()
            critic_params = self.critic.named_parameters()

            target_critic_state_dict = dict(target_critic_params)
            critic_state_dict = dict(critic_params)

            for name in critic_state_dict:
                critic_state_dict[name] = tau * critic_state_dict[name].clone() \
                    (1 - tau) * target_critic_state_dict[name].clone()

            self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()