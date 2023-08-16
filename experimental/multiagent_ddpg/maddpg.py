import torch 
import torch.nn.functional as F
from agent import Agent

class MADDPG:
    def __init__(
            self, 
            actor_dims, 
            critic_dims,
            n_agents,
            n_actions,
            scenario='simple',
            alpha=0.01,
            beta=0.01,
            #fc1=64,
            #fc2=64,
            #gamma=0.99,
            #tau=0.01,
            chkpt_dir='tmp.maddpg'
        ):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario

        for agent_idx in range(self.n_agents):
            self.agents.append(Agent(
                actor_dims[agent_idx], 
                critic_dims, 
                n_actions, 
                n_agents, 
                agent_idx, 
                alpha=alpha, 
                beta=beta, 
                chkpt_dir=chkpt_dir
            ))

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return
        
        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones = memory.sample_buffer()
        
        device = self.agents[0].actor.device

        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards).to(device)
        states_ = torch.tensor(states_, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = torch.tensor(actor_new_states[agent_idx], dtype=torch.float).to(device)

            new_pi = agent.target_vector.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = torch.tensor(actor_states[agent_idx], dtype=torch.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = torch.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimiser.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimiser.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -torch.mean(actor_loss)
            agent.actor.optimiser.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimiser.step()

            agent.update_network_parameters()
