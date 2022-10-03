from mappo.network import ActorNetwork, CriticNetwork
from mappo.memory import PPOMemory
import torch
import numpy as np

class Agent:
    def __init__(self, n_actions, input_dim, gamma=0.99, alpha=0.0003, beta=0.0003, 
        gae_lambda=0.95, policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dim, alpha)
        self.critic = CriticNetwork(input_dim, beta)
        self.memory = PPOMemory(batch_size=batch_size)

        #self.actor_recurrent_cell = (ahxs, acxs)
        #self.critic_recurrent_cell = (chxs, ccxs)

    def remember(self, state, action, probs, vals, reward, done, actor_rcell, critic_rcell):
        self.memory.store_memory(state, action, probs, vals, reward, done, actor_rcell, critic_rcell)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, actor_rcell, critic_rcell):
        # the actor recurrent_cell should be a tuple (hxs, cxs)
        # the critic recurrent_cell should also be a tuple

        # the LSTM layer expects a 3D array, the observation from a gym environment
        # is not even a tensor, first convert to tensor, then add dimensons
        # 1d -> 2d -> 3d
        state = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        state = state.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            dist, actor_rcell = self.actor(state, actor_rcell)
            value, critic_rcell = self.critic(state, critic_rcell)

        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value, actor_rcell, critic_rcell

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, ahx, acx, chx, ccx, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1]* \
                        (1 - int(dones_arr[k])) - values[k])

                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)
                ahx_ = torch.tensor(ahx[batch]).to(self.actor.device)
                acx_ = torch.tensor(acx[batch]).to(self.actor.device)
                chx_ = torch.tensor(chx[batch]).to(self.actor.device)
                ccx_ = torch.tensor(ccx[batch]).to(self.actor.device)
    
                states = states.unsqueeze(0)

                dist, _ = self.actor(states.transpose(0, 1), (ahx_.squeeze(1).transpose(0, 1), acx_.squeeze(1).transpose(0, 1)))
                critic_value, _ = self.critic(states.transpose(0, 1), (chx_.squeeze(1).transpose(0, 1), ccx_.squeeze(1).transpose(0, 1)))

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clip_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 
                    1 + self.policy_clip
                ) * advantage[batch]

                actor_loss = -torch.min(weighted_probs, weighted_clip_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimiser.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimiser.step()

        self.memory.clear_memory()