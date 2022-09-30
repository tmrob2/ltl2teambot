import os
import gym
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


env = gym.make('CartPole-v1')

n_actions = env.action_space.n
STATE_DIM = 3
ACTOR_HIDDEN = 40
CRITIC_HIDDEN = 40

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, hidden_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dims, hidden_dims, batch_first=True)
        self.fc1 = nn.Linear(hidden_dims, n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc1(x)
        x = F.relu(x)
        return x, hidden

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, hidden_size):
        super(CriticNetwork, self).__init__()

        self.lstm = nn.LSTM(input_dims, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 1)
        
        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc1(x)
        return x, hidden

    