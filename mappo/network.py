import os
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class ExtractTensor(nn.Module):
    def forward(self, x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # reshape (batch, hidden)
        return tensor[: , -1, :]

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, hidden_dims=256, fc1_dims=256, 
        fc2_dims=256, chkpt_dir='tmp/ppo') -> None:
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        #self.actor = nn.Sequential(
        #    nn.LSTM(input_dims, hidden_dims),
        #    ExtractTensor(),
        #    nn.Linear(hidden_dims, fc1_dims),
        #    nn.ReLU(),
        #    nn.Linear(fc1_dims, fc2_dims),
        #    nn.ReLU(),
        #    nn.Linear(fc2_dims, n_actions),
        #    nn.Softmax(dim=-1)
        #)
        self.lstm_layer = nn.LSTM(input_dims, hidden_dims, batch_first=True)
        self.fc1 = nn.Linear(hidden_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.action_layer = nn.Linear(fc2_dims, n_actions)
        self.output_layer = nn.Softmax(dim=-1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        self.to(self.device)

    def forward(self, state, recurrent_cell):
        dist, (hxs, cxs) = self.lstm_layer(state, recurrent_cell)
        dist = torch.tanh(self.fc1(dist))
        dist = torch.tanh(self.fc2(dist))
        dist = self.action_layer(dist)
        dist = self.output_layer(dist)
        dist = Categorical(dist)

        return dist, (hxs, cxs)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, beta, hidden_dims=256, fc1_dims=256, 
        fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()
        
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        #self.actor = nn.Sequential(
        #    nn.LSTM(input_dims, hidden_dims),
        #    ExtractTensor(),
        #    nn.Linear(hidden_dims, fc1_dims),
        #    nn.ReLU(),
        #    nn.Linear(fc1_dims, fc2_dims),
        #    nn.ReLU(),
        #    nn.Linear(fc2_dims, 1)
        #)

        self.lstm_layer = nn.LSTM(input_dims, hidden_dims, batch_first=True)
        self.fc1 = nn.Linear(hidden_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.value_layer = nn.Linear(fc2_dims, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, recurrent_cell):
        #value = self.critic(state)
        state_value, (hxs, cxs) = self.lstm_layer(state, recurrent_cell)
        state_value = torch.tanh(self.fc1(state_value))
        state_value = torch.tanh(self.fc2(state_value))
        state_value = self.value_layer(state_value)
        return state_value, (hxs, cxs)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
