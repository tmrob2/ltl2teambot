import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, in_size, hidden_size,
        fc2_dims=256, chkpt_dir='tmp/maa2c'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'ia2c')

        self.fc2_dims = fc2_dims

        self.lstm = nn.LSTM(in_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, self.fc2_dims)
        self.fc2 = nn.Linear(self.fc2_dims, n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=alpha())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, state, hidden):
        x, hidden = self.lstm(x, hidden)
        