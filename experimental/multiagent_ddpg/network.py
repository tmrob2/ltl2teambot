import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(
            self,
            beta, 
            input_dims, 
            fc1_dims,
            fc2_dims,
            n_agents,
            n_actions,
            name, 
            chkpt_dir
        ):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name)
        # full state observation of the entire system
        # full action vectors for each of the agents
        self.fc1 = nn.Linear(input_dims + n_agents*n_actions, fc1_dims) 
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(
            self, 
            alpha, 
            input_dims, 
            fc1_dims, 
            fc2_fdims, 
            n_actions, # what is this when the box is continuous
            name, 
            chkpt_dir
        ):

        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_fdims)
        self.pi = nn.Linear(fc2_fdims, n_actions)

        self.optimiser = optim.Adam(self.parameters(), lr=alpha())
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = torch.softmax(self.pi(x), dim=1)

        return pi
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

