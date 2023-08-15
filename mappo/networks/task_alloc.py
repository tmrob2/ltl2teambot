from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch

class Alloc(nn.Module):
    def __init__(self, num_tasks) -> None:
        super(Alloc, self).__init__()

        self.fc1 = nn.Linear(num_tasks, num_tasks)
    
    def forward(self, mu):
        x = self.fc1(mu)
        x = F.softmax(x, dim=0)
        return x
