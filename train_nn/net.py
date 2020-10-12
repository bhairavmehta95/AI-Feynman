import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, ninputs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(ninputs, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
        
    def forward(self, x):
        return self.model(x)