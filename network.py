import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class Q_estimator(nn.Module):
    """Q(s,a)"""
    def __init__(self, state_dim, action_dim, out=1, fc1=64, fc2=128):
        super(Q_estimator, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, fc1)
        self.fc2 = nn.Linear(fc1+action_dim, fc2)
        self.fc3 = nn.Linear(fc2, out)

    def forward(self, state, action):
        x = self.fc1(state)
        x = torch.tanh(x)
        x = torch.cat([x, action], dim=1)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)

        return x


class mu_estimator(nn.Module):
    """deterministic policy mu(s)=a """
    def __init__(self, state_dim, action_dim, fc1=64, fc2=64):
        super(mu_estimator, self).__init__()

        self.mu =  nn.Sequential(
                                    nn.Linear(state_dim, fc1),
                                    nn.Tanh(),
                                    nn.Linear(fc1, fc2),
                                    nn.Tanh(),
                                    nn.Linear(fc2, action_dim),
                                    nn.Tanh()
                                    )
        
    def forward(self, state):
        x = self.mu(state)

        return x