import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 256
DROPOUT = 0.2
ACTIONS = 3

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc   = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        y = F.relu(self.norm(self.fc(x)))
        y = self.drop(y)
        return x + y

class RegretNet(nn.Module):
    def __init__(self, input_dim, output_dim=ACTIONS):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.Tanh(),
            nn.Dropout(DROPOUT)
        )
        self.trunk = nn.Sequential(
            ResidualBlock(HIDDEN),
            ResidualBlock(HIDDEN),
            ResidualBlock(HIDDEN)
        )
        self.head = nn.Linear(HIDDEN, output_dim)

    def forward(self, x):
        z = self.proj(x)
        z = self.trunk(z)
        return self.head(z)