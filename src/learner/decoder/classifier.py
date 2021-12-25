import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(x):
    return x / (1e-6 + x.pow(2).sum(dim=-1, keepdim=True).sqrt())

class MLP(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.core = nn.Sequential([
            nn.Linear(indim, indim // 2), nn.ReLU(),
            nn.Linear(indim // 2, indim // 2), nn.ReLU(),
            nn.Linear(indim // 2, outdim)
        ])

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.core(x)
