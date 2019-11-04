import torch
import torch.nn as nn

class MF_Model(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MF_Model, self).__init__()
        self.fn1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh())

    def forward(self, X):        
        out = self.fn1(X)
        return out