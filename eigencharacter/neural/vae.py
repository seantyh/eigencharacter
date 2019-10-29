import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_dim):
        super(VAE, self).__init__()
        
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, 500)
        self.fc21 = nn.Linear(500, 50)
        self.fc22 = nn.Linear(500, 50)
        self.fc3 = nn.Linear(50, 500)
        self.fc4 = nn.Linear(500, in_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.in_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar