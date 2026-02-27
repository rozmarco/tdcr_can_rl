import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentHead(nn.Module):
    """
    Produces latent Gaussian distribution and samples z.
    """
    def __init__(self, d_hidden: int, d_latent: int):
        super(LatentHead, self).__init__()

        self.mean = nn.Linear(d_hidden, d_latent)
        self.log_std = nn.Linear(d_hidden, d_latent)

    def forward(self, features):
        mu = self.mean(features)
        log_std = self.log_std(features)

        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, log_std