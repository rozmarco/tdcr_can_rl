import torch
import torch.nn as nn
import torch.nn.functional as F

class RobotFeatureEncoder(nn.Module):
    """
    Encodes robot and obstacle features.
    """
    def __init__(
        self, 
        r_dim: int, 
        d_hidden: int, 
        d_embedding: int,
        **kwargs
    ):
        super(RobotFeatureEncoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(r_dim, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_embedding)
        )

    def forward(self, x):
        return self.mlp(x)