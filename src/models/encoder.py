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
        # Ensure input is on the same device as the model weights.
        # Ray workers serialize tensors to CPU; this guards against the
        # "Expected all tensors to be on the same device" RuntimeError.
        device = next(self.parameters()).device
        x = x.to(device)
        return self.mlp(x)