import torch
import torch.nn as nn

class SpatialTemporalEncoder(nn.Module):
    """
    Encodes robot and obstacle features.
    """
    def __init__(
        self, 
        r_input_size: int, 
        d_hidden: int, 
        d_embedding: int
    ):
        super(SpatialTemporalEncoder, self).__init__()
        
        self.robot_encoder = nn.Sequential(
            nn.Linear(r_input_size, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_embedding)
        )

    def forward(self, x):
        return self.robot_encoder(x)