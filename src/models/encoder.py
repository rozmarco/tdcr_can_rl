import torch
import torch.nn as nn

from .mamba import MambaTransformerLayer

class RobotFeatureEncoder(nn.Module):
    """
    Encodes robot and obstacle features.
    """
    def __init__(
        self, 
        r_input_size: int, 
        d_hidden: int, 
        d_embedding: int,
        d_state: int,
        num_blocks: int
    ):
        super(RobotFeatureEncoder, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(r_input_size, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_embedding)
        )

        self.mamba_block = nn.Sequential(
            *[MambaTransformerLayer(d_embedding, d_state) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.mlp(x)
        x = self.mamba_block(x)
        return x