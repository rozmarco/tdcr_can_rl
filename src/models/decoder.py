import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaTransformerLayer
from .film import FiLM

class Decoder(nn.Module):
    """
    This architecture generates 'one-shot' action trajectories over a specified horizon. 

    Attributes:
        input_proj (nn.Module): Projects raw action noise into the latent manifold.
        blocks (nn.ModuleList): A stack of Mamba-FiLM blocks for deep temporal reasoning.
        output_proj (nn.Module): Projects processed features back to action space.
    """
    def __init__(
        self, 
        input_dim: int,
        output_dim: int, 
        d_cond: int,
        d_hidden: int, 
        d_state: int, 
        num_blocks: int,
        expand: int = 2
    ):
        """
        Args:
            output_dim (int): The dimensionality of the output space.
            d_hidden (int): The hidden dimension (d_model) used throughout the 
                Mamba blocks. This defines the capacity of the 'motion strategist'.
            d_state (int): The state expansion factor for the Mamba blocks, 
                governing the complexity of the internal recurrent dynamics.
            num_blocks (int): The number of repeated Mamba-FiLM layers. Higher 
                values allow for more sophisticated reasoning about long-term 
                consequences of actions.
        """
        super(Decoder, self).__init__()

        self.input_proj = nn.Linear(input_dim, d_hidden)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "film": FiLM(d_cond, d_hidden),
                "mamba": MambaTransformerLayer(d_hidden, d_state, expand),
            }) for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(d_hidden, output_dim)

    def forward(self, x, z):
        # [Batch, Horizon, Output_dim]
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block["film"](x, z)
            x = block["mamba"](x)

        x = self.output_proj(x)

        return x