import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaTransformerLayer
from .film import FiLM

class NoisePredictor(nn.Module):
    """
    This architecture generates 'one-shot' action trajectories over a specified horizon. 

    Attributes:
        input_proj (nn.Module): Projects raw action noise into the latent manifold.
        blocks (nn.ModuleList): A stack of Mamba-FiLM blocks for deep temporal reasoning.
        output_proj (nn.Module): Projects processed features back to action space.
    """
    def __init__(
        self, 
        action_dim: int, 
        d_latent: int, 
        d_state: int, 
        num_blocks: int
    ):
        """
        Args:
            action_dim (int): The dimensionality of the action space.
            d_latent (int): The hidden dimension (d_model) used throughout the 
                Mamba blocks. This defines the capacity of the 'motion strategist'.
            d_state (int): The state expansion factor for the Mamba blocks, 
                governing the complexity of the internal recurrent dynamics.
            num_blocks (int): The number of repeated Mamba-FiLM layers. Higher 
                values allow for more sophisticated reasoning about long-term 
                consequences of actions.
        """
        super(NoisePredictor, self).__init__()

        self.input_proj = nn.Linear(action_dim, d_latent)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "film": FiLM(d_latent, d_latent),
                "mamba": MambaTransformerLayer(d_latent, d_state),
            }) for _ in range(num_blocks)
        ])
        
        self.output_proj = nn.Linear(d_latent, action_dim)

    def forward(self, x, z):
        # [Batch, Horizon, Action_dim]

        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block["film"](x, z)
            x = block["mamba"](x)

        x = self.output_proj(x)
            
        return x