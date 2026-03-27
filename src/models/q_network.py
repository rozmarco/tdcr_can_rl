import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import RobotFeatureEncoder
from .decoder import Decoder

class QNetwork(nn.Module):
    def __init__(
        self, 
        r_dim: int,
        action_dim: int,
        d_embedding: int = 32,
        d_hidden: int = 64,
        d_state: int = 64,
        num_blocks: int = 3,
        **kwargs
    ):
        super(QNetwork, self).__init__()

        self.encoder = RobotFeatureEncoder(
            r_dim=r_dim, 
            d_hidden=d_hidden,
            d_embedding=d_embedding, 
        )
        self.decoder = Decoder(
            input_dim=action_dim,
            output_dim=d_hidden//2,
            d_cond=d_embedding,
            d_hidden=d_hidden,
            d_state=d_state,
            num_blocks=num_blocks
        )
        self.ff = nn.Linear(d_hidden//2, 1)

    def forward(self, s, a):
        # s: [Batch, Horizon, State]
        # a: [Batch, Horizon, Action]

        # Encode the state sequence -> [B, H, d_embedding]
        emb = self.encoder(s)

        # The decoder's FiLM conditioning expects a flat [B, d_embedding] vector,
        # not a sequence. Mean-pool across the horizon to produce a single context
        # vector that summarises the full state trajectory.
        #
        # This was the root cause of the [256, 256, 1, 16] conv1d crash: when H>1
        # (which only became common once reward shaping produced longer episodes),
        # passing [B, H, d_embedding] directly as FiLM conditioning z caused
        # broadcasting to create a 4D tensor before Mamba's conv1d.
        z = emb.mean(dim=1)    # [B, d_embedding]

        # Decode action trajectory conditioned on pooled state context
        emb = self.decoder(a, z)   # [B, H, d_hidden//2]

        r = self.ff(emb)           # [B, H, 1]

        # Return a scalar Q-value per sample by averaging over the horizon
        return r.mean(dim=1)       # [B, 1]