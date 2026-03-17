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
        emb = self.encoder(s)
        emb = self.decoder(a, emb)
        r = self.ff(emb)
        # r: [Batch, Horizon, 1]
        return r