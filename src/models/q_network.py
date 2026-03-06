import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import RobotFeatureEncoder
from .decoder import NoisePredictor

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
        self.encoder = RobotFeatureEncoder(r_dim, d_embedding, d_hidden, d_state, num_blocks)
        self.backbone = NoisePredictor(action_dim, d_hidden, d_state, num_blocks)
        self.ff = nn.Linear(action_dim, 1)

    def _get_discounted(self, r):
        # Cumulative Discounted Reward
        B, H, _ = r.shape

        gamma = torch.tensor([0.99], device=r.device)
        k = torch.arange(0, H, step=1, device=r.device)
        weights = torch.pow(gamma, k).view(1, -1, 1) # [1, H, 1]

        return (weights * r).sum(dim=-1) # [B, H, 1] -> [B, H]

    def forward(self, s, a):
        # s: [Batch, Horizon, State]
        # a: [Batch, Horizon, Action]
        emb = self.encoder(s)
        emb = self.backbone(a, emb)
        out = self.ff(emb)
        # r: [Batch, Horizon, 1]
        return self._get_discounted(out)