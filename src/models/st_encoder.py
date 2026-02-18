import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GraphNeuralNetwork
from .mamba import MambaLayer

class SpatialTemporalEncoder(nn.Module):
    """
    Encodes robot + obstacle features and processes them
    through stacked Mamba layers for temporal modeling.
    """
    def __init__(
        self,
        r_input_size: int,
        o_input_size: int,
        d_embedding: int,
        d_hidden: int,
        d_state: int,
        d_ff: int,
        num_layers: int,
    ):
        super(SpatialTemporalEncoder, self).__init__()

        self.robot_encoder = nn.Linear(r_input_size, d_embedding)
        self.obstacle_encoder = GraphNeuralNetwork(o_input_size, d_embedding)
        self.fusion = nn.Linear(d_embedding * 2, d_hidden)

        self.mamba_layers = nn.ModuleList([
            MambaLayer(d_hidden, d_state, d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x, graph, state):
        """
        Args:
            x: [B, r_input_size]
            graph: PyG graph object
            state: recurrent SSM state

        Returns:
            features: [B, d_hidden]
            new_state: updated SSM state
        """
        robot_feat = self.robot_encoder(x)
        spatial_feat = self.obstacle_encoder(graph.x, graph.edge_index)

        fused = torch.cat([robot_feat, spatial_feat], dim=-1)
        fused = self.fusion(fused)

        for layer in self.mamba_layers:
            fused, state = layer(fused, state)

        return fused, state