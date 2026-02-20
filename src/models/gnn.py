import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

def generate_edges(n: int) -> torch.Tensor:
    """
    Generate an edge_index connecting nodes across multiple agents.
    """
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)

    # Nodes 1 to n-1
    others = torch.arange(1, n)

    # 0 -> others
    edges_out = torch.stack([
        torch.zeros(n - 1, dtype=torch.long),
        others
    ], dim=0)

    # others -> 0
    edges_in = torch.stack([
        others,
        torch.zeros(n - 1, dtype=torch.long)
    ], dim=0)

    # Combine
    edge_index = torch.cat([edges_out, edges_in], dim=1)

    return edge_index

class GraphNeuralNetwork(MessagePassing):
    """
    Attention-Gated Graph Neural Network with Global Attention Pooling.

    This module implements a message-passing graph neural network (GNN)
    designed to model pairwise interactions between a primary agent
    (e.g., a robot) and surrounding entities (e.g., obstacles).

    The architecture consists of:

        1. Linear input projection into a shared embedding space.
        2. Attention-based message passing:
            - Computes pairwise attention scores between nodes.
            - Applies a learned gating mechanism to modulate messages.
            - Aggregates messages using additive aggregation.
        3. Global attention pooling:
            - Computes node-level importance weights.
            - Produces a single pooled graph embedding.

    Key Characteristics:
        - Directed message passing (typically robot ↔ obstacles).
        - Learnable attention mechanism for interaction importance.
        - Feature-wise gating for adaptive message modulation.
        - Graph-level embedding output via attention pooling.

    Args:
        in_channels (int):
            Dimensionality of input node features.
        out_channels (int):
            Dimensionality of projected node embeddings and output features.

    Input:
        x (torch.Tensor):
            Node feature matrix of shape [num_nodes, in_channels].
            Conventionally, node 0 represents the primary agent
            (e.g., robot) and remaining nodes represent surrounding entities.
        edge_index (torch.Tensor):
            Graph connectivity in COO format with shape [2, num_edges].

    Output:
        torch.Tensor:
            Graph-level embedding of shape [1, out_channels],
            obtained via attention-weighted global pooling.

    Methods:
        generate_edges(n: int) -> torch.Tensor:
            Generates bidirectional edges between node 0 and
            all other nodes (star topology).

        forward(x, edge_index):
            Executes projection, message passing, and global pooling.

        message(x_i, x_j, ...):
            Computes attention-weighted and gated messages.

        update(aggr_out):
            Returns aggregated node features.
    """
    def __init__(self, in_channels, out_channels):
        super(GraphNeuralNetwork, self).__init__(aggr='add')

        self.out_channels = out_channels
        self.in_proj = nn.Linear(in_channels, out_channels, bias=False)

        # Attention MLP
        self.att_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, 1)
        )

        # Gating mechanism
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.Sigmoid()
        )

        # Attention Pooling
        self.pool_att = nn.Sequential(
            nn.Linear(out_channels, out_channels//2),
            nn.GELU(),
            nn.Linear(out_channels//2, 1)
        )

    def forward(self, x, edge_index):
        # x: [Nodes, in_channels] (e.g., Node 0 is Robot, others are Obstacles)
        x = self.in_proj(x)
        x = self.propagate(edge_index, x=x)
        x = (torch.softmax(self.pool_att(x), dim=0) * x).sum(dim=0, keepdim=True)  # [1, 64]
        return x

    def message(self, x_i, x_j, index, ptr, size_i):
        # x_i: Robot features
        # x_j: Obstacle features
        
        x_cat = torch.cat([x_i, x_j], dim=-1) # [Edges, heads, 2 * out_channels]
        
        # --- Attention ---
        # Compute pairwise importance
        alpha = self.att_mlp(x_cat)           # [E, 1]
        alpha = softmax(alpha, index, ptr, size_i)

        # --- Gating ---
        gate = self.gate_mlp(x_cat)           # [E, out_channels]

        message = x_j * gate                  # gated message
        message = message * alpha             # attention-weighted
        
        return message

    def update(self, aggr_out):
        # Combine multi-head results back into the d_model dimension
        return aggr_out.view(-1, self.out_channels)