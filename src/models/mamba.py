import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import StateSpaceModel

class MambaBlock(nn.Module):
    """
    Description:
        For time-feature representation learning.
        
    Brief:
        [Input] Temporal features [N, A, T, F]
        [Output] Temporal features [N, A, T, F]

    Args:
        d_model (int): Dimension of the hidden layer
        d_state (int):
    """
    def __init__(self, d_model: int, d_state: int):
        super(MambaBlock, self).__init__()

        self.ln = nn.LayerNorm(d_model)

        # Computation
        self.w1 = nn.Linear(d_model, d_model)
        self.v1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)

        # Local-feature convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_model, 
            out_channels=d_model, 
            kernel_size=3, 
            padding=1,
            groups=d_model,
            bias=False
        )

        self.silu = nn.SiLU()

        # Communication
        self.ssm = StateSpaceModel(d_model, d_state)

    def forward(self, x: torch.Tensor, ssm_state: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Features]
        x_norm = self.ln(x)

        # --- Main branch ---
        x1 = self.w1(x_norm)
        x_conv = self.conv1d(x1.unsqueeze(-1)).squeeze(-1) # [B, C, L]
        x_conv = self.silu(x_conv)
        x_ssm, next_ssm_state = self.ssm(ssm_state, x_conv)

        # --- Gating Mechanism ---
        v = self.silu(self.v1(x_norm))

        # --- Output ---
        out = self.w2(x_ssm * v)
        return out, next_ssm_state

class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_ff: int):
        super(MambaLayer, self).__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x: torch.Tensor, ssm_state: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.ln1(x)
        x, next_ssm_state = self.mamba(x, ssm_state)
        x = x + residual

        residual = x

        x = self.ln2(x)
        x = self.ff(x)
        x = x + residual

        return x, next_ssm_state
