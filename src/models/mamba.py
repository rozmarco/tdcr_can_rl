import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import StateSpaceModel

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super(MambaBlock, self).__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Sequence, Features]

        # --- Main branch ---
        x1 = self.w1(x)
        x_conv = self.conv1d(x1.transpose(2, 1)).transpose(2, 1)
        x_conv = self.silu(x_conv)
        x_ssm = self.ssm(x_conv)

        # --- Gating Mechanism ---
        v = self.silu(self.v1(x))

        # --- Output ---
        out = self.w2(x_ssm * v)
        return out

class MambaTransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, expand: int=2):
        super(MambaTransformerLayer, self).__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * expand),
            nn.GELU(),
            nn.Linear(d_model * expand, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.ln1(x)
        x = self.mamba(x)
        x = x + residual

        residual = x

        x = self.ln2(x)
        x = self.ff(x)
        x = x + residual

        return x