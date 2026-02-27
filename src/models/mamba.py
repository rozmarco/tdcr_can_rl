import torch
import torch.nn as nn
import torch.nn.functional as F

from .ssm import StateSpaceModel

class MambaLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super(MambaLayer, self).__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Sequence, Features]
        x_norm = self.ln(x)

        # --- Main branch ---
        x1 = self.w1(x_norm)
        x_conv = self.conv1d(x1.transpose(2, 1)).transpose(2, 1)
        x_conv = self.silu(x_conv)
        x_ssm = self.ssm(x_conv)

        # --- Gating Mechanism ---
        v = self.silu(self.v1(x_norm))

        # --- Output ---
        out = self.w2(x_ssm * v)
        return out

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super(MambaBlock, self).__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.mamba = MambaLayer(d_model, d_state)

        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*3),
            nn.GELU(),
            nn.Linear(d_model*3, d_model)
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