import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, d_input: int, d_output: int):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(d_input, d_output)
        self.beta  = nn.Linear(d_input, d_output)

    def forward(self, x, z):
        # x: [B, H, d_output]  — sequence being conditioned
        # z: [B, d_input]      — flat conditioning vector (must be 2D)
        #
        # gamma(z) and beta(z) are [B, d_output].
        # Unsqueeze to [B, 1, d_output] so the broadcast across H is
        # explicit and stays 3D — without this, PyTorch auto-broadcasts
        # in a way that can produce 4D tensors before Mamba's conv1d.
        g = self.gamma(z).unsqueeze(1)  # [B, 1, d_output]
        b = self.beta(z).unsqueeze(1)   # [B, 1, d_output]
        return g * x + b                # [B, H, d_output]