import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, d_input: int, d_output: int):
        super(FiLM, self).__init__()
        
        self.gamma = nn.Linear(d_input, d_output)
        self.beta = nn.Linear(d_input, d_output)

    def forward(self, x, z):
        return self.gamma(z) * x + self.beta(z)