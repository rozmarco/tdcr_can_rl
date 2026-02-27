import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_period: float = 10000.0):
        """
        Args:
            d_model: The dimension of the output embedding.
            max_period: Controls the frequency of the embeddings.
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.d_model // 2
        
        exponent = -torch.log(torch.tensor(self.max_period)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * exponent)
        
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        return emb