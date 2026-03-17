import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from .pe import SinusoidalPositionalEncoding

class DiffusionNetwork(nn.Module):
    """
    Handles forward diffusion training and reverse sampling.
    """
    def __init__(
        self,
        noise_predictor,
        d_hidden: int,
        output_dim: int,
        diffusion_steps: int = 30,
        max_period: float = 1000.0
    ):
        super(DiffusionNetwork, self).__init__()

        self.noise_predictor = noise_predictor
        self.output_dim = output_dim
        self.diffusion_steps = diffusion_steps

        self.pe = SinusoidalPositionalEncoding(d_hidden, max_period)

        self._init_schedule()

    def _init_schedule(self):
        steps = self.diffusion_steps
        t = torch.linspace(0, steps, steps + 1) / steps

        alpha_bar = torch.cos((t + 0.008) / 1.008 * torch.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        beta = beta.clamp(0, 0.999)

        alpha = 1. - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def get_time_embedding(self, t):
        return self.pe(t)

    def forward(self, target):
        B = target.shape[0]
        device = target.device

        t = torch.randint(0, self.diffusion_steps, (B,), device=device)
        noise = torch.randn_like(target)

        alpha_bar_t = self.alpha_bar[t]
        sqrt_ab = alpha_bar_t.sqrt().unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_ab = (1.0 - alpha_bar_t).sqrt().unsqueeze(-1).unsqueeze(-1)
        noisy_target = sqrt_ab * target + sqrt_one_minus_ab * noise

        return {
            "noisy_target": noisy_target,
            "target_noise": noise,
            "timestep": t
        }
    
    def reverse(
        self, 
        sequence: torch.Tensor, 
        z: torch.Tensor, 
        t: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        """
        # Reverse process: One-shot noise prediction
        B = z.shape[0]
        device = z.device

        # Predict noise for the full sequence
        if isinstance(t, int):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        else:
            t_tensor = t

        t_embed = self.get_time_embedding(t_tensor)
        z_expanded = z + t_embed.unsqueeze(1)
        predicted_noise = self.noise_predictor(sequence, z_expanded)

        # Coefficients
        alpha_t = self.alpha[t_tensor].view(B,1,1)
        alpha_bar_t = self.alpha_bar[t_tensor].view(B,1,1)
        beta_t = self.beta[t_tensor].view(B,1,1)

        # DDPM update
        sequence = (
            (1 / torch.sqrt(alpha_t)) * 
            (sequence - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)
        )

        # Add noise except for t = 0
        mask = (t_tensor > 0).float().view(B,1,1)
        sequence += mask * torch.sqrt(beta_t) * torch.randn_like(sequence)

        return sequence

    @torch.no_grad()
    def sample(
        self, 
        z: torch.Tensor, 
        horizon: int = 1
    ) -> torch.Tensor:
        """
        """
        B = z.shape[0]
        device = z.device

        sequence = torch.randn((B, horizon, self.output_dim), device=device)

        for t in reversed(range(self.diffusion_steps)):
            sequence = self.reverse(sequence, z, t)

        return sequence