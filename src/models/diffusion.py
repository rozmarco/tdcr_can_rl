import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable

from .pe import SinusoidalPositionalEncoding

class DiffusionNetwork(nn.Module):
    """
    Handles forward diffusion training and reverse sampling.
    """
    def __init__(
        self,
        noise_predictor: nn.Module,
        d_hidden: int,
        action_dim: int,
        diffusion_steps: int = 30,
        max_period: float = 10000.0
    ):
        super(DiffusionNetwork, self).__init__()

        self.noise_predictor = noise_predictor
        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps

        self.pe = SinusoidalPositionalEncoding(d_hidden, max_period)
        self.time_embeddings = nn.Sequential(
            nn.Linear(d_hidden, d_hidden * 2),
            nn.GELU(),
            nn.Linear(d_hidden * 2, d_hidden)
        )

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
        t_raw = self.pe(t)
        t_emb = self.time_embeddings(t_raw)
        return t_emb

    def forward(self, z: torch.Tensor, target_action: torch.Tensor):
        B = z.shape[0]
        device = z.device

        t = torch.randint(0, self.diffusion_steps, (B,), device=device)
        noise = torch.randn_like(target_action)

        alpha_bar_t = self.alpha_bar[t]
        sqrt_ab = alpha_bar_t.sqrt().unsqueeze(-1).unsqueeze(-1)
        sqrt_one_minus_ab = (1.0 - alpha_bar_t).sqrt().unsqueeze(-1).unsqueeze(-1)
        # print(sqrt_ab.shape, target_action.shape, sqrt_one_minus_ab.shape, noise.shape)
        noisy_action = sqrt_ab * target_action + sqrt_one_minus_ab * noise

        t_embed = self.get_time_embedding(t).unsqueeze(1)
        conditioned = z + t_embed
        predicted_noise = self.noise_predictor(noise, conditioned)

        return {
            "predicted_noise": predicted_noise,
            "noisy_action": noisy_action,
            "target_noise": noise,
            "timestep": t
        }

    def sample(
        self, 
        z: torch.Tensor, 
        horizon: int = 1, 
        guide_fn: Optional[Callable] = None,
        scale: float = 0.1
    ) -> torch.Tensor:
        
        B = z.shape[0]
        device = z.device
        sequence = torch.randn(B, horizon, self.action_dim, device=device)

        for t in reversed(range(self.diffusion_steps)):
            # Predict noise for the full sequence
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            t_embed = self.get_time_embedding(t_tensor)
            z_expanded = z + t_embed.unsqueeze(1)
            predicted_noise = self.noise_predictor(sequence, z_expanded)

            # Classifier Guidance
            if guide_fn is not None:
                with torch.enable_grad():
                    sequence.requires_grad_(True)
                    # The guide_fn should return a scalar (like 'distance to obstacle')
                    loss = guide_fn(sequence)
                    grad = torch.autograd.grad(loss, sequence)[0]
                    sequence.requires_grad_(False)
                predicted_noise = predicted_noise + scale * grad

            # DDPM update
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            sequence = (1 / torch.sqrt(alpha_t)) * (sequence - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise)

            if t > 0:
                sequence += torch.sqrt(beta_t) * torch.randn_like(sequence)

        return sequence