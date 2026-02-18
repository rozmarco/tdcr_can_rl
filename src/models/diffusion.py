import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class DiffusionNetwork(nn.Module):
    """
    Handles forward diffusion training and reverse sampling.
    """
    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        action_dim: int,
        diffusion_steps: int = 30,
        time_embed_type: str = "learned"
    ):
        super(DiffusionNetwork, self).__init__()

        self.d_hidden = d_hidden
        self.diffusion_steps = diffusion_steps
        self.time_embed_type = time_embed_type

        # Time embedding
        if time_embed_type == "learned":
            self.time_embed = nn.Embedding(diffusion_steps, d_latent)
        else:
            self.time_embed = None

        # Noise predictor
        self.noise_predictor = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, action_dim)
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


    def _get_time_embedding(self, t):
        if self.time_embed_type == "learned":
            return self.time_embed(t)

        # sinusoidal
        half_dim = self.d_hidden // 2
        emb = torch.exp(
            torch.arange(half_dim, device=t.device)
            * (-torch.log(torch.tensor(10000.0)) / (half_dim - 1))
        )
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


    def forward(self, z, target_action=None):
        """
        If target_action provided -> training mode
        If target_action is None -> sampling mode
        """

        if target_action is not None:
            # Training
            B = z.shape[0]
            device = z.device

            t = torch.randint(0, self.diffusion_steps, (B,), device=device)

            noise = torch.randn_like(target_action)

            alpha_bar_t = self.alpha_bar[t]
            sqrt_ab = alpha_bar_t.sqrt().unsqueeze(-1)
            sqrt_one_minus_ab = (1 - alpha_bar_t).sqrt().unsqueeze(-1)

            noisy_action = sqrt_ab * target_action + sqrt_one_minus_ab * noise

            t_embed = self._get_time_embedding(t)
            conditioned = z + t_embed

            predicted_noise = self.noise_predictor(conditioned)

            return {
                "predicted_noise": predicted_noise,
                "target_noise": noise,
                "timestep": t
            }

        else:
            # Reverse diffusion sampling
            return self.sample(z)

    @torch.no_grad()
    def sample(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        device = z.device

        action_dim = self.noise_predictor[-1].out_features
        action = torch.randn(B, action_dim, device=device)

        for t in reversed(range(self.diffusion_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            t_embed = self._get_time_embedding(t_tensor)
            conditioned = z + t_embed

            predicted_noise = self.noise_predictor(conditioned)

            # Example of classifier-free guidance
            # pred_next_state = self.state_decoder(z)
            # cost = my_cost_function(pred_next_state)  # Differentiable cost function
            # grad = torch.autograd.grad(cost, pred_next_state, retain_graph=True)[0]
            # modified_noise = predicted_noise + lambda_scale * grad

            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.beta[t]

            # DDPM update
            action = (1 / torch.sqrt(alpha_t)) * (
                action - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise
            )

            if t > 0:
                action += torch.sqrt(beta_t) * torch.randn_like(action)

        return action