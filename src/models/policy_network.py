import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .encoder import RobotFeatureEncoder
from .latent import LatentHead
from .decoder import Decoder
from .diffusion import DiffusionNetwork

class LatentPolicyPlanner(nn.Module):
    """
    Latent policy planner using VAE.

    The architecture is fully modular:

        1. SpatialTemporalEncoder: encodes robot and obstacle features, applies temporal dynamics (Mamba layers).
        2. LatentHead: maps fused features to a latent Gaussian distribution (mu, log_std) and samples z.
        3. Decoder: maps latent z back to next robot state.

    Args:
        r_dim (int): Dimension of the robot state input.
        action_dim (int): Dimension of the output action.
        d_embedding (int): Size of per-encoder feature embeddings.
        d_hidden_enc (int): Hidden size after fusion and Mamba layers.
        d_hidden_dec (int): Hidden size for the decoder.
        d_state (int): Size of state representation inside Mamba layers.
        d_latent (int): Size of latent vector for diffusion.
        num_layers_enc (int, optional): Number of Mamba layers in the encoder. Default: 1
        num_layers_dec (int, optional): Number of Mamba layers in the decoder. Default: 1
        diffusion_steps (int, optional): Number of diffusion steps. Default: 30
    """
    def __init__(
        self,
        r_dim: int,
        action_dim: int,
        d_embedding: int = 32,
        d_hidden_enc: int = 64,
        d_hidden_dec: int = 64,
        d_state: int = 32,
        d_latent: int = 64,
        num_blocks_enc: int = 3,
        num_blocks_dec: int = 5,
        expand_dec: int = 2,
        diffusion_steps: int = 30,
        max_period: float = 1000.0,
        **kwargs
    ):
        super(LatentPolicyPlanner, self).__init__()
        self.action_dim = action_dim

        # --- ENCODER ---
        self.encoder = RobotFeatureEncoder(
            r_dim,
            d_hidden_enc,
            d_embedding
        )

        self.latent = LatentHead(d_embedding, d_latent)

        # --- DIFFUSION ---
        # self.diffusion = DiffusionNetwork(
        #     Decoder(d_latent, d_latent, d_latent, d_hidden_dec,
        #             d_state, num_blocks=1, expand=expand_dec),
        #     d_latent,
        #     d_latent,
        #     diffusion_steps,
        #     max_period
        # )

        # --- DECODER ---
        self.decoder = Decoder(
            input_dim=d_embedding,
            output_dim=action_dim,
            d_cond=d_latent,
            d_hidden=d_hidden_dec,
            d_state=d_state,
            num_blocks=num_blocks_dec,
            expand=expand_dec
        )

    def log_pi(self, mu, log_std):
        normal = Normal(mu, torch.exp(log_std))
        # Sample using reparameterization
        u = normal.rsample()
        # Log probability before tanh
        log_prob = normal.log_prob(u).sum(dim=-1, keepdim=True)
        # Tanh correction: log(1 - tanh(u)^2)
        log_prob -= torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return log_prob

    def forward(
        self, 
        x: torch.Tensor, 
        horizon: int = 1
    ) -> torch.Tensor:
        """
        Perform a single-step reverse diffusion update for training the policy.

        Args:
            x (torch.Tensor): Current robot observation/state. Shape: [Batch, Features].
            horizon (int): Number of timesteps to rollout.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - actions (torch.Tensor): Tensor of shape [Batch, Horizon, Action_dim] containing
                the predicted actions after applying tanh.
                - log_pi (torch.Tensor): Tensor of shape [Batch, Horizon] (or compatible) representing
                the log-probabilities of the predicted actions under the policy.
        """
        fused = self.encoder(x)
        z, mu, log_std = self.latent(fused)

        # B, L, D = z.shape
        # device = z.device
        # t = torch.randint(0, self.diffusion.diffusion_steps, (B,), dtype=torch.long, device=device)
        # sequence = torch.randn((B, horizon, D), device=device)
        # latent_plan = self.diffusion.reverse(sequence, z, t)
        # plan = self.decoder(latent_plan, z)
        plan = self.decoder(fused, z)

        return F.tanh(plan), self.log_pi(mu, log_std)
    
    @torch.no_grad()
    def sample(
        self, 
        x: torch.Tensor, 
        horizon: int = 1
    ) -> torch.Tensor:
        """
        Perform a single/multi-step rollout of the policy over a given horizon.

        Args:
            x (torch.Tensor): Current robot observation/state. Shape: [Batch, Features].
            horizon (int): Number of timesteps to rollout.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - actions (torch.Tensor): Tensor of shape [Batch, Horizon, Action_dim] containing
                the predicted actions after applying tanh.
                - log_pi (torch.Tensor): Tensor of shape [Batch, Horizon] (or compatible) representing
                the log-probabilities of the predicted actions under the policy.
        """
        fused = self.encoder(x)
        z, mu, log_std = self.latent(fused)

        # latent_plan = self.diffusion.sample(z, horizon)
        # plan = self.decoder(latent_plan, z)
        plan = self.decoder(fused, z)
        
        return F.tanh(plan), self.log_pi(mu, log_std)