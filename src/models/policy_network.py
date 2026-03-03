import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable

from .encoder import RobotFeatureEncoder
from .latent import LatentHead
from .decoder import NoisePredictor
from .diffusion import DiffusionNetwork

class LatentDiffusionPolicyPlanner(nn.Module):
    """
    Latent diffusion policy planner with modular encoder, latent VAE, and diffusion modules.

    The architecture is fully modular:

        1. SpatialTemporalEncoder: encodes robot and obstacle features, applies temporal dynamics (Mamba layers).
        2. LatentHead: maps fused features to a latent Gaussian distribution (mu, log_std) and samples z.
        3. DiffusionNetwork: predicts noise in latent space and performs forward/reverse diffusion.
        4. State decoder: maps latent z back to next robot state.

    Args:
        r_dim (int): Dimension of the robot state input.
        o_dim (int): Dimension of the obstacle / graph input.
        action_dim (int): Dimension of the output action.
        d_embedding (int): Size of per-encoder feature embeddings.
        d_hidden (int): Hidden size after fusion and Mamba layers.
        d_latent (int): Size of latent vector for diffusion.
        num_layers (int, optional): Number of Mamba layers. Default: 1
        d_ff (int): Feedforward size inside each Mamba layer. Default: 64
        diffusion_steps (int, optional): Number of diffusion steps. Default: 60
        time_embed_type (str, optional): "learned" or "sinusoidal" time embeddings. Default: "learned"
    """
    def __init__(
        self,
        r_dim: int,
        action_dim: int,
        d_embedding: int = 32,
        d_hidden_enc: int = 64,
        d_state: int = 32,
        d_latent: int = 64,
        num_blocks: int = 3,
        num_blocks_dec: int = 5,
        diffusion_steps: int = 30,
        max_period: float = 10000.0
    ):
        super(LatentDiffusionPolicyPlanner, self).__init__()
       
        # --- ENCODER ---
        self.encoder = RobotFeatureEncoder(
            r_dim,
            d_hidden_enc,
            d_embedding,
            d_state,
            num_blocks
        )

        self.latent = LatentHead(d_embedding, d_latent)

        # --- DECODER ---
        self.decoder = NoisePredictor(
            action_dim,
            d_latent,
            d_state,
            num_blocks_dec
        )

        self.diffusion = DiffusionNetwork(
            self.decoder,
            d_latent,
            action_dim,
            diffusion_steps,
            max_period
        )

    def log_pi(self, actions, z):
        # Run the diffusion forward
        output = self.diffusion(z, actions)
        noisy_action = output["noisy_action"]        # [B, H, d_a]
        target_noise = output["target_noise"]        # [B, H, d_a]
        t = output["timestep"]                       # [B]

        # Time embedding for conditioning
        t_embed = self.diffusion.get_time_embedding(t)        # [B, d_z]
        z_expanded = z + t_embed.unsqueeze(1)                 # [B, 1, d_z]
        predicted_noise = self.diffusion.noise_predictor(noisy_action, z_expanded)  # [B, H, d_a]

        # Compute log probability under unit Gaussian
        # log N(x | mu, sigma=1) = -0.5 * (x - mu)^2 - 0.5 * log(2*pi)
        log_prob_per_dim = -0.5 * (target_noise - predicted_noise)**2 - 0.5 * math.log(2*math.pi)
        # Sum over action dimensions and timesteps
        log_pi = log_prob_per_dim.sum(dim=(-1))  # sum over d_a for log-prob per timestep.

        return log_pi  # [B]
    
    def rollout(
        self, 
        x: torch.Tensor, 
        horizon: int = 1,
        guide_fn: Optional[Callable] = None,
        scale: float = 0.1
    ) -> torch.Tensor:
        """
        Perform a multi-step rollout of the policy over a given horizon.

        Args:
            x (torch.Tensor): Current robot observation/state. Shape: [Batch, Features].
            graph (torch.Tensor): Graph or spatial context input. Can be static if obstacles are static.
            horizon (int): Number of timesteps to rollout.

        Returns:
            torch.Tensor: Tensor of shape [Batch, Horizon, Action_dim] containing
                            the predicted actions for each timestep.
        """
        fused = self.encoder(x)
        z, mu, std = self.latent(fused)
        plan = self.diffusion.sample(z, horizon, guide_fn, scale)
        log_pi = self.log_pi(plan, z)
        return F.tanh(plan), log_pi