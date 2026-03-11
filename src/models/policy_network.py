import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        max_period: float = 10000.0,
        **kwargs
    ):
        super(LatentDiffusionPolicyPlanner, self).__init__()
        self.action_dim = action_dim

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
        output = self.diffusion(actions)
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
        z, mu, std = self.latent(fused)

        B = z.shape[0]
        device = z.device
        t = torch.randint(0, self.diffusion.diffusion_steps, (B,), device=device, dtype=torch.long)
        sequence = torch.randn(B, horizon, self.action_dim, device=device)
        plan = self.diffusion.reverse(sequence, z, t)

        return F.tanh(plan), self.log_pi(plan, z)
    
    @torch.no_grad()
    def sample(
        self, 
        x: torch.Tensor, 
        horizon: int = 1
    ) -> torch.Tensor:
        """
        Perform a multi-step rollout of the policy over a given horizon.

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
        z, mu, std = self.latent(fused)
        plan = self.diffusion.sample(z, horizon)
        return F.tanh(plan), self.log_pi(plan, z)