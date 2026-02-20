import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from .st_encoder import SpatialTemporalEncoder
from .diffusion import DiffusionNetwork
from .latent import LatentHead

class LatentDiffusionPolicyNetwork(nn.Module):
    """
    Latent diffusion policy network with modular encoder, latent VAE, and diffusion modules.

    The architecture is fully modular:

        1. SpatialTemporalEncoder: encodes robot and obstacle features, applies temporal dynamics (Mamba layers).
        2. LatentHead: maps fused features to a latent Gaussian distribution (mu, log_std) and samples z.
        3. DiffusionNetwork: predicts noise in latent space and performs forward/reverse diffusion.
        4. State decoder: maps latent z back to next robot state.

    Args:
        r_input_size (int): Dimension of the robot state input.
        o_input_size (int): Dimension of the obstacle / graph input.
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
        r_input_size: int,
        o_input_size: int,
        action_dim: int,
        horizon: int = 1,
        d_embedding: int = 64,
        d_hidden: int = 128,
        d_state: int = 64,
        d_latent: int = 128,
        d_ff: int = 64,
        num_layers: int = 2,
        diffusion_steps: int = 30,
        time_embed_type: str = "learned"
    ):
        super(LatentDiffusionPolicyNetwork, self).__init__()
        self.horizon = horizon
       
        self.encoder = SpatialTemporalEncoder(
            r_input_size,
            o_input_size,
            d_embedding,
            d_hidden,
            d_state,
            d_ff,
            num_layers
        )

        self.latent = LatentHead(d_hidden, d_latent)

        self.diffusion = DiffusionNetwork(
            d_latent,
            d_hidden,
            action_dim,
            diffusion_steps,
            time_embed_type
        )

        self.state_decoder = nn.Linear(d_latent, r_input_size)

    def forward(self, x, graph, state, target_action):
        """
        Full forward pass for training.

        Steps:
            1. Encode robot and obstacle features via SpatialTemporalEncoder.
            2. Map fused features to latent z via LatentHead.
            3. Apply diffusion forward process and predict noise.
            4. Decode latent z to next robot state.

        Args:
            x (torch.Tensor): Current robot observation/state. Shape [B, r_input_size]
            graph (torch.Tensor): Graph or spatial context input.
            state (torch.Tensor): Internal SSM hidden state.
            target_action (torch.Tensor): Ground truth action for diffusion training.

        Returns:
            dict: {
                "predicted_noise": predicted noise in latent space,
                "target_noise": ground-truth noise added to action,
                "latent_mean": mu,
                "latent_log_std": log_std,
                "predicted_next_state": decoded next robot state,
                "timestep": diffusion timestep,
                "ssm_state": updated SSM state
            }
        """
        # Encode features
        fused_feats, ssm_state = self.encoder(x, graph, state)

        # Latent sampling
        z, mu, log_std = self.latent(fused_feats)

        # Diffusion forward
        diffusion_out = self.diffusion(z, target_action)

        # Decode latent to next state
        pred_next_state = self.state_decoder(z)

        diffusion_out.update({
            "latent_mean": mu,
            "latent_log_std": log_std,
            "predicted_next_state": pred_next_state,
            "ssm_state": ssm_state
        })

        return diffusion_out
    
    def rollout(
        self, 
        x: torch.Tensor, 
        graph: torch.Tensor, 
        state: torch.Tensor, 
        horizon: Optional[int] = None
    ) -> torch.Tensor:
        """
        Perform a multi-step rollout of the policy over a given horizon.

        This function generates a sequence of actions by repeatedly calling the
        policy's diffusion-based sampling function. The internal SSM state
        is propagated through each timestep to maintain temporal context.

        Note:
            - `x` (robot state) is not updated here. To perform a physically
                meaningful rollout, `x` should be updated with environment
                dynamics or a learned next-state model.
            - This is effectively a latent rollout: only the action sequence
                and internal state dynamics are propagated.

        Args:
            x (torch.Tensor): Current robot observation/state. Shape: [Batch, Features].
            graph (torch.Tensor): Graph or spatial context input. Can be static if obstacles are static.
            ssm_state (torch.Tensor): Internal SSM hidden state. Shape: [Batch, SSM_hidden_size].
            horizon (int): Number of timesteps to rollout.

        Returns:
            torch.Tensor: Tensor of shape [Batch, Horizon, Action_dim] containing
                            the predicted actions for each timestep.
        """
        actions = []
        r_state = x
        ssm_state = state
        horizon = self.horizon if horizon is None else horizon

        for _ in range(horizon):
            # action, r_state, ssm_state = self.sample(r_state, graph, ssm_state)
            fused, ssm_state = self.encoder(r_state, graph, ssm_state)
            z, _, _ = self.latent(fused)
            action = self.diffusion(z)
            r_state = self.state_decoder(z)
            actions.append(action)

        return torch.stack(actions, dim=1)