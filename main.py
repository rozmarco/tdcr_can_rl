#!/usr/bin/env python3

import yaml
from pathlib import Path

import torch

from tdcr_sim_mujoco.src.utils.config_loader import PROJECT_ROOT

from src.environment.envrunner import EnvRunner
from src.models.policy_network import LatentPolicyPlanner


DEFAULT_CONFIG = {
    "scene": "assets/tdcr_linear_base_5_cyl_real.xml",
    "input_device": "tdcr_keyboard",
    "controller": "tdcr_joint",
    "controller_params": {
        "tension_mode": True
    },
    "control_mode": "joint_space", 
    "width": 1200,
    "height": 900,
    "show_info": True,
    "sim_steps_per_frame": 1,
    "fps": 30,
    "velocity_scale": 0.5,
    "damping_factor": 0.01,
    "disable_gravity": True,
    "verbose": True,
    "target_pose": {
        "pos": [0.1, 0.1, 0],
        "euler": [0, 0, 0]
    },
    "description": "TDCR task-space control with Jacobian IK"
}

if __name__ == "__main__":
    # ----- SCENE METADATA ------
    scene_path = Path(DEFAULT_CONFIG["scene"])
    
    if not scene_path.is_absolute():
        scene_path = PROJECT_ROOT / scene_path

    if not scene_path.exists():
        print(f"Error: Scene file not found: {scene_path}")

    # ----- SETUP POLICY AND ENVIRONMENT -----
    torch.manual_seed(0)

    # TODO: Hardcoded.
    state_dim = 41
    action_dim = 2
    horizon = 1
    render_mode = "human"

    with torch.no_grad():
        # TODO: Hardcoded.
        policy = LatentPolicyPlanner(
            state_dim, 
            action_dim,
            d_embedding=24,
            d_hidden_enc=48,
            d_hidden_dec=48,
            d_state=8,
            d_latent=16,
            num_blocks_enc=1,
            num_blocks_dec=3,
            expand_dec=2,
            diffusion_steps=20,
            max_period=250.0
        )
        policy.eval()

        checkpoint_path = Path("./checkpoints/policy_7000_final.pth")  # TODO: Hardcoded

        if checkpoint_path.is_file():
            policy.load_state_dict(torch.load(checkpoint_path, weights_only=True), strict=True)
            print(f"Loaded weights from {checkpoint_path}")
        else:
            print("No weights found. Using random initialization.")

        # Run environment with policy
        env = EnvRunner(
            "0",
            False,
            scene_path,
            policy,
            horizon=horizon,
            render_mode=render_mode,
            logs_location=None
        )
        env.run_session()