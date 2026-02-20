import copy
import yaml
import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Type

from src.environment.envrunner import EnvRunner
from src.models.policy_network import LatentDiffusionPolicyNetwork
from src.models.q_network import QNetwork
from src.buffers.buffer import ReplayBuffer
from src.sac import SoftActorCritic

from tdcr_sim_mujoco.src.utils.config_loader import PROJECT_ROOT



if __name__ == '__main__':
    PROJECT_ROOT_ = Path(__file__).parent.resolve()

    # Load YAML configurations
    config_file = Path(PROJECT_ROOT_ / "config" / "config.yaml")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Initialization
    policy_network = LatentDiffusionPolicyNetwork(
        r_input_size=310,
        o_input_size=4,
        action_dim=5,
        horizon=config["agent"]["horizon"],
        d_embedding=config["model"]["policy"]["d_embedding"],
        d_hidden=config["model"]["policy"]["d_hidden"],
        d_state=config["model"]["policy"]["d_state"],
        d_latent=config["model"]["policy"]["d_latent"],
        d_ff=config["model"]["policy"]["d_ff"],
        num_layers=config["model"]["policy"]["num_layers"],
        diffusion_steps=config["model"]["policy"]["diffusion_steps"],
        time_embed_type=config["model"]["policy"]["time_embed"]
    )

    q_network1 = QNetwork(
        state_dim=
        action_dim=
        hidden_dim=config["model"]["q_network"]["hidden_dim"]
    )
    q_network2 = QNetwork(
        state_dim=
        action_dim=
        hidden_dim=config["model"]["q_network"]["hidden_dim"]
    )

    buffer = ReplayBuffer(
        max_size=config["buffer"]["max_size"],
        horizon=config["agent"]["horizon"],
        seed=config["seed"]
    )

    sac = SoftActorCritic(
        policy=policy_network,
        q1=q_network1,
        q2=q_network2,
        replay_buffer=buffer,
        optimizer_class=config["agent"]["_target_"],
        policy_lr=config["agent"]["policy_lr"],
        q_lr=config["agent"]["q_lr"],
        batch_size=config["agent"]["batch_size"],
        gamma=config["agent"]["gamma"],
        tau=config["agent"]["tau"],
        alpha=config["agent"]["alpha"],
        seed=config["seed"],
        device=config["device"]
    )

    scene_path = Path(config["scene"])
    if not scene_path.is_absolute():
        scene_path = PROJECT_ROOT / scene_path

    env = EnvRunner(
        scene_path, 
        policy_network, 
        buffer=buffer,
        num_episodes=config["env"]["num_episodes"]
    )

    # Run environment
    env.run_session(is_train=True)

    # Load data
    data_folder = Path(PROJECT_ROOT / "data")
    npz_files = list(data_folder.rglob("*.npz"))
    buffer.load(npz_files)

    # Training loop
    while not buffer.is_empty(): # TODO: Create function
        sac.update()

    # TODO: Save model

    # TODO: Get reward
    
    # TODO: Create function with simulation and training (Compatible with Ray)
