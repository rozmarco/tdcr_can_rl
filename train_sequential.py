#!/usr/bin/env python3
"""Sequential training (no Ray) to verify full pipeline works."""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.environment.env import CustomEnv
from src.environment.envrunner import EnvRunner
from src.models.policy_network import LatentDiffusionPolicyPlanner
from src.models.q_network import QNetwork
from src.buffers.buffer import ReplayBuffer
from src.sac import SoftActorCritic

# Load config
config_file = Path("config/train.yaml")
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

# Setup
scene_path = config["scene"]
r_dim = config["agent"]["r_dim"]
action_dim = config["agent"]["action_dim"]

print(f"Sequential Training (No Ray)")
print(f"Scene: {scene_path}")
print(f"State dim: {r_dim}, Action dim: {action_dim}\n")

# Create networks
policy_network = LatentDiffusionPolicyPlanner(
    r_dim=r_dim,
    action_dim=action_dim,
    **config['model']["policy"]
)

q_network1 = QNetwork(r_dim=r_dim, action_dim=action_dim, **config["model"]["q_network"])
q_network2 = QNetwork(r_dim=r_dim, action_dim=action_dim, **config["model"]["q_network"])

buffer = ReplayBuffer(max_size=config["buffer"]["max_size"], seed=config["seed"])

sac = SoftActorCritic(
    policy=policy_network, q1=q_network1, q2=q_network2,
    replay_buffer=buffer,
    horizon=config["agent"]["horizon"],
    optimizer_str=config["agent"]["optimizer"]["_target_"],
    policy_lr=config["agent"]["policy_lr"],
    q_lr=config["agent"]["q_lr"],
    batch_size=config["agent"]["batch_size"],
    gamma=config["agent"]["gamma"],
    tau=config["agent"]["tau"],
    alpha=config["agent"]["alpha"],
    seed=config["seed"],
    device=config["device"]
)

# Training loop - sequential only
for epoch in range(min(3, config['epochs'])):  # Just 3 epochs for testing
    print(f"\n--- Epoch {epoch+1} ---")
    
    if config['env']['run_env']:
        # Run single environment
        env = CustomEnv(scene_path, "rgb_array", config['env']['frame_skips'], config['env']['timestep'])
        num_episodes = config['env']['num_episodes']
        max_steps = config['env']['max_steps']
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            for step in range(max_steps):
                # Get action from policy  
                from src.utils.data_preprocessor import flatten_state
                r_state = flatten_state(state, config["device"]).view(1, 1, -1)
                
                with torch.no_grad():
                    plan, _ = policy_network.rollout(r_state, config["agent"]["horizon"])
                plan = plan.detach().cpu().numpy().squeeze()
                action = plan[0] if plan.ndim > 1 else plan
                
                # Step
                next_state, reward, term, trunc, _ = env.step(action)  
                episode_reward += reward
                
                # Store in buffer
                if step > 0:  # Skip first step
                    buffer.add(state, action, reward, next_state, term or trunc)
                
                state = next_state
                step_count += 1
                
                if term or trunc:
                    break
            
            print(f"  Episode {episode+1}: steps={step_count}, reward={episode_reward:.3f}")
        
        print(f"Buffer size: {len(buffer)}")
    
    # Training update
    for iteration in range(min(10, len(buffer) // config["agent"]["batch_size"])):
        sac.update()
    
    print(f"SAC update complete ({iteration+1} iterations)")

print("\n✓ Sequential training test passed!")
print("If this works, the issue is definitely Ray-specific.")
print("For full training, consider:")
print("  1. Using fewer Ray workers (num_workers: 1)")
print("  2. Adding timeouts to envrunner.run_episodes()")
print("  3. Checking Ray object store limits")
