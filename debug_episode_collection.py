#!/usr/bin/env python3
"""Debug episode collection to find blocking point."""

import numpy as np
import torch
import yaml
from pathlib import Path
from src.environment.env import CustomEnv
from src.models.policy_network import LatentDiffusionPolicyPlanner
from src.utils.data_preprocessor import flatten_state

# Load config
with open("config/train.yaml") as f:
    config = yaml.safe_load(f)

scene_path = config["scene"]
r_dim = config["agent"]["r_dim"]
action_dim = config["agent"]["action_dim"]

print("1. Creating environment...")
env = CustomEnv(scene_path, "rgb_array", 50, 0.002)

print("2. Resetting environment...")
state, info = env.reset()

print("3. Creating policy network...")
policy = LatentDiffusionPolicyPlanner(
    r_dim=r_dim,
    action_dim=action_dim,
    **config["model"]["policy"]
)
policy.eval()

print("4. Testing single step...")
action = np.array([0.1, 0.05], dtype=np.float32)
next_state, reward, terminated, truncated, info = env.step(action)
print(f"  ✓ Step successful: reward={reward:.4f}")

print("5. Testing policy rollout...")
r_state = flatten_state(state).view(1, 1, -1).to('cpu')
print(f"  Input shape: {r_state.shape}")
with torch.no_grad():
    plan, _ = policy.rollout(r_state, horizon=2)
print(f"  ✓ Rollout successful: plan shape={plan.shape}")

print("6. Testing multi-step episode collection...")
state, _ = env.reset()
for step_i in range(5):
    print(f"  Step {step_i+1}: ", end="",flush=True)
    
    # Get flat state
    r_state = flatten_state(state).view(1, 1, -1).cpu()
    
    # Get action from policy
    with torch.no_grad():
        plan, _ = policy.rollout(r_state, horizon=config["agent"]["horizon"])
    plan = plan.detach().cpu().numpy().squeeze()  # [horizon, action_dim]
    
    # Receding horizon - take first action
    if plan.ndim > 1:
        action = plan[0]
    else:
        action = plan
        
    print(f"action shape: {action.shape}, ", end="", flush=True)
    
    # Step
    state, reward, term, trunc, info = env.step(action)
    print(f"reward={reward:.4f}, done={term or trunc}")
    
    if term or trunc:
        print(f"  Episode ended at step {step_i+1}")
        break

print("\n✓ All debug tests passed - no blocking detected!")
print("\nThe issue is likely in Ray parallelization or buffer operations.")
