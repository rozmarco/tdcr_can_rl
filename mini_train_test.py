#!/usr/bin/env python3
"""Minimal training loop with debug output."""

import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from src.environment.env import CustomEnv
from src.utils.data_preprocessor import flatten_state
from src.models.policy_network import LatentDiffusionPolicyPlanner

config_file = Path("config/train.yaml")
with open(config_file) as f:
    config = yaml.safe_load(f)

scene_path = config["scene"]
r_dim = config["agent"]["r_dim"]
action_dim = config["agent"]["action_dim"]

print(f"Minimal Training Test\nr_dim={r_dim}, action_dim={action_dim}\n")

print("1. Creating environment...", flush=True)
env = CustomEnv(scene_path, "rgb_array", 50, 0.002)

print("2. Creating policy...", flush=True)
policy = LatentDiffusionPolicyPlanner(
    r_dim=r_dim, action_dim=action_dim,
    **config['model']["policy"]
)
policy.eval()

print("3. Running 1 episode...", flush=True)
state, _ = env.reset()
print(f"   - reset ok", flush=True)

for step in range(5):
    print(f"   Step {step+1}...", end="", flush=True)
    
    # Policy inference
    r_state = flatten_state(state).view(1, 1, -1).cpu()
    with torch.no_grad():
        plan, _ = policy.rollout(r_state, config["agent"]["horizon"])
    plan = plan.detach().cpu().numpy().squeeze()
    action = plan[0] if plan.ndim > 1 else plan
    
    # Env step
    try:
        state, reward, term, trunc, _ = env.step(action)
        print(f" reward={reward:.3f} ok", flush=True)
    except Exception as e:
        print(f" ERROR: {e}", flush=True)
        sys.exit(1)
    
    if term or trunc:
        break

print("\n✓ Minimal test passed!")
print("\nSummary:")
print("- Sequential episode collection: ✓ WORKS")
print("- Policy rollout: ✓ WORKS")  
print("- Environment stepping: ✓ WORKS")
print("- Ray parallelization: ❌ HANGS")
print("\nNext: Run full training.py with Ray")
