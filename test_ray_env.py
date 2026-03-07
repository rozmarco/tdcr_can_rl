#!/usr/bin/env python3
"""Simple test of Ray with environment."""

import sys
import yaml
from pathlib import Path
import ray
import torch
import numpy as np

from src.models.policy_network import LatentDiffusionPolicyPlanner
from src.buffers.buffer import ReplayBuffer

# Load config
print("1. Loading config..."  )
config_file = Path("config/train.yaml")
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# Setup
scene_path = config["scene"]  # Should be absolute path now
r_dim = config["agent"]["r_dim"]
action_dim = config["agent"]["action_dim"]

print(f"2. Config loaded - r_dim: {r_dim}, action_dim: {action_dim}")
print(f"   Scene: {scene_path}")

# Create network
print("3. Creating policy network...")
sys.stdout.flush()
policy = LatentDiffusionPolicyPlanner(
    r_dim=r_dim,
    action_dim=action_dim,
    **config["model"]["policy"]
)
print("4. Policy network created")
sys.stdout.flush()

buffer = ReplayBuffer(
    max_size=config["buffer"]["max_size"],
    seed=config["seed"]
)
print("5. Buffer created")
sys.stdout.flush()

print("6. Initializing Ray...")
sys.stdout.flush()
ray.init(ignore_reinit_error=True)
print("7. Ray initialized")
sys.stdout.flush()

print("8. Importing ParallelEnvRunner...")
sys.stdout.flush()
from src.environment.envrunner import ParallelEnvRunner
print("9. ParallelEnvRunner imported")
sys.stdout.flush()

print("10. Creating remote actor...")
sys.stdout.flush()
try:
    runner = ParallelEnvRunner.remote(True, scene_path, policy, config)
    print(f"11. ParallelEnvRunner created: {runner}")
    sys.stdout.flush()
    
    print("12. Running session (this may take a while)...")
    sys.stdout.flush()
    result = ray.get(runner.run_session_remote.remote())
    print(f"13. Session completed: {result}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Ray test passed!")
ray.shutdown()
