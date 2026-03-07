#!/usr/bin/env python3
"""Simple script to test environment initialization."""

import sys
from pathlib import Path

# Test environment initialization
scene_path = "/home/rozilyn/tdcr_can_rl/tdcr_sim_mujoco/assets/tdcr/tdcr_linear_base.xml"

print(f"Scene path exists: {Path(scene_path).exists()}")
print(f"Scene path: {scene_path}")

print("\n1. Testing environment initialization...")
from src.environment.env import CustomEnv

try:
    env = CustomEnv(
        scene_path=scene_path,
        render_mode="rgb_array",
        frame_skips=50,
        timstep=0.002
    )
    print("✓ Environment initialized successfully")
    
    print("\n2. Testing environment reset...")
    state, info = env.reset()
    print(f"✓ Environment reset successfully")
    print(f"  State type: {type(state)}")
    if isinstance(state, dict):
        for key, val in state.items():
            if isinstance(val, list):
                print(f"    {key}: list of {len(val)} items")
            else:
                try:
                    print(f"    {key}: shape {val.shape}")
                except:
                    print(f"    {key}: {type(val)}")
    
    print("\n3. Testing state flattening...")
    from src.utils.data_preprocessor import flatten_state
    flat = flatten_state(state)
    print(f"✓ State flattened successfully")
    print(f"  Flattened state shape: {flat.shape}")
    print(f"  Flattened state dtype: {flat.dtype}")
    
    print("\n4. Testing env.step...")
    import numpy as np
    action = np.array([0.5, 0.2], dtype=np.float32)  # [bend_x, extension_delta]
    next_state, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Environment step successful")
    print(f"  Next state type: {type(next_state)}")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All basic tests passed!")
