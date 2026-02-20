#!/usr/bin/env python3

import re
from pathlib import Path

import torch
import mujoco

from tdcr_sim_mujoco.src.utils.config_loader import PROJECT_ROOT

from src.environment.envrunner import EnvRunner
from src.models.policy_network import LatentDiffusionPolicyNetwork


DEFAULT_CONFIG = {
    "scene": "assets/cylinders5_links100.xml",
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


def get_scene_info(model, scene_path):
    """Extract scene information from model."""

    # This pattern looks for obstacles with numbers
    # ^        : Start of string
    # obstacle : The literal word
    # \d+      : One or more digits (0-9)
    # $        : End of string
    pattern = re.compile(r'^obstacle\d+$')
    obstacle_ids = [model.body(i).name for i in range(model.nbody) if pattern.match(model.body(i).name)]
    num_obstacles = len(obstacle_ids)

    # Identify TDCR Links
    tdcr_link_ids = [model.body(i).name for i in range(model.nbody) if 'link' in model.body(i).name]
    num_links = len(tdcr_link_ids)

    # Identify number of TDCR Segments via Actuators
    num_segments = 0
    for i in range(20):
        try:
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"seg_{i}_ten_0")
            num_segments += 1
        except:
            break

    obs_dim = model.nu + model.nv + num_links + 1

    is_combined = "franka_scene" in str(scene_path).lower() and (
        "tdcr" in str(scene_path).lower() or "ftdcr" in str(scene_path).lower()
    )

    info = {
        "num_bodies": model.nbody,
        "num_joints": model.njnt,
        "num_actuators": model.nu,
        "num_obstacles": num_obstacles,
        "num_links": num_links,
        "num_tdcr_segments": num_segments,
        "obs_dim": obs_dim,
        "is_tdcr": True,
        "is_combined": is_combined,
        "has_gripper": "gripper" in [model.body(i).name for i in range(model.nbody)]
    }

    return info


if __name__ == "__main__":
    # Resolve scene path relative to project root
    scene_path = Path(DEFAULT_CONFIG["scene"])
    
    if not scene_path.is_absolute():
        scene_path = PROJECT_ROOT / scene_path

    if not scene_path.exists():
        print(f"Error: Scene file not found: {scene_path}")
        print(f"Project root: {PROJECT_ROOT}")

    # Load model
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    # Get scene information
    scene_info = get_scene_info(model, scene_path)

    print(f"\nScene information:")
    print(f"  - Bodies: {scene_info['num_bodies']}")
    print(f"  - Joints: {scene_info['num_joints']}")
    print(f"  - Actuators: {scene_info['num_actuators']}")
    print(f"  - Has gripper: {scene_info['has_gripper']}")
    if scene_info.get("is_tdcr", False):
        print(f"  - TDCR segments: {scene_info.get('num_tdcr_segments', 0)}")

    # Simulation 
    with torch.no_grad():
        policy = LatentDiffusionPolicyNetwork(
            scene_info['obs_dim'], 
            3 + 1, # TODO: (3D) 3 positions + 1 radius
            scene_info['num_obstacles']
        )
        # TODO: Load model weights
        policy.eval()
        env = EnvRunner(scene_path, policy)
        env.run_session(is_train=False)
