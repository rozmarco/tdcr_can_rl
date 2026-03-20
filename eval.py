#!/usr/bin/env python3
"""
eval.py  —  Rigorous evaluation of a trained TDCR policy.

Runs the policy across many contact-required goals and reports:
    - Success rate       (% reaching goal within 2 cm)
    - Mean final distance to goal
    - Mean steps to success (for successful episodes)
    - Per-episode breakdown

Usage:
    python eval.py                          # 100 contact goals, no render
    python eval.py --num_goals 50           # 50 contact goals
    python eval.py --render --num_goals 5   # 5 goals with live viewer
    python eval.py --checkpoint checkpoints/policy_best.pth
    python eval.py --goal_idx 3             # single specific goal (debug mode)
"""

import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import torch

from pathlib import Path
from typing import Optional, List

from src.environment.env import CustomEnv
from src.models.policy_network import LatentPolicyPlanner
from src.utils.data_preprocessor import flatten_state
from gymnasium.wrappers import TimeLimit

GOAL_RADIUS_M = 0.02   # 2 cm — matches is_terminal() in env.py

# Obstacle positions (x, y) and radius from XML — all cylinders, radius 0.015 m
OBSTACLES = [
    {"pos": (-0.100,  0.150), "radius": 0.015, "name": "obstacle0"},
    {"pos": (-0.045,  0.260), "radius": 0.015, "name": "obstacle1"},
    {"pos": ( 0.020,  0.150), "radius": 0.015, "name": "obstacle2"},
    {"pos": ( 0.140,  0.150), "radius": 0.015, "name": "obstacle3"},
    {"pos": ( 0.080,  0.260), "radius": 0.015, "name": "obstacle4"},
]


# ------------------------------------------------------------------ #
#  Data loading                                                        #
# ------------------------------------------------------------------ #

def load_contact_goals(npz_path: str):
    """Return tip positions and actuator configs for contact-required goals."""
    ws = np.load(npz_path)
    keys = list(ws.keys())
    print(f"NPZ keys: {keys}")

    if 'contact_at_goal' not in ws:
        raise KeyError(
            f"'contact_at_goal' not found in {npz_path}.\n"
            f"Available keys: {keys}"
        )

    mask = ws['contact_at_goal'].astype(bool)
    idx  = np.where(mask)[0]
    print(f"Contact-required goals: {len(idx)} / {len(mask)} total configs.")
    return idx, ws['tip_positions'][idx, :2].astype(np.float32), ws['actuator_configs'][idx].astype(np.float32)


def sample_goals(npz_path: str, num_goals: int, seed: int, goal_idx: Optional[int] = None):
    """
    Return a list of (goal_xy, actuator_config, workspace_idx) tuples.
    If goal_idx is set, return only that one goal (debug mode).
    """
    rng = np.random.default_rng(seed)
    contact_idx, tip_positions, actuator_configs = load_contact_goals(npz_path)

    if len(contact_idx) == 0:
        raise RuntimeError("No contact-required goals found in workspace npz.")

    if goal_idx is not None:
        i = goal_idx % len(contact_idx)
        return [(tip_positions[i], actuator_configs[i], contact_idx[i])]

    n = min(num_goals, len(contact_idx))
    if num_goals > len(contact_idx):
        print(f"Warning: requested {num_goals} goals but only {len(contact_idx)} available. Using {n}.")
    chosen = rng.choice(len(contact_idx), size=n, replace=False)
    return [(tip_positions[i], actuator_configs[i], contact_idx[i]) for i in chosen]


# ------------------------------------------------------------------ #
#  Policy + env                                                        #
# ------------------------------------------------------------------ #

def load_policy(checkpoint_path: str, config: dict, device: str = 'cpu'):
    policy = LatentPolicyPlanner(
        r_dim=config["agent"]["r_dim"],
        action_dim=config["agent"]["action_dim"],
        **config["model"]["policy"]
    )
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        policy.load_state_dict(
            torch.load(ckpt, map_location='cpu', weights_only=True), strict=True
        )
        print(f"Loaded weights from {ckpt}")
    else:
        print(f"WARNING: checkpoint not found at {ckpt} — using random weights.")
    policy.to(device)
    policy.eval()
    return policy


def make_env(scene_path: str, workspace_npz: str, config: dict, render: bool, max_steps: int):
    raw = CustomEnv(
        scene_path=scene_path,
        workspace_npz=workspace_npz,
        render_mode="human" if render else None,
        frame_skips=config["env"]["frame_skips"],
        timstep=config["env"]["timestep"],
        allow_contact_goals=True,
    )
    return TimeLimit(raw, max_episode_steps=max_steps), raw


# ------------------------------------------------------------------ #
#  Single episode                                                      #
# ------------------------------------------------------------------ #

def run_episode(env, raw_env, policy, goal_xy, max_steps, device, render):
    """
    Run one episode with a fixed goal.
    Returns dict with keys: success, final_dist, steps, tip_trajectory.
    """
    # Reset, then inject the fixed goal and rebuild observation so the
    # policy sees the correct goal_rel_pos from step 0.
    raw_env.goal_pos = np.array(goal_xy, dtype=np.float32)
    state, _ = env.reset()
    raw_env.goal_pos = np.array(goal_xy, dtype=np.float32)
    state = raw_env.get_state()   # rebuild obs with correct goal (fixes ghost-goal bug)

    tip_trajectory = []
    done  = False
    step  = 0
    success_step = None

    while not done and step < max_steps:
        tip_pos = raw_env.data.xpos[raw_env.tip_body_id][:2].copy()
        tip_trajectory.append(tip_pos)

        with torch.no_grad():
            r_state = flatten_state(state, device).view(1, 1, -1)
            plan, _ = policy.sample(r_state, horizon=1)
            action  = plan.detach().cpu().numpy().squeeze()
            if action.ndim > 1:
                action = action[0]

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

        if render:
            env.render()

        final_tip = raw_env.data.xpos[raw_env.tip_body_id][:2].copy()
        dist = np.linalg.norm(final_tip - np.array(goal_xy))
        if dist < GOAL_RADIUS_M and success_step is None:
            success_step = step

    final_tip  = raw_env.data.xpos[raw_env.tip_body_id][:2].copy()
    tip_trajectory.append(final_tip)
    final_dist = np.linalg.norm(final_tip - np.array(goal_xy))
    success    = final_dist < GOAL_RADIUS_M or success_step is not None

    return {
        "success":        success,
        "final_dist_m":   final_dist,
        "final_dist_cm":  final_dist * 100,
        "steps":          step,
        "success_step":   success_step if success_step is not None else (step if success else None),
        "tip_trajectory": np.array(tip_trajectory),
        "goal_xy":        np.array(goal_xy),
    }


# ------------------------------------------------------------------ #
#  Plots                                                               #
# ------------------------------------------------------------------ #

def plot_all_trajectories(results: List[dict],
                          save_path="results/eval_trajectories.png"):
    """Overlay all episode trajectories on one 2D plot, coloured by success."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f0f7fb")
    ax.grid(which="major", linestyle="-", alpha=0.4)

    # --- Draw obstacles from XML constants ---
    for obs in OBSTACLES:
        x, y = obs["pos"]
        r    = obs["radius"]
        ax.add_patch(patches.Circle((x, y), r,
                                     color="#c0392b", alpha=0.85, zorder=4))
        ax.add_patch(patches.Circle((x, y), r,
                                     color="#7b241c", fill=False,
                                     linewidth=1.2, zorder=5))
        ax.text(x, y, obs["name"].replace("obstacle", "O"),
                ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold", zorder=6)

    # --- Draw trajectories ---
    for res in results:
        traj  = res["tip_trajectory"]
        color = "#2ecc71" if res["success"] else "#e74c3c"
        alpha = 0.7 if res["success"] else 0.35
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.2,
                alpha=alpha, zorder=3)
        # Start dot
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o",
                   s=30, zorder=6, alpha=0.6)
        # Goal star + success radius
        ax.scatter(*res["goal_xy"], color=color, marker="*",
                   s=120, zorder=7, alpha=0.9)
        ax.add_patch(patches.Circle(res["goal_xy"], GOAL_RADIUS_M,
                                     color=color, fill=False, linestyle="--",
                                     linewidth=0.8, alpha=0.5, zorder=6))

    # --- Legend ---
    ax.plot([], [], color="#2ecc71", linewidth=2, label="Success")
    ax.plot([], [], color="#e74c3c", linewidth=2, label="Failure")
    ax.scatter([], [], color="gray", marker="*", s=100, label="Goal")
    ax.add_patch(patches.Circle((0, 0), 0.001, color="#c0392b",
                                 alpha=0.85, label="Obstacle"))

    n_success = sum(r["success"] for r in results)
    ax.set_title(f"Tip Trajectories — {n_success}/{len(results)} successful", fontsize=13)
    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.set_aspect("equal")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Trajectory plot saved to {save_path}")


def plot_distance_histogram(results: List[dict], save_path="results/eval_distance_hist.png"):
    """Histogram of final distances to goal."""
    dists = [r["final_dist_cm"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_facecolor("#f0f7fb")
    ax.hist(dists, bins=20, color="#64b5eb", edgecolor="white")
    ax.axvline(GOAL_RADIUS_M * 100, color="red", linestyle="--",
               linewidth=1.5, label=f"Goal threshold ({GOAL_RADIUS_M*100:.0f} cm)")
    ax.set_xlabel("Final distance to goal (cm)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Final Distances to Goal", fontsize=13)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distance histogram saved to {save_path}")


def plot_steps_to_success(results: List[dict], save_path="results/eval_steps.png"):
    """Histogram of steps to success for successful episodes."""
    steps = [r["success_step"] for r in results if r["success"] and r["success_step"] is not None]
    if not steps:
        print("No successful episodes — skipping steps plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_facecolor("#f0f7fb")
    ax.hist(steps, bins=20, color="#2ecc71", edgecolor="white")
    ax.set_xlabel("Steps to reach goal", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Steps to Success (successful episodes only)", fontsize=13)
    ax.grid(axis="y", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Steps-to-success plot saved to {save_path}")


def print_summary(results: List[dict], checkpoint_path: str):
    """Print a clean summary table to console."""
    n          = len(results)
    n_success  = sum(r["success"] for r in results)
    dists      = [r["final_dist_cm"] for r in results]
    succ_steps = [r["success_step"] for r in results if r["success"] and r["success_step"]]

    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)
    print(f"  Checkpoint:           {checkpoint_path}")
    print(f"  Goals evaluated:      {n}")
    print(f"  Goal threshold:       {GOAL_RADIUS_M*100:.0f} cm")
    print("-"*55)
    print(f"  Success rate:         {n_success}/{n}  ({100*n_success/n:.1f}%)")
    print(f"  Mean final dist:      {np.mean(dists):.2f} cm")
    print(f"  Median final dist:    {np.median(dists):.2f} cm")
    print(f"  Std final dist:       {np.std(dists):.2f} cm")
    print(f"  Best final dist:      {np.min(dists):.2f} cm")
    if succ_steps:
        print(f"  Mean steps (success): {np.mean(succ_steps):.0f}")
        print(f"  Min steps (success):  {np.min(succ_steps)}")
    print("="*55 + "\n")

    df = pd.DataFrame([{
        "episode":    i + 1,
        "goal_x":     f"{r['goal_xy'][0]:.4f}",
        "goal_y":     f"{r['goal_xy'][1]:.4f}",
        "dist_cm":    f"{r['final_dist_cm']:.2f}",
        "steps":      r["steps"],
        "success":    "✓" if r["success"] else "✗",
    } for i, r in enumerate(results)])

    print(df.to_string(index=False))
    print()

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/eval_results.csv", index=False)
    print("Per-episode results saved to results/eval_results.csv")


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/policy_best.pth")
    parser.add_argument("--workspace",  type=str, default=None,
                        help="Path to workspace npz (overrides auto-detection)")
    parser.add_argument("--num_goals",  type=int, default=100,
                        help="Number of contact-required goals to evaluate on")
    parser.add_argument("--goal_idx",   type=int, default=None,
                        help="Single goal index for debug/demo (overrides --num_goals)")
    parser.add_argument("--max_steps",  type=int, default=1000)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--render",     action="store_true",
                        help="Enable live MuJoCo viewer (slow — use for demos only)")
    args = parser.parse_args()

    PROJECT_ROOT_ = Path(__file__).parent.resolve()

    with open(PROJECT_ROOT_ / "config" / "train.yaml", "r") as f:
        config = yaml.safe_load(f)

    scene_path = str(PROJECT_ROOT_ / config["scene"])

    # Workspace precedence:
    #   1. --workspace CLI arg
    #   2. tdcr_sim_mujoco/data/workspace_test.npz  (held-out test set)
    #   3. train.yaml workspace_npz                 (training set — optimistic)
    if args.workspace:
        workspace_npz = str(PROJECT_ROOT_ / args.workspace)
        print(f"Using workspace from --workspace arg: {workspace_npz}")
    else:
        test_npz = PROJECT_ROOT_ / "tdcr_sim_mujoco" / "data" / "workspace_test.npz"
        if test_npz.exists():
            workspace_npz = str(test_npz)
            print(f"Using held-out test workspace: {workspace_npz}")
        else:
            workspace_npz = str(PROJECT_ROOT_ / config["workspace_npz"])
            print(f"WARNING: No test workspace found at {test_npz}")
            print(f"Using training workspace — results will be optimistic.")
            print(f"Create a test split with: python split_workspace.py")

    device = 'cpu'

    # --- Sample goals ---
    goals = sample_goals(workspace_npz, args.num_goals, args.seed, args.goal_idx)
    print(f"\nEvaluating on {len(goals)} contact-required goal(s)...\n")

    # --- Load policy ---
    policy = load_policy(args.checkpoint, config, device)

    # --- Build env ---
    env, raw_env = make_env(scene_path, workspace_npz, config, args.render, args.max_steps)

    # --- Run all episodes ---
    results = []
    for i, (goal_xy, actuator_cfg, ws_idx) in enumerate(goals):
        print(f"Episode {i+1}/{len(goals)} | ws_idx={ws_idx} | "
              f"goal=({goal_xy[0]:.4f}, {goal_xy[1]:.4f})", end=" ... ")
        res = run_episode(env, raw_env, policy, goal_xy, args.max_steps, device, args.render)
        results.append(res)
        status = f"✓ {res['final_dist_cm']:.2f}cm in {res['success_step']} steps" \
                 if res["success"] else f"✗ {res['final_dist_cm']:.2f}cm"
        print(status)

    env.close()

    # --- Summary + plots ---
    print_summary(results, args.checkpoint)
    plot_all_trajectories(results)
    plot_distance_histogram(results)
    plot_steps_to_success(results)