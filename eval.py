#!/usr/bin/env python3
"""
eval.py  —  Rigorous evaluation of a trained TDCR policy.

Runs the policy across many goals and reports:
    - Success rate       (% reaching goal within 2 cm)
    - Mean final distance to goal
    - Mean steps to success (for successful episodes)
    - Per-episode breakdown

Usage:
    python eval.py                          # 100 goals from test lookup table, shaped reward
    python eval.py --num_goals 50
    python eval.py --render --num_goals 5   # live viewer (slow)
    python eval.py --render --render_fps 10 # slow render for inspection
    python eval.py --checkpoint checkpoints/policy_best.pth
    python eval.py --goal_idx 3             # single specific goal (debug)
    python eval.py --no_shaping             # disable heuristic shaping even if configured
    python eval.py --lookup_table lookup_tables_test.npz   # override test table path
"""

import argparse
import time
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

def load_goals(npz_path: str):
    """
    Load goal positions from either a lookup table npz or a legacy workspace npz.

    Lookup table format (lookup_tables_*.npz):
        raw_goal_positions  (N, 2)
        goal_to_table_idx   (N,)     — used to filter goals with valid tables

    Legacy workspace format (explored_configs_*.npz):
        tip_positions       (N, 2+)
        actuator_configs    (N, *)

    Returns (original_indices [N,], tip_positions [N, 2])
    """
    ws   = np.load(npz_path, allow_pickle=True)
    keys = list(ws.keys())
    print(f"Workspace keys: {keys}")

    if "raw_goal_positions" in keys:
        # ── Lookup table format ──────────────────────────────────────
        all_positions = np.asarray(ws["raw_goal_positions"], dtype=np.float32)
        N = len(all_positions)

        # Only keep goals that have a valid heuristic table entry.
        # Goals without a table (goal_to_table_idx < 0) would silently
        # fall back to unshaped reward, which makes returns incomparable.
        if "goal_to_table_idx" in keys:
            table_idx = np.asarray(ws["goal_to_table_idx"], dtype=np.int32)
            valid_mask = table_idx >= 0
            idx = np.where(valid_mask)[0].astype(np.int32)
            n_invalid = N - len(idx)
            if n_invalid > 0:
                print(f"  Skipping {n_invalid} goals with no heuristic table entry.")
        else:
            idx = np.arange(N, dtype=np.int32)

        tip_positions = all_positions[idx, :2]
        print(
            f"Lookup table goals: {len(idx)} valid / {N} total  "
            f"x=[{tip_positions[:,0].min():.3f}, {tip_positions[:,0].max():.3f}]  "
            f"y=[{tip_positions[:,1].min():.3f}, {tip_positions[:,1].max():.3f}]"
        )
    else:
        # ── Legacy workspace format ──────────────────────────────────
        tip_positions = ws['tip_positions'][:, :2].astype(np.float32)
        idx = np.arange(len(tip_positions), dtype=np.int32)
        print(f"Legacy workspace goals: {len(idx)}")

    print(f"Total goals available: {len(idx)}")
    return idx, tip_positions


def sample_goals(npz_path: str, num_goals: int, seed: int,
                 goal_idx: Optional[int] = None):
    """
    Return a list of (goal_xy, workspace_idx) tuples.
    If goal_idx is set, return only that one goal (debug mode).
    """
    rng = np.random.default_rng(seed)
    all_idx, tip_positions = load_goals(npz_path)

    if len(all_idx) == 0:
        raise RuntimeError("No goals found in npz.")

    if goal_idx is not None:
        i = goal_idx % len(all_idx)
        return [(tip_positions[i], int(all_idx[i]))]

    n = min(num_goals, len(all_idx))
    if num_goals > len(all_idx):
        print(f"Warning: requested {num_goals} goals but only {len(all_idx)} available. Using {n}.")

    chosen = rng.choice(len(all_idx), size=n, replace=False)
    return [(tip_positions[i], int(all_idx[i])) for i in chosen]


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


def make_env(scene_path: str, lookup_table_npz: str, config: dict,
             render: bool, max_steps: int):
    """
    Create the environment using lookup_table_npz for both goal sampling
    and heuristic reward shaping.

    The env's _load_workspace is called with from_lookup_table=True when
    lookup_table_npz is provided, which reads raw_goal_positions directly
    and preserves the index alignment with goal_to_table_idx — the same
    mechanism used during training with lookup_tables_train.npz.

    For eval we pass lookup_tables_test.npz to both workspace_npz and
    lookup_table_npz so the env samples from the held-out goal set and
    applies shaped rewards using the test table.
    """
    raw = CustomEnv(
        scene_path=scene_path,
        workspace_npz=lookup_table_npz,   # goal sampling from test table
        render_mode="human" if render else None,
        frame_skips=config["env"]["frame_skips"],
        timstep=config["env"]["timestep"],
        allow_contact_goals=True,
        lookup_table_npz=lookup_table_npz, # shaped reward from test table
    )
    return TimeLimit(raw, max_episode_steps=max_steps), raw


def _try_fullscreen(env):
    try:
        import glfw
        win     = env.unwrapped._get_viewer("human").window
        monitor = glfw.get_primary_monitor()
        mode    = glfw.get_video_mode(monitor)
        glfw.set_window_monitor(win, monitor, 0, 0,
                                mode.size.width, mode.size.height,
                                mode.refresh_rate)
        print("[Render] Fullscreen mode set via GLFW.")
        return True
    except Exception:
        pass

    try:
        viewer = env.unwrapped._get_viewer("human")
        viewer.window.maximize()
        print("[Render] Viewer window maximised.")
        return True
    except Exception:
        pass

    print("[Render] Could not set fullscreen automatically — "
          "maximise the window manually.")
    return False


# ------------------------------------------------------------------ #
#  Single episode                                                      #
# ------------------------------------------------------------------ #

def run_episode(env, raw_env, policy, goal_xy, ws_idx, max_steps, device,
                render: bool, render_fps: Optional[float] = None):
    """
    Run one episode with a fixed goal.

    ws_idx is the original row index into the lookup table npz.  It is
    injected into raw_env._current_goal_idx so the shaped reward uses
    the correct heuristic table for this goal — matching exactly what
    happens during training.

    Returns dict with keys: success, final_dist, steps, tip_trajectory,
                             episode_return.
    """
    # Reset and inject the fixed goal
    raw_env.goal_pos = np.array(goal_xy, dtype=np.float32)
    env.reset()
    raw_env.goal_pos          = np.array(goal_xy, dtype=np.float32)
    raw_env._current_goal_idx = int(ws_idx)   # align heuristic table lookup
    raw_env._goal_dwell_steps = 0
    state = raw_env.get_state()

    tip_trajectory = []
    done           = False
    step           = 0
    success_step   = None
    episode_return = 0.0
    frame_interval = (1.0 / render_fps) if (render and render_fps) else None

    while not done and step < max_steps:
        tip_pos = raw_env.data.xpos[raw_env.tip_body_id][:2].copy()
        tip_trajectory.append(tip_pos)

        t_frame_start = time.perf_counter() if frame_interval else None

        with torch.no_grad():
            flat    = flatten_state(state, device)
            r_state = flat.view(1, 1, -1)
            plan, _ = policy.sample(r_state, horizon=1)
            action  = plan.squeeze().cpu().numpy()
            if action.ndim > 1:
                action = action[0]

        state, reward, terminated, truncated, _ = env.step(action)
        episode_return += reward
        done = terminated or truncated
        step += 1

        if render:
            env.render()
            if frame_interval is not None:
                elapsed = time.perf_counter() - t_frame_start
                sleep   = frame_interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        tip_now = raw_env.data.xpos[raw_env.tip_body_id][:2].copy()
        if np.linalg.norm(tip_now - np.array(goal_xy)) < GOAL_RADIUS_M and success_step is None:
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
        "ws_idx":         ws_idx,
        "episode_return": episode_return,
    }


# ------------------------------------------------------------------ #
#  Plots                                                               #
# ------------------------------------------------------------------ #

def plot_all_trajectories(results: List[dict],
                          save_path="results/eval_trajectories.png"):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#f0f7fb")
    ax.grid(which="major", linestyle="-", alpha=0.4)

    for obs in OBSTACLES:
        x, y = obs["pos"]
        r    = obs["radius"]
        ax.add_patch(patches.Circle((x, y), r, color="#c0392b", alpha=0.85, zorder=4))
        ax.add_patch(patches.Circle((x, y), r, color="#7b241c", fill=False,
                                     linewidth=1.2, zorder=5))
        ax.text(x, y, obs["name"].replace("obstacle", "O"),
                ha="center", va="center", fontsize=6.5,
                color="white", fontweight="bold", zorder=6)

    for res in results:
        traj  = res["tip_trajectory"]
        color = "#2ecc71" if res["success"] else "#e74c3c"
        alpha = 0.7 if res["success"] else 0.35
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.2, alpha=alpha, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], color=color, marker="o", s=30, zorder=6, alpha=0.6)
        ax.scatter(*res["goal_xy"], color=color, marker="*", s=120, zorder=7, alpha=0.9)
        ax.add_patch(patches.Circle(res["goal_xy"], GOAL_RADIUS_M,
                                     color=color, fill=False, linestyle="--",
                                     linewidth=0.8, alpha=0.5, zorder=6))

    ax.plot([], [], color="#2ecc71", linewidth=2, label="Success")
    ax.plot([], [], color="#e74c3c", linewidth=2, label="Failure")
    ax.scatter([], [], color="gray", marker="*", s=100, label="Goal")
    ax.add_patch(patches.Circle((0, 0), 0.001, color="#c0392b", alpha=0.85, label="Obstacle"))

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


def plot_distance_histogram(results: List[dict],
                            save_path="results/eval_distance_hist.png"):
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


def plot_steps_to_success(results: List[dict],
                          save_path="results/eval_steps.png"):
    steps = [r["success_step"] for r in results
             if r["success"] and r["success_step"] is not None]
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


def print_summary(results: List[dict], checkpoint_path: str,
                  lookup_table_npz: str):
    n          = len(results)
    n_success  = sum(r["success"] for r in results)
    dists      = [r["final_dist_cm"] for r in results]
    returns    = [r["episode_return"] for r in results]
    succ_steps = [r["success_step"] for r in results
                  if r["success"] and r["success_step"]]

    print("\n" + "="*55)
    print("  EVALUATION SUMMARY")
    print("="*55)
    print(f"  Checkpoint:           {checkpoint_path}")
    print(f"  Goals evaluated:      {n}")
    print(f"  Goal threshold:       {GOAL_RADIUS_M*100:.0f} cm")
    print(f"  Lookup table (test):  {lookup_table_npz}")
    print("-"*55)
    print(f"  Success rate:         {n_success}/{n}  ({100*n_success/n:.1f}%)")
    print(f"  Mean final dist:      {np.mean(dists):.2f} cm")
    print(f"  Median final dist:    {np.median(dists):.2f} cm")
    print(f"  Std final dist:       {np.std(dists):.2f} cm")
    print(f"  Best final dist:      {np.min(dists):.2f} cm")
    print(f"  Mean episode return:  {np.mean(returns):.2f}")
    if succ_steps:
        print(f"  Mean steps (success): {np.mean(succ_steps):.0f}")
        print(f"  Min steps (success):  {np.min(succ_steps)}")
    print("="*55 + "\n")

    df = pd.DataFrame([{
        "episode":  i + 1,
        "ws_idx":   r["ws_idx"],
        "goal_x":   f"{r['goal_xy'][0]:.4f}",
        "goal_y":   f"{r['goal_xy'][1]:.4f}",
        "dist_cm":  f"{r['final_dist_cm']:.2f}",
        "return":   f"{r['episode_return']:.2f}",
        "steps":    r["steps"],
        "success":  "✓" if r["success"] else "✗",
    } for i, r in enumerate(results)])

    print(df.to_string(index=False))
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/eval_results.csv", index=False)
    print("\nPer-episode results saved to results/eval_results.csv")


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str, default="checkpoints/policy_best.pth")
    parser.add_argument("--lookup_table",  type=str, default=None,
                        help="Path to lookup_tables_test.npz (overrides config default). "
                             "Used for both goal sampling and shaped reward.")
    parser.add_argument("--num_goals",     type=int, default=100)
    parser.add_argument("--goal_idx",      type=int, default=None,
                        help="Single goal index for debug (overrides --num_goals)")
    parser.add_argument("--max_steps",     type=int, default=1000)
    parser.add_argument("--seed",          type=int, default=0)
    parser.add_argument("--render",        action="store_true")
    parser.add_argument("--render_fps",    type=float, default=10.0,
                        help="Target FPS for the viewer when --render is set.")
    parser.add_argument("--no_shaping",    action="store_true",
                        help="Disable heuristic reward shaping (uses task reward only). "
                             "Policy behaviour is unaffected; only reported returns change.")
    args = parser.parse_args()

    PROJECT_ROOT_ = Path(__file__).parent.resolve()

    with open(PROJECT_ROOT_ / "config" / "train.yaml", "r") as f:
        config = yaml.safe_load(f)

    scene_path = str(PROJECT_ROOT_ / config["scene"])

    # ── Resolve test lookup table ─────────────────────────────────────────
    # Priority: 1) --lookup_table CLI  2) lookup_tables_test.npz next to train
    #           3) config lookup_table_npz as last resort (will warn)
    if args.lookup_table:
        lookup_table_npz = str(PROJECT_ROOT_ / args.lookup_table)
        print(f"Using lookup table (CLI): {lookup_table_npz}")
    else:
        # Try to find lookup_tables_test.npz in the same directory as the
        # training table so train/test are always kept together.
        train_npz_rel  = config.get("env", {}).get("lookup_table_npz", "lookup_tables_train.npz")
        train_npz_path = PROJECT_ROOT_ / train_npz_rel
        test_candidate = train_npz_path.parent / "lookup_tables_test.npz"

        if test_candidate.exists():
            lookup_table_npz = str(test_candidate)
            print(f"Using held-out test lookup table: {lookup_table_npz}")
        else:
            # Fall back to training table but warn loudly
            lookup_table_npz = str(train_npz_path)
            print(
                "WARNING: lookup_tables_test.npz not found at "
                f"{test_candidate}\n"
                "         Falling back to TRAINING lookup table — "
                "results will be optimistic!\n"
                "         Run generate_lookup_tables_gpu.py with the test "
                "workspace to create lookup_tables_test.npz."
            )

    if args.no_shaping:
        # Pass None so env disables shaping; goal sampling still uses the
        # test table's raw_goal_positions.
        effective_lookup = None
        print("--no_shaping: reward shaping disabled (goal sampling still uses test table).")
    else:
        effective_lookup = lookup_table_npz

    # ── Sample goals from test lookup table ──────────────────────────────
    goals = sample_goals(lookup_table_npz, args.num_goals, args.seed, args.goal_idx)
    print(f"\nEvaluating on {len(goals)} goal(s)...\n")

    device = 'cpu'
    policy = load_policy(args.checkpoint, config, device)
    env, raw_env = make_env(
        scene_path, lookup_table_npz, config,
        args.render, args.max_steps
    )

    # If shaping is disabled, detach the heuristic from the env so
    # get_reward() falls through to r_task only.
    if args.no_shaping:
        raw_env._heuristic = None

    fullscreen_done = False
    results = []

    for i, (goal_xy, ws_idx) in enumerate(goals):
        print(f"Episode {i+1}/{len(goals)} | ws_idx={ws_idx} | "
              f"goal=({goal_xy[0]:.4f}, {goal_xy[1]:.4f})", end=" ... ", flush=True)

        res = run_episode(
            env, raw_env, policy, goal_xy, ws_idx,
            args.max_steps, device,
            render=args.render,
            render_fps=args.render_fps if args.render else None,
        )

        if args.render and not fullscreen_done:
            _try_fullscreen(env)
            fullscreen_done = True

        results.append(res)
        status = (f"✓ {res['final_dist_cm']:.2f}cm in {res['success_step']} steps"
                  if res["success"] else f"✗ {res['final_dist_cm']:.2f}cm")
        print(status)

    env.close()

    print_summary(results, args.checkpoint, lookup_table_npz)
    plot_all_trajectories(results)
    plot_distance_histogram(results)
    plot_steps_to_success(results)