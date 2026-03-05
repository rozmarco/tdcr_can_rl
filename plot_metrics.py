"""
Usage:
    python plot_metrics.py --reward ./logs/monitor --name sac_run1

Description:
    Plot functions register themselves using @plot_metric("name").
    Each decorator automatically creates a CLI flag (--reward, etc.).
    Data is loaded from Stable-Baselines monitor logs.
"""

import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy


# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Registry for plotting functions
PLOT_REGISTRY = {}


def plot_metric(name):
    """Decorator to register plotting functions."""
    def decorator(func):
        PLOT_REGISTRY[name] = func
        return func
    return decorator

def resolve_path(path):
    """Resolve path relative to project root if not absolute."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


@plot_metric("reward")
def plot_reward(path, run_name):

    def load_monitor_rewards(path):
        """
        Loads rewards from a Stable-Baselines monitor log.
        Returns rewards grouped as [steps, runs] for plotting.
        """
        df = pd.read_csv(path, skiprows=1)

        rewards = df["r"].to_numpy()
        lengths = df["l"].to_numpy()

        steps = np.cumsum(lengths)

        return steps, rewards
    
    steps, rewards = load_monitor_rewards(path)

    # convert to shape (N,1) so mean/std logic works
    rewards = rewards.reshape(-1, 1)

    avg_return = rewards.mean(axis=1)
    std_return = rewards.std(axis=1)

    plt.figure(figsize=(8, 5))

    plt.fill_between(
        steps,
        avg_return - std_return,
        avg_return + std_return,
        color="blue",
        alpha=0.2,
        label="Std deviation",
    )

    plt.plot(
        steps,
        avg_return,
        color="blue",
        label="Average Return",
        linewidth=1,      # thickness of the line
        marker='o',       # shape of the points ('o', 's', '^', '*', etc.)
        markersize=3,     # size of the dots
        markeredgewidth=1,       # thickness of dot edge
        markeredgecolor='black'  # color of the dot edge
    )

    plt.grid(which="major", linestyle="--", alpha=0.6)
    plt.grid(which="minor", linestyle=":", alpha=0.3)
    plt.minorticks_on()
    plt.gca().set_facecolor("#f0f0f0")

    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Average Return", fontsize=12)
    plt.title("SAC Training Performance", fontsize=14)
    plt.legend()

    save_path = os.path.join("results", f"{run_name}_reward.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved reward plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")

    parser.add_argument(
        "--name",
        type=str,
        default="run",
        help="Base name for saved figures",
    )

    for plot_name in PLOT_REGISTRY:
        parser.add_argument(
            f"--{plot_name}",
            type=str,
            help=f"Path to monitor log directory for {plot_name}",
        )

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    for plot_name, plot_func in PLOT_REGISTRY.items():
        path = getattr(args, plot_name)
        resolved = resolve_path(path)
        if not resolved.exists():
            print(f"[WARNING] {plot_name} path not found: {resolved}")
            continue
        plot_func(resolved, args.name)


if __name__ == "__main__":
    main()