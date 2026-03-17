"""
Usage:
    python plot_metrics.py --reward ./logs --name sac_run1

Description:
    Plot functions register themselves using @plot_metric("name").
    Each decorator automatically creates a CLI flag (--reward, etc.).
    Data is loaded from Stable-Baselines monitor logs.
"""


import os
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        Returns a dict mapping worker_id → concatenated episode rewards across all runs.
        Assumes CSVs are stored under run directories with worker ID in filename.
        """
        run_dirs = sorted([d for d in os.listdir(path)])
        worker_rewards = {}

        for run in run_dirs:
            run_path = os.path.join(path, run)
            csvs = glob.glob(os.path.join(run_path, "*.csv"))
            for csv in csvs:
                df = pd.read_csv(csv, skiprows=1)
                # extract worker id from filename (example: worker0_monitor.csv)
                worker_id = os.path.basename(csv).split('_')[0]
                if worker_id not in worker_rewards:
                    worker_rewards[worker_id] = []
                avg_reward = df["r"].mean()
                worker_rewards[worker_id].append(avg_reward)  # concatenate across runs

        # Convert lists to arrays
        for w in worker_rewards:
            worker_rewards[w] = np.array(worker_rewards[w])

        return worker_rewards
    
    all_rewards = load_monitor_rewards(path)

    # Calculate the mean/std per run
    worker_ids = sorted(all_rewards.keys())  # sort for consistent order
    rewards_matrix = np.column_stack([all_rewards[w] for w in worker_ids])  # shape: (runs, workers)
    avg_return = np.mean(rewards_matrix, axis=1)
    std_return = np.std(rewards_matrix, axis=1)

    plt.figure(figsize=(7,5))

    x_axis = np.arange(len(avg_return))

    for i, w in enumerate(worker_ids):
        plt.plot(
            x_axis, 
            rewards_matrix[:, i], 
            linestyle=':', 
            alpha=0.5
        )

    plt.plot(
        x_axis,
        avg_return,
        color="#64b5eb",
        label="SAC",
        linewidth=2.0,
        linestyle='-'
    )

    plt.fill_between(
        x_axis,
        avg_return - std_return,
        avg_return + std_return,
        color="#64b5eb",
        alpha=0.2,
    )

    plt.grid(which="major", linestyle="-", alpha=0.5)
    plt.gca().set_facecolor("#f0f7fb")

    plt.xlabel("Number of Iterations", fontsize=12)
    plt.ylabel("Average Return", fontsize=12)
    plt.title("TDCR-Agent", fontsize=14)
    plt.legend(frameon=False, fontsize=11)

    save_path = os.path.join("results", f"average_reward.png")
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

    for plot_name, plot_func in PLOT_REGISTRY.items():
        path = getattr(args, plot_name)
        resolved = resolve_path(path)
        if not resolved.exists():
            print(f"[WARNING] {plot_name} path not found: {resolved}")
            continue
        plot_func(resolved, args.name)


if __name__ == "__main__":
    try:
        main()
    except TypeError:
        print(f"TypeError: Missing required arguments")