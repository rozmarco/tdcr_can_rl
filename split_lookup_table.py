"""
split_lookup_table.py
=====================
Splits lookup_tables_5cyl_all_goals.npz into train and test subsets
using SPATIALLY STRATIFIED sampling — the workspace is divided into a
grid of cells, and goals are sampled from each cell proportionally so
that both splits have even spatial coverage across the full workspace.

Usage
-----
    python split_lookup_table.py \
        --input  lookup_tables_5cyl_all_goals.npz \
        --output-train lookup_tables_train.npz \
        --output-test  lookup_tables_test.npz \
        --test-ratio   0.2 \
        --strat-bins   10 \
        --seed         42

Arguments
---------
    --input          Path to the full lookup table npz.
    --output-train   Output path for the training split.
    --output-test    Output path for the test split.
    --test-ratio     Fraction of goals to reserve for test (default: 0.2).
    --strat-bins     Number of bins per axis for the spatial stratification
                     grid (default: 10, giving a 10x10 = 100 cell grid).
    --seed           Random seed for reproducibility (default: 42).
    --visualize      If set, saves a scatter plot of the split.

Output
------
Each output .npz is a drop-in replacement for the original in CustomEnv.
It contains a remapped H_all with only the unique tables needed for that
split, and a remapped goal_to_table_idx pointing into the new H_all.
"""

import argparse
import numpy as np
from pathlib import Path


def stratified_spatial_split(goal_positions, test_ratio, n_bins, rng):
    """
    Divide the goal positions into a spatial grid and sample test_ratio
    of goals from each cell, returning train and test indices.

    Parameters
    ----------
    goal_positions : (N, 2) array of XY goal positions
    test_ratio     : fraction to assign to test
    n_bins         : number of bins per axis (n_bins x n_bins grid)
    rng            : np.random.Generator

    Returns
    -------
    train_idx, test_idx : arrays of raw goal indices
    """
    N = len(goal_positions)
    x = goal_positions[:, 0]
    y = goal_positions[:, 1]

    # Build stratification grid
    x_edges = np.linspace(x.min() - 1e-9, x.max() + 1e-9, n_bins + 1)
    y_edges = np.linspace(y.min() - 1e-9, y.max() + 1e-9, n_bins + 1)

    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)
    cell_id = x_bin * n_bins + y_bin  # unique cell index per goal

    train_idx  = []
    test_idx   = []
    empty_cells = 0

    for cell in range(n_bins * n_bins):
        cell_goals = np.where(cell_id == cell)[0]
        if len(cell_goals) == 0:
            empty_cells += 1
            continue

        # Shuffle within cell
        cell_goals = rng.permutation(cell_goals)

        # At least 1 test goal per cell if cell has >1 goal
        n_test = max(1, round(len(cell_goals) * test_ratio)) if len(cell_goals) > 1 else 0
        test_idx.extend(cell_goals[:n_test].tolist())
        train_idx.extend(cell_goals[n_test:].tolist())

    train_idx = np.array(train_idx, dtype=np.int32)
    test_idx  = np.array(test_idx,  dtype=np.int32)

    print(f"  Stratification grid : {n_bins}x{n_bins} = {n_bins**2} cells")
    print(f"  Non-empty cells     : {n_bins**2 - empty_cells} / {n_bins**2}")
    print(f"  Train goals         : {len(train_idx)}")
    print(f"  Test goals          : {len(test_idx)}")
    print(f"  Actual test ratio   : {len(test_idx) / N:.3f}  (target: {test_ratio:.3f})")

    return train_idx, test_idx


def build_split_npz(data, raw_indices, split_name):
    """
    Build a split npz dict containing only the H tables needed for
    the given raw_indices, with remapped goal_to_table_idx.

    Parameters
    ----------
    data        : dict-like from np.load of the full lookup table
    raw_indices : (M,) array of raw goal indices for this split
    split_name  : "train" or "test" for logging

    Returns
    -------
    out : dict suitable for np.savez_compressed
    """
    full_goal_to_table = data["goal_to_table_idx"]  # (N_raw,)
    H_all_full         = data["H_all"]               # (N_unique, Nx, Ny, Ntheta)

    # Find which unique table indices are needed for this split
    split_table_indices = full_goal_to_table[raw_indices]  # may contain -1
    valid_mask    = split_table_indices >= 0
    needed_unique = np.unique(split_table_indices[valid_mask])

    print(f"\n  [{split_name}] {len(raw_indices)} raw goals → "
          f"{len(needed_unique)} unique H tables "
          f"(of {H_all_full.shape[0]} total)")

    # Compact H_all — only tables needed for this split
    H_split = H_all_full[needed_unique]  # (N_needed, Nx, Ny, Ntheta)

    # Remap old unique idx → new compact idx
    old_to_new = np.full(H_all_full.shape[0], -1, dtype=np.int32)
    for new_idx, old_idx in enumerate(needed_unique):
        old_to_new[old_idx] = new_idx

    # Build new goal_to_table_idx for this split's raw goals (reindexed 0..M-1)
    new_goal_to_table = np.full(len(raw_indices), -1, dtype=np.int32)
    for i, raw_i in enumerate(raw_indices):
        old_t = full_goal_to_table[raw_i]
        if old_t >= 0:
            new_goal_to_table[i] = old_to_new[old_t]

    n_covered = int((new_goal_to_table >= 0).sum())
    print(f"  [{split_name}] Goals with valid H tables: {n_covered}/{len(raw_indices)}")

    out = {
        # Core lookup data
        "H_all":                  H_split,
        "goal_to_table_idx":      new_goal_to_table,

        # Raw goal data for this split (reindexed 0..M-1)
        "raw_goal_positions":     data["raw_goal_positions"][raw_indices],
        "raw_goal_thetas":        data["raw_goal_thetas"][raw_indices],

        # Maps each split goal back to its original index in the full dataset.
        # Use this to match with workspace_train.npz rows.
        "original_raw_indices":   raw_indices,

        # Unique goal metadata (only the needed subset)
        "unique_goal_positions":  data["unique_goal_positions"][needed_unique],
        "unique_goal_thetas":     data["unique_goal_thetas"][needed_unique],
        "unique_goal_theta_bins": data["unique_goal_theta_bins"][needed_unique],
        "unique_goal_nodes":      data["unique_goal_nodes"][needed_unique],
        "finite_counts":          data["finite_counts"][needed_unique],

        # Grid metadata (unchanged)
        "xlim":           data["xlim"],
        "ylim":           data["ylim"],
        "cell_size":      data["cell_size"],
        "Ntheta":         data["Ntheta"],
        "dtheta":         data["dtheta"],
        "start_position": data["start_position"],
        "start_node":     data["start_node"],
        "heuristic_type": data["heuristic_type"],

        # Split metadata
        "split_name":      np.array(split_name, dtype=object),
        "n_raw_goals":     np.int32(len(raw_indices)),
        "n_unique_tables": np.int32(len(needed_unique)),
    }

    return out


def visualize_split(goal_positions, train_idx, test_idx, output_path):
    """Save a scatter plot of the spatial split."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping visualization.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, indices, title, color in [
        (axes[0], np.arange(len(goal_positions)), "All goals",   "gray"),
        (axes[1], train_idx,                       "Train goals", "steelblue"),
        (axes[2], test_idx,                        "Test goals",  "coral"),
    ]:
        ax.scatter(goal_positions[indices, 0], goal_positions[indices, 1],
                   s=3, alpha=0.5, c=color)
        ax.set_title(f"{title} (n={len(indices)})")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Spatially Stratified Train/Test Split", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"  Visualization saved → {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        type=str, required=True)
    parser.add_argument("--output-train", type=str, default="lookup_tables_train.npz")
    parser.add_argument("--output-test",  type=str, default="lookup_tables_test.npz")
    parser.add_argument("--test-ratio",   type=float, default=0.2)
    parser.add_argument("--strat-bins",   type=int,   default=10)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--visualize",    action="store_true")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    data  = np.load(args.input, allow_pickle=True)
    N_raw = len(data["raw_goal_positions"])
    print(f"Total raw goals     : {N_raw}")
    print(f"Total unique tables : {data['H_all'].shape[0]}")
    print(f"H_all shape         : {data['H_all'].shape}")

    # Spatially stratified split
    print(f"\nSplitting ({args.strat_bins}x{args.strat_bins} grid, "
          f"test_ratio={args.test_ratio}, seed={args.seed}) ...")
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = stratified_spatial_split(
        data["raw_goal_positions"], args.test_ratio, args.strat_bins, rng
    )

    # Build and save train split
    print("\nBuilding train split ...")
    train_out = build_split_npz(data, train_idx, "train")
    np.savez_compressed(args.output_train, **train_out)
    print(f"  Saved → {args.output_train}  "
          f"({Path(args.output_train).stat().st_size / 1e6:.1f} MB)")

    # Build and save test split
    print("\nBuilding test split ...")
    test_out = build_split_npz(data, test_idx, "test")
    np.savez_compressed(args.output_test, **test_out)
    print(f"  Saved → {args.output_test}  "
          f"({Path(args.output_test).stat().st_size / 1e6:.1f} MB)")

    # Spatial coverage summary
    print("\nSpatial coverage check:")
    for name, idx in [("train", train_idx), ("test", test_idx)]:
        pos = data["raw_goal_positions"][idx]
        print(f"  {name}: X=[{pos[:,0].min():.3f}, {pos[:,0].max():.3f}]  "
              f"Y=[{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]")

    print(f"\nNOTE: 'original_raw_indices' is saved in each .npz so you can")
    print(f"      create matching workspace splits from your workspace_train.npz.")

    if args.visualize:
        vis_path = str(Path(args.output_train).parent / "split_visualization.png")
        print(f"\nGenerating visualization ...")
        visualize_split(data["raw_goal_positions"], train_idx, test_idx, vis_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
