"""
heuristic_map.py — BFS-graph cost-to-go heuristic for TDCR reward shaping.

Inspired by:
    Rao et al., "Towards Contact-Aided Motion Planning for
    Tendon-Driven Continuum Robots", arXiv:2402.14175

The original paper precomputes one SE(2) grid per goal using CC-arc backward
search (~20 min/goal × 1,871 goals = intractable). This module implements the
same *idea* — a precomputed cost-to-go that respects robot kinematics and
contact sequences — but uses the BFS exploration data that already exists
rather than re-solving a planning problem from scratch.

Key insight
-----------
The BFS data (explored_configs.npz) is already a discretised shortest-path
tree over the robot's reachable SE(2) workspace. Each node stores:
    - actuator_configs (clark_x_steps, slide_steps) — the BFS coordinates
    - tip_positions    (x, y, z)                    — world-frame tip XY
    - tip_poses        (9,) rotation matrix          — tip heading theta

The BFS step count from home to a node IS the cost-to-go from home.
The cost between any two nodes is the L1 distance in BFS step space
(manhattan distance in clark_x_steps, slide_steps), which approximates
the number of controller ticks needed to move between them.

The heuristic h(s, g) for current state s and goal g is:
    h(s, g) = ||BFS(s) - BFS(g)||_1  * STEPS_TO_METRES

where BFS(s) is the nearest BFS node to the current tip pose s.

This is:
  - Built in ~1 second (one KD-tree build over 1,871 nodes)
  - Kinematically valid (only uses configurations the robot can actually reach)
  - Contact-aware (BFS nodes that required contact to reach are included)
  - Goal-conditioned without needing a separate grid per goal

Potential-based shaping
-----------------------
    Phi(s)    = -h(s, g)
    F(s, s')  = gamma*Phi(s') - Phi(s) = h(s, g) - gamma*h(s', g)

Positive when moving closer to goal in BFS-step space, negative when moving
away. Policy-invariant under the conditions of Ng et al. (1999).

Usage
-----
    from src.environment.heuristic_map import BFSHeuristicMap

    hmap = BFSHeuristicMap(bfs_npz_path)   # builds in ~1 second

    # At episode reset:
    hmap.set_goal(goal_idx)

    # At each step:
    F = hmap.shaping_reward(prev_tip_pos, prev_tip_heading,
                            next_tip_pos, next_tip_heading, gamma=0.99)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Weight applied to the shaping term in get_reward.
SHAPING_WEIGHT = 0.5

# Scale factor: one BFS step ~ how many metres of tip travel.
STEPS_TO_METRES = 0.27 / 67  # (0.27m max tip travel in BFS / 67 BFS steps from home to furthest node)

# SE(2) KD-tree weights: heading (radians) weighted lower than XY (metres).
XY_WEIGHT      = 1.0
HEADING_WEIGHT = 0.1


# ---------------------------------------------------------------------------
# BFSHeuristicMap
# ---------------------------------------------------------------------------

class BFSHeuristicMap:
    """
    Goal-conditioned cost-to-go heuristic built directly from BFS data.

    Parameters
    ----------
    bfs_npz_path : str or Path
        Path to explored_configs.npz produced by bfs_explore.py.
        Must contain: tip_positions (N,3), tip_poses (N,9),
                      actuator_configs (N,2).
    """

    def __init__(self, bfs_npz_path: str):
        bfs_npz_path = Path(bfs_npz_path)
        assert bfs_npz_path.exists(), f"BFS file not found: {bfs_npz_path}"

        data = np.load(bfs_npz_path)

        self.positions  = data["tip_positions"].astype(np.float32)[:, :2]  # (N, 2)
        self.bfs_coords = data["actuator_configs"].astype(np.float32)      # (N, 2)

        if "tip_poses" in data:
            rot = data["tip_poses"].astype(np.float32)   # (N, 9) row-major
            self.headings = np.arctan2(rot[:, 3], rot[:, 0]).astype(np.float32)
        else:
            self.headings = np.zeros(len(self.positions), dtype=np.float32)

        self.n_nodes = len(self.positions)

        # Build KD-tree over weighted SE(2) coords for fast nearest-node lookup
        tree_pts = self._to_tree_coords(self.positions, self.headings)
        self._pose_tree = cKDTree(tree_pts)

        self._goal_idx: int = 0
        self._goal_bfs: np.ndarray = self.bfs_coords[0]

        print(
            f"[Heuristic] BFSHeuristicMap built — {self.n_nodes} nodes, "
            f"clark=[{self.bfs_coords[:,0].min():.0f}, "
            f"{self.bfs_coords[:,0].max():.0f}]  "
            f"slide=[{self.bfs_coords[:,1].min():.0f}, "
            f"{self.bfs_coords[:,1].max():.0f}]"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_goal(self, goal_idx: int):
        """Set the current goal index. Call at every episode reset."""
        self._goal_idx = int(goal_idx)
        self._goal_bfs = self.bfs_coords[self._goal_idx]

    def cost_to_go(self, tip_pos: np.ndarray, tip_heading: float) -> float:
        """
        Estimated cost from current tip pose to current goal,
        in units of metres (BFS steps * STEPS_TO_METRES).
        """
        nearest_idx = self._nearest_node(tip_pos, tip_heading)
        state_bfs   = self.bfs_coords[nearest_idx]
        step_dist   = float(np.sum(np.abs(state_bfs - self._goal_bfs)))
        return step_dist * STEPS_TO_METRES

    def shaping_reward(
        self,
        prev_pos: np.ndarray, prev_heading: float,
        next_pos: np.ndarray, next_heading: float,
        gamma: float = 0.99,
    ) -> float:
        """
        Potential-based shaping: F = h(s,g) - gamma*h(s',g).
        Positive when moving toward goal in BFS-step space.
        Capped at [-2, +2] to avoid dominating the base reward.
        """
        h_prev = self.cost_to_go(prev_pos, prev_heading)
        h_next = self.cost_to_go(next_pos, next_heading)
        raw    = SHAPING_WEIGHT * (h_prev - gamma * h_next)
        return float(np.clip(raw, -2.0, 2.0))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tree_coords(
        self, positions: np.ndarray, headings: np.ndarray
    ) -> np.ndarray:
        return np.column_stack([
            positions[:, 0] * XY_WEIGHT,
            positions[:, 1] * XY_WEIGHT,
            headings         * HEADING_WEIGHT,
        ]).astype(np.float32)

    def _nearest_node(self, tip_pos: np.ndarray, tip_heading: float) -> int:
        query = np.array([
            float(tip_pos[0]) * XY_WEIGHT,
            float(tip_pos[1]) * XY_WEIGHT,
            float(tip_heading) * HEADING_WEIGHT,
        ], dtype=np.float32)
        _, idx = self._pose_tree.query(query)
        return int(idx)


# ---------------------------------------------------------------------------
# Validation script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Validates the BFS npz and confirms the heuristic map builds correctly.
    No precomputation needed — the map is built at runtime from the npz.

    Usage:
        python -m src.environment.heuristic_map \\
            --bfs tdcr_sim_mujoco/exploration_data/explored_configs_5cyl.npz
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate BFS npz and test BFSHeuristicMap"
    )
    parser.add_argument("--bfs", required=True, help="Path to explored_configs.npz")
    args = parser.parse_args()

    print(f"Loading: {args.bfs}\n")
    hmap = BFSHeuristicMap(args.bfs)

    print("\nSample cost-to-go queries:")
    indices = [0, hmap.n_nodes // 4, hmap.n_nodes // 2, hmap.n_nodes - 1]
    for goal_idx in indices:
        hmap.set_goal(goal_idx)
        gpos = hmap.positions[goal_idx]
        gh   = hmap.headings[goal_idx]

        c_self = hmap.cost_to_go(gpos, gh)
        c_home = hmap.cost_to_go(np.array([0.004, 0.1], dtype=np.float32), 0.0)

        print(
            f"  goal[{goal_idx:4d}]  "
            f"pos=({gpos[0]:.3f},{gpos[1]:.3f})  "
            f"theta={np.degrees(gh):.1f}deg  "
            f"h(self)={c_self:.4f}  h(home)={c_home:.4f}"
        )

    print("\nOK — no precomputation needed, ready for training.")