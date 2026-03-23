import os
import numpy as np


class PoseLookupTable:
    def __init__(self, npz_path: str, train_goal_count: int | None = None):
        data = np.load(npz_path, allow_pickle=True)

        # Main lookup tensor: [num_goals, Nx, Ny, Ntheta]
        self.H_all = np.asarray(data["H_all"], dtype=np.float32)

        # Metadata
        self.cell_size = float(np.asarray(data["cell_size"]).item())
        self.Ntheta = int(np.asarray(data["Ntheta"]).item())
        self.dtheta = float(np.asarray(data["dtheta"]).item())

        fname = os.path.basename(npz_path).lower()

        # Infer bounds + goal arrays from filename convention
        if "cc_reward" in fname:
            self.xlim = (-0.2, 0.2)
            self.ylim = (-0.1, 0.4)
            self.goal_positions_xy = np.asarray(data["goal_positions_snapped"], dtype=np.float32)
            self.goal_theta = np.asarray(data["goal_thetas"], dtype=np.float32)
        elif "distance" in fname:
            self.xlim = (-0.2, 0.2)
            self.ylim = (-0.1, 0.45)
            self.goal_positions_xy = np.asarray(data["goal_positions_xy_snapped"], dtype=np.float32)
            self.goal_theta = np.asarray(data["goal_theta_raw"], dtype=np.float32)
        else:
            raise ValueError(
                "Could not infer xlim/ylim + goal key format from filename. "
                "Expected filename to contain either 'cc_reward' or 'distance'."
            )

        self.num_goals, self.Nx, self.Ny, self.H_Ntheta = self.H_all.shape

        if self.H_Ntheta != self.Ntheta:
            raise ValueError(
                f"Mismatch: H_all has Ntheta={self.H_Ntheta}, metadata says Ntheta={self.Ntheta}"
            )

        if len(self.goal_positions_xy) != self.num_goals:
            raise ValueError(
                f"goal_positions_xy length ({len(self.goal_positions_xy)}) "
                f"does not match H_all num_goals ({self.num_goals})"
            )

        if len(self.goal_theta) != self.num_goals:
            raise ValueError(
                f"goal_theta length ({len(self.goal_theta)}) "
                f"does not match H_all num_goals ({self.num_goals})"
            )

        if train_goal_count is None:
            self.train_goal_indices = np.arange(self.num_goals, dtype=np.int32)
        else:
            n = min(int(train_goal_count), self.num_goals)
            if n <= 0:
                raise ValueError("train_goal_count must be >= 1")
            self.train_goal_indices = np.arange(n, dtype=np.int32)

    @staticmethod
    def wrap_angle(theta: float) -> float:
        return (theta + 2.0 * np.pi) % (2.0 * np.pi)

    def angle_diff(self, a: float, b: float) -> float:
        d = self.wrap_angle(a) - self.wrap_angle(b)
        return (d + np.pi) % (2.0 * np.pi) - np.pi

    def bin_from_theta(self, theta: float) -> int:
        theta = self.wrap_angle(theta)
        return int(np.round(theta / self.dtheta)) % self.Ntheta

    def world_to_cell(self, x: float, y: float):
        # Nearest cell center, not floor
        ix = int(np.round((x - self.xlim[0]) / self.cell_size - 0.5))
        iy = int(np.round((y - self.ylim[0]) / self.cell_size - 0.5))

        if ix < 0 or ix >= self.Nx or iy < 0 or iy >= self.Ny:
            return None
        return ix, iy

    def cell_center(self, ix: int, iy: int):
        x = self.xlim[0] + (ix + 0.5) * self.cell_size
        y = self.ylim[0] + (iy + 0.5) * self.cell_size
        return float(x), float(y)

    def get_goal_pose(self, goal_idx: int):
        if not (0 <= goal_idx < self.num_goals):
            raise IndexError(f"goal_idx={goal_idx} out of range [0, {self.num_goals})")

        gx, gy = self.goal_positions_xy[goal_idx]
        gtheta = self.goal_theta[goal_idx]
        return float(gx), float(gy), float(gtheta)

    def sample_train_goal_idx(self) -> int:
        return int(np.random.choice(self.train_goal_indices))

    def lookup_cost(
        self,
        goal_idx: int,
        x: float,
        y: float,
        theta: float,
        unreachable_value: float = np.inf,
        search_radius: int = 1,
    ):
        if not (0 <= goal_idx < self.num_goals):
            raise IndexError(f"goal_idx={goal_idx} out of range [0, {self.num_goals})")
    
        cell = self.world_to_cell(x, y)
        if cell is None:
            return float(unreachable_value), False, None
    
        ix, iy = cell
        it = self.bin_from_theta(theta)
    
        # Exact nearest discretized lookup first
        cost = float(self.H_all[goal_idx, ix, iy, it])
        if np.isfinite(cost):
            return cost, True, (ix, iy, it)
    
        # Fallback: search nearby bins for nearest finite value
        best = None
        best_dist = np.inf
    
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                nix, niy = ix + dx, iy + dy
                if nix < 0 or nix >= self.Nx or niy < 0 or niy >= self.Ny:
                    continue
    
                for dt in range(-search_radius, search_radius + 1):
                    nit = (it + dt) % self.Ntheta
                    c = float(self.H_all[goal_idx, nix, niy, nit])
    
                    if not np.isfinite(c):
                        continue
    
                    # weighted squared distance in SE(2) index space
                    d = dx * dx + dy * dy + dt * dt
                    if d < best_dist:
                        best_dist = d
                        best = (c, (nix, niy, nit))
    
        if best is None:
            return float(unreachable_value), False, (ix, iy, it)
    
        return best[0], True, best[1]

    def lookup_cost_batch(
        self,
        goal_idx: int,
        states,
        unreachable_value: float = np.inf,
    ):
        if not (0 <= goal_idx < self.num_goals):
            raise IndexError(f"goal_idx={goal_idx} out of range [0, {self.num_goals})")

        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError(f"states must have shape (N, 3), got {states.shape}")

        N = states.shape[0]
        costs = np.full(N, unreachable_value, dtype=np.float32)
        valid = np.zeros(N, dtype=bool)

        for i in range(N):
            x, y, theta = states[i]
            cell = self.world_to_cell(float(x), float(y))
            if cell is None:
                continue

            ix, iy = cell
            it = self.bin_from_theta(float(theta))
            c = self.H_all[goal_idx, ix, iy, it]

            if np.isfinite(c):
                costs[i] = float(c)
                valid[i] = True

        return costs, valid


if __name__ == "__main__":
    LOOKUP_FILE = "/workspace/lookup_tables_5cyl_pose_distance_free.npz"

    lut = PoseLookupTable(LOOKUP_FILE, train_goal_count=400)

    goal_idx = 100
    goal_x, goal_y, goal_theta = lut.get_goal_pose(goal_idx)
    print("goal:", goal_x, goal_y, goal_theta)

    robot_x, robot_y, robot_theta = -0.19, 0.3, 1.57
    cost, is_valid, idx = lut.lookup_cost(goal_idx, robot_x, robot_y, robot_theta)

    print("cost:", cost)
    print("valid:", is_valid)
    print("idx:", idx)

    sampled_goal = lut.sample_train_goal_idx()
    print("sampled train goal idx:", sampled_goal)