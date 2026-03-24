import os
import numpy as np


class MultiPoseLookupTable:
    def __init__(self, npz_paths, train_goal_count=None):
        if isinstance(npz_paths, str):
            npz_paths = [npz_paths]
        if not npz_paths:
            raise ValueError("npz_paths cannot be empty")

        H_list = []
        goal_xy_list = []
        goal_theta_list = []

        self.cell_size = None
        self.Ntheta = None
        self.dtheta = None
        self.xlim = None
        self.ylim = None

        total_goals = 0

        for path in npz_paths:
            data = np.load(path, allow_pickle=True)
            H = np.asarray(data["H_all"], dtype=np.float32)

            cell_size = float(np.asarray(data["cell_size"]).item())
            Ntheta = int(np.asarray(data["Ntheta"]).item())
            dtheta = float(np.asarray(data["dtheta"]).item())

            fname = os.path.basename(path).lower()
            if "rao" in fname:
                xlim = (-0.25, 0.25)
                ylim = (-0.1, 0.4)
                goal_xy = np.asarray(data["goal_positions_snapped"], dtype=np.float32)
                goal_theta = np.asarray(data["goal_thetas_raw"], dtype=np.float32)
            elif "lookup_tables" in fname:
                xlim = (-0.25, 0.25)
                ylim = (-0.1, 0.4)
                goal_xy = np.asarray(data["goal_positions_xy_snapped"], dtype=np.float32)
                goal_theta = np.asarray(data["goal_thetas_raw"], dtype=np.float32)
            else:
                raise ValueError(f"Could not infer file format from filename: {path}")

            if H.shape[0] != len(goal_xy) or H.shape[0] != len(goal_theta):
                raise ValueError(f"Goal count mismatch inside file: {path}")

            if self.cell_size is None:
                self.cell_size = cell_size
                self.Ntheta = Ntheta
                self.dtheta = dtheta
                self.xlim = xlim
                self.ylim = ylim
                self.Nx = H.shape[1]
                self.Ny = H.shape[2]
            else:
                if not np.isclose(self.cell_size, cell_size):
                    raise ValueError(f"cell_size mismatch in {path}")
                if self.Ntheta != Ntheta:
                    raise ValueError(f"Ntheta mismatch in {path}")
                if not np.isclose(self.dtheta, dtheta):
                    raise ValueError(f"dtheta mismatch in {path}")
                if self.xlim != xlim or self.ylim != ylim:
                    raise ValueError(f"xlim/ylim mismatch in {path}")
                if self.Nx != H.shape[1] or self.Ny != H.shape[2]:
                    raise ValueError(f"grid shape mismatch in {path}")

            H_list.append(H)
            goal_xy_list.append(goal_xy)
            goal_theta_list.append(goal_theta)
            total_goals += H.shape[0]

        self.H_all = np.concatenate(H_list, axis=0)
        self.goal_positions_xy = np.concatenate(goal_xy_list, axis=0)
        self.goal_theta = np.concatenate(goal_theta_list, axis=0)

        self.num_goals = self.H_all.shape[0]
        
        if train_goal_count is None:
            self.train_goal_indices = np.arange(self.num_goals, dtype=np.int32)
        else:
            n = min(int(train_goal_count), self.num_goals)
            if n <= 0:
                raise ValueError("train_goal_count must be >= 1")
        
            # sample a random subset from ALL loaded goals
            self.train_goal_indices = np.random.choice(
                self.num_goals,
                size=n,
                replace=False
            ).astype(np.int32)
            '''
        if train_goal_count is None:
            self.train_goal_indices = np.arange(self.num_goals, dtype=np.int32)
        else:
            n = min(int(train_goal_count), self.num_goals)
            if n <= 0:
                raise ValueError("train_goal_count must be >= 1")
            self.train_goal_indices = np.arange(n, dtype=np.int32)'''

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
        ix = int(np.round((x - self.xlim[0]) / self.cell_size - 0.5))
        iy = int(np.round((y - self.ylim[0]) / self.cell_size - 0.5))
        if ix < 0 or ix >= self.Nx or iy < 0 or iy >= self.Ny:
            return None
        return ix, iy

    def get_goal_pose(self, goal_idx: int):
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
        cell = self.world_to_cell(x, y)
        if cell is None:
            return float(unreachable_value), False, None

        ix, iy = cell
        it = self.bin_from_theta(theta)

        cost = float(self.H_all[goal_idx, ix, iy, it])
        if np.isfinite(cost):
            return cost, True, (ix, iy, it)

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
                    d = dx * dx + dy * dy + dt * dt
                    if d < best_dist:
                        best_dist = d
                        best = (c, (nix, niy, nit))

        if best is None:
            return float(unreachable_value), False, (ix, iy, it)

        return best[0], True, best[1]