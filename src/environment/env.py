import re
import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

from tdcr_sim_mujoco.src.controllers import (
    LinearBaseStiffnessController,
    TDCRJointController
)

# ---------------------------------------------------------------------------
# Physical action limits — sourced directly from explored_configs npz metadata
# ---------------------------------------------------------------------------
CLARK_MAX     = 20.0   # clark_max from npz
EXTENSION_MIN = 0.1    # slide_min_m from npz (robot cannot retract below 0.1 m)
EXTENSION_MAX = 0.37   # slide_max_m from npz

# Slew-rate caps: maximum change allowed in one 100 ms control cycle.
MAX_CLARK_STEP = 2.0   # Clark units per cycle  (~10% of full range)
MAX_EXT_STEP   = 0.01  # metres per cycle       (~2.7% of full range)

# Reward shaping scale.
SHAPING_SCALE = 1.0

# Dwell: number of consecutive steps the tip must remain inside the goal
# radius before the episode is considered a success.  5 steps = 500 ms.
DWELL_REQUIRED = 3


# ---------------------------------------------------------------------------
# HeuristicTable
# ---------------------------------------------------------------------------

class HeuristicTable:
    """
    Loads the lookup table produced by generate_lookup_tables_gpu.py and
    exposes a single method:  phi(x, y, goal_idx) -> float

    The potential Φ(s) = min over θ-bins of H_all[table_idx, ix, iy, :]
    following the smoothing procedure from Rao et al. Section III-D.

    Shaped reward:
        r_shaped = r_task + SHAPING_SCALE * (phi(s_t) - phi(s_{t+1}))
    """

    INF_SUBSTITUTE = 1e4

    def __init__(
        self,
        npz_path: str,
        H_all: np.ndarray = None,
        goal_to_table_idx: np.ndarray = None,
    ):
        # Always load the npz for grid metadata (xlim, ylim, cell_size).
        # These are tiny scalars — cheap regardless of how the big arrays arrive.
        d = np.load(npz_path, allow_pickle=True)

        if H_all is not None and goal_to_table_idx is not None:
            # Pre-loaded from Ray shared memory — skip the 221MB array load.
            self.H_all             = H_all
            self.goal_to_table_idx = goal_to_table_idx
        else:
            # Local / non-Ray path — load everything from disk as before.
            self.H_all             = d["H_all"]
            self.goal_to_table_idx = d["goal_to_table_idx"]

        self.xlim      = (float(d["xlim"][0]), float(d["xlim"][1]))
        self.ylim      = (float(d["ylim"][0]), float(d["ylim"][1]))
        self.cell_size = float(d["cell_size"])
        _, self.Nx, self.Ny, _ = self.H_all.shape

        n_tables = self.H_all.shape[0]
        n_raw    = len(self.goal_to_table_idx)
        n_valid  = int((self.goal_to_table_idx >= 0).sum())
        print(
            f"[HeuristicTable] Loaded {n_tables} unique tables "
            f"covering {n_valid}/{n_raw} raw goals  "
            f"(H shape: {self.H_all.shape})"
        )

    def _world_to_cell(self, x: float, y: float):
        ix = int((x - self.xlim[0]) / self.cell_size)
        iy = int((y - self.ylim[0]) / self.cell_size)
        ix = max(0, min(self.Nx - 1, ix))
        iy = max(0, min(self.Ny - 1, iy))
        return ix, iy

    def phi(self, x: float, y: float, goal_idx: int) -> float:
        tidx = int(self.goal_to_table_idx[goal_idx])
        if tidx < 0:
            return self.INF_SUBSTITUTE
        ix, iy  = self._world_to_cell(x, y)
        h_slice = self.H_all[tidx, ix, iy, :]
        val     = float(np.min(h_slice))
        return val if np.isfinite(val) else self.INF_SUBSTITUTE

    def has_table(self, goal_idx: int) -> bool:
        return int(self.goal_to_table_idx[goal_idx]) >= 0


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CustomEnv(MujocoEnv):
    """
    Custom MuJoCo environment for a continuum manipulator (TDCR) with a
    linear sliding base, trained with SAC for goal-reaching tasks.

    Both goal positions and the heuristic lookup tables are loaded from a
    single pre-computed npz file (lookup_tables_train.npz), which is produced
    by generate_lookup_tables_gpu.py and contains:

        raw_goal_positions    (N_raw, 2)  — workspace XY positions
        H_all                 (N_unique, Nx, Ny, Ntheta)
        goal_to_table_idx     (N_raw,)    — maps each raw goal to H_all row
        xlim, ylim, cell_size, Ntheta     — grid metadata

    Pass the same file path for both workspace_npz and lookup_table_npz.
    The legacy separate workspace_npz path is still supported for backwards
    compatibility; if lookup_table_npz is None the env falls back to
    workspace_npz for goal sampling and disables reward shaping.

    Success criterion
    -----------------
    The episode terminates successfully only after the tip remains within
    the 2 cm goal radius for DWELL_REQUIRED consecutive steps (default 5,
    i.e. 500 ms).  A single accidental crossing no longer counts.

    Contact handling
    ----------------
    The flat contact_bonus has been removed.  Contact that lies on the
    geometrically optimal path is already rewarded implicitly via the
    heuristic shaping term Φ(s_t) - Φ(s_{t+1}), which was computed with
    contact-aware motion primitives.  A small contact_penalty discourages
    pathological obstacle-pushing that the flat bonus used to incentivise.

    Action space (2-D, continuous, [-1, 1]):
        action[0]  ->  Clark X target  mapped to [-CLARK_MAX,  +CLARK_MAX]
        action[1]  ->  extension target mapped to [EXTENSION_MIN, EXTENSION_MAX]
    """

    def __init__(
        self,
        scene_path: str,
        workspace_npz: str,
        render_mode: str = "human",
        frame_skips: int = 50,
        timstep: float = 0.002,
        n_curv_bins: int = 10,
        n_contact_bins: int = 10,
        allow_contact_goals: bool = False,
        lookup_table_npz: str = None,
        H_all: np.ndarray = None,
        goal_to_table_idx: np.ndarray = None,
        **kwargs
    ):
        self.n_curv_bins    = n_curv_bins
        self.n_contact_bins = n_contact_bins

        # ── Decide which file to use for goal sampling ────────────────────
        use_lookup = (lookup_table_npz is not None) or (H_all is not None)
        goal_source = lookup_table_npz if lookup_table_npz is not None else workspace_npz
        self._load_workspace(goal_source, allow_contact_goals,
                            from_lookup_table=use_lookup)
        self.n_obstacles = 0

        super().__init__(
            model_path=str(scene_path),
            render_mode=render_mode,
            observation_space=self._build_observation_space(),
            frame_skip=frame_skips,
            max_geom=1000,
            **kwargs
        )

        self._cache_ids()
        self.observation_space = self._build_observation_space()

        self.base_pos = np.array([0.0, 0.0])

        self._current_goal_idx: int = 0
        self.goal_pos = self._sample_goal_from_workspace()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.linear_base_ctrl = LinearBaseStiffnessController(
            self.model, self.data, inside_stiffness=50.0
        )
        self.linear_base_ctrl.update_stiffness()

        self.tdcr_controller = TDCRJointController(
            self.model, self.data, clark_speed_scale=0.001, fps=100
        )

        self.model.opt.timestep = timstep

        # ── Heuristic table ───────────────────────────────────────────────
        self._heuristic: HeuristicTable | None = None
        if lookup_table_npz is not None or H_all is not None:
            # npz_path is needed for grid metadata (xlim, ylim, cell_size) even when
            # arrays are pre-loaded — fall back to workspace_npz since they're the same file
            self._heuristic = HeuristicTable(
                npz_path=lookup_table_npz if lookup_table_npz is not None else workspace_npz,
                H_all=H_all,
                goal_to_table_idx=goal_to_table_idx,
            )
        else:
            print("[CustomEnv] lookup_table_npz not provided — running with original reward function...")

        self._prev_tip_pos: np.ndarray = np.zeros(2, dtype=np.float64)

        # ── Dwell counter ─────────────────────────────────────────────────
        self._goal_dwell_steps: int = 0

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def _load_workspace(self, npz_path: str, allow_contact_goals: bool,
                        from_lookup_table: bool = False):
        ws = np.load(npz_path, allow_pickle=True)

        if from_lookup_table:
            all_positions = np.asarray(ws["raw_goal_positions"], dtype=np.float32)
            N = len(all_positions)
            idx = np.arange(N, dtype=np.int32)

            self._ws_tip_pos      = all_positions[:, :2]
            self._ws_actuators    = np.zeros((N, 2), dtype=np.float32)
            self._ws_original_idx = idx

            print(
                f"Workspace loaded from lookup table: {N} raw goals  "
                f"x=[{self._ws_tip_pos[:,0].min():.3f}, {self._ws_tip_pos[:,0].max():.3f}]  "
                f"y=[{self._ws_tip_pos[:,1].min():.3f}, {self._ws_tip_pos[:,1].max():.3f}]"
            )
        else:
            mask = np.ones(len(ws['tip_positions']), dtype=bool)
            idx  = np.where(mask)[0]
            self._ws_tip_pos      = ws['tip_positions'][idx, :2].astype(np.float32)
            self._ws_actuators    = ws['actuator_configs'][idx].astype(np.float32)
            self._ws_original_idx = idx.astype(np.int32)

            print(
                f"Workspace loaded (legacy): {len(idx)} configs  "
                f"x=[{self._ws_tip_pos[:,0].min():.3f}, {self._ws_tip_pos[:,0].max():.3f}]  "
                f"y=[{self._ws_tip_pos[:,1].min():.3f}, {self._ws_tip_pos[:,1].max():.3f}]"
            )

    def _sample_goal_from_workspace(self) -> np.ndarray:
        local_idx = np.random.randint(len(self._ws_tip_pos))
        self._current_goal_idx = int(self._ws_original_idx[local_idx])
        return self._ws_tip_pos[local_idx].copy()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _cache_ids(self):
        possible_tip_names = ["tdcr_tip", "EE_pos", "ee_pos", "end_effector"]
        self.tip_body_id = None
        for name in possible_tip_names:
            try:
                self.tip_body_id = self.model.body(name).id
                break
            except KeyError:
                continue

        if self.tip_body_id is None:
            body_names = [self.model.body(i).name for i in range(self.model.nbody)]
            self.tip_body_id = (
                self.model.body("link_25").id if "link_25" in body_names
                else self.model.body("link_0").id
            )

        self.link_body_ids = [
            i for i in range(self.model.nbody)
            if "link" in self.model.body(i).name
        ]

        self.obstacle_body_ids = [
            i for i in range(self.model.nbody)
            if re.fullmatch(r"obstacle\d+", self.model.body(i).name)
        ]
        self.obstacle_body_name = [
            self.model.body(i).name for i in self.obstacle_body_ids
        ]
        self.n_obstacles = len(self.obstacle_body_ids)

        self.linear_jnt_ids = [
            j for j in range(self.model.njnt)
            if self.model.joint(j).name and "linear_" in self.model.joint(j).name
        ]
        self.linear_actuator_ids = [
            a for a in range(self.model.nu)
            if self.model.actuator(a).name and "linear_" in self.model.actuator(a).name
        ]

    def _build_observation_space(self):
        return spaces.Dict({
            "tendon_length":   spaces.Box(-np.inf, np.inf, shape=(3,),                   dtype=np.float32),
            "extension":       spaces.Box(-np.inf, np.inf, shape=(1,),                   dtype=np.float32),
            "curvature_hist":  spaces.Box(0.0,     np.inf, shape=(self.n_curv_bins,),     dtype=np.float32),
            "contact_hist":    spaces.Box(0.0,     np.inf, shape=(self.n_contact_bins,),  dtype=np.float32),
            "goal_rel_pos":    spaces.Box(-np.inf, np.inf, shape=(2,),                   dtype=np.float32),
            "obstacle_pos":    spaces.Box(-np.inf, np.inf, shape=(self.n_obstacles * 2,), dtype=np.float32),
            "obstacle_radius": spaces.Box(0.0,     np.inf, shape=(self.n_obstacles,),     dtype=np.float32),
        })

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _compute_curvature_histogram(self):
        angles = []
        for i in range(len(self.link_body_ids) - 1):
            dx = self.data.xpos[self.link_body_ids[i+1]] - self.data.xpos[self.link_body_ids[i]]
            angles.append(np.arctan2(dx[1], dx[0]))
        hist, _ = np.histogram(
            np.array(angles), bins=self.n_curv_bins, range=(-np.pi, np.pi), density=True
        )
        return hist.astype(np.float32)

    def _compute_contact_histogram(self):
        contact_s = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for geom_id in [con.geom1, con.geom2]:
                body_id = self.model.geom_bodyid[geom_id]
                if body_id in self.link_body_ids:
                    s = self.link_body_ids.index(body_id) / max(len(self.link_body_ids) - 1, 1)
                    contact_s.append(s)
        hist, _ = np.histogram(contact_s, bins=self.n_contact_bins, range=(0.0, 1.0))
        return hist.astype(np.float32)

    def _get_extension(self):
        return self.linear_base_ctrl.get_slide_position()

    def _compute_obstacle_pos(self):
        tip_pos = self.data.xpos[self.tip_body_id][:2]
        rel = []
        for name in self.obstacle_body_name:
            actual_pos = self.model.body(name).ipos[:2]
            rel.extend(actual_pos - tip_pos)
        return np.array(rel, dtype=np.float32)

    def _get_obstacle_radius(self):
        radii = []
        for name in self.obstacle_body_name:
            body = self.model.body(name)
            g_id = self.model.body_geomadr[body.id]
            radii.append(self.model.geom_size[g_id][0])
        return np.array(radii, dtype=np.float32)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self):
        tip_pos = self.data.xpos[self.tip_body_id][:2]
        return {
            "tendon_length":   self.data.ten_length[:3].copy().astype(np.float32),
            "extension":       np.array([self._get_extension()], dtype=np.float32),
            "curvature_hist":  self._compute_curvature_histogram(),
            "contact_hist":    self._compute_contact_histogram(),
            "goal_rel_pos":    (self.goal_pos - tip_pos).astype(np.float32),
            "obstacle_pos":    self._compute_obstacle_pos(),
            "obstacle_radius": self._get_obstacle_radius(),
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def get_reward(self, obs: dict, action: np.ndarray,
                   prev_tip_pos: np.ndarray, next_tip_pos: np.ndarray) -> float:
        dist = np.linalg.norm(obs["goal_rel_pos"])

        if self._goal_dwell_steps >= DWELL_REQUIRED:
            goal_bonus = 100.0
        else:
            goal_bonus = 0.0

        dist_reward     = -(dist ** 2)
        contact_penalty = -0.02 * float(np.sum(obs["contact_hist"]))
        action_penalty  = -0.0001 * float(np.sum(action ** 2))
        time_penalty    = -0.0001

        r_task = dist_reward + goal_bonus + contact_penalty + action_penalty + time_penalty

        if self._heuristic is None:
            return r_task

        if not self._heuristic.has_table(self._current_goal_idx):
            return r_task

        phi_prev = self._heuristic.phi(
            float(prev_tip_pos[0]), float(prev_tip_pos[1]),
            self._current_goal_idx
        )
        phi_next = self._heuristic.phi(
            float(next_tip_pos[0]), float(next_tip_pos[1]),
            self._current_goal_idx
        )

        PHI_CLIP = 50
        phi_prev = min(phi_prev, PHI_CLIP)
        phi_next = min(phi_next, PHI_CLIP)

        shaping = SHAPING_SCALE * (phi_prev - phi_next)
        return r_task + shaping

    # ------------------------------------------------------------------
    # Termination  (dwell-based)
    # ------------------------------------------------------------------

    def is_terminal(self, obs: dict) -> bool:
        inside = np.linalg.norm(obs["goal_rel_pos"]) < 0.02
        if inside:
            self._goal_dwell_steps += 1
        else:
            self._goal_dwell_steps = 0
        return self._goal_dwell_steps >= DWELL_REQUIRED

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _remap_action(self, action: np.ndarray):
        clark_target = float(action[0]) * CLARK_MAX
        ext_target   = EXTENSION_MIN + (float(action[1]) + 1.0) * 0.5 * (EXTENSION_MAX - EXTENSION_MIN)
        return clark_target, ext_target

    def _apply_clark_target(self, clark_target: float):
        kinematics = self.tdcr_controller.kinematics
        seg_idx    = self.tdcr_controller.current_segment
        current_x  = kinematics.goal_clark_coords[seg_idx * 2]
        delta      = np.clip(clark_target - current_x, -MAX_CLARK_STEP, MAX_CLARK_STEP)
        clark_increment = np.zeros(2 * self.tdcr_controller.n_segments)
        clark_increment[seg_idx * 2] = delta
        target_tendons = kinematics.clark_coords_increment_to_tendon(clark_increment)
        for i, (act_id, length) in enumerate(
            zip(self.tdcr_controller.tendon_actuator_ids, target_tendons)
        ):
            pretension = (
                self.tdcr_controller.pretension_lengths[i]
                if self.tdcr_controller.pretension_lengths is not None else 0.0
            )
            self.data.ctrl[act_id] = pretension + length

    def _apply_extension_target(self, ext_target: float):
        current = self.linear_base_ctrl.get_slide_target()
        delta   = np.clip(ext_target - current, -MAX_EXT_STEP, MAX_EXT_STEP)
        self.linear_base_ctrl.set_slide_target(current + delta)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = np.zeros_like(self.init_qvel)
        qpos += np.random.uniform(-0.01, 0.01, size=qpos.shape)
        self.set_state(qpos, qvel)
        self.goal_pos = self._sample_goal_from_workspace()
        self._prev_tip_pos     = self.data.xpos[self.tip_body_id][:2].copy()
        self._goal_dwell_steps = 0
        return self.get_state()

    def step(self, action: np.ndarray):
        prev_tip_pos = self.data.xpos[self.tip_body_id][:2].copy()

        clark_target, ext_target = self._remap_action(action)
        self._apply_clark_target(clark_target)
        self._apply_extension_target(ext_target)
        self.linear_base_ctrl.update_stiffness()
        self.do_simulation(self.data.ctrl, self.frame_skip)

        next_tip_pos = self.data.xpos[self.tip_body_id][:2].copy()
        self._prev_tip_pos = next_tip_pos

        next_state = self.get_state()
        terminated = self.is_terminal(next_state)
        reward     = self.get_reward(next_state, action, prev_tip_pos, next_tip_pos)
        return next_state, reward, terminated, False, {}

    def _print_init_state(self):
        print("\n--- Environment Initialized ---")
        print(f"  qpos shape:        {self.data.qpos.shape}")
        print(f"  ctrl shape:        {self.data.ctrl.shape}")
        print(f"  num links:         {len(self.link_body_ids)}")
        print(f"  num obstacles:     {self.n_obstacles}")
        print(f"  workspace configs: {len(self._ws_tip_pos)}")
        print(f"  CLARK_MAX:         {CLARK_MAX}")
        print(f"  EXTENSION range:   [{EXTENSION_MIN}, {EXTENSION_MAX}]")
        print(f"  MAX_CLARK_STEP:    {MAX_CLARK_STEP} / cycle")
        print(f"  MAX_EXT_STEP:      {MAX_EXT_STEP} / cycle")
        print(f"  Dwell required:    {DWELL_REQUIRED} steps")
        print(f"  Heuristic shaping: {'enabled' if self._heuristic else 'disabled'}")
        print(f"-------------------------------\n")