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
# The heuristic H is in units of arc-length (metres) weighted by lambda_progress.
# Typical values are O(0.1–2.0). This scale factor brings the shaping signal
# into the same range as dist_reward (which is O(-0.001) near goal, O(-1) far).
# Start at 1.0 and tune if the shaping dominates or is invisible.
SHAPING_SCALE = 1.0


# ---------------------------------------------------------------------------
# HeuristicTable — thin wrapper around the precomputed lookup table npz.
# Kept here so env.py has no external import dependency.
# ---------------------------------------------------------------------------

class HeuristicTable:
    """
    Loads the lookup table produced by generate_lookup_tables_gpu.py and
    exposes a single method:  phi(x, y, goal_idx) -> float

    The potential Φ(s) is defined as:
        phi = min over θ-bins of H_all[table_idx, ix, iy, :]
    which is the smoothing procedure from Rao et al. Section III-D.

    Shaped reward:
        r_shaped = r_task + SHAPING_SCALE * (phi(s_t) - phi(s_{t+1}))

    Note the sign convention:  H is cost-to-go (smaller = closer to goal),
    so *decreasing* H means progress, and the shaped reward should be
    *positive* when H decreases.  Hence phi_prev - phi_next (not next - prev).
    """

    # Substituted when a state is outside the grid or H is inf.
    # Using a large-but-finite value avoids inf arithmetic in the reward.
    INF_SUBSTITUTE = 1e4

    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.H_all             = d["H_all"]               # (N_unique, Nx, Ny, Ntheta)
        self.goal_to_table_idx = d["goal_to_table_idx"]   # (N_raw,)  int32, -1 = no table
        self.xlim      = (float(d["xlim"][0]), float(d["xlim"][1]))
        self.ylim      = (float(d["ylim"][0]), float(d["ylim"][1]))
        self.cell_size = float(d["cell_size"])
        _, self.Nx, self.Ny, _ = self.H_all.shape

        n_tables   = self.H_all.shape[0]
        n_raw      = len(self.goal_to_table_idx)
        n_valid    = int((self.goal_to_table_idx >= 0).sum())
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
        """
        Potential Φ(s) for end-effector at world position (x, y),
        for the goal with index goal_idx in the original workspace npz.

        Returns INF_SUBSTITUTE if:
          - goal_idx has no precomputed table  (goal_to_table_idx == -1)
          - the cell is unreachable in the Dijkstra search  (H == inf)
        """
        tidx = int(self.goal_to_table_idx[goal_idx])
        if tidx < 0:
            # No table for this goal — caller should fall back to unshaped reward
            return None

        ix, iy   = self._world_to_cell(x, y)
        h_slice  = self.H_all[tidx, ix, iy, :]   # (Ntheta,)
        val      = float(np.min(h_slice))
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

    Goal positions are sampled from a pre-validated workspace npz file
    (explored_configs_5cyl_labelled.npz).

    Reward shaping
    --------------
    When a lookup_table_npz path is provided, the shaped reward is:

        r = r_task  +  SHAPING_SCALE * (Φ(s_t) - Φ(s_{t+1}))

    where Φ(s) = min_θ H[goal_table_idx, ix, iy, :] from Rao et al.
    If no table exists for the sampled goal, the reward falls back to r_task
    with no shaping applied — training remains stable.

    Action space (2-D, continuous, [-1, 1]):
        action[0]  ->  Clark X target  mapped to [-CLARK_MAX,  +CLARK_MAX]
        action[1]  ->  extension target mapped to [EXTENSION_MIN, EXTENSION_MAX]

    Observation space keys  (must match flatten_state and r_dim=41 for 5 cylinders):
        tendon_length  (3,)
        extension      (1,)
        curvature_hist (n_curv_bins,)    default 10
        contact_hist   (n_contact_bins,) default 10
        goal_rel_pos   (2,)
        obstacle_pos   (n_obstacles*2,)  relative XY per obstacle
        obstacle_radius(n_obstacles,)    radius per obstacle
        --- total for 5 cylinders: 3+1+10+10+2+10+5 = 41  (matches r_dim) ---
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
        lookup_table_npz: str = None,   # ← new: path to precomputed H tables
        **kwargs
    ):
        """
        Args:
            scene_path:          Absolute path to the MuJoCo XML model file.
            workspace_npz:       Path to explored_configs_5cyl_labelled.npz.
            render_mode:         "human", "rgb_array", or None.
            frame_skips:         Physics steps per control cycle (50 x 0.002 s = 100 ms).
            timstep:             Physics integration step in seconds.
            n_curv_bins:         Bins for the curvature histogram observation.
            n_contact_bins:      Bins for the contact histogram observation.
            allow_contact_goals: Include configs where tip contacts obstacle at goal.
            lookup_table_npz:    Path to lookup_tables_5cyl_all_goals.npz produced
                                 by generate_lookup_tables_gpu.py.
                                 If None, reward shaping is disabled and the original
                                 reward function is used unchanged.
            **kwargs:            Passed through to MujocoEnv.
        """
        self.n_curv_bins    = n_curv_bins
        self.n_contact_bins = n_contact_bins

        # Load workspace before super().__init__ so obstacle count is known.
        self._load_workspace(workspace_npz, allow_contact_goals)

        # Temporarily set n_obstacles=0; _cache_ids() sets the real value.
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

        # Tracks the current goal index into the workspace npz.
        # Set by _sample_goal_from_workspace() on every reset.
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

        # ── Heuristic table (optional) ────────────────────────────────────
        self._heuristic: HeuristicTable | None = None
        if lookup_table_npz is not None:
            self._heuristic = HeuristicTable(lookup_table_npz)
        else:
            print(
                "[CustomEnv] lookup_table_npz not provided — "
                "running with original reward function (no heuristic shaping)."
            )

        # Stores tip XY at the START of each step so we can compute Φ(s_t).
        # Initialised to zeros; overwritten at the top of step().
        self._prev_tip_pos: np.ndarray = np.zeros(2, dtype=np.float64)

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def _load_workspace(self, npz_path: str, allow_contact_goals: bool):
        """Load and filter the pre-validated configuration workspace.

        Sets:
            self._ws_tip_pos        (N, 2) float32  — XY tip positions
            self._ws_actuators      (N, 2) float32  — [clark_x, slide_m]
            self._ws_original_idx   (N,)   int32    — original row index in npz
                                                       used to look up goal_to_table_idx
        """
        ws   = np.load(npz_path)
        mask = np.ones(len(ws['tip_positions']), dtype=bool)
        # if not allow_contact_goals:
        #     mask &= ~ws['contact_at_goal']

        idx = np.where(mask)[0]
        self._ws_tip_pos       = ws['tip_positions'][idx, :2].astype(np.float32)
        self._ws_actuators     = ws['actuator_configs'][idx].astype(np.float32)
        self._ws_original_idx  = idx.astype(np.int32)   # maps local idx → npz row

        print(
            f"Workspace loaded: {len(idx)} configs  "
            f"Tip x=[{self._ws_tip_pos[:,0].min():.3f}, {self._ws_tip_pos[:,0].max():.3f}]  "
            f"y=[{self._ws_tip_pos[:,1].min():.3f}, {self._ws_tip_pos[:,1].max():.3f}]"
        )

    def _sample_goal_from_workspace(self) -> np.ndarray:
        """
        Sample a random goal from the workspace.

        Sets self._current_goal_idx to the original npz row index,
        which is what goal_to_table_idx in the lookup table is keyed on.
        """
        local_idx = np.random.randint(len(self._ws_tip_pos))
        self._current_goal_idx = int(self._ws_original_idx[local_idx])
        return self._ws_tip_pos[local_idx].copy()

    # ------------------------------------------------------------------
    # Initialisation helpers  (unchanged)
    # ------------------------------------------------------------------

    def _cache_ids(self):
        """Cache body/joint/actuator IDs. Sets self.n_obstacles."""
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
    # Observation helpers  (unchanged)
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
            body  = self.model.body(name)
            g_id  = self.model.body_geomadr[body.id]
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
        """
        Compute the (optionally shaped) reward for one transition.

        Task reward  (unchanged from original):
            r_task = dist_reward + goal_bonus + contact_bonus
                     + action_penalty + time_penalty

        Shaped reward (when lookup table is available for this goal):
            r = r_task  +  SHAPING_SCALE * (Φ(s_t) - Φ(s_{t+1}))

        The shaping term is positive when H decreases (robot made progress)
        and negative when H increases (robot moved away from goal).

        Falls back to r_task alone if:
          - No lookup table was loaded (lookup_table_npz=None)
          - This episode's goal has no precomputed table (goal_to_table_idx=-1)
          - H is inf at either tip position (unreachable cell)

        Args:
            obs:           next_state dict (same as before)
            action:        action taken
            prev_tip_pos:  world XY of tip BEFORE the physics step  (s_t)
            next_tip_pos:  world XY of tip AFTER  the physics step  (s_{t+1})
        """
        # ── Task reward (identical to original get_reward) ────────────────
        dist           = np.linalg.norm(obs["goal_rel_pos"])
        goal_bonus     = 1000.0 if dist < 0.02 else 0.0
        dist_reward    = -(dist ** 2)
        contact_bonus  =  0.01   * np.sum(obs["contact_hist"])
        action_penalty = -0.0001 * np.sum(action ** 2)
        time_penalty   = -0.0001
        r_task = dist_reward + goal_bonus + contact_bonus + action_penalty + time_penalty

        # ── Heuristic shaping  Φ(s_t) - Φ(s_{t+1}) ──────────────────────
        
        if self._heuristic is None:
            return r_task

        # Fast check: skip if no table exists for this goal
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

        # Safety clip (prevents huge spikes from unreachable states)
        PHI_CLIP = 1e3
        phi_prev = min(phi_prev, PHI_CLIP)
        phi_next = min(phi_next, PHI_CLIP)

        shaping = SHAPING_SCALE * (phi_prev - phi_next)

        return r_task + shaping

    def is_terminal(self, obs):
        return np.linalg.norm(obs["goal_rel_pos"]) < 0.02

    # ------------------------------------------------------------------
    # Action application  (unchanged)
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
        # Reset prev tip pos so the first step doesn't use a stale value
        self._prev_tip_pos = self.data.xpos[self.tip_body_id][:2].copy()
        return self.get_state()

    def step(self, action: np.ndarray):
        # ── Record tip position BEFORE physics step  (s_t) ───────────────
        prev_tip_pos = self.data.xpos[self.tip_body_id][:2].copy()

        # ── Apply action + advance simulation ────────────────────────────
        clark_target, ext_target = self._remap_action(action)
        self._apply_clark_target(clark_target)
        self._apply_extension_target(ext_target)
        self.linear_base_ctrl.update_stiffness()
        self.do_simulation(self.data.ctrl, self.frame_skip)

        # ── Record tip position AFTER physics step  (s_{t+1}) ────────────
        next_tip_pos = self.data.xpos[self.tip_body_id][:2].copy()
        self._prev_tip_pos = next_tip_pos   # store for reference if needed

        # ── Build next state, compute shaped reward ───────────────────────
        next_state = self.get_state()
        reward     = self.get_reward(next_state, action, prev_tip_pos, next_tip_pos)
        terminated = self.is_terminal(next_state)
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
        print(f"  Heuristic shaping: {'enabled' if self._heuristic else 'disabled'}")
        print(f"-------------------------------\n")