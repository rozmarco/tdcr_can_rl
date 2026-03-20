"""
CustomEnv — Planar TDCR Gymnasium environment.

Observation space: flat Box(33,) — compatible with the existing
flatten_state / policy / Q-network pipeline.

Flat observation layout (fixed order, 33 floats total):
  [0:2]   clark_state    — normalised [clark_x/20, slide_norm] ∈ [-1,1]
  [2:5]   tendon_length  — raw MuJoCo ten_length[:3]
  [5:15]  curvature_hist — 10-bin planar bending angle histogram
  [15:25] contact_hist   — 10-bin robot–obstacle contact histogram
  [25:27] goal_rel_pos   — goal_xy − tip_xy
  [27:29] tip_vel        — tip XY velocity
  [29:31] tip_orient     — [sin(θ), cos(θ)] of tip heading in XY plane
  [31:33] goal_rel_angle — [sin(Δθ), cos(Δθ)] where Δθ = goal_θ − tip_θ

Action space: Box(2,) ∈ [-1, 1]
  action[0]  Δ Clark x   (planar bending)
  action[1]  Δ slide     (linear base extension)

Both actions are accumulated into tracked state variables clamped to the
same limits used during BFS exploration:
  Clark x  ∈ [–20, +20]
  Slide    ∈ [0.10 m, 0.37 m]

Orientation is represented as sin/cos of the planar heading angle extracted
from the tip body's rotation matrix (xmat). This avoids the ±π discontinuity
of a raw angle and keeps all observation components bounded.

The goal heading is extracted from tip_poses saved by bfs_explore.py:
  rot = tip_poses[i].reshape(3, 3)   # row-major 3×3
  heading = arctan2(rot[1, 0], rot[0, 0])   # angle of local X-axis in world XY
"""

import re
import random
import numpy as np

from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv
import mujoco

from src.controllers.tdcr_joint_controller import TDCRJointController
from src.controllers.linear_base_controller import LinearBaseStiffnessController


# ---------------------------------------------------------------------------
# Workspace limits  —  must match BFS exploration parameters exactly
# ---------------------------------------------------------------------------

CLARK_MIN = -20.0
CLARK_MAX  =  20.0

SLIDE_MIN_M = 0.10
SLIDE_MAX_M = 0.37

# Physical step sizes per normalised action unit — match BFS values:
#   clark_speed_scale=10.0, fps=100, ticks_per_step=10
#   → Δclark per BFS step = 10.0 * (1/100) * 10 = 1.0
#   linear_base_speed=0.002 m/tick, ticks_per_step=10
#   → Δslide per BFS step = 0.002 * 10 = 0.02 m
CLARK_DELTA_SCALE = 1.0
SLIDE_DELTA_SCALE = 0.02

# Controller parameters — must match BFS / teleop config exactly
TENDON_DISTANCE_MM = 4.0
CLARK_SPEED_SCALE  = 10.0
ANGLE_OFFSET_RAD   = np.array([0.0])
CONTROLLER_FPS     = 100.0
LINEAR_BASE_SPEED  = 0.002
INSIDE_STIFFNESS   = 50.0
TICKS_PER_STEP     = 10

# Flat observation size — update r_dim in train.yaml to match (was 29, now 33)
OBS_DIM = 33

# Reward weight for orientation error relative to position error.
# A value of 0.3 means orientation contributes ~23% of the dense reward signal
# at maximum misalignment (1 - cos(π) = 2), which keeps position convergence
# as the primary driver while still incentivising correct tip orientation.
ORIENTATION_REWARD_WEIGHT = 0.3

# Angular threshold for goal success (radians). ~11 degrees.
ORIENTATION_GOAL_THRESHOLD = 0.2

# Penalty weight on action magnitude — discourages large commands.
ACTION_PENALTY_WEIGHT = 0.05

# Penalty weight on action *change* between steps — discourages rapid reversals
# that destabilise the physics (NaN/Inf in QACC). This is the most important
# stability penalty: it directly costs the agent for thrashing the robot.
SMOOTHNESS_PENALTY_WEIGHT = 0.1


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def heading_from_xmat(xmat: np.ndarray) -> float:
    """
    Extract the planar heading angle (radians) from a MuJoCo body rotation
    matrix stored as a flat 9-element row-major array (data.xmat[body_id]).

    The heading is the angle of the body's local X-axis projected onto the
    world XY plane:  θ = arctan2(R[1,0], R[0,0])
    """
    rot = xmat.reshape(3, 3)
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def heading_from_rotmat_flat(rot_flat: np.ndarray) -> float:
    """
    Same as heading_from_xmat but accepts the flattened rotation matrix
    stored in the BFS npz file (tip_poses column, 9 floats, row-major).
    """
    rot = rot_flat.reshape(3, 3)
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def sincos(angle: float) -> np.ndarray:
    """Return [sin(angle), cos(angle)] as float32."""
    return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)


def angle_diff(target: float, current: float) -> float:
    """Signed angular difference (target − current) wrapped to (−π, π]."""
    diff = target - current
    return float((diff + np.pi) % (2 * np.pi) - np.pi)


# ---------------------------------------------------------------------------
# Curriculum Goal Sampler
# ---------------------------------------------------------------------------

class CurriculumGoalSampler:
    """
    Samples BFS-verified tip positions (and headings) with an expanding
    radius curriculum.

    goal_pool  : (N, 2)  XY positions
    goal_angles: (N,)    planar heading angles in radians

    Starts within `initial_radius` of home. Every `expand_every` episodes,
    if rolling success rate >= `expand_threshold`, radius grows by `radius_step`.
    """

    def __init__(
        self,
        goal_pool: np.ndarray,
        goal_angles: np.ndarray,
        home_pos: np.ndarray,
        initial_radius: float = 0.05,
        radius_step: float    = 0.03,
        max_radius: float     = 0.60,
        expand_every: int     = 200,
        expand_threshold: float = 0.50,
    ):
        self.goal_pool        = goal_pool          # (N, 2)
        self.goal_angles      = goal_angles        # (N,)
        self.home_pos         = home_pos
        self.radius           = initial_radius
        self.radius_step      = radius_step
        self.max_radius       = max_radius
        self.expand_every     = expand_every
        self.expand_threshold = expand_threshold

        self._episode          = 0
        self._recent_successes = []
        self._dists            = np.linalg.norm(goal_pool - home_pos, axis=1)

    def update_home(self, home_pos: np.ndarray):
        self.home_pos = home_pos.copy()
        self._dists   = np.linalg.norm(self.goal_pool - home_pos, axis=1)

    def sample(self) -> tuple:
        """Return (goal_xy, goal_heading) for a randomly selected goal."""
        mask       = self._dists <= self.radius
        candidates = np.where(mask)[0]
        if len(candidates) == 0:
            idx = int(np.argmin(self._dists))
        else:
            idx = int(candidates[np.random.randint(len(candidates))])
        return self.goal_pool[idx].copy(), float(self.goal_angles[idx])

    def report_episode(self, success: bool):
        self._recent_successes.append(float(success))
        self._episode += 1
        if len(self._recent_successes) > self.expand_every:
            self._recent_successes.pop(0)

        if self._episode % self.expand_every == 0 and self.radius < self.max_radius:
            rate = float(np.mean(self._recent_successes))
            if rate >= self.expand_threshold:
                old         = self.radius
                self.radius = min(self.radius + self.radius_step, self.max_radius)
                print(
                    f"[Curriculum] ep={self._episode:>6,}  success={rate:.2f}  "
                    f"radius {old:.3f} → {self.radius:.3f} m  "
                    f"pool={self.pool_size:>4,} goals"
                )
            else:
                print(
                    f"[Curriculum] ep={self._episode:>6,}  success={rate:.2f}  "
                    f"radius held at {self.radius:.3f} m  "
                    f"(need {self.expand_threshold:.2f})"
                )

    @property
    def pool_size(self) -> int:
        return int(np.sum(self._dists <= self.radius))

    @property
    def info(self) -> dict:
        return {
            "curriculum_radius":    self.radius,
            "curriculum_episode":   self._episode,
            "curriculum_pool_size": self.pool_size,
        }


# ---------------------------------------------------------------------------
# Custom MuJoCo Environment
# ---------------------------------------------------------------------------

class CustomEnv(MujocoEnv):
    """
    Planar TDCR reaching environment with flat Box observation space.

    OBS_DIM is now 33 (was 29). Update r_dim in train.yaml accordingly.

    The two new observation components are:
      tip_orient    [29:31]  — current tip heading as [sin, cos]
      goal_rel_angle [31:33] — heading error (goal − tip) as [sin, cos]

    The goal is now a (position, heading) pair. Terminal success requires
    both position AND orientation to be within threshold.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Fixed observation layout — positions within the flat vector
    OBS_LAYOUT = {
        "clark_state":     (0,  2),
        "tendon_length":   (2,  5),
        "curvature_hist":  (5,  15),
        "contact_hist":    (15, 25),
        "goal_rel_pos":    (25, 27),
        "tip_vel":         (27, 29),
        "tip_orient":      (29, 31),   # NEW: [sin(θ_tip), cos(θ_tip)]
        "goal_rel_angle":  (31, 33),   # NEW: [sin(Δθ), cos(Δθ)]
    }

    def __init__(
        self,
        scene_path: str,
        render_mode: str    = "human",
        frame_skips: int    = 50,
        timestep: float     = 0.002,
        n_curv_bins: int    = 10,
        n_contact_bins: int = 10,
        max_episode_steps: int = 500,
        # BFS goal curriculum
        goal_data_path: str     = None,
        initial_radius: float   = 0.05,
        radius_step: float      = 0.03,
        expand_every: int       = 200,
        expand_threshold: float = 0.50,
        **kwargs,
    ):
        """
        Args:
            scene_path:          Absolute path to the MuJoCo XML file.
            render_mode:         "human" | "rgb_array" | None.
            frame_skips:         Physics steps per agent action (50 → 10 Hz).
            timestep:            Physics integration step in seconds.
            n_curv_bins:         Curvature histogram bins (must equal 10 to match OBS_DIM).
            n_contact_bins:      Contact histogram bins  (must equal 10 to match OBS_DIM).
            max_episode_steps:   Hard truncation limit.
            goal_data_path:      Path to BFS explored_configs.npz. If None,
                                 falls back to rejection sampling (no orientation goal).
            initial_radius:      Starting curriculum radius (m).
            radius_step:         Curriculum expansion per success gate (m).
            expand_every:        Episodes between curriculum checks.
            expand_threshold:    Rolling success rate required to expand.
        """
        assert n_curv_bins == 10 and n_contact_bins == 10, \
            "Bin counts must be 10 to match the fixed OBS_DIM=33. " \
            "If you change them, update OBS_DIM and OBS_LAYOUT accordingly."

        self.n_curv_bins       = n_curv_bins
        self.n_contact_bins    = n_contact_bins
        self.max_episode_steps = max_episode_steps
        self._step_count       = 0
        self.goal_threshold    = 0.02   # 2 cm position threshold

        # Tracked actuator state — clamped to workspace limits every step
        self.clark_x   = 0.0
        self.slide_pos = SLIDE_MIN_M

        super().__init__(
            model_path=str(scene_path),
            render_mode=render_mode,
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
            ),
            frame_skip=frame_skips,
            max_geom=1000,
            **kwargs,
        )

        self.model.opt.timestep = timestep
        self._cache_ids()
        self._init_controllers()

        # 2-DOF action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Goal: XY position + heading angle
        self.goal_pos     = np.zeros(2, dtype=np.float32)
        self.goal_heading = 0.0   # radians

        self.goal_sampler  = None
        self._goal_pool    = None
        self._goal_angles  = None
        self._home_pos_set = False

        if goal_data_path is not None:
            self._load_bfs_goals(
                goal_data_path, initial_radius, radius_step,
                expand_every, expand_threshold,
            )

        self._last_terminated  = False
        self._last_pos_reached = False   # position-only flag for curriculum
        self._prev_action      = np.zeros(2, dtype=np.float32)  # for smoothness penalty

    # -----------------------------------------------------------------------
    # Controller initialisation
    # -----------------------------------------------------------------------

    def _init_controllers(self):
        """Initialise controllers with identical params to BFS exploration."""
        self._joint_ctrl = TDCRJointController(
            self.model,
            data=self.data,
            tendon_distance_mm=TENDON_DISTANCE_MM,
            angle_offset_rad_ccw=ANGLE_OFFSET_RAD,
            clark_speed_scale=CLARK_SPEED_SCALE,
            fps=CONTROLLER_FPS,
            apply_extensibility_constraint=True,
        )

        lb_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "linear_base_slide"
        )
        if lb_jnt >= 0:
            self._linear_base_ctrl = LinearBaseStiffnessController(
                self.model, self.data, inside_stiffness=INSIDE_STIFFNESS
            )
            self._linear_base_ctrl.update_stiffness()
        else:
            self._linear_base_ctrl = None
            print("[Warning] No linear_base_slide joint found.")

    # -----------------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------------

    def _load_bfs_goals(
        self, path, initial_radius, radius_step, expand_every, expand_threshold
    ):
        """
        Load BFS goal pool from npz.

        Extracts both XY positions and planar heading angles from the saved
        tip_poses (N, 9) rotation matrices. Each rotation matrix row is:
            [R00 R01 R02  R10 R11 R12  R20 R21 R22]
        The heading is arctan2(R10, R00) — angle of the local X-axis in world XY.
        """
        bfs = np.load(path)

        self._goal_pool = bfs["tip_positions"].astype(np.float32)[:, :2]  # (N, 2)

        # Extract planar heading from each saved rotation matrix
        if "tip_poses" in bfs:
            rot_mats = bfs["tip_poses"].astype(np.float32)          # (N, 9)
            self._goal_angles = np.arctan2(
                rot_mats[:, 3],   # R[1,0]  (row-major: index 3)
                rot_mats[:, 0],   # R[0,0]  (row-major: index 0)
            ).astype(np.float32)                                     # (N,)
        else:
            print("[Warning] tip_poses not found in BFS npz — goal headings set to 0.")
            self._goal_angles = np.zeros(len(self._goal_pool), dtype=np.float32)

        dists       = np.linalg.norm(self._goal_pool, axis=1)
        n_top       = max(1, len(self._goal_pool) // 100)
        approx_home = self._goal_pool[np.argsort(dists)[:n_top]].mean(axis=0)

        self.goal_sampler = CurriculumGoalSampler(
            goal_pool        = self._goal_pool,
            goal_angles      = self._goal_angles,
            home_pos         = approx_home,
            initial_radius   = initial_radius,
            radius_step      = radius_step,
            max_radius       = SLIDE_MAX_M + 0.05,
            expand_every     = expand_every,
            expand_threshold = expand_threshold,
        )
        print(
            f"[BFS] Loaded {len(self._goal_pool):,} goals from {path}\n"
            f"[BFS] Curriculum radius={initial_radius:.3f} m  "
            f"({self.goal_sampler.pool_size} goals in range)\n"
            f"[BFS] Heading range: [{self._goal_angles.min():.2f}, "
            f"{self._goal_angles.max():.2f}] rad"
        )

    def _cache_ids(self):
        # End-effector
        self._tip_body_name = None
        for name in ("EE_pos", "tdcr_tip"):
            try:
                self.tip_body_id    = self.model.body(name).id
                self._tip_body_name = name
                break
            except Exception:
                pass
        assert self._tip_body_name is not None, \
            "Could not find EE_pos or tdcr_tip in model."

        # Links (sorted numerically)
        self.link_body_ids = sorted(
            [
                i for i in range(self.model.nbody)
                if re.fullmatch(r"link_\d+", self.model.body(i).name)
            ],
            key=lambda i: int(self.model.body(i).name.split("_")[1]),
        )

        # Obstacles
        self.obstacle_body_ids   = [
            i for i in range(self.model.nbody)
            if re.fullmatch(r"obstacle\d+", self.model.body(i).name)
        ]
        self.obstacle_body_names = [
            self.model.body(i).name for i in self.obstacle_body_ids
        ]

    # -----------------------------------------------------------------------
    # Observation components
    # -----------------------------------------------------------------------

    def _get_tip_pos(self) -> np.ndarray:
        return self.data.xpos[self.tip_body_id][:2].copy()

    def _get_tip_vel(self) -> np.ndarray:
        return self.data.subtree_linvel[self.tip_body_id][:2].copy()

    def _get_tip_heading(self) -> float:
        """Planar heading angle of the tip body in radians."""
        return heading_from_xmat(self.data.xmat[self.tip_body_id])

    def _get_clark_state(self) -> np.ndarray:
        """Normalised [clark_x, slide] both remapped to [-1, 1]."""
        clark_norm = self.clark_x / CLARK_MAX
        slide_norm = (self.slide_pos - SLIDE_MIN_M) / (SLIDE_MAX_M - SLIDE_MIN_M) * 2.0 - 1.0
        return np.array([clark_norm, slide_norm], dtype=np.float32)

    def _compute_curvature_histogram(self) -> np.ndarray:
        angles = [
            np.arctan2(
                (self.data.xpos[self.link_body_ids[i + 1]]
                 - self.data.xpos[self.link_body_ids[i]])[1],
                (self.data.xpos[self.link_body_ids[i + 1]]
                 - self.data.xpos[self.link_body_ids[i]])[0],
            )
            for i in range(len(self.link_body_ids) - 1)
        ]
        hist, _ = np.histogram(
            np.array(angles, dtype=np.float32),
            bins=self.n_curv_bins,
            range=(-np.pi, np.pi),
            density=True,
        )
        return np.nan_to_num(hist, nan=0.0).astype(np.float32)

    def _compute_contact_histogram(self) -> np.ndarray:
        contact_s       = []
        link_id_set     = set(self.link_body_ids)
        obstacle_id_set = set(self.obstacle_body_ids)

        for i in range(self.data.ncon):
            con    = self.data.contact[i]
            bodies = {self.model.geom_bodyid[con.geom1],
                      self.model.geom_bodyid[con.geom2]}
            if not (bodies & link_id_set) or not (bodies & obstacle_id_set):
                continue
            for geom_id in [con.geom1, con.geom2]:
                body_id = self.model.geom_bodyid[geom_id]
                if body_id in link_id_set:
                    idx = self.link_body_ids.index(body_id)
                    contact_s.append(idx / max(len(self.link_body_ids) - 1, 1))

        hist, _ = np.histogram(contact_s, bins=self.n_contact_bins, range=(0.0, 1.0))
        return hist.astype(np.float32)

    # -----------------------------------------------------------------------
    # State — returns flat np.ndarray(33,)
    # -----------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """
        Returns a flat float32 array of shape (33,).

        Layout is fixed and documented in OBS_LAYOUT / the module docstring.
        """
        tip_pos     = self._get_tip_pos()
        tip_heading = self._get_tip_heading()
        delta_theta = angle_diff(self.goal_heading, tip_heading)

        obs = np.concatenate([
            self._get_clark_state(),                                        # [0:2]
            self.data.ten_length[:3].copy().astype(np.float32),             # [2:5]
            self._compute_curvature_histogram(),                            # [5:15]
            self._compute_contact_histogram(),                              # [15:25]
            (self.goal_pos - tip_pos).astype(np.float32),                   # [25:27]
            self._get_tip_vel().astype(np.float32),                         # [27:29]
            sincos(tip_heading),                                            # [29:31]
            sincos(delta_theta),                                            # [31:33]
        ], dtype=np.float32)

        return obs

    # -----------------------------------------------------------------------
    # Convenience: extract a named component from a flat obs vector
    # -----------------------------------------------------------------------

    @staticmethod
    def extract(obs: np.ndarray, key: str) -> np.ndarray:
        """
        Extract a named component from a flat observation vector.

        Usage:
            goal_vec = CustomEnv.extract(obs, "goal_rel_pos")
            orient   = CustomEnv.extract(obs, "tip_orient")
        """
        start, end = CustomEnv.OBS_LAYOUT[key]
        return obs[..., start:end]

    # -----------------------------------------------------------------------
    # Reward
    # -----------------------------------------------------------------------

    def get_reward(self, obs: np.ndarray, action: np.ndarray, prev_action: np.ndarray) -> float:
        # --- Position error ---
        goal_rel = self.extract(obs, "goal_rel_pos")
        dist     = float(np.linalg.norm(goal_rel))

        # --- Orientation error ---
        # goal_rel_angle = [sin(Δθ), cos(Δθ)].  cos(Δθ) ∈ [-1, 1].
        # orientation_error = 1 - cos(Δθ) ∈ [0, 2], zero when perfectly aligned.
        goal_rel_angle    = self.extract(obs, "goal_rel_angle")
        cos_delta         = float(goal_rel_angle[1])          # index 1 = cos
        orientation_error = 1.0 - cos_delta                   # ∈ [0, 2]

        # --- Dense shaping (primary signal) ---
        # Position drives the main signal; orientation is weighted at 0.3.
        # At max misalignment (π rad) orientation_error = 2, contributing
        # 0.3 * 2 = 0.6 to the penalty — meaningful but not position-dominating.
        reward = -(dist + ORIENTATION_REWARD_WEIGHT * orientation_error)

        # --- Success bonus — requires BOTH position and orientation ---
        if dist < self.goal_threshold and orientation_error < (1.0 - np.cos(ORIENTATION_GOAL_THRESHOLD)):
            reward += 50.0

        # --- Action magnitude penalty (discourages large commands) ---
        reward -= ACTION_PENALTY_WEIGHT * float(np.sum(action ** 2))

        # --- Smoothness penalty (discourages rapid reversals that cause NaN/Inf) ---
        # Penalises the change in action between consecutive steps. At max reversal
        # ([1,1] → [-1,-1]) this contributes -0.1 * 8 = -0.8, making thrashing
        # meaningfully costly relative to the dense position reward.
        action_delta = action - prev_action
        reward -= SMOOTHNESS_PENALTY_WEIGHT * float(np.sum(action_delta ** 2))

        # --- Small time penalty ---
        reward -= 0.01

        return reward

    # -----------------------------------------------------------------------
    # Terminal / truncation
    # -----------------------------------------------------------------------

    def is_terminal(self, obs: np.ndarray) -> bool:
        pos_ok = float(np.linalg.norm(self.extract(obs, "goal_rel_pos"))) < self.goal_threshold

        # cos(Δθ) is the second element of goal_rel_angle
        cos_delta   = float(self.extract(obs, "goal_rel_angle")[1])
        orient_ok   = cos_delta > np.cos(ORIENTATION_GOAL_THRESHOLD)

        return pos_ok and orient_ok

    def is_truncated(self) -> bool:
        return self._step_count >= self.max_episode_steps

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset_model(self) -> np.ndarray:
        # Report episode outcome to curriculum — uses position-only success
        # so the curriculum expands based on reaching the XY goal, independently
        # of whether orientation was also matched. This prevents the curriculum
        # from stalling because the joint pos+orient task is too hard early on.
        if self.goal_sampler is not None and self._step_count > 0:
            self.goal_sampler.report_episode(self._last_pos_reached)

        # Reset physics
        self.set_state(self.init_qpos.copy(), np.zeros_like(self.init_qvel))
        self.data.ctrl[:] = 0.0

        # Reset controllers to home
        self._joint_ctrl.reset_to_home()
        if self._linear_base_ctrl is not None:
            self._linear_base_ctrl.set_slide_target(SLIDE_MIN_M)
            self._linear_base_ctrl.update_stiffness()
            self._linear_base_ctrl.apply_ctrl()

        # Reset tracked state
        self.clark_x   = 0.0
        self.slide_pos = SLIDE_MIN_M

        # Settle physics (mirrors BFS SETTLE_STEPS=200)
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        # Measure real home position on first reset
        if self.goal_sampler is not None and not self._home_pos_set:
            actual_home = self._get_tip_pos()
            self.goal_sampler.update_home(actual_home)
            self._home_pos_set = True
            print(f"[BFS] Home tip position: {actual_home}")

        self.goal_pos, self.goal_heading = self._sample_valid_goal()
        self._step_count           = 0
        self._last_terminated      = False
        self._last_pos_reached     = False   # position-only flag for curriculum
        self._prev_action          = np.zeros(2, dtype=np.float32)

        return self.get_state()

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(self, action: np.ndarray):
        """
        Apply 2-DOF action through the same controller pipeline as BFS.

        Runs TICKS_PER_STEP=10 controller ticks per RL step, each calling
        joint_ctrl.compute_target_qpos() and linear_base_ctrl.update_stiffness(),
        mirroring bfs_explore.apply_config() exactly.
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Accumulate and clamp actuator state
        new_clark = float(np.clip(
            self.clark_x + action[0] * CLARK_DELTA_SCALE, CLARK_MIN, CLARK_MAX
        ))
        new_slide = float(np.clip(
            self.slide_pos + action[1] * SLIDE_DELTA_SCALE, SLIDE_MIN_M, SLIDE_MAX_M
        ))

        # Convert total delta into per-tick normalised command for joint controller
        clark_delta_total = new_clark - self.clark_x
        per_tick_clark    = clark_delta_total / TICKS_PER_STEP
        x_cmd = float(np.clip(
            per_tick_clark / (CLARK_SPEED_SCALE * (1.0 / CONTROLLER_FPS)),
            -1.0, 1.0,
        ))

        slide_delta_total = new_slide - self.slide_pos
        per_tick_slide    = slide_delta_total / TICKS_PER_STEP

        slide_target = (
            self._linear_base_ctrl.get_slide_target()
            if self._linear_base_ctrl else self.slide_pos
        )

        # Run controller ticks
        for _ in range(TICKS_PER_STEP):
            command = {
                "x":                 x_cmd,
                "y":                 0.0,
                "segment":           0,
                "reset_home":        False,
                "linear_base_delta": (1.0 if per_tick_slide > 0
                                      else -1.0 if per_tick_slide < 0
                                      else 0.0),
            }

            target_qpos = self._joint_ctrl.compute_target_qpos(command, self.data)
            self.data.ctrl[self._joint_ctrl.tendon_actuator_ids] = (
                target_qpos[self._joint_ctrl.tendon_actuator_ids]
            )

            if self._linear_base_ctrl is not None:
                if abs(per_tick_slide) > 1e-9:
                    slide_target = float(np.clip(
                        slide_target + per_tick_slide, SLIDE_MIN_M, SLIDE_MAX_M
                    ))
                    self._linear_base_ctrl.set_slide_target(slide_target)
                self._linear_base_ctrl.update_stiffness()
                self._linear_base_ctrl.apply_ctrl()

            mujoco.mj_step(self.model, self.data)

        self.clark_x   = new_clark
        self.slide_pos = new_slide
        self._step_count += 1

        next_state             = self.get_state()
        reward                 = self.get_reward(next_state, action, self._prev_action)
        terminated             = self.is_terminal(next_state)
        truncated              = self.is_truncated()
        self._last_terminated  = terminated
        self._prev_action      = action.copy()
        # Position-only success — used by curriculum so it can expand even when
        # orientation is not yet matched (prevents curriculum from never advancing)
        self._last_pos_reached = (
            float(np.linalg.norm(self.extract(next_state, "goal_rel_pos"))) < self.goal_threshold
        )

        info = {
            "step":              self._step_count,
            "dist_to_goal":      float(np.linalg.norm(self.extract(next_state, "goal_rel_pos"))),
            "orient_error_deg":  float(np.degrees(np.arccos(np.clip(
                                     self.extract(next_state, "goal_rel_angle")[1], -1.0, 1.0
                                 )))),
            "clark_x":           self.clark_x,
            "slide_pos":         self.slide_pos,
        }
        if self.goal_sampler is not None:
            info.update(self.goal_sampler.info)

        return next_state, reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Goal sampling
    # -----------------------------------------------------------------------

    def _sample_valid_goal(self) -> tuple:
        """Return (goal_xy, goal_heading). Falls back to rejection sampling if no BFS pool."""
        if self.goal_sampler is not None:
            return self.goal_sampler.sample()   # (np.ndarray(2,), float)

        # Fallback: rejection sampling (no orientation from BFS — heading fixed to 0)
        for _ in range(500):
            goal = np.array([
                random.uniform(0.05, SLIDE_MAX_M + 0.05),
                random.uniform(-0.25, 0.25),
            ], dtype=np.float32)
            if self._is_goal_valid(goal):
                return goal, 0.0

        return np.array([0.15, 0.0], dtype=np.float32), 0.0

    def _is_inside_obstacle(self, goal: np.ndarray) -> bool:
        for bid in self.obstacle_body_ids:
            center = self.data.xpos[bid][:2]
            radius = float(self.model.geom_size[self.model.body_geomadr[bid], 0])
            if np.linalg.norm(goal - center) < radius + 0.01:
                return True
        return False

    def _is_reachable(self, goal: np.ndarray) -> bool:
        return float(np.linalg.norm(goal - self._get_tip_pos())) <= SLIDE_MAX_M + 0.05

    def _is_goal_valid(self, goal: np.ndarray) -> bool:
        return not self._is_inside_obstacle(goal) and self._is_reachable(goal)

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def print_init_state(self):
        print("\n─── Environment Initialised ─────────────────────────────────")
        print(f"  Tip body            : {self._tip_body_name} (id={self.tip_body_id})")
        print(f"  Links               : {len(self.link_body_ids)}")
        print(f"  Obstacles           : {len(self.obstacle_body_ids)}  {self.obstacle_body_names}")
        print(f"  Obs space           : Box({OBS_DIM},)  flat float32")
        print(f"  Action space        : {self.action_space}")
        print(f"  Clark range         : [{CLARK_MIN}, {CLARK_MAX}]")
        print(f"  Slide range         : [{SLIDE_MIN_M} m, {SLIDE_MAX_M} m]")
        print(f"  Max episode steps   : {self.max_episode_steps}")
        print(f"  Orient weight       : {ORIENTATION_REWARD_WEIGHT}")
        print(f"  Orient threshold    : {np.degrees(ORIENTATION_GOAL_THRESHOLD):.1f} deg")
        if self.goal_sampler is not None:
            print(f"  BFS goal pool       : {len(self._goal_pool):,} positions")
            print(f"  Curriculum radius   : {self.goal_sampler.radius:.3f} m")
            print(f"  Goals in radius     : {self.goal_sampler.pool_size}")
        else:
            print(f"  Goal sampling       : rejection sampling (no BFS pool)")
        print("─────────────────────────────────────────────────────────────\n")