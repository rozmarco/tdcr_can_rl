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
# Keeps the physics from seeing violent step changes while letting the
# policy reason in absolute target space.
MAX_CLARK_STEP = 2.0   # Clark units per cycle  (~10% of full range)
MAX_EXT_STEP   = 0.01  # metres per cycle       (~2.7% of full range)


class CustomEnv(MujocoEnv):
    """
    Custom MuJoCo environment for a continuum manipulator (TDCR) with a
    linear sliding base, trained with SAC for goal-reaching tasks.

    Goal positions are sampled from a pre-validated workspace npz file
    (explored_configs_5cyl_labelled.npz).  Only configurations where
    contact_at_goal=False are used by default.

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
            **kwargs:            Passed through to MujocoEnv.
        """
        self.n_curv_bins   = n_curv_bins
        self.n_contact_bins = n_contact_bins

        # Load workspace before super().__init__ so obstacle count is known
        # when _build_observation_space() is called.
        self._load_workspace(workspace_npz, allow_contact_goals)

        # Temporarily set n_obstacles=0; _cache_ids() (called after super) sets the real value.
        # _build_observation_space() uses self.n_obstacles, so we need a placeholder.
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

        # Rebuild observation space now that n_obstacles is known from the model
        self.observation_space = self._build_observation_space()

        self.base_pos = np.array([0.0, 0.0])
        self.goal_pos = self._sample_goal_from_workspace()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Initialize controllers
        self.linear_base_ctrl = LinearBaseStiffnessController(
            self.model, self.data, inside_stiffness=50.0
        )
        self.linear_base_ctrl.update_stiffness()

        self.tdcr_controller = TDCRJointController(
            self.model, self.data, clark_speed_scale=0.001, fps=100
        )

        self.model.opt.timestep = timstep  # 2 ms / 500 Hz

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def _load_workspace(self, npz_path: str, allow_contact_goals: bool):
        """Load and filter the pre-validated configuration workspace.

        Sets:
            self._ws_tip_pos   (N, 2) float32  — XY tip positions in world frame
            self._ws_actuators (N, 2) float32  — [clark_x, slide_m]
        """
        ws   = np.load(npz_path)
        mask = np.ones(len(ws['tip_positions']), dtype=bool)
        #if not allow_contact_goals:
        #    mask &= ~ws['contact_at_goal']

        idx = np.where(mask)[0]
        self._ws_tip_pos   = ws['tip_positions'][idx, :2].astype(np.float32)
        self._ws_actuators = ws['actuator_configs'][idx].astype(np.float32)

        print(
            f"Workspace loaded: {len(idx)} configs "
            #f"({'incl.' if allow_contact_goals else 'excl.'} contact-at-goal). "
            f"Tip x=[{self._ws_tip_pos[:,0].min():.3f}, {self._ws_tip_pos[:,0].max():.3f}]  "
            f"y=[{self._ws_tip_pos[:,1].min():.3f}, {self._ws_tip_pos[:,1].max():.3f}]"
        )

    def _sample_goal_from_workspace(self) -> np.ndarray:
        idx = np.random.randint(len(self._ws_tip_pos))
        return self._ws_tip_pos[idx].copy()

    # ------------------------------------------------------------------
    # Initialisation helpers
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
        """
        Fixed-size observation space.

        Must stay in sync with get_state() and flatten_state().
        For 5 cylinders: 3+1+10+10+2+10+5 = 41  (== r_dim in train.yaml).
        """
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
            body  = self.model.body(name)
            g_id  = self.model.body_geomadr[body.id]
            radii.append(self.model.geom_size[g_id][0])
        return np.array(radii, dtype=np.float32)

    # ------------------------------------------------------------------
    # State / reward / terminal
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

    def get_reward(self, obs, action):
        dist           = np.linalg.norm(obs["goal_rel_pos"])
        goal_bonus     = 1000.0 if dist < 0.02 else 0.0
        dist_reward    = -(dist ** 2)
        contact_bonus  =  0.01   * np.sum(obs["contact_hist"])
        action_penalty = -0.0001 * np.sum(action ** 2)
        time_penalty   = -0.0001
        return dist_reward + goal_bonus + contact_bonus + action_penalty + time_penalty

    def is_terminal(self, obs):
        return np.linalg.norm(obs["goal_rel_pos"]) < 0.02

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _remap_action(self, action: np.ndarray):
        """Map [-1, 1] policy outputs to physical coordinates.

        Returns:
            clark_target : float in [-CLARK_MAX, +CLARK_MAX]
            ext_target   : float in [EXTENSION_MIN, EXTENSION_MAX]
        """
        clark_target = float(action[0]) * CLARK_MAX
        ext_target   = EXTENSION_MIN + (float(action[1]) + 1.0) * 0.5 * (EXTENSION_MAX - EXTENSION_MIN)
        return clark_target, ext_target

    def _apply_clark_target(self, clark_target: float):
        """Slew-rate-limited write of Clark X target to tendon actuators."""
        kinematics = self.tdcr_controller.kinematics
        seg_idx    = self.tdcr_controller.current_segment

        current_x = kinematics.goal_clark_coords[seg_idx * 2]
        delta     = np.clip(clark_target - current_x, -MAX_CLARK_STEP, MAX_CLARK_STEP)

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
        """Slew-rate-limited write of extension target to linear base."""
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
        return self.get_state()

    def step(self, action: np.ndarray):
        clark_target, ext_target = self._remap_action(action)
        self._apply_clark_target(clark_target)
        self._apply_extension_target(ext_target)
        self.linear_base_ctrl.update_stiffness()
        self.do_simulation(self.data.ctrl, self.frame_skip)

        next_state = self.get_state()
        reward     = self.get_reward(next_state, action)
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
        print(f"-------------------------------\n")