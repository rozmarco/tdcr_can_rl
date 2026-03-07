import re
import random
import sys
import importlib.util
from pathlib import Path
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

# Add tdcr_sim_mujoco directory to path
tdcr_sim_mujoco_path = Path(__file__).parent.parent.parent / "tdcr_sim_mujoco"
if str(tdcr_sim_mujoco_path) not in sys.path:
    sys.path.insert(0, str(tdcr_sim_mujoco_path))

# Pre-load tdcr_kinematics package into sys.modules so that relative imports work
tdcr_kinematics_path = tdcr_sim_mujoco_path / "src" / "tdcr_kinematics"
if "src.tdcr_kinematics" not in sys.modules:
    # Load the __init__ file
    init_path = tdcr_kinematics_path / "__init__.py"
    spec = importlib.util.spec_from_file_location("src.tdcr_kinematics", init_path)
    tdcr_kinematics_pkg = importlib.util.module_from_spec(spec)
    sys.modules["src.tdcr_kinematics"] = tdcr_kinematics_pkg
    spec.loader.exec_module(tdcr_kinematics_pkg)

# Now import linear base controller directly
linear_base_module_path = tdcr_sim_mujoco_path / "src" / "controllers" / "linear_base_controller.py"
spec = importlib.util.spec_from_file_location("linear_base_controller", linear_base_module_path)
linear_base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linear_base_module)
LinearBaseStiffnessController = linear_base_module.LinearBaseStiffnessController

# Now import TDCR joint controller (should work now that tdcr_kinematics is in sys.modules)
tdcr_joint_module_path = tdcr_sim_mujoco_path / "src" / "controllers" / "tdcr_joint_controller.py"
spec = importlib.util.spec_from_file_location("tdcr_joint_controller", tdcr_joint_module_path)
tdcr_joint_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tdcr_joint_module)
TDCRJointController = tdcr_joint_module.TDCRJointController


class CustomEnv(MujocoEnv):
    """
    A custom MuJoCo environment for robotic simulation using Gymnasium.

    This environment simulates a flexible continuum manipulator using a multi-body 
    approximation (links and joints). It is optimized for tasks involving obstacle 
    avoidance and goal reaching in 3D space.
    """
    def __init__(
        self, 
        scene_path: str,
        render_mode: str = "human",
        frame_skips: int = 50,
        timstep: float = 0.002,
        n_curv_bins: int = 10, 
        n_contact_bins: int = 10,
        **kwargs
    ):
        """
        Initializes the MuJoCo simulation environment.

        Args:
            scene_path: Absolute path to the XML model file.
            render_mode: Options include "human" for GUI, "rgb_array" for off-screen pixels, or None.
            frame_skip (int): The number of physics steps (at 500 Hz) to perform before 
                returning control to the agent. A value of 25 results in a 20 Hz control 
                cycle (50 steps * 0.002s = 0.1s (100ms) per action). Thus the agent is 
                running at a 100ms control period, meaning it "sees" the world and 
                updates its command only 10 times per second, while the physics engine 
                maintains stability by calculating the robot's motion every 2ms during 
                that interval.
            timestep (float): The physics integration discretization (in seconds). 
                0.002s (500 Hz) provides stable dynamics for tendon interactions.
            max_geom (int): The maximum number of geometric primitives (geoms) allowed 
                in the internal MuJoCo render buffer. Increasing this prevents 
                rendering artifacts in complex scenes with many objects.
            **kwargs: Additional keyword arguments passed to the MujocoEnv base class.
        """
        self.n_curv_bins = n_curv_bins
        self.n_contact_bins = n_contact_bins

        super().__init__(
            model_path=str(scene_path), 
            render_mode=render_mode,
            observation_space=self._build_observation_space(),
            frame_skip=frame_skips,
            max_geom=1000,
            **kwargs
        )

        self.goal_pos = np.array([0.5, 0.0], dtype=np.float32)
        self._cache_ids()
        self._print_init_state()
        self.base_pos = np.array([0.0, 0.0])
        self.max_reach = 0.5   # example — measure from model
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),   # bend_x, extension_delta (planar system)
            dtype=np.float32
        )

        # Initialize linear base controller
        self.linear_base_ctrl = LinearBaseStiffnessController(
            self.model, self.data, inside_stiffness=50.0
        )
        # Run initial stiffness update
        self.linear_base_ctrl.update_stiffness()

        # Initialize TDCR joint controller for Clark coordinate control
        self.tdcr_controller = TDCRJointController(
            self.model, self.data, clark_speed_scale=0.001, fps=100
        )

        self.model.opt.timestep = timstep # Set discretization to 2ms (500Hz).

    def _cache_ids(self):
        """Cache IDs so we don't string-search every step."""
        # Try different possible end-effector names
        possible_tip_names = ["tdcr_tip", "EE_pos", "ee_pos", "end_effector"]
        self.tip_body_id = None
        for name in possible_tip_names:
            try:
                self.tip_body_id = self.model.body(name).id
                break
            except KeyError:
                continue
        
        if self.tip_body_id is None:
            # If no tip found, use the last link
            self.tip_body_id = self.model.body("link_25").id if "link_25" in [self.model.body(i).name for i in range(self.model.nbody)] else self.model.body("link_0").id

        self.link_body_ids = [
            i for i in range(self.model.nbody)
            if "link" in self.model.body(i).name
        ]

        self.obstacle_body_ids = [
            i for i in range(self.model.nbody)
            if re.fullmatch(r"obstacle\d+", self.model.body(i).name)
        ]

        self.linear_jnt_ids = []
        for j in range(self.model.njnt):
            name = self.model.joint(j).name
            if name is not None and "linear_" in name:
                self.linear_jnt_ids.append(j)

        self.linear_actuator_ids = []
        for a in range(self.model.nu):
            name = self.model.actuator(a).name
            if name is not None and "linear_" in name:
                self.linear_actuator_ids.append(a)

        self.obstacle_body_name = [
            self.model.body(i).name for i in range(self.model.nbody)
            if re.fullmatch(r"obstacle\d+", self.model.body(i).name)
        ]

    def _build_observation_space(self):
        """
        Fixed-size observation regardless of:
        - number of links
        - number of contacts
        """
        return spaces.Dict({
            "tendon_length": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "extension": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),

            "curvature_hist": spaces.Box(
                0.0, np.inf, shape=(self.n_curv_bins,), dtype=np.float32
            ),

            "contact_hist": spaces.Box(
                0.0, np.inf, shape=(self.n_contact_bins,), dtype=np.float32
            ),

            "goal_rel_pos": spaces.Box(
                -np.inf, np.inf, shape=(2,), dtype=np.float32
            ),
        })

    def _compute_curvature_histogram(self):
        """
        Approximate curvature via relative orientations between links.
        """
        angles = []

        for i in range(len(self.link_body_ids) - 1):
            b0 = self.link_body_ids[i]
            b1 = self.link_body_ids[i + 1]

            x0 = self.data.xpos[b0]
            x1 = self.data.xpos[b1]

            dx = x1 - x0
            theta = np.arctan2(dx[1], dx[0])
            angles.append(theta)

        angles = np.array(angles)
        hist, _ = np.histogram(
            angles,
            bins=self.n_curv_bins,
            range=(-np.pi, np.pi),
            density=True,
        )

        return hist.astype(np.float32)
    
    def _compute_contact_histogram(self):
        """
        Histogram of contact locations along the backbone.
        """
        contact_s = []

        for i in range(self.data.ncon):
            con = self.data.contact[i]

            b1 = con.geom1
            b2 = con.geom2

            for geom_id in [b1, b2]:
                body_id = self.model.geom_bodyid[geom_id]
                if body_id in self.link_body_ids:
                    link_index = self.link_body_ids.index(body_id)
                    s = link_index / max(len(self.link_body_ids) - 1, 1)
                    contact_s.append(s)

        hist, _ = np.histogram(
            contact_s,
            bins=self.n_contact_bins,
            range=(0.0, 1.0),
        )

        return hist.astype(np.float32)
    
    def _get_extension(self):
        # Use linear base controller's slide position
        return self.linear_base_ctrl.get_slide_position()
    
    def _compute_obstacle_pos(self):
        tip_pos = self.data.xpos[self.tip_body_id][:2]
        obstacles_rel = []

        for name in self.obstacle_body_name:
            body = self.model.body(name)
            actual_pos = body.ipos[:2]
            rel_pos = actual_pos - tip_pos
            obstacles_rel.extend(rel_pos)

        return np.array(obstacles_rel)

    def _get_obstacle_radius(self):
        obstacles_radius = []

        for name in self.obstacle_body_name:
            body = self.model.body(name)
            g_id = self.model.body_geomadr[body.id]
            radius = self.model.geom_size[g_id][0]
            obstacles_radius.append(radius)

        return np.array(obstacles_radius)
    
    def _print_init_state(self):
        print("\n--- Environment Initialized ---")
        print(f"qpos shape:        {self.data.qpos.shape}")
        print(f"ctrl shape:        {self.data.ctrl.shape}")
        print(f"num links:         {len(self.link_body_ids)}")
        print(f"num obstacles:     {len(self.obstacle_body_ids)}")
        print(f"--------------------------------\n")

    def get_state(self):
        """
        Observation fed to the neural network.
        """
        tip_pos = self.data.xpos[self.tip_body_id][:2]

        # Tendons (actuation-relevant)
        tendon_length = self.data.ten_length[:3].copy()

        # Extension (average of linear joints)
        extension = np.array([self._get_extension()], dtype=np.float32)

        # Histograms
        curvature_hist = self._compute_curvature_histogram()
        contact_hist = self._compute_contact_histogram()

        # Goal position
        goal_rel = self.goal_pos - tip_pos

        # Obstacles
        obstacle_pos = self._compute_obstacle_pos()
        obstacle_radius = self._get_obstacle_radius()

        return {
            "tendon_length": tendon_length.astype(np.float32),
            "extension": extension,
            "curvature_hist": curvature_hist,
            "contact_hist": contact_hist,
            "goal_rel_pos": goal_rel.astype(np.float32),
            "obstacle_pos": obstacle_pos,
            "obstacle_radius": obstacle_radius,
        }

    def get_reward(self, obs, action):

        dist = np.linalg.norm(obs["goal_rel_pos"])

        goal_threshold = 0.02
        goal_bonus = 10.0 if dist < goal_threshold else 0.0

        distance_reward = -dist**2

        contact_bonus = 0.01 * np.sum(obs["contact_hist"])


        # Quadratic control regularization.
        # Encourages smooth tendon/extension adjustments and prevents
        # the policy from exploiting actuator saturation or producing
        # high-frequency oscillations.
        action_penalty = 0.001 * np.sum(action**2) 

        time_penalty = -0.01

        return (
            distance_reward
            + goal_bonus
            + contact_bonus
            - action_penalty
            + time_penalty
        )

    def is_terminal(self, obs):
        return np.linalg.norm(obs["goal_rel_pos"]) < 0.02

    def reset_model(self):


        qpos = self.init_qpos.copy()
        qvel = np.zeros_like(self.init_qvel)

        # Small randomization in tendon space
        noise_scale = 0.01
        qpos += np.random.uniform(-noise_scale, noise_scale, size=qpos.shape)

        self.set_state(qpos, qvel)

        # Randomize goal
        self.goal_pos = self._sample_valid_goal()

        return self.get_state()

    def step(self, action: np.ndarray):
        """
        Action:
        [bend_x, extension_delta] (planar system using Clark coordinates)
        """
        bend_x, extension_delta = action

        # Create command for TDCR controller (Clark coordinates)
        command = {
            "x": bend_x,  # Left/right bending (Clark x-coordinate)
            "y": 0.0,     # No vertical bending (planar system)
            "segment": 0, # Control first segment
            "reset_home": False,
            "linear_base_delta": extension_delta  # For compatibility, but we handle extension separately
        }

        # Get target tendon positions from TDCR controller
        tendon_targets = self.tdcr_controller.compute_target_qpos(command, self.data)

        # Apply tendon targets
        self.data.ctrl[self.tdcr_controller.tendon_actuator_ids] = tendon_targets[self.tdcr_controller.tendon_actuator_ids]

        # Handle extension via linear base controller
        lb_speed = 0.002  # Same as in teleop config
        if abs(extension_delta) > 0.01:
            new_target = self.linear_base_ctrl.get_slide_target() + extension_delta * lb_speed
            self.linear_base_ctrl.set_slide_target(new_target)
        self.linear_base_ctrl.update_stiffness()

        self.do_simulation(self.data.ctrl, self.frame_skip)
        
        next_state = self.get_state()
        reward = self.get_reward(next_state, action)
        terminated = self.is_terminal(next_state)
        truncated = False
        info = {}
        
        return next_state, reward, terminated, truncated, info
    
    def _sample_valid_goal(self):
        max_reach = self.max_reach   # define this once

        for _ in range(100):  # avoid infinite loop
            goal = np.array([
                random.uniform(0.3, 0.6),
                random.uniform(-0.2, 0.2),
            ])

            if not self._is_goal_valid(goal, max_reach):
                continue

            return goal

        raise RuntimeError("Failed to sample valid goal.")
    def _is_inside_obstacle(self, goal):
        for body_id in self.obstacle_body_ids:
            center = self.data.xpos[body_id][:2]
            geom_id = self.model.body_geomadr[body_id]
            radius = self.model.geom_size[geom_id][0]

            if np.linalg.norm(goal - center) < radius + 0.01:
                return True
        return False
    
    def _is_reachable(self, goal, max_reach):
        return np.linalg.norm(goal - self.base_pos) <= max_reach
    
    def _is_goal_valid(self, goal, max_reach):

        if self._is_inside_obstacle(goal):
            return False

        if not self._is_reachable(goal, max_reach):
            return False

        return True
