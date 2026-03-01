import random
import numpy as np
import mujoco

from typing import Dict, Tuple, Any

from gymnasium.envs.mujoco import MujocoEnv


# Custom Aliases
State = Dict[str, Dict[str, Any]]
NextState = Dict[str, Dict[str, Any]]
StepResults = Tuple[NextState, float, bool, bool, Dict[str, Any]]


class CustomEnv(MujocoEnv):
    """
    A custom MuJoCo environment for robotic simulation using Gymnasium.

    This environment simulates a flexible continuum manipulator using a multi-body 
    approximation (links and joints). It is optimized for tasks involving obstacle 
    avoidance and goal reaching in 3D space.
    """
    def __init__(
        self, 
        scene_path:  str,
        render_mode: str = "human",
        frame_skips: int = 50,
        timstep: float = 0.002,
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
        super().__init__(
            model_path = str(scene_path), 
            render_mode = render_mode,
            observation_space = None,
            frame_skip = frame_skips,
            max_geom = 1000,
            **kwargs
        )

        self.model.opt.timestep = timstep # Set discretization to 2ms (500Hz).
        
        self._print_init_state()

    def _print_init_state(self):
        # Generalized State
        print(f"Gen Positions (qpos):      {self.data.qpos.shape}")
        print(f"Actuators outputs:         {self.data.ctrl}")
        #  [Forward/Backward, Rightward Curvature, Downward Curvature, Leftward Curvature, Topward Curvative]

        # Cartesian State (World Space)
        print(f"Body Positions (xpos):       {self.data.xpos.shape}")
        print(f"Body Rotations (xmat):       {self.data.xmat.shape}")
        print(f"Body Orientation (xquat):    {self.data.xquat.shape}")
        print(f"Geom Positions (geom_xpos):  {self.data.geom_xpos.shape}")

        # Contacts
        print(f"Number of Contacts (ncon):   {self.data.ncon}")

        # Tendons
        print(f"Tendon Length:               {self.data.ten_length}")
        print(f"{'------------------------------':^40}\n")

        # for i in range(self.model.nbody):
        #     name = self.model.body(i).name
        #     print(f"Body ID {i} | Name: {name}")

    def get_state(self) -> State:
        # Identify obstacles position relative to tdcr_tip
        effector_id = self.model.body('tdcr_tip').id
        effector_pos = self.data.site_xpos[effector_id]

        obstacle_names = ['obstacle0', 'obstacle1', 'obstacle2', 'obstacle3', 'obstacle4'] # TODO: Require more robust method of getting obstacle names in world
        obstacles_rel = []
        obstacles_radius = []

        # Get the obstacle's position relative to the EFF and radius
        for name in obstacle_names:
            body = self.model.body(name)

            actual_pos = body.ipos
            rel_pos = actual_pos - effector_pos
            obstacles_rel.extend(rel_pos)

            # TODO: Need a way to verify the obstacles_rel calculations are correct

            g_id = self.model.body_geomadr[body.id]
            radius = self.model.geom_size[g_id][0]
            obstacles_radius.append(radius)

        # Get the link's radius
        tdcr_radius = []
        for i in range(self.model.nbody): # TODO: Require more robust method
            body = self.model.body(i)
            if 'link' in body.name:
                g_id = self.model.body_geomadr[body.id]
                radius = self.model.geom_size[g_id][0]
                tdcr_radius.append(radius)

        tdcr_links = []
        # Get the link poses
        for i in range(self.model.nbody):
            body = self.model.body(i)
            if ('link' in body.name) and (body.name is not 'link0'): # link0 is a fixed link, so does not move.
                g_id = self.model.body_geomadr[body.id]
                # TODO: World-coordinate-independent link state

        # Change to numpy
        obstacles_rel = np.array(obstacles_rel)
        obstacles_radius = np.array(obstacles_radius)
        tdcr_radius = np.array(tdcr_radius)
    
        return {
            "robot": {
                "ten_length": self.data.ten_length.copy(),
                "qfrc": self.data.qfrc_constraint.copy(),
                "link_radii": tdcr_radius,                   # TODO: Lots of redundant info here, but radius can be important. 
                                                             #       If all radius is 0.006, then input scalar to reduce dim.
                                                             # TODO: Link pose vs Body Curvature. Does constant curvature give more useful info than link pose?
            },
            "obstacles": {
                "rel_positions": obstacles_rel.reshape(-1, 3), # (N, 3)
                "radii": obstacles_radius.reshape(-1, 1),      # (N, 1)
            },
            "goal": {
                "position_x": 0.0,         # TODO: Eff to goal position (relative)
                "position_y": 0.0,
                "orientation": 0.0         # TODO: Eff to goal orientation (relative)
            }
        }
    
    def get_reward(
        self, 
        state: State,
        action: np.ndarray,
        next_state: NextState
    ) -> float:
        # TODO: Penalty for 'each second not at goal' set to 0.001
        return 0
    
    def is_terminal(
        self, 
        state: State,
        goal_pose
    ) -> bool:
        # TODO: Check if state is terminal here
        return False

    def reset_model(self): 
        case = random.choice([0, 1])

        if case == 0:
            # TODO: Randomize initial pose and end pose
            pass
        
        else:
            # TODO: Start from current pose and set random end pose
            pass
        
        # TODO: How can we guarantee a solution for random end pose

        # Resets the simulation state
        self.set_state(self.init_qpos, self.init_qvel) # TODO: Randomize initial qpos, qvel

        return self.get_state()

    def step(
        self, 
        action: np.ndarray
    ) -> StepResults:
        state = self.get_state()

        # Bypass do_simulation and step mujoco sim
        self._step_mujoco_simulation(action, self.frame_skip)
        
        next_state = self.get_state()
        reward = self.get_reward(state, action, next_state)
        terminated = self.is_terminal(next_state, None)
        truncated = False
        info = {}
        
        return next_state, reward, terminated, truncated, info
    
    def _step_mujoco_simulation(self, ctrl, n_frames):
        """
        Step over the MuJoCo simulation.
        """
        # Identify all actuator indices belonging to Group 1 and Group 2 as defined in the XML,
        # and assign control to motors
        group1_indices = [i for i in range(self.model.nu) if self.model.actuator_group[i] == 1]
        group2_indices = [i for i in range(self.model.nu) if self.model.actuator_group[i] == 2]

        self.data.ctrl[group1_indices] = ctrl[0]
        self.data.ctrl[group2_indices] = ctrl[1]

        # Identify all tendons defined in the XML, then assign controls to actuator
        ten0_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'seg_0_ten_0')
        ten1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'seg_0_ten_1')

        self.data.ctrl[ten0_id] = ctrl[2]
        self.data.ctrl[ten1_id] = ctrl[3]

        # Step the simulator
        mujoco.mj_step(self.model, self.data, nstep=n_frames)

        mujoco.mj_rnePostConstraint(self.model, self.data)