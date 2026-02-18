import numpy as np
import random

from gymnasium.envs.mujoco import MujocoEnv

class CustomEnv(MujocoEnv):
    """
    A custom MuJoCo environment for robotic simulation using Gymnasium.

    Args:
        scene_path (str or Path): Path to the MuJoCo XML configuration file.
        render_mode (str): The mode used for rendering. 
    """
    def __init__(
        self, 
        scene_path,
        render_mode:str="human"
    ):
        super().__init__(
            model_path=str(scene_path), 
            frame_skip=100,
            observation_space=None,
            render_mode=render_mode,
            max_geom=1000
        )
        self._print_init_state()

    def _print_init_state(self):
        # print(dir(self.data))

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

        # Body ID 1 | Name: target_pose
        # Body ID 3 | Name: obstacles
        # Body ID 4 | Name: obstacle0
        # Body ID 5 | Name: obstacle1
        # Body ID 6 | Name: obstacle2
        # Body ID 7 | Name: obstacle3
        # Body ID 8 | Name: obstacle4
        # Body ID 9 | Name: extensible_base
        # Body ID 10 | Name: tdcr_mount
        # Body ID 112 | Name: tdcr_tip

    def get_state(self):
        # Identify obstacles position relative to tdcr_tip
        effector_id = self.model.body('tdcr_tip').id
        effector_pos = self.data.site_xpos[effector_id]

        obstacle_names = ['obstacle0', 'obstacle1', 'obstacle2', 'obstacle3', 'obstacle4'] # TODO: Require more robust method
        obstacles_rel = []
        obstacles_radius = []

        for name in obstacle_names:
            body = self.model.body(name)

            actual_pos = body.pos
            rel_pos = actual_pos - effector_pos
            obstacles_rel.extend(rel_pos)

            g_id = self.model.body_geomadr[body.id]
            radius = self.model.geom_size[g_id][0]
            obstacles_radius.append(radius)

        # Identify the link's radius
        tdcr_radius = []
        for i in range(self.model.nbody): # TODO: Require more robust method
            body = self.model.body(i)
            if 'link' in body.name:
                g_id = self.model.body_geomadr[body.id]
                radius = self.model.geom_size[g_id][0]
                tdcr_radius.append(radius)

        # Change to numpy
        obstacles_rel = np.array(obstacles_rel)
        obstacles_radius = np.array(obstacles_radius)
        tdcr_radius = np.array(tdcr_radius)

        # TODO: Get the variable for extension, because currently may not be extending

        # Prep for output
        robot_state = np.concatenate([
            self.data.ten_length.flatten(),
            self.data.qfrc_constraint.flatten()  # Force applied on each joint
            # TODO: Eff to goal position
            # TODO: Eff to goal orientation
        ]).astype(np.float32)
        obstacle_state = np.concatenate([
            # TODO: Link pose vs Body Curvature (Assume constant-curvature, kappa)
            # self.data.qpos.flatten(),         # Body position
                                                # Body orientation
            tdcr_radius.ravel(),                # Body radius # TODO: If all radius are 0.006, then input scalar to reduce dim
            obstacles_rel.ravel(),              # Obstacle position relative to EEF
            obstacles_radius.ravel(),           # Obstacle radius
        ])

        return {
            "robot_state": robot_state,
            "obstacle_state": obstacle_state
        }
    
        # return {
        #     "robot": {
        #         "ten_length": self.data.ten_length.copy(),
        #         "qfrc": self.data.qfrc_constraint.copy(),
        #         "link_radii": tdcr_radius, # Array of size (num_links,)
        #         "link_poses": self.data.xpos[link_ids].copy(), # Get all link positions
        #     },
        #     "obstacles": {
        #         "rel_positions": obstacles_rel.reshape(-1, 3), # (N, 3)
        #         "radii": obstacles_radius, # (N,)
        #     },
        #     "goal": {
        #         "position": self.goal_pos - effector_pos,
        #         "orientation": 
        #     }
        # }
    
    def get_reward(self, state, action, next_state):
        return 0
    
    def is_terminal(self, goal_pose):
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
        self.set_state(self.init_qpos, self.init_qvel)

        return self.get_state()

    def step(self, action):
        state = self.get_state()

        # Step the physics
        self.do_simulation(action, self.frame_skip)
        
        next_state = self.get_state()
        reward = self.get_reward(state, action, next_state)
        terminated = self.is_terminal(next_state)
        truncated = False
        info = {}
        
        return next_state, reward, terminated, truncated, info