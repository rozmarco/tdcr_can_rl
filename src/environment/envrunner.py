import time
import uuid

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import ray

import mujoco
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

from src.environment.env import CustomEnv
from src.buffers.buffer import ReplayBuffer
from src.utils.data_preprocessor import flatten_state


class EnvRunner:
    """
    Manages the execution and lifecycle of episodes within an environment
    using a specific policy.
    """
    def __init__(
        self,
        name: str,
        is_train: bool,
        scene_path: str,
        workspace_npz: str,
        policy_net: torch.nn.Module,
        horizon: int = 1,
        num_episodes: int = 1,
        max_steps: int = 25000,
        frame_skips: int = 50,
        timestep: float = 0.002,
        render_mode: str = "human",
        allow_contact_goals: bool = False,
        lookup_table_npz: str = None,       # used only for local (non-Ray) runs
        H_all: np.ndarray = None,           # pre-loaded array from Ray shared memory
        goal_to_table_idx: np.ndarray = None,  # pre-loaded array from Ray shared memory
        buffer: ReplayBuffer = None,
        data_dir: str = "data",
        logs_dir: str = "logs",
        logs_location=None,
        seed: int = 42,
        device='cpu'
    ):
        assert horizon >= 1, f"Horizon must be at least 1, but got {horizon}"

        self.is_train    = is_train
        self.scene_path  = scene_path
        self.policy_net  = policy_net
        self.horizon     = horizon
        self.buffer      = buffer if buffer is not None else ReplayBuffer()
        self.num_episodes = num_episodes
        self.max_steps   = max_steps
        self.data_dir    = Path(data_dir)
        self.logs_dir    = Path(logs_dir)
        self.seed        = seed
        self.device      = device
        self.render_mode = render_mode

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.env = CustomEnv(
            scene_path=scene_path,
            workspace_npz=workspace_npz,
            render_mode=render_mode,
            frame_skips=frame_skips,
            timstep=timestep,
            allow_contact_goals=allow_contact_goals,
            # Pass pre-loaded arrays if available, otherwise fall back to loading from path.
            # CustomEnv / HeuristicTable must accept these kwargs (see Step 2 below).
            lookup_table_npz=lookup_table_npz,
            H_all=H_all,
            goal_to_table_idx=goal_to_table_idx,
        )
        self.env = TimeLimit(self.env, max_episode_steps=self.max_steps)

        if logs_location is not None:
            run_folder_path = self.logs_dir / Path(logs_location)
            run_folder_path.mkdir(parents=True, exist_ok=True)
            self.monitor_file = run_folder_path / f"worker{name}_monitor"
            self.env = Monitor(self.env, filename=str(self.monitor_file))

    def _get_action(self, state: Dict) -> np.ndarray:
        r_state = flatten_state(state, self.device).view(1, 1, -1)
        plan, _ = self.policy_net.sample(r_state, self.horizon)
        plan    = plan.detach().cpu().numpy().squeeze()

        action = plan[0] if plan.ndim > 1 else plan
        return action

    def _save_buffer(self):
        filename = (
            f"{self.data_dir}/{str(uuid.uuid4())[:8]}_{time.strftime('%Y%m%d_%H%M')}"
        )
        if self.is_train:
            self.buffer.save(filename)
            self.buffer.clear()

    def run_episodes(self):
        for episode in range(self.num_episodes):
            state, info = self.env.reset(seed=self.seed)
            done        = False
            step_count  = 0

            while not done:
                action = self._get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if self.is_train and step_count >= 1:
                    flat_state      = flatten_state(state).cpu().numpy()
                    flat_next_state = flatten_state(next_state).cpu().numpy()
                    self.buffer.add(flat_state, action, reward, flat_next_state, done)

                step_count += 1
                state = next_state

                if self.render_mode == "human":
                    self.env.render()

            self._save_buffer()

    def run_session(self):
        try:
            self.run_episodes()

        except (KeyboardInterrupt, mujoco.FatalError, Exception) as e:
            if isinstance(e, KeyboardInterrupt):
                print(f"\n\033[96m[INFO] Simulation stopped by user.\033[0m")
            elif isinstance(e, mujoco.FatalError):
                print(f"\n\033[1;31m[CRITICAL] MuJoCo Engine crashed! {e}\033[0m")
            else:
                print(f"\n\033[93m[ERROR] Unexpected error: {e}\033[0m")
                import traceback
                traceback.print_exc()
            self._save_buffer()

        finally:
            self.env.close()


@ray.remote
class ParallelEnvRunner(EnvRunner):
    def __init__(
        self,
        name,
        is_train,
        scene_path,
        workspace_npz,
        policy_state_dict,
        config,
        logs_location,
        H_all_ref=None,             # Ray object ref — resolved once below
        goal_to_table_idx_ref=None, # Ray object ref — resolved once below
        lookup_table_npz=None,
    ):
        from src.models.policy_network import LatentPolicyPlanner

        policy_net = LatentPolicyPlanner(
            r_dim=config["agent"]["r_dim"],
            action_dim=config["agent"]["action_dim"],
            **config["model"]["policy"]
        )
        policy_net.load_state_dict(policy_state_dict)
        policy_net.eval()

        # Resolve the Ray object refs into plain numpy arrays here, once per worker.
        # ray.get() on a ref that lives in the object store is zero-copy via shared
        # memory — no 221MB allocation per worker.
        # After (fixed)
        H_all = ray.get(H_all_ref) if isinstance(H_all_ref, ray.ObjectRef) else H_all_ref
        goal_to_table_idx = ray.get(goal_to_table_idx_ref) if isinstance(goal_to_table_idx_ref, ray.ObjectRef) else goal_to_table_idx_ref

        super().__init__(
            name=name,
            is_train=is_train,
            scene_path=scene_path,
            workspace_npz=workspace_npz,
            lookup_table_npz=lookup_table_npz,  # ← add (needed for _load_workspace branch)
            H_all=H_all,
            goal_to_table_idx=goal_to_table_idx,
            policy_net=policy_net,
            horizon=config["agent"]["horizon"],
            num_episodes=config["env"]["num_episodes"],
            max_steps=config["env"]["max_steps"],
            frame_skips=config["env"]["frame_skips"],
            timestep=config["env"]["timestep"],
            render_mode=config["env"]["render"],
            allow_contact_goals=config["env"].get("allow_contact_goals", False),
            seed=config["seed"],
            device=config["device"],
            logs_location=logs_location,
        )

    def run_session_remote(self):
        return self.run_session()