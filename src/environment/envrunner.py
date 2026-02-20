import torch
import torch.nn as nn
import numpy as np

import concurrent.futures
import multiprocessing
import threading
import traceback
import signal

from typing import Optional, Dict

from src.environment.env import CustomEnv
from src.utils.data_preprocessor import format_state
from src.buffers.buffer import ReplayBuffer

class EnvRunner:
    """
    Manages the execution and lifecycle of episodes within an environment 
    using a specific policy.
    """

    def __init__(
        self, 
        scene_path: str,
        policy: nn.Module,
        render_mode: str = "human",
        buffer: Optional[ReplayBuffer] = None,
        num_episodes: int = 1,
        seed: int = 42
    ):
        """
        Initialize the environment runner

        Args
        ----
        scene_path : str
            Path to the simulation configuration file.
        policy : nn.Module
            The neural network model used to determine actions.
        num_episodes : int, optional
            Total number of episodes to execute, by default 1.
        seed : int, optional
            Random seed for reproducibility, by default 42.
        """

        self.scene_path = scene_path
        self.policy = policy
        self.buffer = buffer
        self.num_episodes = num_episodes
        self.seed = seed

        self.env = CustomEnv(scene_path, render_mode)

    def get_action(self, state: Dict) -> np.ndarray:
        r_state, graph, ssm_state = format_state(state)

        action = self.policy.rollout(r_state, graph, ssm_state, horizon=1)
        action = action.detach().numpy().squeeze()

        # If n_samples >> 1, there are several methods of selecting the candidate action:
        # 1. Importance sampling
        # 2. MAP approximation
        # 3. Reward choosing
        action = action[0]

        return action
    
    def reset_env(self):
        self.env.reset()
    
    def run_episodes(self, is_train: bool):
        for episode in range(self.num_episodes):
            print(f"\nStarting Episode {episode + 1}")
            state, info = self.env.reset(seed=self.seed)
            done = False

            while not done: # TODO: Should limit it
                action = self.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if is_train and self.buffer is not None:
                    self.buffer.add(state, action, reward, next_state, done)

                state = next_state
                self.env.render()

        self.reset_env()

    def run_session(self, is_train: bool):
        try:
            self.run_episodes(is_train)
        except KeyboardInterrupt as e:
            pass
        finally:
            self.env.close()

'''
class SimulationConcurrency:
    def __init__(
        self, 
        tasks: List[SimulationTask],
        max_workers: Optional[int] = None
    ):
        self.tasks = tasks
        self.max_workers = max_workers
        self.results = []
        self.executor = None

    def _concurrency_handler(self, signum, frame):
        if self.executor is not None:
            try:
                self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        raise KeyboardInterrupt

    def run(self) -> List:
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._concurrency_handler)
        
        ctx = multiprocessing.get_context("spawn")

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=ctx
        ) as executor:
            self.executor = executor

            futures = [executor.submit(task) for task in self.tasks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    self.results.append(result)

                except KeyboardInterrupt:
                    break

                except Exception as e:
                    print(f"[Error within concurrent simulation] {e}")
                    traceback.print_exc()

                finally:
                    self.executor = None

        return self.results
'''

''' Reference
How do we convert a desired shape (curvature + orientation) into how much each tendon should be pulled?

### Physics Settings
- **Simulation**: 500 Hz (2ms timestep) for smooth contact dynamics
- **Control**: 20 Hz (25 frame skip) for stable teleoperation
- **Contact**: Tuned for soft compliance without penetration
- **Episode Length**: 1000 steps maximum (50 seconds at 20 Hz)

# Set simulation timestep to 1000 Hz (matching scene default)
self.model.opt.timestep = 0.001  # 1000 Hz for stable contact dynamics
'''