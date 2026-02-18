import torch
import torch_geometric

import concurrent.futures
import multiprocessing
import threading
import traceback
import signal

from .env import CustomEnv

# TODO: Data collection start / end points in do_simulation

class EnvRunner:
    def __init__(
        self, 
        scene_path, 
        policy,
        buffer=None,
        num_episodes=1, 
        seed=42
    ):
        self.scene_path = scene_path
        self.policy = policy
        self.buffer = buffer
        self.num_episodes = num_episodes
        self.seed = seed

        self.env = CustomEnv(scene_path=self.scene_path)

    def get_action(self, state):
        # TODO: Complete this section after xml and states are finalized
        device = self.policy.device

        robot_tensor = torch.as_tensor(state['robot_state'], dtype=torch.float32, device=device).unsqueeze(0)
        
        obstacles_tensor = torch.as_tensor(state['obstacle_state'], dtype=torch.float32, device=device)
        obstacles_tensor_temp = torch.rand((5, 50)) # TODO: Format obstacles_tensor into each obstacle
        edge_index = self.policy.encoder.obstacle_encoder.generate_edges(5)
        graph = torch_geometric.data.Data(x=obstacles_tensor_temp, edge_index=edge_index)
        
        ssm_state = torch.randn((64, 1))

        action = self.policy.rollout(robot_tensor, graph, ssm_state, horizon=1)
        action = action.detach().numpy().squeeze()

        # If n_samples >> 1, there are several methods of selecting the candidate action:
        # 1. Importance sampling
        # 2. MAP approximation
        # 3. Reward choosing
        action = action[0]

        return action
    
    def reset_env(self):
        self.env.reset()
    
    def do_simulation(self):
        for episode in range(self.num_episodes):
            print(f"\nStarting Episode {episode + 1}")

            state, info = self.env.reset(seed=self.seed)
            done = False

            # while not done: # TODO
            for step in range(50):
                if done:
                    break

                action = self.get_action(state)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                if self.buffer is not None:
                    self.buffer.add(state, action, reward, next_state, done)

                state = next_state

                self.env.render()

        self.reset_env()

    def run_session(self):
        try:
            self.do_simulation()
        finally:
            self.env.close()

'''
class SimulationTask:
    def __init__(
        self,
        session_resources: SessionResources,
        session_config: SessionFactory
    ):
        super().__init__()
        self.session_resources = session_resources
        self.session_config = session_config

    def _handle_interrupt(self, signum, frame):
        raise KeyboardInterrupt

    def __call__(self):
        if multiprocessing.current_process().name != 'MainProcess':
            signal.signal(signal.SIGINT, self._handle_interrupt)

        runner = SessionRunner(self.session_resources, self.session_config)
        
        try: 
            runner.run_session()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            raise e

class SimulationConcurrency:
    def __init__(
        self, 
        tasks: List[SimulationTask],
        max_workers: Optional[int] = None
    ):
        super().__init__()
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