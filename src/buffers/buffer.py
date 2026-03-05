import os
import numpy as np

from typing import Union, List

from src.buffers.base import BaseReplayBuffer
from src.utils.sampler import NumpySampler

class ReplayBuffer(BaseReplayBuffer):
    """
    A buffer for storing and sampling transitions.
    """
    def __init__(
        self, 
        max_size: int = 1000000,
        seed: int = 42
    ):
        self.max_size = max_size

        self.ptr = 0
        self.size = 0
        
        self.state = None
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = None

        self.samples_consumed = 0

        self._rng = np.random.default_rng(seed)
        self.sampler = NumpySampler(rng=self._rng)

    def _init_buffers(self, action: np.ndarray):
        """
        Helper to create arrays based on the shapes of the first sample.
        """
        self.state = np.empty((self.max_size, 1), dtype=object)
        self.next_state = np.empty((self.max_size, 1), dtype=object)
        self.action = np.zeros((self.max_size, *np.shape(action)), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=bool)
    
    def add(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: Union[np.ndarray, float], 
        next_state: np.ndarray,
        done: Union[bool, int]
    ):
        """
        Adds a new transition to the replay buffer. If the buffer is full, 
        the oldest transition is overwritten.

        Args:
            state (np.ndarray): The observation/state at the current timestep.
            action (np.ndarray): The action taken by the agent.
            reward (float or np.ndarray): The scalar reward received.
            next_state (np.ndarray): The observation/state after the action.
            done (bool or int): Terminal flag (1 or True if episode ended).

        Returns:
            None
        """
        # Assumption: If state is empty, then recent buffer initialization
        if self.state is None:
            self._init_buffers(action)

        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, horizon: int=1):
        """
        Randomly samples a batch of experience sequences from the buffer.

        Args:
            batch_size (int): Number of independent sequences to sample.
            horizon (int): The length of each sequence (n-steps). Defaults to 1.

        Returns:
            tuple: A tuple containing (states, actions, rewards, next_states, dones).
                Each element is a NumPy array of shape (batch_size, horizon, dim).
        """
        s_batch, a_batch, r_batch, ns_batch, d_batch = [], [], [], [], []

        if horizon < 1 or self.size < horizon:
            return s_batch, a_batch, r_batch, ns_batch, d_batch
        
        # Sample for indices (Without Replacement)
        indices = self.sampler.sample(batch_size, self.size - horizon + 1)
        self.samples_consumed += len(indices)

        for start_idx in indices:
            # Check for 'done' within this specific horizon
            chunk_dones = self.done[start_idx : start_idx + horizon].flatten()
            done_indices = np.where(chunk_dones)[0]
            
            # Truncate at the first 'done' found
            actual_len = (done_indices[0] + 1) if done_indices.size > 0 else horizon
            effective_end = start_idx + actual_len

            s_batch.append(self.state[start_idx : effective_end])
            a_batch.append(self.action[start_idx : effective_end])
            r_batch.append(self.reward[start_idx : effective_end])
            ns_batch.append(self.next_state[start_idx : effective_end])
            d_batch.append(self.done[start_idx : effective_end])

        return s_batch, a_batch, r_batch, ns_batch, d_batch
    
    def __len__(self):
        return self.size
    
    def clear(self):
        """
        Resets pointers. 
        New data will overwrite old entries starting from index 0.
        Does not reset the buffer to an empty state.
        """
        self.ptr = 0
        self.size = 0
        self.samples_consumed = 0
        # self.state.fill(None) 
        # self.action.fill(0)
        # self.next_state.fill(None) 
        # self.reward.fill(0)
        # self.done.fill(0)
        
    def can_sample(self, batch_size):
        """
        Checks if there is enough data left to form a batch.
        """
        return (self.size - self.samples_consumed) >= batch_size

    def save(self, file_path: str):
        """
        Save the buffer to disk.
        """
        if self.size <= 0:
            return

        if not file_path.endswith('.npz'):
            file_path += '.npz'
            
        np.savez_compressed(
            file_path,
            state=self.state[:self.size],
            action=self.action[:self.size],
            reward=self.reward[:self.size],
            next_state=self.next_state[:self.size],
            done=self.done[:self.size],
            ptr=self.ptr,
            size=self.size
        )
    
    def load(self, file_paths: Union[str, List[str]]):
        """
        Load buffer from disk.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        for path in file_paths:
            if not path.endswith('.npz'):
                path += '.npz'

            if not os.path.exists(path):
                continue

            with np.load(path, allow_pickle=True) as data:

                n_samples = len(data['state'])
                for i in range(n_samples):
                    self.add(
                        data['state'][i], 
                        data['action'][i], 
                        data['reward'][i], 
                        data['next_state'][i], 
                        data['done'][i]
                    )