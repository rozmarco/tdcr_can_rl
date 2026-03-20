import os
import numpy as np

from typing import Union, List
from src.buffers.base import BaseReplayBuffer


class ReplayBuffer(BaseReplayBuffer):
    """
    Replay buffer for SAC. States are flat float32 numpy arrays.
    Uses simple random sampling with replacement — standard for off-policy RL.
    """
    def __init__(self, max_size: int = 1000000, seed: int = 42):
        self.max_size = max_size
        self.ptr      = 0
        self.size     = 0

        self.state      = None
        self.next_state = None
        self.action     = None
        self.reward     = None
        self.done       = None

        self._rng = np.random.default_rng(seed)

    def _init_buffers(self, state: np.ndarray, action: np.ndarray):
        state_dim  = state.shape[0]
        action_dim = np.shape(action)
        self.state      = np.zeros((self.max_size, state_dim),   dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim),   dtype=np.float32)
        self.action     = np.zeros((self.max_size, *action_dim), dtype=np.float32)
        self.reward     = np.zeros((self.max_size, 1),           dtype=np.float32)
        self.done       = np.zeros((self.max_size, 1),           dtype=bool)

    def add(self, state, action, reward, next_state, done):
        if self.state is None:
            self._init_buffers(state, action)
        self.state[self.ptr]      = state
        self.action[self.ptr]     = action
        self.reward[self.ptr]     = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr]       = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, horizon: int = 1):
        """
        Random sample with replacement. Standard SAC sampling.
        Returns lists of (horizon, dim) arrays, one per batch item.
        """
        assert self.size > horizon, "Not enough data to sample"

        # Sample random start indices
        indices = self._rng.integers(0, self.size - horizon, size=batch_size)

        s_batch, a_batch, r_batch, ns_batch, d_batch = [], [], [], [], []
        for start_idx in indices:
            chunk_dones  = self.done[start_idx : start_idx + horizon].flatten()
            done_indices = np.where(chunk_dones)[0]
            actual_len   = (done_indices[0] + 1) if done_indices.size > 0 else horizon
            end          = start_idx + actual_len

            s_batch.append(self.state[start_idx:end])
            a_batch.append(self.action[start_idx:end])
            r_batch.append(self.reward[start_idx:end])
            ns_batch.append(self.next_state[start_idx:end])
            d_batch.append(self.done[start_idx:end])

        return s_batch, a_batch, r_batch, ns_batch, d_batch

    def can_sample(self, horizon: int = 1) -> bool:
        return self.size > horizon

    def __len__(self):
        return self.size

    def clear(self):
        self.ptr  = 0
        self.size = 0

    def save(self, file_path: str):
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
        )

    def load(self, file_paths: Union[str, List[str]]):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        for path in file_paths:
            if not os.path.exists(path):
                continue
            with np.load(path, allow_pickle=True) as data:
                n = len(data['state'])
                if self.state is None:
                    self._init_buffers(data['state'][0], data['action'][0])
                idx = np.arange(self.ptr, self.ptr + n) % self.max_size
                self.state[idx]      = data['state']
                self.action[idx]     = data['action']
                self.reward[idx]     = data['reward']
                self.next_state[idx] = data['next_state']
                self.done[idx]       = data['done']
                self.ptr  = (self.ptr + n) % self.max_size
                self.size = min(self.size + n, self.max_size)