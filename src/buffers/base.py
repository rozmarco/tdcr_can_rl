from abc import ABC, abstractmethod
from typing import Any, Tuple

class BaseReplayBuffer(ABC):
    """
    Abstract interface for all Replay Buffer implementations.
    Ensures consistency across different sampling strategies.
    """

    @abstractmethod
    def add(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """Add a single transition to the buffer."""
        pass

    @abstractmethod
    def sample(self, batch_size: int, horizon: int = 1) -> Tuple:
        """Return a batch of sequences from the buffer."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the current number of elements in the buffer."""
        pass