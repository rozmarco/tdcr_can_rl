import numpy as np

class NumpySampler:
    def __init__(
        self, 
        rng: np.random.Generator,
        auto_resample: bool = False
    ):
        self.rng = rng
        self.auto_resample = auto_resample
        self.indices = np.array([], dtype=int)
        self._initialized = False

    def reset(self):
        """
        Clears the current sampling state to allow for a fresh shuffle. 
        
        This is primarily used to combat the 'exhaustion' state: once the sampler 
        has returned all indices, calling this ensures that the next sampling 
        request will look at the current buffer size and generate a new 
        shuffled deck including any newly added data.
        """
        self.indices = np.array([], dtype=int)
        self._initialized = False

    def sample(self, batch_size: int, size: int) -> np.ndarray:
        """
        Random sampling without replacement.
        """
        # INITIALIZATION
        if (not self._initialized) or (self.auto_resample and len(self.indices) == 0):
            if size > 0:
                self.indices = self.rng.permutation(size)
                self._initialized = True

        # EXHAUSTION CHECK
        if len(self.indices) == 0:
            return np.array([], dtype=int)     

        # SAMPLE: Take what is available
        actual_batch = min(batch_size, len(self.indices))
        
        batch_indices = self.indices[:actual_batch]
        self.indices = self.indices[actual_batch:]
        
        return batch_indices