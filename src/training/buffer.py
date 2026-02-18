import numpy as np
import os

# TODO: Save in a way where retrieving sequences is easy (horizon)
#       To re-iterate, we predict a sequence of actions but only take and record (replay buffer) 
#       the next action a_t. Then during training (update) we use a sequence of single-steps to 
#       train the model (actor and critic).

class ReplayBuffer:
    def __init__(
        self, 
        state_dim: int,
        action_dim: int,
        max_size: int=1000000
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind]
        )
    
    def save(self, file_path):
        """
        Save the buffer to disk.
        """
        # TODO: Save data in csv
        pass
    
    def load(self, file_path):
        """
        Load buffer from disk.
        """
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = # TODO Load data using csv
                self.state = data['state']
                self.action = data['action']
                self.reward = data['reward']
                self.next_state = data['next_state']
                self.done = data['done']
                self.ptr = data['ptr']
                self.size = data['size']
                self.max_size = self.state.shape[0]
        else:
            print(f"No file found at {file_path}")


if __name__ == "__main__":
    buffer = ReplayBuffer(state_dim=4, action_dim=1, max_size=1000)
    buffer.add(np.array([1,2,3,4]), np.array([0]), 1.0, np.array([1,2,3,5]), False)
    s, a, r, ns, d = buffer.sample(1)
    print("Sampled:", s, a, r, ns, d)