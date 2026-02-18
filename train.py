import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from src.training.sac import SoftActorCritic
from src.training.mse import MeanSquaredError

class Trainer:
    def __init__(self):
        super().__init__()
        pass

    def train():
        return

if __name__ == '__main__':
    # Load YAML configurations

    # Initialize model

    # Train model (SAC + MSE)

    # Plot reward per epoch to show convergence
    import matplotlib.pyplot as plt

    steps = np.arange(len(rewards))
    avg_return = rewards.mean(axis=1)
    std_return = rewards.std(axis=1)

    plt.figure(figsize=(8, 5))

    # Plot with shaded std deviation
    plt.fill_between(steps, avg_return - std_return, avg_return + std_return, 
                    color='blue', alpha=0.2, label='Std deviation')

    # Plot average return
    plt.plot(steps, avg_return, color='blue', label='Average Return', linewidth=2)
    plt.grid(which='major', linestyle='--', alpha=0.6)
    plt.grid(which='minor', linestyle=':', alpha=0.3)
    plt.minorticks_on()
    plt.gca().set_facecolor('#f0f0f0')
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Average Return", fontsize=12)
    plt.title("SAC Training Performance", fontsize=14)
    plt.legend()
    plt.show()