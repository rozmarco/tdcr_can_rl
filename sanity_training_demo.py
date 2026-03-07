#!/usr/bin/env python3
"""
TDCR RL Sanity Training Demo
Produces clear "signs of life" for interim report.

Outputs:
- Episode reward curve
- Distance-to-goal curve
- Replay buffer growth
"""

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.environment.env import CustomEnv
from src.models.policy_network import LatentDiffusionPolicyPlanner
from src.models.q_network import QNetwork
from src.buffers.buffer import ReplayBuffer
from src.sac import SoftActorCritic
from src.utils.data_preprocessor import flatten_state


# =====================================================
# CONFIG
# =====================================================

print("\nLoading config...")

config = yaml.safe_load(open("config/train.yaml"))

torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

scene = config["scene"]
device = config["device"]

r_dim = config["agent"]["r_dim"]
action_dim = config["agent"]["action_dim"]

# SMALL training for quick results
NUM_EPISODES = 50
MAX_STEPS = 30
UPDATE_STEPS = 20

print(f"State dim: {r_dim}")
print(f"Action dim: {action_dim}")
print(f"Episodes: {NUM_EPISODES}\n")


# =====================================================
# ENVIRONMENT
# =====================================================

env = CustomEnv(
    scene,
    render_mode=None,   # faster
    frame_skips=config["env"]["frame_skips"],
    timstep=config["env"]["timestep"]
)


# =====================================================
# NETWORKS
# =====================================================

policy = LatentDiffusionPolicyPlanner(
    r_dim=r_dim,
    action_dim=action_dim,
    **config["model"]["policy"]
)

q1 = QNetwork(
    r_dim=r_dim,
    action_dim=action_dim,
    **config["model"]["q_network"]
)

q2 = QNetwork(
    r_dim=r_dim,
    action_dim=action_dim,
    **config["model"]["q_network"]
)

buffer = ReplayBuffer(
    max_size=50000,
    seed=config["seed"]
)

sac = SoftActorCritic(
    policy=policy,
    q1=q1,
    q2=q2,
    replay_buffer=buffer,
    horizon=config["agent"]["horizon"],
    optimizer_str=config["agent"]["optimizer"]["_target_"],
    policy_lr=config["agent"]["policy_lr"],
    q_lr=config["agent"]["q_lr"],
    batch_size=config["agent"]["batch_size"],
    gamma=config["agent"]["gamma"],
    tau=config["agent"]["tau"],
    alpha=config["agent"]["alpha"],
    seed=config["seed"],
    device=device
)


# =====================================================
# TRACKING
# =====================================================

episode_rewards = []
episode_distances = []
buffer_sizes = []


print("Starting training...\n")


# =====================================================
# TRAIN LOOP
# =====================================================

for episode in range(NUM_EPISODES):

    state, _ = env.reset()

    total_reward = 0

    for step in range(MAX_STEPS):

        r_state = flatten_state(state, device).view(1,1,-1)

        with torch.no_grad():
            plan, _ = policy.rollout(r_state, config["agent"]["horizon"])

        plan = plan.detach().cpu().numpy().squeeze()

        action = plan[0] if plan.ndim > 1 else plan

        next_state, reward, term, trunc, _ = env.step(action)

        total_reward += reward

        if step > 0:
            buffer.add(state, action, reward, next_state, term or trunc)

        state = next_state

        if term or trunc:
            break


    # distance to goal
    final_dist = np.linalg.norm(state["goal_rel_pos"])

    episode_rewards.append(total_reward)
    episode_distances.append(final_dist)
    buffer_sizes.append(len(buffer))

    print(
        f"Episode {episode+1:02d} | "
        f"Reward: {total_reward:7.3f} | "
        f"Dist: {final_dist:.3f} m | "
        f"Buffer: {len(buffer)}"
    )


    # =================================================
    # TRAINING STEP
    # =================================================

    min_buffer = 200 

    if len(buffer) > min_buffer:

        for _ in range(UPDATE_STEPS):

            s, a, r, ns, d = buffer.sample(
                config["agent"]["batch_size"],
                config["agent"]["horizon"]
            )

            if len(s) == 0:
                print("Sample empty — skipping update")
                continue

            sac.update()

        print("SAC update step executed")

    else:
        print(f"Buffer not large enough for training yet ({len(buffer)})")
# =====================================================
# VISUALIZATION
# =====================================================

print("\nGenerating report plots...")

episodes = np.arange(NUM_EPISODES)

fig, axs = plt.subplots(1,3, figsize=(15,4))

# Reward curve
axs[0].plot(episodes, episode_rewards, marker='o')
axs[0].set_title("Episode Reward")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")
axs[0].grid(True)

# Distance curve
axs[1].plot(episodes, episode_distances, marker='o')
axs[1].set_title("Distance to Goal")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Distance (m)")
axs[1].grid(True)

# Buffer growth
axs[2].plot(episodes, buffer_sizes, marker='o')
axs[2].set_title("Replay Buffer Size")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Transitions")
axs[2].grid(True)

plt.tight_layout()

Path("reports").mkdir(exist_ok=True)

plot_path = "reports/rl_sanity_training.png"

plt.savefig(plot_path, dpi=150)

print(f"\nSaved figure: {plot_path}")

print("\n✓ RL pipeline sanity check complete.")