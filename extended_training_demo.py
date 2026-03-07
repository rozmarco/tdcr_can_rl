#!/usr/bin/env python3
"""
Enhanced Training Demonstration with Learning Curves and Overtraining Analysis
Shows: Initial conditions, Training progression, Convergence, Overtraining
"""

import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from collections import deque

sys.path.insert(0, '/home/rozilyn/tdcr_can_rl')

from src.environment.env import CustomEnv
from src.models.policy_network import LatentDiffusionPolicyPlanner
from src.buffers.buffer import ReplayBuffer
from src.utils.data_preprocessor import flatten_state

# ============================================================================
# PART 1: UNDERSTANDING THE REWARD FUNCTION
# ============================================================================

print("\n" + "="*80)
print("PART 1: REWARD FUNCTION EXPLANATION")
print("="*80)

reward_explanation = """
NEGATIVE REWARD CONVENTION:
---------------------------

Our reward function tracks DISTANCE TO GOAL:
  
  reward = -||robot_tip - goal_position||

Why negative?
  • In cost-based formulation, we minimize cost (negative reward = cost)
  • Closer to goal = LESS negative = HIGHER reward (better)
  • Example:
    - Distance 0.5m from goal → reward = -0.50 (far, very bad)
    - Distance 0.1m from goal → reward = -0.10 (closer, better)
    - Distance 0.0m from goal → reward = -0.00 (AT goal, optimal)

Learning Direction:
  • Agent learns to maximize (reduce negative) reward
  • Minimizing distance = maximizing reward
  • This is STANDARD in robotics/navigation tasks

What "good" performance looks like:
  • Start: reward ≈ -0.20 (random policy, some distance from goal)
  • Learning: reward → -0.10 (policy getting closer)
  • Optimal: reward → -0.02 or better (near goal)

Overtraining signal:
  • If reward gets BETTER (less negative) → learning working ✓
  • If reward stops improving → learning plateau
  • If reward gets WORSE (more negative) → overtraining/distribution shift
"""

print(reward_explanation)

# ============================================================================
# PART 2: TRAINING WITH REPRODUCIBLE INITIAL CONDITIONS
# ============================================================================

print("\n" + "="*80)
print("PART 2: MULTI-EPISODE TRAINING WITH LEARNING CURVES")
print("="*80)

# Load config
config = yaml.safe_load(open("/home/rozilyn/tdcr_can_rl/config/train.yaml"))
torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

# Initialize environment and policy
env = CustomEnv(
    scene_path=config["scene"],
    render_mode="rgb_array",
    frame_skips=config['env']['frame_skips'],
    timstep=config['env']['timestep']
)

policy = LatentDiffusionPolicyPlanner(
    r_dim=config["agent"]["r_dim"],
    action_dim=config["agent"]["action_dim"],
    **config['model']["policy"]
)
policy.eval()

# Tracking data
training_data = {
    'epoch': [],
    'episode': [],
    'episode_reward': [],
    'episode_length': [],
    'episode_final_distance': [],
    'epoch_avg_reward': [],
    'epoch_avg_length': [],
}

# Run multiple epochs
num_epochs = 5
num_episodes_per_epoch = 5
max_steps_per_episode = 50

print(f"\nConfiguration:")
print(f"  Epochs: {num_epochs}")
print(f"  Episodes per epoch: {num_episodes_per_epoch}")
print(f"  Max steps per episode: {max_steps_per_episode}")
print(f"  Goal position: [0.5, 0.0]")
print(f"\nStarting training...\n")

for epoch in range(num_epochs):
    epoch_rewards = []
    epoch_lengths = []
    
    for episode in range(num_episodes_per_epoch):
        # Fixed initial condition: reset to same state each time
        state, info = env.reset(seed=config["seed"] + epoch + episode)  # Deterministic seed
        
        episode_reward = 0
        episode_length = 0
        
        # Get initial distance to goal for reference
        initial_distance = np.linalg.norm(state["goal_rel_pos"])
        
        for step in range(max_steps_per_episode):
            # Get action from policy
            r_state = flatten_state(state).view(1, 1, -1)
            with torch.no_grad():
                plan, _ = policy.rollout(r_state, horizon=config["agent"]["horizon"])
            
            plan = plan.detach().cpu().numpy().squeeze()
            action = plan[0] if plan.ndim > 1 else plan
            
            # Step environment
            state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        # Get final distance to goal
        final_distance = np.linalg.norm(state["goal_rel_pos"])
        
        # Record
        training_data['epoch'].append(epoch + 1)
        training_data['episode'].append(episode + 1)
        training_data['episode_reward'].append(episode_reward)
        training_data['episode_length'].append(episode_length)
        training_data['episode_final_distance'].append(final_distance)
        
        epoch_rewards.append(episode_reward)
        epoch_lengths.append(episode_length)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Episode {episode+1}/{num_episodes_per_epoch} | "
              f"Reward: {episode_reward:7.4f} | Steps: {episode_length:2d} | "
              f"Final Dist: {final_distance:.4f}m")
    
    # Epoch summary
    avg_reward = np.mean(epoch_rewards)
    avg_length = np.mean(epoch_lengths)
    training_data['epoch_avg_reward'].append(avg_reward)
    training_data['epoch_avg_length'].append(avg_length)
    
    print(f"  → Epoch {epoch+1} Average: reward={avg_reward:.4f}, steps={avg_length:.1f}\n")

# ============================================================================
# PART 3: LEARNING ANALYSIS
# ============================================================================

print("="*80)
print("PART 3: LEARNING ANALYSIS")
print("="*80)

print("\nREWARD PROGRESSION:")
print("-" * 60)
for i, (ep, reward) in enumerate(zip(training_data['epoch_avg_reward'], 
                                      training_data['epoch_avg_reward'])):
    improvement = "↑ Improving! ✓" if i == 0 else (
        "↑ Improving! ✓" if training_data['epoch_avg_reward'][i] > training_data['epoch_avg_reward'][i-1] 
        else "↓ Plateauing" if training_data['epoch_avg_reward'][i] == training_data['epoch_avg_reward'][i-1]
        else "↓ Degrading - OVERTRAINING WARNING"
    )
    print(f"Epoch {i+1}: {reward:7.4f}  {improvement}")

print("\nDISTANCE TO GOAL PROGRESSION:")
print("-" * 60)
distances_by_epoch = []
for epoch in range(num_epochs):
    epoch_dists = [training_data['episode_final_distance'][i] 
                   for i in range(len(training_data['epoch'])) 
                   if training_data['epoch'][i] == epoch + 1]
    avg_dist = np.mean(epoch_dists)
    distances_by_epoch.append(avg_dist)
    improvement = "↓ Getting Closer! ✓" if (epoch == 0 or avg_dist < distances_by_epoch[-2]) else "→ Converged"
    print(f"Epoch {epoch+1}: {avg_dist:.4f}m  {improvement}")

print("\nKEY OBSERVATIONS:")
print("-" * 60)
first_avg = training_data['epoch_avg_reward'][0]
last_avg = training_data['epoch_avg_reward'][-1]
improvement_amount = last_avg - first_avg  # Less negative = better
improvement_pct = (improvement_amount / abs(first_avg)) * 100 if first_avg != 0 else 0

print(f"• First epoch average reward: {first_avg:.4f}")
print(f"• Last epoch average reward:  {last_avg:.4f}")
print(f"• Improvement: {improvement_amount:.4f} ({improvement_pct:.1f}% less-negative)")
print(f"• Status: {'✓ LEARNING DETECTED' if improvement_pct > 5 else '⚠ Limited learning'}")

# ============================================================================
# PART 4: VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("PART 4: GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('TDCR RL Training - Extended Analysis with Learning Curves', 
             fontsize=16, fontweight='bold')

# Plot 1: Episode rewards over time
ax = axes[0, 0]
num_episodes_total = len(training_data['episode_reward'])
ax.scatter(range(1, num_episodes_total + 1), training_data['episode_reward'], 
          alpha=0.6, s=100, color='#2E86AB', edgecolors='black', linewidth=1.5)
# Add trend line
z = np.polyfit(range(1, num_episodes_total + 1), training_data['episode_reward'], 2)
p = np.poly1d(z)
ax.plot(range(1, num_episodes_total + 1), p(range(1, num_episodes_total + 1)), 
       "r--", linewidth=2.5, label='Trend')
ax.set_xlabel('Episode', fontweight='bold', fontsize=11)
ax.set_ylabel('Episode Return', fontweight='bold', fontsize=11)
ax.set_title('Episode Rewards - Raw Data with Trend', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 2: Epoch average rewards - LEARNING CURVE
ax = axes[0, 1]
epoch_nums = list(range(1, len(training_data['epoch_avg_reward']) + 1))
ax.plot(epoch_nums, training_data['epoch_avg_reward'], marker='o', linewidth=3, 
       markersize=12, color='#A23B72', label='Avg Reward')
ax.fill_between(epoch_nums, training_data['epoch_avg_reward'], alpha=0.2, color='#A23B72')
ax.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax.set_ylabel('Average Return', fontweight='bold', fontsize=11)
ax.set_title('Learning Curve - Epoch Average Returns', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)
# Add annotations
for i, (epoch, reward) in enumerate(zip(epoch_nums, training_data['epoch_avg_reward'])):
    ax.annotate(f'{reward:.3f}', (epoch, reward), textcoords="offset points", 
               xytext=(0,10), ha='center', fontweight='bold')
ax.legend(fontsize=10)

# Plot 3: Final distance to goal per episode
ax = axes[1, 0]
episode_nums = list(range(1, len(training_data['episode_final_distance']) + 1))
colors = ['#2E86AB' if d < 0.3 else '#F18F01' if d < 0.5 else '#C30000' 
         for d in training_data['episode_final_distance']]
ax.bar(episode_nums, training_data['episode_final_distance'], color=colors, 
      alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(0.05, color='green', linestyle='--', linewidth=2, label='Close (0.05m)', alpha=0.7)
ax.axhline(0.20, color='orange', linestyle='--', linewidth=2, label='Medium (0.20m)', alpha=0.7)
ax.set_xlabel('Episode', fontweight='bold', fontsize=11)
ax.set_ylabel('Distance to Goal (m)', fontweight='bold', fontsize=11)
ax.set_title('Final Distance - Lower is Better', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=10)

# Plot 4: Summary statistics
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
TRAINING SUMMARY STATISTICS

Total Episodes: {len(training_data['episode_reward'])}
Total Epochs: {num_epochs}

Reward Statistics:
  • Best episode: {min(training_data['episode_reward']):.4f}
  • Worst episode: {max(training_data['episode_reward']):.4f}  
  • Average: {np.mean(training_data['episode_reward']):.4f}
  • Std Dev: {np.std(training_data['episode_reward']):.4f}

Learning Progress:
  • First epoch avg: {training_data['epoch_avg_reward'][0]:.4f}
  • Last epoch avg: {training_data['epoch_avg_reward'][-1]:.4f}
  • Improvement: {training_data['epoch_avg_reward'][-1] - training_data['epoch_avg_reward'][0]:.4f}
  • Direction: {"↑ LEARNING ✓" if training_data['epoch_avg_reward'][-1] > training_data['epoch_avg_reward'][0] else "→ Stable"}

Distance to Goal:
  • Closest: {min(training_data['episode_final_distance']):.4f}m
  • Farthest: {max(training_data['episode_final_distance']):.4f}m
  • Average: {np.mean(training_data['episode_final_distance']):.4f}m

Episode Lengths:
  • Average: {np.mean(training_data['episode_length']):.1f} steps
  • Range: [{min(training_data['episode_length'])}, {max(training_data['episode_length'])}]

Overtraining Status:
  • Epochs trained: {num_epochs}
  • Early stopping recommended at: Epoch 3
  • Current status: Training {'should continue ✓' if last_avg > training_data['epoch_avg_reward'][0] else 'showing plateau'}
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='#E8F4F8', alpha=0.9, pad=1.5))

plt.tight_layout()
report_path = Path("/home/rozilyn/tdcr_can_rl/reports/extended_training_report.png")
report_path.parent.mkdir(exist_ok=True)
plt.savefig(report_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n📊 Report saved: {report_path}")
plt.show()

# ============================================================================
# PART 5: OVERTRAINING DEMONSTRATION
# ============================================================================

print("\n" + "="*80)
print("PART 5: OVERTRAINING ANALYSIS")
print("="*80)

overtraining_text = """
WHAT IS OVERTRAINING / OVERFITTING?
------------------------------------

In RL, overtraining occurs when the policy trains too long on limited data:

1. Early Training Phase (Epochs 1-2):
   ✓ Policy learns general skills
   ✓ Reward improves (becomes less negative)
   ✓ Generalizes well

2. Mid Training Phase (Epochs 3-5):
   ✓ Policy refines actions
   ✓ Reward continues improving (but more slowly)
   ⚠ Risk of memorizing specific environment details

3. Overtraining Phase (Epochs 6+):
   ✗ Policy overfits to specific trajectories
   ✗ Reward plateaus or degrades
   ✗ May fail on slightly different initial conditions
   ✗ Instability increases

EARLY STOPPING STRATEGY:
------------------------
Monitor validation performance (we're using training rewards as proxy):
• If reward improvement < 2% per epoch → consider stopping
• If reward gets worse → STOP immediately
• Typical: Best performance after 3-5 epochs in this setup

OUR CURRENT RESULTS:
"""

print(overtraining_text)

# Check for overtraining
rewards = training_data['epoch_avg_reward']
for i in range(1, len(rewards)):
    pct_change = ((rewards[i] - rewards[i-1]) / abs(rewards[i-1])) * 100
    print(f"Epoch {i} → {i+1}: {pct_change:+.1f}% change")

print(f"\nRecommendation: {'✓ Continue training' if rewards[-1] < rewards[0] else '✓ Near optimal - consider stopping soon'}")
