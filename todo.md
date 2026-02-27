# Reduce num of links to 15 (else OOM)

# Plot reward per epoch to show convergence
# Plot chosen diffused path on map

# Create documentation using Sphinx
# Create README.md


# Experiments
# 2. Diffusion multi-step (predict N-step) with reward selection
# 3. Diffusion multi-step (predict N-step) with reward selection and classifier-free guidance (reward gradient?)
# 4. Diffusion multi-step (predict N-step) with reward selection and MCTS (Lookahead, Receding Horizon, Sampling-based optimization)


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
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Average Return", fontsize=12)
plt.title("SAC Training Performance", fontsize=14)
plt.legend()
plt.show()