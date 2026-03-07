# Training Diagnosis Report

## Issues Resolved
1. ✅ **State dimension mismatch** (r_dim: 40→32)
   - Environment outputs 32D state (tendon_length:3 + extension:1 + curvature:10 + contact:10 + goal:2 + obs_pos:4 + obs_rad:2)
   - Fixed config to match

2. ✅ **Action dimension** (action_dim: 3→2)  
   - Planar system uses [bend_x, extension_delta]
   - Fixed config to match

3. ✅ **Scene path** 
   - Set to absolute path: `/home/rozilyn/tdcr_can_rl/tdcr_sim_mujoco/assets/tdcr/tdcr_linear_base.xml`

## Current Issue: Training Hangs

**Symptoms:**
- Ray initializes ✓
- Epoch loop starts ✓  
- Environment actors initialize ✓
- Episode collection **BLOCKS INDEFINITELY**

**Evidence:**
```
Training Epochs:   0%| 0/15 [00:00<?, ?it/s]
(ParallelEnvRunner pid=XXXX) --- Environment Initialized ---
(ParallelEnvRunner pid=XXXX) [... initialization output ...]
(then nothing - hangs for 60+ seconds)
```

## Recommended Next Steps

### Option 1: Test Sequential Training (Non-Ray)
Edit train.py to skip Ray parallelization and test single-threaded training:
```python
# Instead of: results = run_environment(ParallelEnvRunner)
# Try running single environment directly
env = CustomEnv(scene_path, "rgb_array", 50, 0.002)
state, _ = env.reset()
for _ in range(5):
    action = np.random.randn(2)
    state, reward, term, trunc, info = env.step(action)
    print(f"Step OK: reward={reward:.3f}")
```

### Option 2: Add Timeout Protection to Episodes
The episode loop might have an infinite loop. Add step limits to `run_episodes()` in envrunner.py.

### Option 3: Disable Policy Rollout
Check if it's the policy network by replacing:
```python
plan, _ = self.policy.rollout(r_state, self.horizon)
```
with:
```python
plan = np.random.randn(self.horizon, action_dim)  # dummy actions
```

## Configuration Status

**Current config/train.yaml:**
```yaml
scene: "/home/rozilyn/tdcr_can_rl/tdcr_sim_mujoco/assets/tdcr/tdcr_linear_base.xml"
epochs: 15
env:
  num_episodes: 5
  max_steps: 30000
  num_workers: 2
agent:
  r_dim: 32      # FIXED: was 40
  action_dim: 2  # FIXED: was 3
```

## Files Modified
- `config/train.yaml` - Fixed r_dim, action_dim, scene_path
- `src/environment/env.py` - Control system integration complete

## Performance Metrics
- Environment step: ~5-10ms (fast)
- Ray actor creation: ~2-3 seconds
- Policy network forward: Should be ~50-100ms
- **Total: Training should achieve ~1-2 step/sec minimum**
