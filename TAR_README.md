# Trajectory Adherence Regularization (TAR) Implementation

## Overview

This implementation adds **Trajectory Adherence Regularization (TAR)** to offline preference-based reinforcement learning algorithms. TAR addresses the problem of "optimistic trajectory stitching" by penalizing policies that deviate from dataset trajectories.

## Key Idea

TAR uses a **trajectory-level quadratic penalty**:

```
L_TAR = E_τ [ (Σ_t ||π(s_t) - a_t||)² ]
```

where `||·||` is the L2 norm (Euclidean distance with sqrt).

This design:
1. **Encourages trajectory coherence**: Penalizes cumulative deviation across entire trajectories
2. **Limits stitching frequency**: The quadratic form (sum²) strongly discourages many small deviations
3. **Allows sparse switching**: Permits few high-value deviations but prevents excessive stitching

**Important**: The formula uses L2 distance (with sqrt) for each step, then sums them, and finally squares the sum:
- Step distance: `d_t = ||π(s_t) - a_t|| = sqrt(Σ_i (π_i - a_i)²)`
- Trajectory distance: `D(τ) = Σ_t d_t`
- TAR loss: `L_TAR = E_τ [(D(τ))²]`

## Implementation Structure

### 1. TrajectoryBuffer (`offlinepbrl/buffer/traj_buffer.py`)

A general-purpose buffer that:
- Loads d4rl format datasets
- Automatically detects trajectory boundaries
- Samples complete trajectories with variable lengths
- Maintains backward compatibility with transition-level sampling

**Key methods:**
- `load_dataset(dataset)`: Loads and detects trajectories from d4rl data
- `sample_trajectories(batch_size)`: Samples complete trajectories with masking
- `sample(batch_size)`: Standard transition sampling (for compatibility)

### 2. MFPolicyTrainer (`offlinepbrl/policy_trainer/mf_policy_trainer.py`)

**Updated** to support TrajectoryBuffer:
- Automatically detects if buffer is TrajectoryBuffer
- When `traj_batch_size` is set and buffer is TrajectoryBuffer:
  - Calls `buffer.sample_trajectories()` for complete trajectories
  - Adds trajectory batch to learning batch as `"trajectory"` key
- Maintains backward compatibility with ReplayBuffer and APPO's segment sampling

**Key change:**
```python
if self._supports_traj_sampling:
    # TrajectoryBuffer: sample complete trajectories
    traj_batch = self.buffer.sample_trajectories(self._traj_batch_size)
    batch["trajectory"] = traj_batch
```

### 3. IPL-AWAC-TAR (`offlinepbrl/policy/preference/ipl_awac_tar.py`)

Extends IPLAWACPolicy with TAR regularization:
- Inherits all IPL-AWAC functionality
- Adds `compute_tar_loss()` for trajectory-level penalty
- Applies TAR in a separate actor update step

**Key parameters:**
- `tar_coef`: TAR regularization coefficient (default: 1.0)
- `tar_clip_max`: Maximum per-step deviation to clip outliers (default: 100.0)

### 3. BT-AWAC-TAR (`offlinepbrl/policy/preference/bt_awac_tar.py`)

Combines Bradley-Terry preference learning, AWAC, and TAR:
- Bradley-Terry reward model learning from preferences
- AWAC policy optimization with learned rewards
- TAR trajectory-level regularization

**Key parameters:**
- `rm_stop_epoch`: When to stop training reward model (default: 200)
- `policy_start_epoch`: When to start training policy (default: 200)
- `tar_coef`: TAR regularization coefficient (default: 1.0)

## Usage

### IPL-AWAC-TAR Example

```bash
cd /home/fsj/workspace/OfflinePbRL
python run_example/gym/run_ipl_awac_tar.py \
    --task walker2d-medium-expert-v2 \
    --seed 0 \
    --tar_coef 1.0 \
    --traj_batch_size 8
```

**Key hyperparameters:**
- `--tar_coef`: TAR regularization strength (higher = more conservative)
- `--traj_batch_size`: Number of trajectories per batch for TAR
- `--tar_clip_max`: Maximum per-step deviation clipping
- `--temperature`: AWAC temperature (0.3 for IPL)
- `--reward_reg`: IPL reward regularization (0.5)

### BT-AWAC-TAR Example

```bash
cd /home/fsj/workspace/OfflinePbRL
python run_example/gym/run_bt_awac_tar.py \
    --task walker2d-medium-expert-v2 \
    --seed 0 \
    --tar_coef 1.0 \
    --traj_batch_size 8
```

**Key hyperparameters:**
- `--tar_coef`: TAR regularization strength
- `--temperature`: AWAC temperature (3.0 for BT-AWAC)
- `--rm_stop_epoch`: Stop training reward model after this epoch (200)
- `--policy_start_epoch`: Start training policy from this epoch (200)

## Algorithm Flow

### Training Loop

For each epoch:
1. **Sample batches:**
   - Transition batch for critic/actor updates
   - Preference batch for reward learning (BT) or IPL
   - Trajectory batch for TAR regularization

2. **Learn phase:**
   - **Reward learning** (if applicable):
     - BT: Train reward model on preferences
     - IPL: Joint learning with inverse Bellman
   
   - **Policy learning**:
     - Update critics with Bellman targets
     - Update actor with advantage weighting
   
   - **TAR regularization**:
     - Compute trajectory-level deviations
     - Apply quadratic penalty to actor

3. **Evaluate:** Periodically evaluate policy performance

## Design Decisions

### Why TrajectoryBuffer instead of specialized methods?

- **Generality**: Can be reused for any trajectory-level algorithm
- **Flexibility**: Supports both transition and trajectory sampling
- **Maintainability**: Single source of truth for trajectory handling
- **Extensibility**: Easy to add new trajectory-based methods

### Why separate actor update for TAR?

- **Modularity**: TAR is applied after standard policy updates
- **Flexibility**: Easy to enable/disable TAR independently
- **Stability**: Separates different learning signals
- **Simplicity**: Minimal changes to base policy classes

### Why L2 norm (with sqrt) in the formula?

- **Proper distance metric**: `||π(s_t) - a_t||` is the Euclidean distance in action space
- **Physical interpretation**: Measures actual "distance" traveled from dataset actions
- **Mathematical correctness**: Aligns with the theoretical derivation
- **Not just squared error**: Unlike BC which uses `Σ ||·||²`, we use `(Σ ||·||)²`

### Why quadratic penalty (Σ||·||)² vs linear Σ||·||?

- **Strong adherence**: Linear penalty `Σ ||π(s_t) - a_t||` treats each step independently
- **Sparse switching**: Quadratic `(Σ ||·||)²` heavily penalizes accumulated deviations
- **Theory**: Penalty scales with k² (number of switches), not k
- **Different from BC**: BC uses `Σ ||·||²`, we use `(Σ ||·||)²` - the squaring is outside!

## Metrics Logged

**TAR-specific:**
- `tar/loss`: Raw TAR loss value
- `tar/avg_cumulative_distance`: Average Σ_t ||π(s_t) - a_t||
- `tar/avg_step_distance`: Average per-step L2 distance
- `tar/max_cumulative_distance`: Maximum trajectory distance
- `loss/tar_weighted`: TAR loss × tar_coef

**Standard metrics:**
- `loss/actor`, `loss/q1`, `loss/q2`: Policy learning losses
- `loss/preference`, `loss/reward_model`: Reward learning losses
- `eval/episode_reward`, `eval/normalized_episode_reward`: Performance

## Hyperparameter Tuning Guide

### tar_coef (TAR strength)
- **Low (0.1-0.5)**: Allows more deviation, higher performance ceiling but less stable
- **Medium (1.0-2.0)**: Balanced trade-off (recommended starting point)
- **High (5.0+)**: Very conservative, almost like BC but with sparse improvements

### traj_batch_size
- Smaller (4-8): Less compute, more variance
- Larger (16-32): More stable gradients, higher compute cost

### tar_clip_max
- Prevents single-step outliers from dominating
- Default 100.0 works well for most tasks
- Increase if actions have large magnitude

## Expected Results

TAR should:
1. **Improve stability**: Less performance variance across seeds
2. **Reduce over-optimization**: Prevents exploitation of reward model errors
3. **Maintain performance**: Only allows valuable deviations from dataset

Trade-off:
- More conservative than base algorithm
- Slightly lower ceiling in some cases
- Much better worst-case performance

## Files Created

```
offlinepbrl/
├── buffer/
│   ├── traj_buffer.py              # NEW: TrajectoryBuffer class
│   └── __init__.py                 # UPDATED: Added TrajectoryBuffer export
├── policy/
│   ├── preference/
│   │   ├── ipl_awac_tar.py        # NEW: IPL-AWAC with TAR
│   │   └── bt_awac_tar.py         # NEW: BT-AWAC with TAR
│   └── __init__.py                # UPDATED: Added new policies
├── policy_trainer/
│   └── mf_policy_trainer.py       # UPDATED: Support TrajectoryBuffer
run_example/
└── gym/
    ├── run_ipl_awac_tar.py        # NEW: IPL-AWAC-TAR example
    └── run_bt_awac_tar.py         # NEW: BT-AWAC-TAR example
```

## Next Steps

1. **Test on D4RL benchmarks**: Run experiments to validate performance
2. **Hyperparameter sensitivity**: Analyze tar_coef and traj_batch_size
3. **Extend to other algorithms**: Apply TAR to IQL, CQL, etc.
4. **Theoretical analysis**: Formal bounds on switching frequency

## References

Based on the theoretical framework described in the paper:
"轨迹依从性正则化：一种抑制离线偏好强化学习中过度拼接的保守方法"
