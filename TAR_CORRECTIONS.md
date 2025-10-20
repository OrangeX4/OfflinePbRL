# TAR Implementation - Key Corrections

## Summary of Changes

Based on feedback, two critical corrections were made to the TAR implementation:

### 1. ✅ Corrected TAR Formula

**Before (Incorrect):**
```
L_TAR = E_τ [ (Σ_t ||π(s_t) - a_t||²)² ]
```
This was summing squared distances, then squaring again.

**After (Correct):**
```
L_TAR = E_τ [ (Σ_t ||π(s_t) - a_t||)² ]
```
Where `||·||` is the L2 norm (Euclidean distance with sqrt).

**Implementation:**
```python
# Compute per-step L2 distances (with sqrt)
step_distances = torch.norm(policy_actions - action_flat, p=2, dim=-1)

# Sum distances across trajectory
traj_cumulative_distance = step_distances.sum(dim=1)

# Square the sum
tar_loss = (traj_cumulative_distance.pow(2)).mean()
```

**Why this matters:**
- The correct formula measures the actual Euclidean distance in action space
- Step distance: `d_t = sqrt(Σ_i (π_i - a_i)²)` (proper L2 norm)
- Trajectory distance: `D(τ) = Σ_t d_t` (sum of distances)
- TAR loss: `(D(τ))²` (square of the sum)

**Key difference from BC:**
- BC: `Σ_t ||π(s_t) - a_t||²` (sum of squared distances)
- TAR: `(Σ_t ||π(s_t) - a_t||)²` (squared sum of distances)
- The squaring happens **outside** the sum in TAR!

### 2. ✅ Modified MFPolicyTrainer to Support TrajectoryBuffer

**Before:**
- Example scripts had custom training loops
- Code duplication between IPL and BT examples
- Not using the standard trainer infrastructure

**After:**
- MFPolicyTrainer automatically detects TrajectoryBuffer
- Seamless integration with existing trainer
- No custom training loops needed

**Changes to `mf_policy_trainer.py`:**

1. **Import TrajectoryBuffer:**
```python
from offlinepbrl.buffer import ReplayBuffer, PrefBuffer, TrajectoryBuffer
```

2. **Detect buffer type in __init__:**
```python
self._supports_traj_sampling = isinstance(buffer, TrajectoryBuffer)
```

3. **Conditional trajectory sampling in train loop:**
```python
if self._traj_batch_size is not None:
    if self._supports_traj_sampling:
        # TrajectoryBuffer: sample complete trajectories
        traj_batch = self.buffer.sample_trajectories(self._traj_batch_size)
        batch["trajectory"] = traj_batch
    elif self._segment_size is not None:
        # ReplayBuffer with segment sampling (for APPO)
        traj_batch = self.buffer.sample_trajectory(self._traj_batch_size, self._segment_size)
        batch["traj"] = traj_batch
```

**Benefits:**
- Clean separation of concerns
- Reusable across all TAR variants
- Maintains backward compatibility with APPO
- Standard trainer handles all logging and evaluation

**Example scripts now use standard trainer:**
```python
policy_trainer = MFPolicyTrainer(
    policy=policy,
    eval_env=env,
    buffer=traj_buffer,  # TrajectoryBuffer instead of ReplayBuffer
    logger=logger,
    epoch=args.epoch,
    step_per_epoch=args.step_per_epoch,
    batch_size=args.batch_size,
    eval_episodes=args.eval_episodes,
    lr_scheduler=lr_scheduler,
    pref_buffer=pref_buffer,
    pref_batch_size=args.pref_batch_size,
    eval_freq=args.eval_freq,
    traj_batch_size=args.traj_batch_size,  # Enable trajectory sampling
)

policy_trainer.train()
```

## Files Modified

### Core Implementation
1. `offlinepbrl/policy/preference/ipl_awac_tar.py`
   - Corrected `compute_tar_loss()` to use L2 norm with sqrt
   - Updated metric names to reflect "distance" not "deviation"

2. `offlinepbrl/policy/preference/bt_awac_tar.py`
   - Same TAR formula correction as IPL-AWAC-TAR

3. `offlinepbrl/policy_trainer/mf_policy_trainer.py`
   - Added TrajectoryBuffer import
   - Added `_supports_traj_sampling` flag
   - Conditional trajectory sampling based on buffer type

### Example Scripts
4. `run_example/gym/run_ipl_awac_tar.py`
   - Removed custom training loop
   - Now uses standard MFPolicyTrainer
   - Much simpler and cleaner

5. `run_example/gym/run_bt_awac_tar.py`
   - Same simplification as IPL example
   - Added missing MFPolicyTrainer import

### Documentation
6. `TAR_README.md`
   - Updated formula documentation
   - Added explanation of L2 norm usage
   - Clarified difference from BC
   - Documented MFPolicyTrainer changes

## Verification

All files pass error checking:
```
✅ offlinepbrl/buffer/traj_buffer.py - No errors
✅ offlinepbrl/policy/preference/ipl_awac_tar.py - No errors
✅ offlinepbrl/policy/preference/bt_awac_tar.py - No errors
✅ offlinepbrl/policy_trainer/mf_policy_trainer.py - No errors
✅ run_example/gym/run_ipl_awac_tar.py - No errors
✅ run_example/gym/run_bt_awac_tar.py - No errors
```

## Testing

To test the corrected implementation:

```bash
# IPL-AWAC-TAR
python run_example/gym/run_ipl_awac_tar.py \
    --task walker2d-medium-expert-v2 \
    --tar_coef 1.0 \
    --traj_batch_size 8

# BT-AWAC-TAR  
python run_example/gym/run_bt_awac_tar.py \
    --task walker2d-medium-expert-v2 \
    --tar_coef 1.0 \
    --traj_batch_size 8
```

## Impact of Corrections

### Mathematical Impact
The corrected formula should:
- Provide stronger regularization (L2 norm is typically larger than individual components)
- More accurately measure trajectory-level deviation
- Better align with theoretical derivations

### Code Impact
The MFPolicyTrainer modifications:
- Eliminate code duplication
- Make TAR easier to extend to other algorithms (IQL, CQL, etc.)
- Maintain clean separation between buffer, policy, and trainer
- Future-proof for additional trajectory-level methods

## Next Steps

1. ✅ Formula corrected to use proper L2 norm
2. ✅ Trainer modified to support TrajectoryBuffer
3. ✅ Example scripts simplified
4. ⏭️ Run experiments to validate performance
5. ⏭️ Consider extending TAR to IQL, CQL variants
