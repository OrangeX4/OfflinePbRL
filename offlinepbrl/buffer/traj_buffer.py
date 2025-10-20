import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict


class TrajectoryBuffer:
    """
    A buffer for storing and sampling complete trajectories.
    This is designed for trajectory-level operations like TAR (Trajectory Adherence Regularization).
    """
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        max_trajectory_length: Optional[int] = None,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size  # Maximum number of trajectories
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype
        self.max_trajectory_length = max_trajectory_length  # Maximum length per trajectory (None = dynamic)

        self._ptr = 0
        self._size = 0  # Current number of trajectories

        # Store transitions (will be allocated after loading dataset)
        self.observations = None
        self.next_observations = None
        self.actions = None
        self.rewards = None
        self.terminals = None

        # Trajectory metadata
        self.trajectory_starts = []  # List of trajectory start indices (in transition space)
        self.trajectory_lengths = []  # List of trajectory lengths

        self.device = torch.device(device)

    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        """Load a d4rl-format dataset and automatically detect trajectory boundaries."""
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        # Detect trajectory boundaries
        self._detect_trajectories()
        
        # Update size to number of trajectories
        self._size = len(self.trajectory_starts)
        self._ptr = self._size

    def _detect_trajectories(self) -> None:
        """Detect trajectory boundaries based on terminal flags and observation discontinuities."""
        self.trajectory_starts = []
        self.trajectory_lengths = []

        num_transitions = len(self.observations)
        traj_start = 0
        for i in range(num_transitions):
            # Check if this is the end of a trajectory
            is_terminal = self.terminals[i, 0] > 0.5
            is_last = (i == num_transitions - 1)
            
            # Check for observation discontinuity (next obs != obs[i+1])
            is_discontinuous = False
            if not is_last:
                obs_diff = np.linalg.norm(
                    self.next_observations[i] - self.observations[i + 1]
                )
                is_discontinuous = obs_diff > 1e-6

            if is_terminal or is_discontinuous or is_last:
                traj_len = i - traj_start + 1
                self.trajectory_starts.append(traj_start)
                self.trajectory_lengths.append(traj_len)
                traj_start = i + 1

        print(f"Detected {len(self.trajectory_starts)} trajectories")
        print(f"Average trajectory length: {np.mean(self.trajectory_lengths):.2f}")
        print(f"Min/Max trajectory length: {np.min(self.trajectory_lengths)}/{np.max(self.trajectory_lengths)}")

    def sample_trajectories(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample complete trajectories.
        
        Args:
            batch_size: Number of trajectories to sample
            
        Returns:
            Dictionary containing trajectory data with shape [batch_size, max_traj_len, ...]
            Also includes 'trajectory_lengths' and 'trajectory_mask' for handling variable lengths
        """
        if self._size == 0:
            raise ValueError("No trajectories available. Did you call load_dataset()?")

        # Sample trajectory indices
        traj_indices = np.random.choice(self._size, size=batch_size, replace=True)
        
        # Determine max trajectory length for this batch
        if self.max_trajectory_length is not None:
            max_len = self.max_trajectory_length
        else:
            max_len = max(self.trajectory_lengths[idx] for idx in traj_indices)
        
        # Prepare padded arrays
        obs_batch = np.zeros((batch_size, max_len) + self.obs_shape, dtype=self.obs_dtype)
        next_obs_batch = np.zeros((batch_size, max_len) + self.obs_shape, dtype=self.obs_dtype)
        action_batch = np.zeros((batch_size, max_len, self.action_dim), dtype=self.action_dtype)
        reward_batch = np.zeros((batch_size, max_len, 1), dtype=np.float32)
        terminal_batch = np.zeros((batch_size, max_len, 1), dtype=np.float32)
        mask_batch = np.zeros((batch_size, max_len), dtype=np.float32)
        lengths = np.zeros(batch_size, dtype=np.int32)
        
        # Fill in trajectory data
        for i, traj_idx in enumerate(traj_indices):
            start = self.trajectory_starts[traj_idx]
            length = self.trajectory_lengths[traj_idx]
            
            # Truncate if trajectory is longer than max_len
            actual_length = min(length, max_len)
            end = start + actual_length
            
            obs_batch[i, :actual_length] = self.observations[start:end]
            next_obs_batch[i, :actual_length] = self.next_observations[start:end]
            action_batch[i, :actual_length] = self.actions[start:end]
            reward_batch[i, :actual_length] = self.rewards[start:end]
            terminal_batch[i, :actual_length] = self.terminals[start:end]
            mask_batch[i, :actual_length] = 1.0
            lengths[i] = actual_length
        
        return {
            "observations": torch.tensor(obs_batch).to(self.device),
            "actions": torch.tensor(action_batch).to(self.device),
            "next_observations": torch.tensor(next_obs_batch).to(self.device),
            "terminals": torch.tensor(terminal_batch).to(self.device),
            "rewards": torch.tensor(reward_batch).to(self.device),
            "trajectory_mask": torch.tensor(mask_batch).to(self.device),
            "trajectory_lengths": torch.tensor(lengths).to(self.device),
        }

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample transitions (for compatibility with existing code)."""
        num_transitions = len(self.observations)
        batch_indexes = np.random.randint(0, num_transitions, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }

    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize observations."""
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample_all(self) -> Dict[str, np.ndarray]:
        """Sample all data."""
        num_transitions = len(self.observations)
        return {
            "observations": self.observations[:num_transitions].copy(),
            "actions": self.actions[:num_transitions].copy(),
            "next_observations": self.next_observations[:num_transitions].copy(),
            "terminals": self.terminals[:num_transitions].copy(),
            "rewards": self.rewards[:num_transitions].copy()
        }

    def update_all_rewards(self, rewards: np.ndarray) -> None:
        """Update all rewards."""
        num_transitions = len(self.observations)
        assert len(rewards) == num_transitions
        self.rewards[:num_transitions] = rewards.reshape(-1, 1)

    @property
    def size(self) -> int:
        """Return the number of trajectories (not transitions)."""
        return self._size
    
    @property
    def num_transitions(self) -> int:
        """Return the total number of transitions."""
        return len(self.observations) if self.observations is not None else 0


if __name__ == '__main__':
    # Test TrajectoryBuffer with real d4rl dataset
    print("=" * 80)
    print("Testing TrajectoryBuffer with real d4rl dataset")
    print("=" * 80)
    
    import gym
    import d4rl
    from offlinepbrl.utils.load_dataset import qlearning_dataset
    
    # Create environment and load dataset
    env_name = "hopper-medium-v2"
    print(f"\n1. Loading environment: {env_name}")
    env = gym.make(env_name)
    dataset = qlearning_dataset(env)
    
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    
    print(f"   - Observation shape: {obs_shape}")
    print(f"   - Action dimension: {action_dim}")
    print(f"   - Dataset size: {len(dataset['observations'])}")
    
    # Test 1: Load dataset and detect trajectories
    print("\n2. Testing load_dataset() and trajectory detection...")
    traj_buffer = TrajectoryBuffer(
        buffer_size=10000,  # Max number of trajectories (not transitions)
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        device="cpu"
    )
    traj_buffer.load_dataset(dataset)
    print(f"   ✓ Buffer size after loading: {traj_buffer.size} trajectories")
    print(f"   ✓ Number of transitions: {traj_buffer.num_transitions}")
    print(f"   - Number of trajectories detected: {len(traj_buffer.trajectory_starts)}")
    print(f"   - Average trajectory length: {np.mean(traj_buffer.trajectory_lengths):.2f}")
    print(f"   - Min trajectory length: {np.min(traj_buffer.trajectory_lengths)}")
    print(f"   - Max trajectory length: {np.max(traj_buffer.trajectory_lengths)}")
    
    assert traj_buffer.num_transitions == len(dataset["observations"]), "Number of transitions mismatch!"
    assert len(traj_buffer.trajectory_starts) > 0, "No trajectories detected!"
    
    # Verify trajectories cover all data
    total_traj_length = sum(traj_buffer.trajectory_lengths)
    print(f"   - Total trajectory length sum: {total_traj_length}")
    assert total_traj_length == traj_buffer.num_transitions, "Trajectory coverage mismatch!"
    
    # Test 2: Sample trajectories
    print("\n3. Testing sample_trajectories()...")
    batch_size = 8
    traj_batch = traj_buffer.sample_trajectories(batch_size)
    print(f"   ✓ Sampled {batch_size} trajectories")
    print(f"   - Batch keys: {list(traj_batch.keys())}")
    print(f"   - Observations shape: {traj_batch['observations'].shape}")
    print(f"   - Actions shape: {traj_batch['actions'].shape}")
    print(f"   - Trajectory lengths: {traj_batch['trajectory_lengths'].tolist()}")
    print(f"   - Trajectory mask shape: {traj_batch['trajectory_mask'].shape}")
    
    assert traj_batch['observations'].shape[0] == batch_size, "Batch size mismatch!"
    assert traj_batch['trajectory_mask'].shape[0] == batch_size, "Mask batch size mismatch!"
    
    # Verify mask is correct
    for i in range(batch_size):
        traj_len = traj_batch['trajectory_lengths'][i].item()
        mask = traj_batch['trajectory_mask'][i]
        assert mask[:traj_len].sum() == traj_len, f"Mask incorrect for trajectory {i}!"
        assert mask[traj_len:].sum() == 0, f"Mask has non-zero padding for trajectory {i}!"
    print(f"   ✓ Trajectory masks are correct")
    
    # Test 3: Sample transitions (compatibility mode)
    print("\n4. Testing sample() for transition sampling...")
    trans_batch_size = 256
    trans_batch = traj_buffer.sample(trans_batch_size)
    print(f"   ✓ Sampled {trans_batch_size} transitions")
    print(f"   - Observations shape: {trans_batch['observations'].shape}")
    print(f"   - Actions shape: {trans_batch['actions'].shape}")
    assert trans_batch['observations'].shape[0] == trans_batch_size, "Transition batch size mismatch!"
    assert trans_batch['observations'].shape[1:] == obs_shape, "Observation shape mismatch!"
    
    # Test 4: Sample all
    print("\n5. Testing sample_all()...")
    all_data = traj_buffer.sample_all()
    print(f"   ✓ Sampled all data")
    print(f"   - Total samples: {len(all_data['observations'])}")
    assert len(all_data['observations']) == traj_buffer.num_transitions, "Sample all size mismatch!"
    
    # Test 5: Normalize observations
    print("\n6. Testing normalize_obs()...")
    norm_buffer = TrajectoryBuffer(
        buffer_size=10000,  # Max number of trajectories
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        device="cpu"
    )
    norm_buffer.load_dataset(dataset)
    obs_mean, obs_std = norm_buffer.normalize_obs()
    print(f"   ✓ Normalized observations")
    print(f"   - Obs mean shape: {obs_mean.shape}")
    print(f"   - Obs std shape: {obs_std.shape}")
    
    normalized_mean = norm_buffer.observations.mean()
    normalized_std = norm_buffer.observations.std()
    print(f"   - Normalized data mean: {normalized_mean:.4f} (should be ~0)")
    print(f"   - Normalized data std: {normalized_std:.4f} (should be ~1)")
    assert abs(normalized_mean) < 0.1, "Normalization mean not close to 0!"
    
    # Test 6: Update rewards
    print("\n7. Testing update_all_rewards()...")
    new_rewards = np.random.randn(traj_buffer.num_transitions)
    traj_buffer.update_all_rewards(new_rewards)
    print(f"   ✓ Updated all rewards")
    print(f"   - New rewards shape: {traj_buffer.rewards.shape}")
    assert np.allclose(traj_buffer.rewards.flatten(), new_rewards), "Update rewards failed!"
    
    # Test 7: Verify trajectory boundaries
    print("\n8. Verifying trajectory boundary detection...")
    # Check that trajectories don't overlap
    for i in range(len(traj_buffer.trajectory_starts) - 1):
        start_i = traj_buffer.trajectory_starts[i]
        len_i = traj_buffer.trajectory_lengths[i]
        start_next = traj_buffer.trajectory_starts[i + 1]
        assert start_i + len_i == start_next, f"Trajectory {i} and {i+1} don't connect properly!"
    print(f"   ✓ All trajectory boundaries are valid")
    
    # Test 8: Sample specific trajectories and verify they're complete
    print("\n9. Verifying sampled trajectories are complete...")
    traj_batch = traj_buffer.sample_trajectories(3)
    for i in range(3):
        traj_len = traj_batch['trajectory_lengths'][i].item()
        # Check that terminal flag is set at the end
        terminals = traj_batch['terminals'][i, :traj_len]
        # At least one terminal should be set in the trajectory
        assert terminals.sum() > 0, f"Trajectory {i} has no terminal flags!"
    print(f"   ✓ Sampled trajectories have valid terminal flags")
    
    # Test 9: Test with different d4rl environment
    print("\n10. Testing with different environment (walker2d-medium-v2)...")
    try:
        env2 = gym.make("walker2d-medium-v2")
        dataset2 = qlearning_dataset(env2)
        obs_shape2 = env2.observation_space.shape
        action_dim2 = np.prod(env2.action_space.shape)
        
        traj_buffer2 = TrajectoryBuffer(
            buffer_size=10000,  # Max number of trajectories
            obs_shape=obs_shape2,
            obs_dtype=np.float32,
            action_dim=action_dim2,
            action_dtype=np.float32,
            device="cpu"
        )
        traj_buffer2.load_dataset(dataset2)
        print(f"   ✓ Loaded walker2d dataset")
        print(f"   - Number of trajectories: {len(traj_buffer2.trajectory_starts)}")
        print(f"   - Average trajectory length: {np.mean(traj_buffer2.trajectory_lengths):.2f}")
        
        # Sample trajectories
        traj_batch2 = traj_buffer2.sample_trajectories(5)
        print(f"   ✓ Sampled 5 trajectories from walker2d")
        print(f"   - Trajectory lengths: {traj_batch2['trajectory_lengths'].tolist()}")
    except Exception as e:
        print(f"   ⚠ Could not test walker2d: {e}")
    
    # Test 10: Test with max_trajectory_length parameter
    print("\n11. Testing max_trajectory_length parameter...")
    max_traj_len = 100
    traj_buffer_fixed = TrajectoryBuffer(
        buffer_size=10000,
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        max_trajectory_length=max_traj_len,
        device="cpu"
    )
    traj_buffer_fixed.load_dataset(dataset)
    print(f"   ✓ Created buffer with max_trajectory_length={max_traj_len}")
    
    # Sample trajectories and verify they respect max length
    traj_batch_fixed = traj_buffer_fixed.sample_trajectories(8)
    print(f"   - Sampled trajectory shape: {traj_batch_fixed['observations'].shape}")
    print(f"   - Trajectory lengths: {traj_batch_fixed['trajectory_lengths'].tolist()}")
    assert traj_batch_fixed['observations'].shape[1] == max_traj_len, "Max trajectory length not respected!"
    assert all(length <= max_traj_len for length in traj_batch_fixed['trajectory_lengths'].tolist()), \
        "Some trajectories exceed max_trajectory_length!"
    print(f"   ✓ All trajectories respect max_trajectory_length")
    
    # Test dynamic max length (None)
    print("\n12. Testing dynamic max_trajectory_length (None)...")
    traj_buffer_dynamic = TrajectoryBuffer(
        buffer_size=10000,
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        max_trajectory_length=None,  # Dynamic
        device="cpu"
    )
    traj_buffer_dynamic.load_dataset(dataset)
    traj_batch_dynamic = traj_buffer_dynamic.sample_trajectories(8)
    dynamic_max_len = traj_batch_dynamic['observations'].shape[1]
    print(f"   ✓ Dynamic max length: {dynamic_max_len}")
    print(f"   - Trajectory lengths: {traj_batch_dynamic['trajectory_lengths'].tolist()}")
    assert dynamic_max_len == max(traj_batch_dynamic['trajectory_lengths'].tolist()), \
        "Dynamic max length should match the longest trajectory in batch!"
    print(f"   ✓ Dynamic max_trajectory_length works correctly")
    
    print("\n" + "=" * 80)
    print("All TrajectoryBuffer tests passed! ✓")
    print("=" * 80)
