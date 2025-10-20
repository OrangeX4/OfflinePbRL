import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
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

        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_trajectory(self, batch_size: int, segment_size: int) -> Dict[str, torch.Tensor]:
        """Sample trajectory segments for APPO algorithm.
        
        Args:
            batch_size: Number of trajectory pairs to sample (each pair will have 2*batch_size segments)
            segment_size: Length of each trajectory segment
            
        Returns:
            Dictionary containing sampled trajectory data
        """
        # Assume each trajectory has 500 steps (following local/APPO implementation)
        traj_length = 500
        num_traj = self._size // traj_length
        
        if num_traj < 1:
            raise ValueError(f"Not enough data for trajectory sampling. Need at least {traj_length} samples, got {self._size}")
        
        # Sample trajectory pairs (2 * batch_size segments total)
        traj_indices = np.random.choice(num_traj, 2 * batch_size, replace=True)
        
        # For each trajectory, sample a random starting point for the segment
        start_indices = []
        for traj_idx in traj_indices:
            traj_start = traj_idx * traj_length
            # Make sure we don't go beyond trajectory boundaries
            max_start = traj_start + traj_length - segment_size
            start_pos = np.random.randint(traj_start, max_start)
            segment_indices = list(range(start_pos, start_pos + segment_size))
            start_indices.extend(segment_indices)
        
        return {
            "observations": torch.tensor(self.observations[start_indices]).to(self.device),
            "actions": torch.tensor(self.actions[start_indices]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[start_indices]).to(self.device),
            "terminals": torch.tensor(self.terminals[start_indices]).to(self.device),
            "rewards": torch.tensor(self.rewards[start_indices]).to(self.device),
            # Metadata to reconstruct segment grouping
            "segment_size": torch.tensor(segment_size, device=self.device),
            "pair_batch_size": torch.tensor(batch_size, device=self.device)
        }

    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }

    def update_all_rewards(self, rewards: np.ndarray) -> None:
        assert len(rewards) == self._size
        self.rewards[:self._size] = rewards.reshape(-1, 1)


if __name__ == '__main__':
    # Test ReplayBuffer with real d4rl dataset
    print("=" * 80)
    print("Testing ReplayBuffer with real d4rl dataset")
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
    
    # Test 1: Load dataset
    print("\n2. Testing load_dataset()...")
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        device="cpu"
    )
    buffer.load_dataset(dataset)
    print(f"   ✓ Buffer size after loading: {buffer._size}")
    assert buffer._size == len(dataset["observations"]), "Buffer size mismatch!"
    
    # Test 2: Sample batch
    print("\n3. Testing sample()...")
    batch_size = 256
    batch = buffer.sample(batch_size)
    print(f"   ✓ Sampled batch size: {batch_size}")
    print(f"   - Batch keys: {list(batch.keys())}")
    print(f"   - Observations shape: {batch['observations'].shape}")
    print(f"   - Actions shape: {batch['actions'].shape}")
    assert batch['observations'].shape[0] == batch_size, "Batch size mismatch!"
    assert batch['observations'].shape[1:] == obs_shape, "Observation shape mismatch!"
    assert batch['actions'].shape == (batch_size, action_dim), "Action shape mismatch!"
    
    # Test 3: Sample trajectory (for APPO)
    print("\n4. Testing sample_trajectory()...")
    try:
        traj_batch_size = 4
        segment_size = 50
        traj_batch = buffer.sample_trajectory(traj_batch_size, segment_size)
        expected_size = 2 * traj_batch_size * segment_size
        print(f"   ✓ Sampled trajectory batch")
        print(f"   - Pair batch size: {traj_batch_size}")
        print(f"   - Segment size: {segment_size}")
        print(f"   - Total observations: {traj_batch['observations'].shape[0]}")
        print(f"   - Expected total: {expected_size}")
        assert traj_batch['observations'].shape[0] == expected_size, "Trajectory batch size mismatch!"
    except ValueError as e:
        print(f"   ⚠ Trajectory sampling failed (expected for small datasets): {e}")
    
    # Test 4: Sample all
    print("\n5. Testing sample_all()...")
    all_data = buffer.sample_all()
    print(f"   ✓ Sampled all data")
    print(f"   - Total samples: {len(all_data['observations'])}")
    assert len(all_data['observations']) == buffer._size, "Sample all size mismatch!"
    
    # Test 5: Add single transition
    print("\n6. Testing add()...")
    original_size = buffer._size
    obs = dataset['observations'][0]
    next_obs = dataset['next_observations'][0]
    action = dataset['actions'][0]
    reward = dataset['rewards'][0]
    terminal = dataset['terminals'][0]
    
    # Create a new buffer with room for more data
    small_buffer = ReplayBuffer(
        buffer_size=100,
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        device="cpu"
    )
    small_buffer.add(obs, next_obs, action, reward, terminal)
    print(f"   ✓ Added single transition")
    print(f"   - Buffer size: {small_buffer._size}")
    assert small_buffer._size == 1, "Add single transition failed!"
    
    # Test 6: Add batch
    print("\n7. Testing add_batch()...")
    batch_size = 10
    small_buffer.add_batch(
        dataset['observations'][:batch_size],
        dataset['next_observations'][:batch_size],
        dataset['actions'][:batch_size],
        dataset['rewards'][:batch_size].reshape(-1, 1),
        dataset['terminals'][:batch_size].reshape(-1, 1)
    )
    print(f"   ✓ Added batch of {batch_size} transitions")
    print(f"   - Buffer size: {small_buffer._size}")
    assert small_buffer._size == 1 + batch_size, "Add batch failed!"
    
    # Test 7: Normalize observations
    print("\n8. Testing normalize_obs()...")
    norm_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
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
    print(f"   - Mean value: {obs_mean.mean():.4f}")
    print(f"   - Std mean: {obs_std.mean():.4f}")
    
    # Verify normalization
    normalized_mean = norm_buffer.observations.mean()
    normalized_std = norm_buffer.observations.std()
    print(f"   - Normalized data mean: {normalized_mean:.4f} (should be ~0)")
    print(f"   - Normalized data std: {normalized_std:.4f} (should be ~1)")
    assert abs(normalized_mean) < 0.1, "Normalization mean not close to 0!"
    
    # Test 8: Update rewards
    print("\n9. Testing update_all_rewards()...")
    new_rewards = np.random.randn(buffer._size)
    buffer.update_all_rewards(new_rewards)
    print(f"   ✓ Updated all rewards")
    print(f"   - New rewards shape: {buffer.rewards.shape}")
    assert np.allclose(buffer.rewards[:buffer._size].flatten(), new_rewards), "Update rewards failed!"
    
    print("\n" + "=" * 80)
    print("All ReplayBuffer tests passed! ✓")
    print("=" * 80)