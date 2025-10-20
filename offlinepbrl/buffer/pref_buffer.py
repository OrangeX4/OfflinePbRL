import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict


# Preference Buffer for storing pairs of trajectories with labels
class PrefBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        max_traj_len: int = 200,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype
        self.max_traj_len = max_traj_len

        self._ptr = 0
        self._size = 0

        # Trajectory 1
        self.observations_1 = np.zeros((self._max_size, self.max_traj_len) + self.obs_shape, dtype=obs_dtype)
        self.actions_1 = np.zeros((self._max_size, self.max_traj_len, self.action_dim), dtype=action_dtype)
        self.rewards_1 = np.zeros((self._max_size, self.max_traj_len), dtype=np.float32)
        self.timesteps_1 = np.zeros((self._max_size, self.max_traj_len), dtype=np.int32)
        self.terminals_1 = np.zeros((self._max_size, self.max_traj_len), dtype=np.float32)
        
        # Trajectory 2
        self.observations_2 = np.zeros((self._max_size, self.max_traj_len) + self.obs_shape, dtype=obs_dtype)
        self.actions_2 = np.zeros((self._max_size, self.max_traj_len, self.action_dim), dtype=action_dtype)
        self.rewards_2 = np.zeros((self._max_size, self.max_traj_len), dtype=np.float32)
        self.timesteps_2 = np.zeros((self._max_size, self.max_traj_len), dtype=np.int32)
        self.terminals_2 = np.zeros((self._max_size, self.max_traj_len), dtype=np.float32)
        
        # Start indices and labels
        self.start_indices_1 = np.zeros((self._max_size,), dtype=np.int32)
        self.start_indices_2 = np.zeros((self._max_size,), dtype=np.int32)
        self.labels = np.zeros((self._max_size, 2), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs_1: np.ndarray,
        action_1: np.ndarray,
        reward_1: np.ndarray,
        timestep_1: np.ndarray,
        terminal_1: np.ndarray,
        obs_2: np.ndarray,
        action_2: np.ndarray,
        reward_2: np.ndarray,
        timestep_2: np.ndarray,
        terminal_2: np.ndarray,
        start_idx_1: int,
        start_idx_2: int,
        label: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations_1[self._ptr] = np.array(obs_1).copy()
        self.actions_1[self._ptr] = np.array(action_1).copy()
        self.rewards_1[self._ptr] = np.array(reward_1).copy()
        self.timesteps_1[self._ptr] = np.array(timestep_1).copy()
        self.terminals_1[self._ptr] = np.array(terminal_1).copy()
        
        self.observations_2[self._ptr] = np.array(obs_2).copy()
        self.actions_2[self._ptr] = np.array(action_2).copy()
        self.rewards_2[self._ptr] = np.array(reward_2).copy()
        self.timesteps_2[self._ptr] = np.array(timestep_2).copy()
        self.terminals_2[self._ptr] = np.array(terminal_2).copy()
        
        self.start_indices_1[self._ptr] = start_idx_1
        self.start_indices_2[self._ptr] = start_idx_2
        self.labels[self._ptr] = np.array(label).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        # Extract trajectory 1 data
        self.observations_1 = np.array(dataset["observations"], dtype=self.obs_dtype)
        self.actions_1 = np.array(dataset["actions"], dtype=self.action_dtype)
        self.rewards_1 = np.array(dataset["rewards"], dtype=np.float32)
        self.timesteps_1 = np.array(dataset["timestep"], dtype=np.int32)
        self.start_indices_1 = np.array(dataset["start_indices"], dtype=np.int32)
        
        # Extract trajectory 2 data
        self.observations_2 = np.array(dataset["observations_2"], dtype=self.obs_dtype)
        self.actions_2 = np.array(dataset["actions_2"], dtype=self.action_dtype)
        self.rewards_2 = np.array(dataset["rewards_2"], dtype=np.float32)
        self.timesteps_2 = np.array(dataset["timestep_2"], dtype=np.int32)
        self.start_indices_2 = np.array(dataset["start_indices_2"], dtype=np.int32)
        
        # Extract labels
        self.labels = np.array(dataset["labels"], dtype=np.float32)
        
        # Compute terminals from timesteps (terminal when timestep resets or at end)
        self.terminals_1 = self._compute_terminals(self.timesteps_1)
        self.terminals_2 = self._compute_terminals(self.timesteps_2)

        self._ptr = len(self.observations_1)
        self._size = len(self.observations_1)
    
    def _compute_terminals(self, timesteps: np.ndarray) -> np.ndarray:
        """Compute terminal flags from timesteps"""
        terminals = np.zeros_like(timesteps, dtype=np.float32)
        for i in range(len(timesteps)):
            for j in range(len(timesteps[i]) - 1):
                # Terminal if next timestep is 0 or decreases
                if timesteps[i, j + 1] <= timesteps[i, j]:
                    terminals[i, j] = 1.0
            # Last timestep is always terminal
            terminals[i, -1] = 1.0
        return terminals

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "obs_1": torch.tensor(self.observations_1[batch_indexes]).to(self.device),
            "action_1": torch.tensor(self.actions_1[batch_indexes]).to(self.device),
            "reward_1": torch.tensor(self.rewards_1[batch_indexes]).to(self.device),
            "timestep_1": torch.tensor(self.timesteps_1[batch_indexes]).to(self.device),
            "terminal_1": torch.tensor(self.terminals_1[batch_indexes]).to(self.device),
            
            "obs_2": torch.tensor(self.observations_2[batch_indexes]).to(self.device),
            "action_2": torch.tensor(self.actions_2[batch_indexes]).to(self.device),
            "reward_2": torch.tensor(self.rewards_2[batch_indexes]).to(self.device),
            "timestep_2": torch.tensor(self.timesteps_2[batch_indexes]).to(self.device),
            "terminal_2": torch.tensor(self.terminals_2[batch_indexes]).to(self.device),
            
            "start_indices_1": torch.tensor(self.start_indices_1[batch_indexes]).to(self.device),
            "start_indices_2": torch.tensor(self.start_indices_2[batch_indexes]).to(self.device),
            "label": torch.tensor(self.labels[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "obs_1": self.observations_1[:self._size].copy(),
            "action_1": self.actions_1[:self._size].copy(),
            "reward_1": self.rewards_1[:self._size].copy(),
            "timestep_1": self.timesteps_1[:self._size].copy(),
            "terminal_1": self.terminals_1[:self._size].copy(),
            
            "obs_2": self.observations_2[:self._size].copy(),
            "action_2": self.actions_2[:self._size].copy(),
            "reward_2": self.rewards_2[:self._size].copy(),
            "timestep_2": self.timesteps_2[:self._size].copy(),
            "terminal_2": self.terminals_2[:self._size].copy(),
            
            "start_indices_1": self.start_indices_1[:self._size].copy(),
            "start_indices_2": self.start_indices_2[:self._size].copy(),
            "label": self.labels[:self._size].copy()
        }

    @property
    def size(self) -> int:
        return self._size
    
    @property
    def max_size(self) -> int:
        return self._max_size


if __name__ == '__main__':
    # Test PrefBuffer with real d4rl dataset and RLHF labels
    print("=" * 80)
    print("Testing PrefBuffer with real d4rl dataset and RLHF labels")
    print("=" * 80)
    
    import gym
    import d4rl
    from offlinepbrl.utils.load_dataset import qlearning_dataset, load_rlhf_dataset
    
    # Create environment and load datasets
    env_name = "hopper-medium-v2"
    print(f"\n1. Loading environment: {env_name}")
    env = gym.make(env_name)
    dataset = qlearning_dataset(env)
    
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    
    print(f"   - Observation shape: {obs_shape}")
    print(f"   - Action dimension: {action_dim}")
    print(f"   - Dataset size: {len(dataset['observations'])}")
    
    # Try to load RLHF dataset
    print("\n2. Loading RLHF dataset...")
    try:
        rlhf_dataset = load_rlhf_dataset(
            env, 
            dataset,
            fake_label=True,  # Use scripted labels for testing
            num_query=2000,
            len_query=200
        )
        print(f"   ✓ RLHF dataset loaded")
        print(f"   - Keys: {list(rlhf_dataset.keys())}")
        print(f"   - Observations shape: {rlhf_dataset['observations'].shape}")
        print(f"   - Observations_2 shape: {rlhf_dataset['observations_2'].shape}")
        print(f"   - Labels shape: {rlhf_dataset['labels'].shape}")
        
        # Test 1: Load dataset
        print("\n3. Testing load_dataset()...")
        pref_buffer = PrefBuffer(
            buffer_size=len(rlhf_dataset["observations"]),
            obs_shape=obs_shape,
            obs_dtype=np.float32,
            action_dim=action_dim,
            action_dtype=np.float32,
            max_traj_len=rlhf_dataset["observations"].shape[1],
            device="cpu"
        )
        pref_buffer.load_dataset(rlhf_dataset)
        print(f"   ✓ Buffer size after loading: {pref_buffer.size}")
        print(f"   - Max trajectory length: {pref_buffer.max_traj_len}")
        assert pref_buffer.size == len(rlhf_dataset["observations"]), "Buffer size mismatch!"
        
        # Test 2: Sample batch
        print("\n4. Testing sample()...")
        batch_size = 8
        batch = pref_buffer.sample(batch_size)
        print(f"   ✓ Sampled batch size: {batch_size}")
        print(f"   - Batch keys: {list(batch.keys())}")
        print(f"   - Obs_1 shape: {batch['obs_1'].shape}")
        print(f"   - Obs_2 shape: {batch['obs_2'].shape}")
        print(f"   - Action_1 shape: {batch['action_1'].shape}")
        print(f"   - Action_2 shape: {batch['action_2'].shape}")
        print(f"   - Label shape: {batch['label'].shape}")
        
        assert batch['obs_1'].shape[0] == batch_size, "Batch size mismatch!"
        assert batch['obs_1'].shape[1] == pref_buffer.max_traj_len, "Trajectory length mismatch!"
        assert batch['label'].shape == (batch_size, 2), "Label shape mismatch!"
        
        # Test 3: Verify labels sum to 1 (or 1.0 for ties)
        print("\n5. Verifying label validity...")
        label_sums = batch['label'].sum(dim=1)
        print(f"   - Label sums (should be 1.0): min={label_sums.min():.2f}, max={label_sums.max():.2f}")
        assert torch.allclose(label_sums, torch.ones_like(label_sums), atol=1e-5), "Labels don't sum to 1!"
        print(f"   ✓ All labels valid")
        
        # Test 4: Sample all
        print("\n6. Testing sample_all()...")
        all_data = pref_buffer.sample_all()
        print(f"   ✓ Sampled all data")
        print(f"   - Total samples: {len(all_data['obs_1'])}")
        print(f"   - Keys: {list(all_data.keys())}")
        assert len(all_data['obs_1']) == pref_buffer.size, "Sample all size mismatch!"
        
        # Test 5: Add single preference pair
        print("\n7. Testing add()...")
        small_buffer = PrefBuffer(
            buffer_size=10,
            obs_shape=obs_shape,
            obs_dtype=np.float32,
            action_dim=action_dim,
            action_dtype=np.float32,
            max_traj_len=50,
            device="cpu"
        )
        
        # Create dummy trajectory pair
        traj_len = 50
        obs_1 = np.random.randn(traj_len, *obs_shape).astype(np.float32)
        action_1 = np.random.randn(traj_len, action_dim).astype(np.float32)
        reward_1 = np.random.randn(traj_len).astype(np.float32)
        timestep_1 = np.arange(traj_len, dtype=np.int32)
        terminal_1 = np.zeros(traj_len, dtype=np.float32)
        terminal_1[-1] = 1.0
        
        obs_2 = np.random.randn(traj_len, *obs_shape).astype(np.float32)
        action_2 = np.random.randn(traj_len, action_dim).astype(np.float32)
        reward_2 = np.random.randn(traj_len).astype(np.float32)
        timestep_2 = np.arange(traj_len, dtype=np.int32)
        terminal_2 = np.zeros(traj_len, dtype=np.float32)
        terminal_2[-1] = 1.0
        
        label = np.array([1.0, 0.0])  # Prefer trajectory 1
        
        small_buffer.add(
            obs_1, action_1, reward_1, timestep_1, terminal_1,
            obs_2, action_2, reward_2, timestep_2, terminal_2,
            0, 0, label
        )
        print(f"   ✓ Added single preference pair")
        print(f"   - Buffer size: {small_buffer.size}")
        assert small_buffer.size == 1, "Add single pair failed!"
        
        # Test 6: Verify terminal computation
        print("\n8. Testing _compute_terminals()...")
        test_timesteps = np.array([
            [0, 1, 2, 3, 4],
            [0, 1, 0, 1, 2],  # Reset at index 2
        ])
        terminals = pref_buffer._compute_terminals(test_timesteps)
        print(f"   ✓ Terminal computation")
        print(f"   - Test timesteps:\n{test_timesteps}")
        print(f"   - Computed terminals:\n{terminals}")
        # Last timestep should always be terminal
        assert terminals[0, -1] == 1.0, "Last timestep not terminal!"
        assert terminals[1, -1] == 1.0, "Last timestep not terminal!"
        # Check reset detection
        assert terminals[1, 1] == 1.0, "Reset not detected!"
        
        print("\n" + "=" * 80)
        print("All PrefBuffer tests passed! ✓")
        print("=" * 80)
        
    except ValueError as e:
        print(f"\n⚠ Warning: Could not load RLHF dataset: {e}")
        print("This is expected if the label files are not available.")
        print("PrefBuffer structure is correct, but full testing requires label data.")
        
        # Still do basic buffer tests without real data
        print("\n3. Testing basic PrefBuffer functionality without real labels...")
        pref_buffer = PrefBuffer(
            buffer_size=10,
            obs_shape=obs_shape,
            obs_dtype=np.float32,
            action_dim=action_dim,
            action_dtype=np.float32,
            max_traj_len=50,
            device="cpu"
        )
        
        # Add a dummy preference pair
        traj_len = 50
        obs_1 = np.random.randn(traj_len, *obs_shape).astype(np.float32)
        action_1 = np.random.randn(traj_len, action_dim).astype(np.float32)
        reward_1 = np.random.randn(traj_len).astype(np.float32)
        timestep_1 = np.arange(traj_len, dtype=np.int32)
        terminal_1 = np.zeros(traj_len, dtype=np.float32)
        terminal_1[-1] = 1.0
        
        obs_2 = np.random.randn(traj_len, *obs_shape).astype(np.float32)
        action_2 = np.random.randn(traj_len, action_dim).astype(np.float32)
        reward_2 = np.random.randn(traj_len).astype(np.float32)
        timestep_2 = np.arange(traj_len, dtype=np.int32)
        terminal_2 = np.zeros(traj_len, dtype=np.float32)
        terminal_2[-1] = 1.0
        
        label = np.array([1.0, 0.0])
        
        pref_buffer.add(
            obs_1, action_1, reward_1, timestep_1, terminal_1,
            obs_2, action_2, reward_2, timestep_2, terminal_2,
            0, 0, label
        )
        
        print(f"   ✓ Added dummy preference pair")
        print(f"   - Buffer size: {pref_buffer.size}")
        assert pref_buffer.size == 1, "Basic add failed!"
        
        # Sample
        batch = pref_buffer.sample(1)
        print(f"   ✓ Sampled batch")
        assert batch['obs_1'].shape == (1, 50, *obs_shape), "Sample shape mismatch!"
        
        print("\n" + "=" * 80)
        print("Basic PrefBuffer tests passed! ✓")
        print("(Full tests require RLHF label data)")
        print("=" * 80)
