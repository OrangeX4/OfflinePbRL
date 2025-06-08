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
