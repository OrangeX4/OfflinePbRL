import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional
from abc import ABC, abstractmethod
from offlinepbrl.nets.activation import get_activation


class BaseRewardModel(nn.Module, ABC):
    """Abstract base class for reward models"""
    
    @abstractmethod
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        pass
    
    @abstractmethod
    def select_reward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        pass


class RewardModel(BaseRewardModel):
    def __init__(
        self, 
        backbone: nn.Module, 
        activation: str = "tanh",
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 1).to(device)
        
        activation_fn = get_activation(activation)
        self.activation = activation_fn()

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        rewards = self.last(logits)
        rewards = self.activation(rewards)
        return rewards

    def select_reward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Select reward using the reward model"""
        return self.forward(obs, actions)


class GaussianRewardModel(BaseRewardModel):
    """Gaussian reward model that outputs both mean and log variance"""
    
    def __init__(
        self, 
        backbone: nn.Module, 
        activation: str = "identity",
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        
        # Output both mean and logvar
        self.last = nn.Linear(latent_dim, 2).to(device)
        
        activation_fn = get_activation(activation)
        self.activation = activation_fn()

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            obs = torch.cat([obs, actions], dim=1)
        logits = self.backbone(obs)
        output = self.last(logits)  # [batch_size, 2] where [:, 0] = mean, [:, 1] = logvar
        
        # Apply activation only to mean, keep logvar unbounded
        if self.activation is not None:
            output = torch.cat([
                self.activation(output[..., 0:1]),  # mean with activation
                output[..., 1:2]  # logvar without activation
            ], dim=-1)
        
        return output

    def select_reward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Select reward using the reward model (returns mean only for policy learning)"""
        output = self.forward(obs, actions)
        return output[..., 0:1]  # return mean only