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