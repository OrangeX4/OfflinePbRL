import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional
from abc import ABC, abstractmethod
import copy
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


class EnsembleRewardModel(BaseRewardModel):
    """Ensemble of reward models that aggregates predictions"""
    
    def __init__(
        self, 
        base_reward_model: BaseRewardModel, 
        ensemble_num: int,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        
        self.device = torch.device(device)
        self.ensemble_num = ensemble_num
        
        # Create ensemble using random copies
        self.members = nn.ModuleList([
            self._create_random_copy(base_reward_model) 
            for _ in range(ensemble_num)
        ])
    
    def _create_random_copy(self, base_model: BaseRewardModel) -> BaseRewardModel:
        """Create a random copy of the base model"""
        model_copy = copy.deepcopy(base_model)
        random_state_dict = {
            k: torch.randn_like(v)
            for k, v in base_model.state_dict().items()
        }
        model_copy.load_state_dict(random_state_dict)
        model_copy.to(self.device)
        return model_copy
    
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Get predictions from all members
        outputs = []
        for reward_model in self.members:
            output = reward_model(obs, actions)
            outputs.append(output)
        
        # Stack predictions
        stacked_outputs = torch.stack(outputs, dim=0)
        return stacked_outputs
    
    def select_reward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Select reward using the ensemble (returns averaged prediction)"""
        return self.forward(obs, actions).mean(dim=0)