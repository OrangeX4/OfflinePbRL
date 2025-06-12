import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import numpy as np


class BTWrapper:
    """
    Bradley-Terry wrapper for adding preference learning to any base policy
    """

    def __init__(
        self,
        base_policy: Optional[Any] = None,
        reward_model: nn.Module = None,
        reward_model_optim: torch.optim.Optimizer = None,
        reward_reg: float = 0.0,
        rm_stop_epoch: Optional[int] = None,
        policy_start_epoch: Optional[int] = None,
    ) -> None:
        self.base_policy = base_policy
        self.reward_model = reward_model
        self.reward_model_optim = reward_model_optim
        self.reward_reg = reward_reg
        self.rm_stop_epoch = rm_stop_epoch
        self.policy_start_epoch = policy_start_epoch
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        
        # Handle base_policy=None case
        if base_policy is not None:
            # Store original learn method before copying attributes
            original_learn = self.learn
            original_select_reward = self.select_reward
            
            # Copy all attributes from base policy
            self.__dict__.update(base_policy.__dict__)
            self.__class__ = base_policy.__class__
            
            # Restore our learn method and store base learn method
            self.learn = original_learn
            self._base_learn = base_policy.learn
            self.select_reward = original_select_reward
        else:
            self._base_learn = None

    def select_reward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Select reward using the reward model"""
        if hasattr(self.reward_model, 'select_reward'):
            return self.reward_model.select_reward(obs, actions)
        else:
            return self.reward_model(obs, actions)

    def learn(self, batch: Dict, epoch=None, step=None) -> Dict[str, float]:
        replay_batch, pref_batch = batch["replay"], batch["pref"]
        
        # Determine if BT learning should happen
        should_train_rm = True
        if epoch is not None and self.rm_stop_epoch is not None:
            should_train_rm = epoch < self.rm_stop_epoch
        
        # Determine if base policy learning should happen
        should_train_policy = True
        if epoch is not None and self.policy_start_epoch is not None:
            should_train_policy = epoch >= self.policy_start_epoch
        
        result = {}
        
        # BT preference learning
        if should_train_rm:
            # Extract preference data
            F_B, F_S = pref_batch["obs_1"].shape[0:2]
            F_S -= 1
            
            # Preference learning
            pref_obs_1 = pref_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1)
            pref_obs_2 = pref_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1)
            pref_action_1 = pref_batch["action_1"][:, :-1].reshape(F_B*F_S, -1)
            pref_action_2 = pref_batch["action_2"][:, :-1].reshape(F_B*F_S, -1)
            
            reward_1 = self.select_reward(pref_obs_1, pref_action_1).reshape(F_B, F_S)
            reward_2 = self.select_reward(pref_obs_2, pref_action_2).reshape(F_B, F_S)
            
            logits = reward_2.sum(dim=-1) - reward_1.sum(dim=-1)
            labels = pref_batch["label"][:, 1].float()
            pref_loss = self.reward_criterion(logits, labels).mean()
            
            # Regularization loss
            reg_loss = (reward_1.square().mean() + reward_2.square().mean()) / 2
            
            # Total reward model loss
            reward_model_loss = pref_loss + self.reward_reg * reg_loss
            
            self.reward_model_optim.zero_grad()
            reward_model_loss.backward()
            self.reward_model_optim.step()

            with torch.no_grad():
                reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()
                # Compute reward statistics
                rewards_all = torch.cat([reward_1.flatten(), reward_2.flatten()])
                rewards_win = torch.where(labels.unsqueeze(-1) == 1, reward_2, reward_1).flatten()
                rewards_lose = torch.where(labels.unsqueeze(-1) == 0, reward_2, reward_1).flatten()

            # Add BT-specific metrics
            result.update({
                "loss/preference": pref_loss.item(),
                "loss/reward_model": reward_model_loss.item(),
                "loss/reg": reg_loss.item(),
                "misc/reward_acc": reward_accuracy.item(),
                "info/reward_mean": rewards_all.mean().item(),
                "info/reward_std": rewards_all.std().item(),
                "info/reward_win_mean": rewards_win.mean().item(),
                "info/reward_win_std": rewards_win.std().item(),
                "info/reward_lose_mean": rewards_lose.mean().item(),
                "info/reward_lose_std": rewards_lose.std().item(),
                "info/reward_mean_diff": torch.abs(rewards_win.mean() - rewards_lose.mean()).item(),
            })

        # Train base policy on replay batch if it exists and should train
        if self._base_learn is not None and should_train_policy:
            # Replace rewards in replay_batch with select_reward results
            with torch.no_grad():
                replay_rewards = self.select_reward(replay_batch["observations"], replay_batch["actions"])
                replay_batch["rewards"] = replay_rewards
            base_result = self._base_learn(replay_batch)
            result.update(base_result)

        return result
