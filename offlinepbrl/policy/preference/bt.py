import torch
import torch.nn as nn
from typing import Dict, Any


class BTWrapper:
    """
    Bradley-Terry wrapper for adding preference learning to any base policy
    """

    def __init__(
        self,
        base_policy,
        reward_model: nn.Module,
        reward_model_optim: torch.optim.Optimizer,
        reward_reg: float = 0.5,
    ) -> None:
        # Copy all attributes from base policy
        self.__dict__.update(base_policy.__dict__)
        self.__class__ = base_policy.__class__
        
        # Add BT-specific attributes
        self.reward_model = reward_model
        self.reward_model_optim = reward_model_optim
        self.reward_reg = reward_reg
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        
        # Store original learn method
        self._base_learn = base_policy.learn

    def learn(self, batch: Dict) -> Dict[str, float]:
        replay_batch, pref_batch = batch["replay"], batch["pref"]
        
        # Extract preference data
        F_B, F_S = pref_batch["obs_1"].shape[0:2]
        F_S -= 1
        
        # Preference learning
        pref_obs_1 = pref_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1)
        pref_obs_2 = pref_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1)
        pref_action_1 = pref_batch["action_1"][:, :-1].reshape(F_B*F_S, -1)
        pref_action_2 = pref_batch["action_2"][:, :-1].reshape(F_B*F_S, -1)
        
        reward_1 = self.reward_model(pref_obs_1, pref_action_1).reshape(F_B, F_S)
        reward_2 = self.reward_model(pref_obs_2, pref_action_2).reshape(F_B, F_S)
        
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

        # Train base policy on replay batch
        base_result = self._base_learn(replay_batch)

        # Merge results
        result = base_result.copy()
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
        
        return result
