import torch
import torch.nn as nn
from typing import Dict, Any, Iterable, Optional
import numpy as np

from offlinepbrl.modules.reward_module import BaseRewardModel, EnsembleRewardModel


class BTWrapper:
    """
    Bradley-Terry wrapper for adding preference learning to any base policy
    """

    def __init__(
        self,
        base_policy: Optional[Any] = None,
        reward_model: BaseRewardModel = None,
        reward_model_optim: torch.optim.Optimizer | Iterable[torch.optim.Optimizer] = None,
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
            original_learn_pref_batch = self._learn_pref_batch
            
            # Copy all attributes from base policy
            self.__dict__.update(base_policy.__dict__)
            self.__class__ = base_policy.__class__
            
            # Restore our learn method and store base learn method
            self.learn = original_learn
            self._base_learn = base_policy.learn
            self._learn_pref_batch = original_learn_pref_batch
        else:
            self._base_learn = None

    def _learn_pref_batch(
        self,
        pref_batch: Dict,
        reward_model: BaseRewardModel,
        reward_model_optim: torch.optim.Optimizer,
        prefix: str = "",
    ) -> Dict[str, float]:
        # Extract preference data
        F_B, F_S = pref_batch["obs_1"].shape[0:2]
        F_S -= 1
        
        # Preference learning
        pref_obs_1 = pref_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1)
        pref_obs_2 = pref_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1)
        pref_action_1 = pref_batch["action_1"][:, :-1].reshape(F_B*F_S, -1)
        pref_action_2 = pref_batch["action_2"][:, :-1].reshape(F_B*F_S, -1)
        
        reward_1 = reward_model.select_reward(pref_obs_1, pref_action_1).reshape(F_B, F_S)
        reward_2 = reward_model.select_reward(pref_obs_2, pref_action_2).reshape(F_B, F_S)
        
        logits = reward_2.sum(dim=-1) - reward_1.sum(dim=-1)
        labels = pref_batch["label"][:, 1].float()
        pref_loss = self.reward_criterion(logits, labels).mean()
        
        # Regularization loss
        reg_loss = (reward_1.square().mean() + reward_2.square().mean()) / 2
        
        # Total reward model loss
        reward_model_loss = pref_loss + self.reward_reg * reg_loss

        if reward_model_optim:
            reward_model_optim.zero_grad()
            reward_model_loss.backward()
            reward_model_optim.step()

        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()
            # Compute reward statistics
            rewards_all = torch.cat([reward_1.flatten(), reward_2.flatten()])
            rewards_win = torch.where(labels.unsqueeze(-1) == 1, reward_2, reward_1).flatten()
            rewards_lose = torch.where(labels.unsqueeze(-1) == 0, reward_2, reward_1).flatten()

        # Build metrics dict
        metrics = {
            f"{prefix}loss/preference": pref_loss.item(),
            f"{prefix}loss/reward_model": reward_model_loss.item(),
            f"{prefix}loss/reg": reg_loss.item(),
            f"{prefix}misc/reward_acc": reward_accuracy.item(),
            f"{prefix}info/reward_mean": rewards_all.mean().item(),
            f"{prefix}info/reward_std": rewards_all.std().item(),
            f"{prefix}info/reward_win_mean": rewards_win.mean().item(),
            f"{prefix}info/reward_win_std": rewards_win.std().item(),
            f"{prefix}info/reward_lose_mean": rewards_lose.mean().item(),
            f"{prefix}info/reward_lose_std": rewards_lose.std().item(),
            f"{prefix}info/reward_mean_diff": torch.abs(rewards_win.mean() - rewards_lose.mean()).item(),
        }

        return metrics

    def learn(self, batch, epoch=None, step=None) -> Dict[str, float]:
        # Handle different input formats
        if isinstance(batch, dict) and "replay" in batch:
            # Called with combined batch from trainer
            replay_batch = batch["replay"]
            pref_batch = batch.get("pref", None)
            traj_batch = batch.get("traj", None)
        else:
            # Called directly with replay batch only
            replay_batch = batch
            pref_batch = None
            traj_batch = None
        
        # Determine if BT learning should happen
        should_train_rm = True
        if epoch is not None and self.rm_stop_epoch is not None:
            should_train_rm = epoch < self.rm_stop_epoch
        
        # Determine if base policy learning should happen
        should_train_policy = True
        if epoch is not None and self.policy_start_epoch is not None:
            should_train_policy = epoch >= self.policy_start_epoch
        
        result = {}
        
        # BT preference learning - only if pref_batch is provided
        if should_train_rm and pref_batch is not None:
            if isinstance(pref_batch, (list, tuple)):
                # Handle ensemble case with list of pref_batches
                assert isinstance(self.reward_model_optim, Iterable)
                assert isinstance(self.reward_model, EnsembleRewardModel)
                assert len(pref_batch) == self.reward_model.ensemble_num == len(self.reward_model_optim)
                for i, pref in enumerate(pref_batch):
                    reward_model = self.reward_model.members[i]
                    optim = self.reward_model_optim[i]

                    metrics = self._learn_pref_batch(pref, reward_model, optim, prefix=f"ens_{i}/")
                    result.update(metrics)

                # Also compute aggregate metrics across ensemble
                metrics = self._learn_pref_batch(pref_batch[0], self.reward_model, None, prefix="")
                result.update(metrics)
            else:
                # Normal case: pref_batch is not a list
                metrics = self._learn_pref_batch(pref_batch, self.reward_model, self.reward_model_optim, prefix="")
                result.update(metrics)
        
        # Train base policy on replay batch if it exists and should train
        if self._base_learn is not None and should_train_policy:
            # Replace rewards in replay_batch with select_reward results
            with torch.no_grad():
                replay_rewards = self.reward_model.select_reward(replay_batch["observations"], replay_batch["actions"])
                replay_batch["rewards"] = replay_rewards
            
            # Prepare batch for base policy
            if traj_batch is not None:
                # Update trajectory batch rewards too
                with torch.no_grad():
                    traj_rewards = self.reward_model.select_reward(traj_batch["observations"], traj_batch["actions"])
                    traj_batch["rewards"] = traj_rewards
                # Pass combined batch to base policy
                base_batch = {"replay": replay_batch, "traj": traj_batch}
                base_result = self._base_learn(base_batch, epoch=epoch, step=step)
            else:
                # Pass replay batch directly
                base_result = self._base_learn(replay_batch, epoch=epoch, step=step)
            result.update(base_result)

        return result
