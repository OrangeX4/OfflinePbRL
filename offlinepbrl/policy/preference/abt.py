import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import numpy as np

from offlinepbrl.modules.reward_module import BaseRewardModel
from offlinepbrl.policy.preference.bt import BTWrapper


class AdversarialBTWrapper(BTWrapper):
    """
    Adversarial Bradley-Terry Wrapper (Alternating Optimization)
    
    Implements a max-min game for learning a robust reward function from preferences.
    The process alternates between:
    1. Policy Update (max step): Train the policy on rewards from the current reward model.
    2. Adversarial Reward Update (min step): Update the reward model to minimize the
       policy's performance while staying consistent with preferences.
    """

    def __init__(
        self,
        base_policy: Optional[Any] = None,
        reward_model: BaseRewardModel = None,
        reward_model_optim: torch.optim.Optimizer = None,
        reward_reg: float = 0.1,
        reward_bias: float = 0.5,
        adversarial_weight: float = 0.1,
        rm_stop_epoch: Optional[int] = None,
        policy_start_epoch: Optional[int] = None,
    ) -> None:
        # Call BTWrapper's __init__ but without passing our own learn method
        # We will define our own learn method from scratch
        super().__init__(
            base_policy=base_policy,
            reward_model=reward_model,
            reward_model_optim=reward_model_optim,
            reward_reg=reward_reg,
            rm_stop_epoch=rm_stop_epoch,
            policy_start_epoch=policy_start_epoch
        )
        self.reward_bias = reward_bias
        self.adversarial_weight = adversarial_weight

    def learn(self, batch: Dict, epoch=None, step=None) -> Dict[str, float]:
        replay_batch, pref_batch = batch["replay"], batch["pref"]
        
        should_train_policy = True
        if epoch is not None and self.policy_start_epoch is not None:
            should_train_policy = epoch >= self.policy_start_epoch
            
        result = {}

        # --- Step A: Policy Optimization (max step) ---
        if self._base_learn is not None and should_train_policy:
            with torch.no_grad():
                replay_rewards = self.reward_model.select_reward(replay_batch["observations"], replay_batch["actions"])
                replay_batch["rewards"] = replay_rewards
            base_result = self._base_learn(replay_batch)
            result.update(base_result)

        # --- Step B: Adversarial Reward Update (min step) ---
        # This step updates the reward model to be pessimistic about the policy's performance.
        
        # 1. Preference consistency loss (same as in BTWrapper)
        F_B, F_S = pref_batch["obs_1"].shape[0:2]
        F_S -= 1
        
        pref_obs_1 = pref_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1)
        pref_obs_2 = pref_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1)
        pref_action_1 = pref_batch["action_1"][:, :-1].reshape(F_B*F_S, -1)
        pref_action_2 = pref_batch["action_2"][:, :-1].reshape(F_B*F_S, -1)
        
        reward_1 = self.reward_model.select_reward(pref_obs_1, pref_action_1).reshape(F_B, F_S)
        reward_2 = self.reward_model.select_reward(pref_obs_2, pref_action_2).reshape(F_B, F_S)
        
        logits = reward_2.sum(dim=-1) - reward_1.sum(dim=-1)
        labels = pref_batch["label"][:, 1].float()
        pref_loss = self.reward_criterion(logits, labels).mean()
        
        reg_loss = ((reward_1 - self.reward_bias).square().mean() + (reward_2 - self.reward_bias).square().mean()) / 2
        
        # 2. Adversarial loss
        # We want to minimize the policy's expected return.
        # We approximate V(pi) by the average reward on the replay buffer,
        # as seen by the current reward model.
        replay_rewards_adv = self.reward_model.select_reward(replay_batch["observations"], replay_batch["actions"])
        adversarial_loss = replay_rewards_adv.mean() # This is our approximation of V(pi, R_theta)
        
        # 3. Total reward model loss
        # The goal is to minimize V(pi) while satisfying preference constraints.
        # So we add V(pi) to the loss, which we then minimize.
        reward_model_loss = pref_loss + self.reward_reg * reg_loss + self.adversarial_weight * adversarial_loss
        
        self.reward_model_optim.zero_grad()
        reward_model_loss.backward()
        self.reward_model_optim.step()

        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()
            rewards_all = torch.cat([reward_1.flatten(), reward_2.flatten()])
            rewards_win = torch.where(labels.unsqueeze(-1) == 1, reward_2, reward_1).flatten()
            rewards_lose = torch.where(labels.unsqueeze(-1) == 0, reward_2, reward_1).flatten()

        result.update({
            "loss/preference": pref_loss.item(),
            "loss/reward_model": reward_model_loss.item(),
            "loss/reg": reg_loss.item(),
            "loss/adversarial": adversarial_loss.item(),
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
