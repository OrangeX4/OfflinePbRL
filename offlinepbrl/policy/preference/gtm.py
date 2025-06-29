import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import numpy as np


def gaussian_cdf(x):
    """Gaussian cumulative distribution function"""
    return 0.5 * (1 + torch.special.erf(x / np.sqrt(2)))


class GaussianTMWrapper:
    """
    Gaussian Thurstone-Mosteller wrapper for adding preference learning to any base policy
    """

    def __init__(
        self,
        base_policy: Optional[Any] = None,
        reward_model: nn.Module = None,
        reward_model_optim: torch.optim.Optimizer = None,
        reward_reg: float = 0.0,
        reward_ent_reg: float = 0.1,
        entropy_threshold: float = 1.0,
        reg_type: str = "transition",  # "transition" or "trajectory"
        rm_stop_epoch: Optional[int] = None,
        policy_start_epoch: Optional[int] = None,
    ) -> None:
        self.base_policy = base_policy
        self.reward_model = reward_model
        self.reward_model_optim = reward_model_optim
        self.reward_reg = reward_reg
        self.reward_ent_reg = reward_ent_reg
        self.entropy_threshold = entropy_threshold
        self.reg_type = reg_type
        self.rm_stop_epoch = rm_stop_epoch
        self.policy_start_epoch = policy_start_epoch
        
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
        """Select reward using the reward model (returns mean only for policy learning)"""
        if hasattr(self.reward_model, 'select_reward'):
            return self.reward_model.select_reward(obs, actions)
        else:
            # For Gaussian reward model, return only the mean component
            output = self.reward_model(obs, actions)
            if output.shape[-1] == 2:  # mean and logvar
                return output[..., 0:1]  # return mean only
            else:
                return output

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
        
        # Gaussian BT preference learning with Thurstone-Mosteller model
        if should_train_rm:
            # Extract preference data
            F_B, F_S = pref_batch["obs_1"].shape[0:2]
            F_S -= 1
            
            # Reshape preference data
            pref_obs_1 = pref_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1)
            pref_obs_2 = pref_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1)
            pref_action_1 = pref_batch["action_1"][:, :-1].reshape(F_B*F_S, -1)
            pref_action_2 = pref_batch["action_2"][:, :-1].reshape(F_B*F_S, -1)
            
            # Get reward mean and logvar for both trajectories
            reward_output_1 = self.reward_model(pref_obs_1, pref_action_1)  # [F_B*F_S, 2]
            reward_output_2 = self.reward_model(pref_obs_2, pref_action_2)  # [F_B*F_S, 2]
            
            # Split mean and logvar
            reward_mean_1 = reward_output_1[..., 0].reshape(F_B, F_S)  # [F_B, F_S]
            reward_logvar_1 = reward_output_1[..., 1].reshape(F_B, F_S)  # [F_B, F_S]
            reward_mean_2 = reward_output_2[..., 0].reshape(F_B, F_S)  # [F_B, F_S]
            reward_logvar_2 = reward_output_2[..., 1].reshape(F_B, F_S)  # [F_B, F_S]
            
            # Thurstone-Mosteller model
            # Mean difference: μ = Σ_t μ_1(s_t, a_t) - Σ_t μ_2(s_t, a_t)
            mu_diff = reward_mean_1.sum(dim=1) - reward_mean_2.sum(dim=1)  # [F_B]
            
            # Variance sum: σ² = Σ_t σ²_1(s_t, a_t) + Σ_t σ²_2(s_t, a_t)
            var_sum = torch.exp(reward_logvar_1).sum(dim=1) + torch.exp(reward_logvar_2).sum(dim=1)  # [F_B]
            std_sum = torch.sqrt(var_sum + 1e-8)  # [F_B]
            
            # Gaussian CDF: P(τ¹ ≻ τ²) = Φ(μ_diff / σ_sum)
            probs = gaussian_cdf(mu_diff / std_sum)  # [F_B]
            
            # Cross-entropy loss
            labels = pref_batch["label"][:, 1].float()  # [F_B]
            pref_loss = -(labels * torch.log(probs + 1e-8) + (1 - labels) * torch.log(1 - probs + 1e-8)).mean()
            
            # Regularization based on entropy
            if self.reg_type == "transition":
                # Transition-level entropy regularization: h(N(μ_t, σ_t²)) = 0.5 * log(2πe * σ_t²)
                entropy_1 = 0.5 * torch.log(2 * np.pi * np.e * torch.exp(reward_logvar_1) + 1e-8)  # [F_B, F_S]
                entropy_2 = 0.5 * torch.log(2 * np.pi * np.e * torch.exp(reward_logvar_2) + 1e-8)  # [F_B, F_S]
                avg_entropy = (entropy_1.mean() + entropy_2.mean()) / 2
            else:  # trajectory
                # Trajectory-level entropy regularization: h(N(μ, σ²)) = 0.5 * log(2πe * Σ_t σ_t²)
                entropy_1 = 0.5 * torch.log(2 * np.pi * np.e * torch.exp(reward_logvar_1).sum(dim=1) + 1e-8)  # [F_B]
                entropy_2 = 0.5 * torch.log(2 * np.pi * np.e * torch.exp(reward_logvar_2).sum(dim=1) + 1e-8)  # [F_B]
                avg_entropy = (entropy_1.mean() + entropy_2.mean()) / 2
            
            # Entropy regularization loss: max(0, η - h(N(μ, σ²)))
            reward_ent_loss = torch.clamp(self.entropy_threshold - avg_entropy, min=0.0)
            
            # Regularization loss (original L2 regularization)
            reg_loss = (reward_mean_1.square().mean() + reward_mean_2.square().mean()) / 2
            
            # Total reward model loss
            reward_model_loss = pref_loss + self.reward_reg * reg_loss + self.reward_ent_reg * reward_ent_loss

            self.reward_model_optim.zero_grad()
            reward_model_loss.backward()
            self.reward_model_optim.step()

            with torch.no_grad():
                reward_accuracy = ((probs > 0.5) == labels).float().mean()
                # Compute reward statistics (using means)
                rewards_all = torch.cat([reward_mean_1.flatten(), reward_mean_2.flatten()])
                rewards_win = torch.where(labels.unsqueeze(-1) == 1, reward_mean_2, reward_mean_1).flatten()
                rewards_lose = torch.where(labels.unsqueeze(-1) == 0, reward_mean_2, reward_mean_1).flatten()
                
                # Variance statistics
                var_all = torch.cat([torch.exp(reward_logvar_1).flatten(), torch.exp(reward_logvar_2).flatten()])

            # Add Gaussian BT-specific metrics
            result.update({
                "loss/preference": pref_loss.item(),
                "loss/reward_model": reward_model_loss.item(),
                "loss/reg": reg_loss.item(),
                "loss/reward_ent_loss": reward_ent_loss.item(),
                "misc/reward_acc": reward_accuracy.item(),
                "misc/avg_entropy": avg_entropy.item(),
                "misc/trajectory_std": std_sum.mean().item(),
                "info/reward_mean": rewards_all.mean().item(),
                "info/reward_std": rewards_all.std().item(),
                "info/reward_var_mean": var_all.mean().item(),
                "info/reward_var_std": var_all.std().item(),
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
