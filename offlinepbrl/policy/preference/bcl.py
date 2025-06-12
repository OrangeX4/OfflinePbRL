import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from offlinepbrl.policy import SACPolicy


class BCLPolicy(SACPolicy):
    """
    Bidirectional Conservative Learning
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        reward_model: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
        reward_model_optim: torch.optim.Optimizer,
        action_space: gym.spaces.Space,
        tau: float = 0.005,
        gamma: float  = 0.99,
        alpha: Union[float, Tuple[float, torch.Tensor, torch.optim.Optimizer]] = 0.2,
        cql_weight: float = 1.0,
        temperature: float = 1.0,
        max_q_backup: bool = False,
        deterministic_backup: bool = True,
        with_lagrange: bool = True,
        lagrange_threshold: float = 10.0,
        cql_alpha_lr: float = 1e-4,
        num_repeart_actions:int = 10,
        reward_reg: float = 0.5,
        reward_sync_weight: float = 1.0,
        reward_sync_start_epoch: int = 50,
    ) -> None:
        super().__init__(
            actor,
            critic1,
            critic2,
            actor_optim,
            critic1_optim,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            alpha=alpha
        )

        self.action_space = action_space
        self.reward_model = reward_model
        self.reward_model_optim = reward_model_optim
        self._cql_weight = cql_weight
        self._temperature = temperature
        self._max_q_backup = max_q_backup
        self._deterministic_backup = deterministic_backup
        self._with_lagrange = with_lagrange
        self._lagrange_threshold = lagrange_threshold

        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)

        self._num_repeat_actions = num_repeart_actions
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.reward_reg = reward_reg
        self.reward_sync_weight = reward_sync_weight
        self.reward_sync_start_epoch = reward_sync_start_epoch

    def calc_pi_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        act, log_prob = self.actforward(obs_pi)

        q1 = self.critic1(obs_to_pred, act)
        q2 = self.critic2(obs_to_pred, act)

        return q1 - log_prob.detach(), q2 - log_prob.detach()

    def calc_random_values(
        self,
        obs: torch.Tensor,
        random_act: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.critic1(obs, random_act)
        q2 = self.critic2(obs, random_act)

        log_prob1 = np.log(0.5**random_act.shape[-1])
        log_prob2 = np.log(0.5**random_act.shape[-1])

        return q1 - log_prob1, q2 - log_prob2

    def learn(self, batch: Dict, epoch=None, step=None) -> Dict[str, float]:
        replay_batch, pref_batch = batch["replay"], batch["pref"]
        
        # Extract preference data
        F_B, F_S = pref_batch["obs_1"].shape[0:2]
        F_S -= 1
        
        # Process replay batch
        obss, actions, next_obss, terminals = replay_batch["observations"], replay_batch["actions"], \
            replay_batch["next_observations"], replay_batch["terminals"]
        batch_size = obss.shape[0]
        
        # update actor
        a, log_probs = self.actforward(obss)
        q1a, q2a = self.critic1(obss, a), self.critic2(obss, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs.detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        
        # compute td error
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss = next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss)
                tmp_next_q1 = self.critic1_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                next_q = torch.min(tmp_next_q1, tmp_next_q2)
        else:
            with torch.no_grad():
                next_actions, next_log_probs = self.actforward(next_obss)
                next_q = torch.min(
                    self.critic1_old(next_obss, next_actions),
                    self.critic2_old(next_obss, next_actions)
                )
                if not self._deterministic_backup:
                    next_q -= self._alpha * next_log_probs

        with torch.no_grad():
            replay_rewards = self.reward_model(obss, actions)
        target_q = replay_rewards + self._gamma * (1 - terminals) * next_q
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()

        # compute conservative loss
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])

        # CQL-style conservative loss for Q networks
        obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obss, tmp_obss)
        next_obs_pi_value1, next_obs_pi_value2 = self.calc_pi_values(tmp_next_obss, tmp_obss)
        random_value1, random_value2 = self.calc_random_values(tmp_obss, random_actions)

        obs_pi_value1 = obs_pi_value1.reshape(batch_size, self._num_repeat_actions, 1)
        obs_pi_value2 = obs_pi_value2.reshape(batch_size, self._num_repeat_actions, 1)
        next_obs_pi_value1 = next_obs_pi_value1.reshape(batch_size, self._num_repeat_actions, 1)
        next_obs_pi_value2 = next_obs_pi_value2.reshape(batch_size, self._num_repeat_actions, 1)
        random_value1 = random_value1.reshape(batch_size, self._num_repeat_actions, 1)
        random_value2 = random_value2.reshape(batch_size, self._num_repeat_actions, 1)
        
        # cat_q shape: (batch_size, 3 * num_repeat, 1)
        cat_q1 = torch.cat([obs_pi_value1, next_obs_pi_value1, random_value1], 1)
        cat_q2 = torch.cat([obs_pi_value2, next_obs_pi_value2, random_value2], 1)

        # Add CQL conservative loss to critics
        conservative_loss1 = \
            torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q1.mean() * self._cql_weight
        conservative_loss2 = \
            torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q2.mean() * self._cql_weight
        
        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss1_scaled = cql_alpha * (conservative_loss1 - self._lagrange_threshold)
            conservative_loss2_scaled = cql_alpha * (conservative_loss2 - self._lagrange_threshold)

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -(conservative_loss1_scaled + conservative_loss2_scaled) * 0.5
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()
        
        critic1_loss = critic1_loss + conservative_loss1
        critic2_loss = critic2_loss + conservative_loss2

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

        # Reward model training with bidirectional learning
        # 1. Preference learning (BT model)
        pref_obs_1 = pref_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1)
        pref_obs_2 = pref_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1)
        pref_action_1 = pref_batch["action_1"][:, :-1].reshape(F_B*F_S, -1)
        pref_action_2 = pref_batch["action_2"][:, :-1].reshape(F_B*F_S, -1)
        
        reward_1 = self.reward_model(pref_obs_1, pref_action_1).reshape(F_B, F_S)
        reward_2 = self.reward_model(pref_obs_2, pref_action_2).reshape(F_B, F_S)
        
        logits = reward_2.sum(dim=-1) - reward_1.sum(dim=-1)
        labels = pref_batch["label"][:, 1].float()
        pref_loss = self.reward_criterion(logits, labels).mean()
        
        # 2. Reward synchronization: r ≈ Q - γV (Soft Bellman Operator)
        # Apply to original replay data and sampled data from CQL
        with torch.no_grad():
            # For original replay data
            current_q1 = self.critic1(obss, actions)
            current_q2 = self.critic2(obss, actions)
            current_q = torch.min(current_q1, current_q2)
            
            # Target reward from Soft Bellman: r = Q - γV_next
            target_reward_orig = current_q - self._gamma * (1 - terminals) * next_q
            
            # For sampled data from CQL conservative loss computation
            sampled_obs_actions, _ = self.actforward(tmp_obss)
            sampled_next_obs_actions, _ = self.actforward(tmp_next_obss)
            
            # Q values for sampled actions
            q1_sampled_obs = self.critic1(tmp_obss, sampled_obs_actions)
            q2_sampled_obs = self.critic2(tmp_obss, sampled_obs_actions)
            q_sampled_obs = torch.min(q1_sampled_obs, q2_sampled_obs)
            
            q1_sampled_next = self.critic1(tmp_obss, sampled_next_obs_actions)  # Use tmp_obss as current state
            q2_sampled_next = self.critic2(tmp_obss, sampled_next_obs_actions)
            q_sampled_next = torch.min(q1_sampled_next, q2_sampled_next)
            
            q1_random = self.critic1(tmp_obss, random_actions)
            q2_random = self.critic2(tmp_obss, random_actions)
            q_random = torch.min(q1_random, q2_random)
            
            # V values for next states (all using tmp_next_obss as next state)
            next_v_sampled_obs = torch.min(
                self.critic1_old(tmp_next_obss, sampled_obs_actions),
                self.critic2_old(tmp_next_obss, sampled_obs_actions)
            )
            next_v_sampled_next = torch.min(
                self.critic1_old(tmp_next_obss, sampled_next_obs_actions),
                self.critic2_old(tmp_next_obss, sampled_next_obs_actions)
            )
            next_v_random = torch.min(
                self.critic1_old(tmp_next_obss, random_actions),
                self.critic2_old(tmp_next_obss, random_actions)
            )
            
            # Target rewards for sampled data (assume non-terminal for sampled data)
            target_reward_sampled_obs = q_sampled_obs - self._gamma * next_v_sampled_obs
            target_reward_sampled_next = q_sampled_next - self._gamma * next_v_sampled_next
            target_reward_random = q_random - self._gamma * next_v_random
        
        # Predicted rewards from reward model
        pred_reward_orig = self.reward_model(obss, actions)
        pred_reward_sampled_obs = self.reward_model(tmp_obss, sampled_obs_actions)
        pred_reward_sampled_next = self.reward_model(tmp_obss, sampled_next_obs_actions)
        pred_reward_random = self.reward_model(tmp_obss, random_actions)
        
        # Reward synchronization loss on all data points
        reward_sync_loss = (
            ((pred_reward_orig - target_reward_orig).pow(2)).mean() +
            ((pred_reward_sampled_obs - target_reward_sampled_obs).pow(2)).mean() +
            ((pred_reward_sampled_next - target_reward_sampled_next).pow(2)).mean() +
            ((pred_reward_random - target_reward_random).pow(2)).mean()
        ) / 4
        
        # 3. Regularization loss
        reg_loss = (reward_1.square().mean() + reward_2.square().mean()) / 2
        
        if epoch is not None and epoch >= self.reward_sync_start_epoch:
            reward_sync_weight = self.reward_sync_weight
        else:
            reward_sync_weight = 0.0
        # Total reward model loss
        reward_model_loss = pref_loss + self.reward_reg * reg_loss + reward_sync_weight * reward_sync_loss
        
        self.reward_model_optim.zero_grad()
        reward_model_loss.backward()
        self.reward_model_optim.step()

        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()
            # Compute reward statistics
            rewards_all = torch.cat([reward_1.flatten(), reward_2.flatten()])
            rewards_win = torch.where(labels.unsqueeze(-1) == 1, reward_2, reward_1).flatten()
            rewards_lose = torch.where(labels.unsqueeze(-1) == 0, reward_2, reward_1).flatten()

        result =  {
            "loss/actor": actor_loss.item(),
            "loss/critic1": critic1_loss.item(),
            "loss/critic2": critic2_loss.item(),
            "loss/preference": pref_loss.item(),
            "loss/reward_sync": reward_sync_loss.item(),
            "loss/q_conservative1": conservative_loss1.item(),
            "loss/q_conservative2": conservative_loss2.item(),
            "loss/reward_model": reward_model_loss.item(),
            "loss/reg": reg_loss.item(),
            "misc/reward_acc": reward_accuracy.item(),
            "misc/q1": q1.mean().item(),
            "misc/q2": q2.mean().item(),
            "misc/next_q": next_q.mean().item(),
            "misc/replay_rewards": replay_rewards.mean().item(),
            "misc/target_reward": target_reward_orig.mean().item(),
            "misc/pred_reward": pred_reward_orig.mean().item(),
            "info/reward_mean": rewards_all.mean().item(),
            "info/reward_std": rewards_all.std().item(),
            "info/reward_win_mean": rewards_win.mean().item(),
            "info/reward_win_std": rewards_win.std().item(),
            "info/reward_lose_mean": rewards_lose.mean().item(),
            "info/reward_lose_std": rewards_lose.std().item(),
            "info/reward_mean_diff": torch.abs(rewards_win.mean() - rewards_lose.mean()).item(),
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        if self._with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        
        return result

