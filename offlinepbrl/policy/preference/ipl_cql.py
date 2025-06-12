import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple, Optional
from offlinepbrl.policy import SACPolicy


class IPLCQLPolicy(SACPolicy):
    """
    Inverse Preference Learning Conservative Q-Learning
    """

    def __init__(
        self,
        actor: nn.Module,
        critic1: nn.Module,
        critic2: nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic1_optim: torch.optim.Optimizer,
        critic2_optim: torch.optim.Optimizer,
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
        q_reg: float = 0.0,
        reg_replay_weight: Optional[float] = None,
        actor_replay_weight: Optional[float] = None,
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
        self._cql_weight = cql_weight
        self._temperature = temperature
        self._max_q_backup = max_q_backup
        self._deterministic_backup = deterministic_backup
        self._with_lagrange = with_lagrange
        self._lagrange_threshold = lagrange_threshold

        self.cql_log_alpha = torch.zeros(1, requires_grad=True, device=self.actor.device)
        self.cql_alpha_optim = torch.optim.Adam([self.cql_log_alpha], lr=cql_alpha_lr)

        self._num_repeat_actions = num_repeart_actions
        
        # IPL specific parameters
        self.reward_reg = reward_reg
        self.q_reg = q_reg
        self.reg_replay_weight = reg_replay_weight
        self.actor_replay_weight = actor_replay_weight
        self.reward_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

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

    def learn(self, batch: Dict) -> Dict[str, float]:

        replay_batch, pref_batch = batch["replay"], batch["pref"]

        F_B, F_S = pref_batch["obs_1"].shape[0:2]
        F_S -= 1
        R_B = replay_batch["observations"].shape[0]
        split = [F_B*F_S, F_B*F_S, R_B]

        # Concatenate preference and replay data
        obs = torch.concat([
            pref_batch["obs_1"][:, :-1].reshape(F_B*F_S, -1),
            pref_batch["obs_2"][:, :-1].reshape(F_B*F_S, -1),
            replay_batch["observations"],
        ], dim=0)
        
        next_obs = torch.concat([
            pref_batch["obs_1"][:, 1:].reshape(F_B*F_S, -1),
            pref_batch["obs_2"][:, 1:].reshape(F_B*F_S, -1),
            replay_batch["next_observations"],
        ], dim=0)
        
        action = torch.concat([
            pref_batch["action_1"][:, :-1].reshape(F_B*F_S, -1),
            pref_batch["action_2"][:, :-1].reshape(F_B*F_S, -1),
            replay_batch["actions"],
        ], dim=0)
        
        terminal = torch.concat([
            pref_batch["terminal_1"][:, :-1].reshape(F_B*F_S, -1),
            pref_batch["terminal_2"][:, :-1].reshape(F_B*F_S, -1),
            replay_batch["terminals"],
        ], dim=0)

        batch_size = obs.shape[0]
        
        # update actor
        a, log_probs = self.actforward(obs)
        q1a, q2a = self.critic1(obs, a), self.critic2(obs, a)
        actor_loss = (self._alpha * log_probs - torch.min(q1a, q2a))
        
        if self.actor_replay_weight is not None:
            a1, a2, ar = torch.split(actor_loss, split, dim=0)
            actor_loss_fb = (a1.mean() + a2.mean()) / 2
            actor_loss_re = ar.mean()
            actor_loss = (1 - self.actor_replay_weight) * actor_loss_fb + self.actor_replay_weight * actor_loss_re
        else:
            actor_loss = actor_loss.mean()
            
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
        
        # compute next Q values for inverse Bellman operator
        with torch.no_grad():
            next_actions, next_log_probs = self.actforward(next_obs)
            next_q = torch.min(
                self.critic1_old(next_obs, next_actions),
                self.critic2_old(next_obs, next_actions)
            )
            if not self._deterministic_backup:
                next_q -= self._alpha * next_log_probs

        # compute Q predictions and inverse Bellman rewards
        q1_pred = self.critic1(obs.detach(), action)
        q2_pred = self.critic2(obs.detach(), action)
        
        # Inverse Bellman: reward = Q - γ * Q_next
        reward1 = q1_pred - (1 - terminal) * self._gamma * next_q
        reward2 = q2_pred - (1 - terminal) * self._gamma * next_q
        reward = torch.stack([reward1, reward2], dim=0)  # Shape: [2, total_size, 1]
        reward = reward.squeeze(-1)  # Shape: [2, total_size]
        
        # Split rewards for preference learning
        r1, r2, rr = torch.split(reward, split, dim=1)
        E = r1.shape[0]
        r1, r2 = r1.reshape(E, F_B, F_S), r2.reshape(E, F_B, F_S)
        
        # Bradley-Terry model for preferences
        logits = r2.sum(dim=-1) - r1.sum(dim=-1)  # Shape: [E, F_B]
        labels = pref_batch["label"][:, 1].float().unsqueeze(0).expand(E, -1)  # Shape: [E, F_B]
        pref_loss = self.reward_criterion(logits, labels).mean()
        
        # Regularization losses
        reg_loss_fb = (r1.square().mean() + r2.square().mean()) / 2
        reg_loss_re = rr.square().mean()
        if self.reg_replay_weight is not None:
            reg_loss = (1 - self.reg_replay_weight) * reg_loss_fb + self.reg_replay_weight * reg_loss_re
        else:
            reg_loss = reward.square().mean()
        
        # Q regularization loss
        q_reg_loss = q1_pred.square().mean() + q2_pred.square().mean()
        
        # CQL conservative loss (simplified for IPL)
        conservative_loss = 0.0
        if self._cql_weight > 0:
            # Use only replay data for CQL loss to maintain conservative property
            obs_replay = replay_batch["observations"]
            action_replay = replay_batch["actions"]
            batch_size_replay = obs_replay.shape[0]
            
            random_actions = torch.FloatTensor(
                batch_size_replay * self._num_repeat_actions, action_replay.shape[-1]
            ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
            
            tmp_obs_replay = obs_replay.unsqueeze(1) \
                .repeat(1, self._num_repeat_actions, 1) \
                .view(batch_size_replay * self._num_repeat_actions, obs_replay.shape[-1])
                
            obs_pi_value1, obs_pi_value2 = self.calc_pi_values(tmp_obs_replay, tmp_obs_replay)
            random_value1, random_value2 = self.calc_random_values(tmp_obs_replay, random_actions)
            
            obs_pi_value1 = obs_pi_value1.reshape(batch_size_replay, self._num_repeat_actions, 1)
            obs_pi_value2 = obs_pi_value2.reshape(batch_size_replay, self._num_repeat_actions, 1)
            random_value1 = random_value1.reshape(batch_size_replay, self._num_repeat_actions, 1)
            random_value2 = random_value2.reshape(batch_size_replay, self._num_repeat_actions, 1)
            
            cat_q1 = torch.cat([obs_pi_value1, random_value1], 1)
            cat_q2 = torch.cat([obs_pi_value2, random_value2], 1)
            
            q1_replay = self.critic1(obs_replay, action_replay)
            q2_replay = self.critic2(obs_replay, action_replay)
            
            conservative_loss1 = \
                torch.logsumexp(cat_q1 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
                q1_replay.mean() * self._cql_weight
            conservative_loss2 = \
                torch.logsumexp(cat_q2 / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
                q2_replay.mean() * self._cql_weight
                
            conservative_loss = conservative_loss1 + conservative_loss2
        
        critic_loss = pref_loss + self.reward_reg * reg_loss + self.q_reg * q_reg_loss + conservative_loss
        
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        critic_loss.backward()
        self.critic1_optim.step()
        self.critic2_optim.step()

        self._sync_weight()

        with torch.no_grad():
            reward_accuracy = ((logits > 0) == torch.round(labels)).float().mean()

        metrics = {
            "loss/actor": actor_loss.item(),
            "loss/preference": pref_loss.item(),
            "loss/reg": reg_loss.item(),
            "loss/q_reg": q_reg_loss.item(),
            "loss/conservative": conservative_loss.item() if isinstance(conservative_loss, torch.Tensor) else conservative_loss,
            "misc/reward_value": reward.mean().item(),
            "misc/reward_acc": reward_accuracy.item(),
            "misc/q1": q1_pred.mean().item(),
            "misc/q2": q2_pred.mean().item(),
            "misc/next_q": next_q.mean().item(),
        }
        
        if self.actor_replay_weight is not None:
            metrics.update({
                "detail/actor_loss_fb": actor_loss_fb.item(),
                "detail/actor_loss_re": actor_loss_re.item(),
            })
        if self.reg_replay_weight is not None:
            metrics.update({
                "detail/reg_loss_fb": reg_loss_fb.item(),
                "detail/reg_loss_re": reg_loss_re.item()
            })
        if self._is_auto_alpha:
            metrics["loss/alpha"] = alpha_loss.item()
            metrics["alpha"] = self._alpha.item()

        return metrics
