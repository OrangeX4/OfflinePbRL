import numpy as np
import torch
import torch.nn as nn
import gym

from torch.nn import functional as F
from typing import Dict, Union, Tuple
from offlinepbrl.policy import SACPolicy


class CPRLPolicy(SACPolicy):
    """
    Conservative Preference Reward Learning
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

    def calc_reward_values(
        self,
        obs_pi: torch.Tensor,
        obs_to_pred: torch.Tensor,
        next_obs: torch.Tensor
    ) -> torch.Tensor:
        act, log_prob = self.actforward(obs_pi)
        reward = self.reward_model(obs_to_pred, act)
        with torch.no_grad():
            next_v = torch.min(
                self.critic1_old(next_obs, act),
                self.critic2_old(next_obs, act)
            )
        return reward + self._gamma * next_v - log_prob.detach()

    def calc_random_reward_values(
        self,
        obs: torch.Tensor,
        random_act: torch.Tensor,
        next_obs: torch.Tensor
    ) -> torch.Tensor:
        reward = self.reward_model(obs, random_act)
        with torch.no_grad():
            next_v = torch.min(
                self.critic1_old(next_obs, random_act),
                self.critic2_old(next_obs, random_act)
            )
        log_prob = np.log(0.5**random_act.shape[-1])
        return reward + self._gamma * next_v - log_prob

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
        
        # Extract preference data
        F_B, F_S = pref_batch["obs_1"].shape[0:2]
        F_S -= 1
        
        # Process replay batch
        obss, actions, next_obss, rewards, terminals = replay_batch["observations"], replay_batch["actions"], \
            replay_batch["next_observations"], replay_batch["rewards"], replay_batch["terminals"]
        batch_size = obss.shape[0]
        
        # compute conservative loss for reward model training
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        
        tmp_obss = obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, obss.shape[-1])
        tmp_next_obss = next_obss.unsqueeze(1) \
            .repeat(1, self._num_repeat_actions, 1) \
            .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
        
        # update actor with both original and sampled observations
        all_obss = torch.cat([obss, tmp_obss])
        a_all, log_probs_all = self.actforward(all_obss)
        q1a_all, q2a_all = self.critic1(all_obss, a_all), self.critic2(all_obss, a_all)
        actor_loss = (self._alpha * log_probs_all - torch.min(q1a_all, q2a_all)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self._is_auto_alpha:
            log_probs = log_probs_all[:batch_size].detach() + self._target_entropy
            alpha_loss = -(self._log_alpha * log_probs).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self._alpha = self._log_alpha.detach().exp()
        
        # compute td error
        if self._max_q_backup:
            with torch.no_grad():
                tmp_next_obss_backup = next_obss.unsqueeze(1) \
                    .repeat(1, self._num_repeat_actions, 1) \
                    .view(batch_size * self._num_repeat_actions, next_obss.shape[-1])
                tmp_next_actions, _ = self.actforward(tmp_next_obss_backup)
                tmp_next_q1 = self.critic1_old(tmp_next_obss_backup, tmp_next_actions) \
                    .view(batch_size, self._num_repeat_actions, 1) \
                    .max(1)[0].view(-1, 1)
                tmp_next_q2 = self.critic2_old(tmp_next_obss_backup, tmp_next_actions) \
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

        target_q = rewards + self._gamma * (1 - terminals) * next_q
        
        obs_reward_value = self.calc_reward_values(tmp_obss, tmp_obss, tmp_next_obss)
        next_obs_reward_value = self.calc_reward_values(tmp_next_obss, tmp_obss, tmp_next_obss)
        random_reward_value = self.calc_random_reward_values(tmp_obss, random_actions, tmp_next_obss)

        obs_reward_value = obs_reward_value.reshape(batch_size, self._num_repeat_actions, 1)
        next_obs_reward_value = next_obs_reward_value.reshape(batch_size, self._num_repeat_actions, 1)
        random_reward_value = random_reward_value.reshape(batch_size, self._num_repeat_actions, 1)
        
        cat_q_pred = torch.cat([obs_reward_value, next_obs_reward_value, random_reward_value], 1)
        q_pred = self.reward_model(obss, actions) + self._gamma * (1 - terminals) * next_q.detach()

        conservative_loss = \
            torch.logsumexp(cat_q_pred / self._temperature, dim=1).mean() * self._cql_weight * self._temperature - \
            q_pred.mean() * self._cql_weight
        
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
        reward_model_loss = pref_loss + self.reward_reg * reg_loss + conservative_loss
        
        if self._with_lagrange:
            cql_alpha = torch.clamp(self.cql_log_alpha.exp(), 0.0, 1e6)
            conservative_loss_scaled = cql_alpha * (conservative_loss - self._lagrange_threshold)
            reward_model_loss = pref_loss + self.reward_reg * reg_loss + conservative_loss_scaled

            self.cql_alpha_optim.zero_grad()
            cql_alpha_loss = -conservative_loss_scaled
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optim.step()
        
        self.reward_model_optim.zero_grad()
        reward_model_loss.backward()
        self.reward_model_optim.step()

        # Update critics with both original data and sampled actions from conservative computation
        # Get the sampled actions that were used in conservative loss computation
        with torch.no_grad():
            sampled_obs_actions, _ = self.actforward(tmp_obss)
            sampled_next_obs_actions, _ = self.actforward(tmp_next_obss)
        
        # Compute targets for sampled actions
        with torch.no_grad():
            sampled_next_v_obs = torch.min(
                self.critic1_old(tmp_next_obss, sampled_obs_actions),
                self.critic2_old(tmp_next_obss, sampled_obs_actions)
            )
            sampled_next_v_next = torch.min(
                self.critic1_old(tmp_next_obss, sampled_next_obs_actions),
                self.critic2_old(tmp_next_obss, sampled_next_obs_actions)
            )
            random_next_v = torch.min(
                self.critic1_old(tmp_next_obss, random_actions),
                self.critic2_old(tmp_next_obss, random_actions)
            )
            
            # Targets for sampled actions using reward model
            sampled_target_obs = self.reward_model(tmp_obss, sampled_obs_actions) + self._gamma * sampled_next_v_obs
            sampled_target_next = self.reward_model(tmp_obss, sampled_next_obs_actions) + self._gamma * sampled_next_v_next
            sampled_target_random = self.reward_model(tmp_obss, random_actions) + self._gamma * random_next_v
        
        # Compute Q values for original data
        q1, q2 = self.critic1(obss, actions), self.critic2(obss, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        
        # Add loss for sampled actions
        q1_sampled_obs = self.critic1(tmp_obss, sampled_obs_actions)
        q1_sampled_next = self.critic1(tmp_obss, sampled_next_obs_actions)
        q1_random = self.critic1(tmp_obss, random_actions)
        
        q2_sampled_obs = self.critic2(tmp_obss, sampled_obs_actions)
        q2_sampled_next = self.critic2(tmp_obss, sampled_next_obs_actions)
        q2_random = self.critic2(tmp_obss, random_actions)
        
        critic1_loss += ((q1_sampled_obs - sampled_target_obs).pow(2)).mean()
        critic1_loss += ((q1_sampled_next - sampled_target_next).pow(2)).mean()
        critic1_loss += ((q1_random - sampled_target_random).pow(2)).mean()
        
        critic2_loss += ((q2_sampled_obs - sampled_target_obs).pow(2)).mean()
        critic2_loss += ((q2_sampled_next - sampled_target_next).pow(2)).mean()
        critic2_loss += ((q2_random - sampled_target_random).pow(2)).mean()

        # update critic
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        self._sync_weight()

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
            "loss/conservative": conservative_loss.item(),
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
        }

        if self._is_auto_alpha:
            result["loss/alpha"] = alpha_loss.item()
            result["alpha"] = self._alpha.item()
        if self._with_lagrange:
            result["loss/cql_alpha"] = cql_alpha_loss.item()
            result["cql_alpha"] = cql_alpha.item()
        
        return result

