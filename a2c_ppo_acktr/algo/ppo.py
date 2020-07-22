import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pdb import set_trace as st


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts, pred_loss=False, full_hidden=False, num_processes=16, device='cpu'):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        pred_err_epoch = 0
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch, full_hidden=full_hidden)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, infos_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, full_recurrent_hidden_states_batch = sample

                obs_batch = obs_batch.to(device)

                # Reshape to do in a single forward pass for all steps
                if full_hidden and not self.actor_critic.base.is_leaf: #for pi_1
                    print("updating pi_1")
                    values, action_log_probs, dist_entropy, rnn_hxs, all_hxs = self.actor_critic.evaluate_actions(
                        obs_batch, full_recurrent_hidden_states_batch, masks_batch,
                        actions_batch, info=infos_batch, process_rnn_hxs=full_hidden, N=num_processes//self.num_mini_batch,
                        device=device)
                else:
                    print("updating pi_2")
                    values, action_log_probs, dist_entropy, rnn_hxs, all_hxs = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch, info=infos_batch, process_rnn_hxs=False, N=num_processes//self.num_mini_batch,
                        device=device)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                total_loss = value_loss * self.value_loss_coef + action_loss - \
                 dist_entropy * self.entropy_coef

                if pred_loss: #only for pi_2
                    assert len(infos_batch.shape) == 2
                    gate = infos_batch[:, 0].unsqueeze(dim=-1)
                    # gate, _ = torch.split(infos_batch, 1, dim=-1)
                    gate = gate.squeeze().bool()

                    with torch.no_grad():
                        # currenlt only works for non-perminant memory
                        if obs_batch.ndim > 3 and self.actor_critic.base.gate_input != 'hid':
                            obs_batch = obs_batch/255
                            obs_batch = self.actor_critic.base.cnn(obs_batch)
                        if self.actor_critic.base.persistent:
                            # torch.Size([1026, 128]) torch.Size([128])
                            # TODO: fix it
                            full_hxs = self.actor_critic.process_rnn_hxs(full_recurrent_hidden_states_batch, masks_batch, N=num_processes//self.num_mini_batch, device=device)
                            h1, h2 = torch.chunk(full_hxs, 2, dim = -1)
                            capts = self.actor_critic.base.cell.capture(torch.cat((obs_batch[gate], h2[gate]), dim = -1))
                            capts, _ = torch.chunk(capts, 2, dim = -1)
                        else:
                            capts = self.actor_critic.base.cell.capture(obs_batch[gate])

                    # capts = capts[]
                    all_hxs = all_hxs[gate]
                    # import pdb; pdb.set_trace()

                    pred_err = torch.nn.functional.mse_loss(all_hxs, capts)
                    # import pdb; pdb.set_trace()
                    total_loss += pred_err

                    pred_err_epoch += pred_err.item()

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        pred_err_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, pred_err_epoch
