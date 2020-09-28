import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from pdb import set_trace as st

def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, device='cpu'):
        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds1 = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_preds2 = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns1 = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns2 = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs1 = torch.zeros(num_steps, num_processes, 1)
        self.action_log_probs2 = torch.zeros(num_steps, num_processes, 1)
        self.device = device
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        info_size = action_shape
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.infos = torch.zeros(num_steps, num_processes, info_size) #[last_action]
        self.decisions = torch.zeros(num_steps, num_processes, 1).long() #[action1]

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds1 = self.value_preds1.to(device)
        self.value_preds2 = self.value_preds2.to(device)
        self.returns1 = self.returns1.to(device)
        self.returns2 = self.returns2.to(device)
        self.action_log_probs1 = self.action_log_probs1.to(device)
        self.action_log_probs2 = self.action_log_probs2.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, infos, decisions, ops=False):
        self.obs[self.step + 1].copy_(obs)
        self.recurrent_hidden_states[self.step +
                                     1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        if ops:
            self.infos[self.step].copy_(infos)
            self.decisions[self.step].copy_(decisions)
            self.action_log_probs1[self.step].copy_(action_log_probs[0])
            self.action_log_probs2[self.step].copy_(action_log_probs[1])
            self.value_preds1[self.step].copy_(value_preds[0])
            self.value_preds2[self.step].copy_(value_preds[1])
        else:
            self.action_log_probs2[self.step].copy_(action_log_probs)
            self.value_preds2[self.step].copy_(value_preds)

        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])


    def compute_returns_single(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        value_preds,
                        returns,
                        use_proper_time_limits=True):
        if use_proper_time_limits:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * value_preds[
                        step + 1] * self.masks[step +
                                               1] - value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    gae = gae * self.bad_masks[step + 1]
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    returns[step] = (returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                        + (1 - self.bad_masks[step + 1]) * value_preds[step]
        else:
            if use_gae:
                value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.size(0))):
                    delta = self.rewards[step] + gamma * value_preds[
                        step + 1] * self.masks[step +
                                               1] - value_preds[step]
                    gae = delta + gamma * gae_lambda * self.masks[step +
                                                                  1] * gae
                    returns[step] = gae + value_preds[step]
            else:
                returns[-1] = next_value
                for step in reversed(range(self.rewards.size(0))):
                    returns[step] = returns[step + 1] * \
                        gamma * self.masks[step + 1] + self.rewards[step]

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True,
                        ops=False):
        if ops:
            self.compute_returns_single(next_value[0],
                                        use_gae,
                                        gamma,
                                        gae_lambda,
                                        self.value_preds1,
                                        self.returns1,
                                        use_proper_time_limits
                                        )
            self.compute_returns_single(next_value[1],
                                        use_gae,
                                        gamma,
                                        gae_lambda,
                                        self.value_preds2,
                                        self.returns2,
                                        use_proper_time_limits)
        else:
            self.compute_returns_single(next_value,
                                        use_gae,
                                        gamma,
                                        gae_lambda,
                                        self.value_preds2,
                                        self.returns2,
                                        use_proper_time_limits)

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None,
                               is_leaf=True):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)
        for indices in sampler:
            obs_batch = self.obs[:-1].view(-1, *self.obs.size()[2:])[indices]
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]

            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            if is_leaf:
                actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
                value_preds_batch = self.value_preds2[:-1].view(-1, 1)[indices]
                return_batch = self.returns2[:-1].view(-1, 1)[indices]
                old_action_log_probs_batch = self.action_log_probs2.view(-1, 1)[indices]
                info_batch = torch.cat((self.infos.view(-1, self.infos.size(-1))[indices], self.decisions.view(-1, 1)[indices]), dim=-1).to(self.device)
            else:
                actions_batch = self.decisions.view(-1, 1)[indices].to(self.device)
                value_preds_batch = self.value_preds1[:-1].view(-1, 1)[indices]
                return_batch = self.returns1[:-1].view(-1, 1)[indices]
                old_action_log_probs_batch = self.action_log_probs1.view(-1, 1)[indices]
                info_batch=None

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, info_batch

    # def recurrent_generator(self, advantages, num_mini_batch):
    #     num_processes = self.rewards.size(1)
    #     assert num_processes >= num_mini_batch, (
    #         "PPO requires the number of processes ({}) "
    #         "to be greater than or equal to the number of "
    #         "PPO mini batches ({}).".format(num_processes, num_mini_batch))
    #     num_envs_per_batch = num_processes // num_mini_batch
    #     perm = torch.randperm(num_processes)
    #     for start_ind in range(0, num_processes, num_envs_per_batch):
    #         obs_batch = []
    #         recurrent_hidden_states_batch = []
    #         actions_batch = []
    #         value_preds_batch = []
    #         return_batch = []
    #         masks_batch = []
    #         old_action_log_probs_batch = []
    #         adv_targ = []
    #
    #         for offset in range(num_envs_per_batch):
    #             ind = perm[start_ind + offset]
    #             obs_batch.append(self.obs[:-1, ind])
    #             recurrent_hidden_states_batch.append(
    #                 self.recurrent_hidden_states[0:1, ind])
    #             actions_batch.append(self.actions[:, ind])
    #             value_preds_batch.append(self.value_preds[:-1, ind])
    #             return_batch.append(self.returns[:-1, ind])
    #             masks_batch.append(self.masks[:-1, ind])
    #             old_action_log_probs_batch.append(
    #                 self.action_log_probs[:, ind])
    #             adv_targ.append(advantages[:, ind])
    #
    #         T, N = self.num_steps, num_envs_per_batch
    #         # These are all tensors of size (T, N, -1)
    #         obs_batch = torch.stack(obs_batch, 1)
    #         actions_batch = torch.stack(actions_batch, 1)
    #         value_preds_batch = torch.stack(value_preds_batch, 1)
    #         return_batch = torch.stack(return_batch, 1)
    #         masks_batch = torch.stack(masks_batch, 1)
    #         old_action_log_probs_batch = torch.stack(
    #             old_action_log_probs_batch, 1)
    #         adv_targ = torch.stack(adv_targ, 1)
    #
    #         # States is just a (N, -1) tensor
    #         recurrent_hidden_states_batch = torch.stack(
    #             recurrent_hidden_states_batch, 1).view(N, -1)
    #
    #         # Flatten the (T, N, ...) tensors to (T * N, ...)
    #         obs_batch = _flatten_helper(T, N, obs_batch)
    #         actions_batch = _flatten_helper(T, N, actions_batch)
    #         value_preds_batch = _flatten_helper(T, N, value_preds_batch)
    #         return_batch = _flatten_helper(T, N, return_batch)
    #         masks_batch = _flatten_helper(T, N, masks_batch)
    #         old_action_log_probs_batch = _flatten_helper(T, N, \
    #                 old_action_log_probs_batch)
    #         adv_targ = _flatten_helper(T, N, adv_targ)
    #
    #         yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
    #             value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
