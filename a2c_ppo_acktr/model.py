
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from pdb import set_trace as st


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1
 
    @property
    def output_size(self):
        return self._hidden_size


class Identity(nn.Module):
    def forward(self, x):
        return x


class OpsPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, is_leaf, base=None, base_kwargs={}):
        super(OpsPolicy, self).__init__()

        if len(obs_shape) == 3:
            self.base = OpsBase(obs_shape, action_space, is_leaf=is_leaf, mode='cnn', **base_kwargs)
        else:
            self.base = OpsBase(obs_shape[0], action_space, is_leaf=is_leaf, **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False, info=None):
        value, actor_features, rnn_hxs, _ = self.base(inputs, rnn_hxs, masks, info=info)
        dist = self.dist(actor_features) #[16, 64]

        if deterministic:
            action = dist.mode() #[16, 3]
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, info=None):
        value, _, _, _ = self.base(inputs, rnn_hxs, masks, info=info)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, info=None, process_rnn_hxs=False, N=None, device='cpu'):
        if process_rnn_hxs:
            rnn_hxs = self.process_rnn_hxs(rnn_hxs, masks, N=N, device=device)

        value, actor_features, rnn_hxs, all_hxs = self.base(inputs, rnn_hxs, masks, info=info)
        dist = self.dist(actor_features)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, all_hxs

    def process_rnn_hxs(self, hxs, masks, N, device):
        T = int(hxs.shape[0])
        has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        has_zeros = [0] + has_zeros + [T]
        all_hxs = []
        dim = hxs.shape[-1]
        for i in range(len(has_zeros) - 1):
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]
            all_hxs.append(hxs[start_idx:end_idx]*masks[start_idx:end_idx])

        all_hxs = torch.cat(all_hxs, dim=0)
        return all_hxs


init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))
class Dynamics(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super(Dynamics, self).__init__()

        self.model = nn.Sequential(
            init_(nn.Linear(obs_dim + act_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        )

    def forward(self, x, a):
        return self.model(torch.cat([x, a], dim=-1))


class OpsCell(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64, persistent=False, pred_mode='pred_model'):
        super(OpsCell, self).__init__()
        self.hidden_size = hidden_size
        self.persistent = persistent
        self.pred_mode = pred_mode

        if self.persistent:
            cap_in_dim = obs_dim + hidden_size
            cap_out_dim = 2 * hidden_size
        else:
            cap_in_dim = obs_dim
            cap_out_dim = hidden_size

        self.capture = nn.Sequential(
            init_(nn.Linear(cap_in_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, cap_out_dim)), nn.Tanh()
        )

        self.predict = Dynamics(obs_dim=cap_out_dim, act_dim=act_dim, hidden_size=cap_out_dim)

        for name, param in self.predict.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h, g, a, full_hidden = False):
        # B x T x I
        # B x T x H
        # also keep another hidden state, so make h' = [h, h_persistent]
        outs = []
        capt = []

        # h1: short memory
        # h2: persistent memory
        if self.persistent:
            h1, h2 = torch.chunk(h, 2, dim = -1)
        if full_hidden:
            hs = [h]

        for t in range(g.shape[0]):
            if self.persistent:
                z1 = self.capture(torch.cat((x[t], h2[0]), dim = -1)) #2 * hidden_size
                z2 = self.predict(torch.cat((h1[0], h2[0]), dim = -1), a[t]) #2 * hidden_size

                z1_1, z1_2 = torch.chunk(z1, 2, dim = -1)
                z2_1, z2_2 = torch.chunk(z2, 2, dim = -1)

                h1 = (1-g[t]) * z1_1 + g[t] * z2_1
                h1 = h1.unsqueeze(0)

                h2 = (1-g[t]) * z1_2 + g[t] * z2_2
                h2 = h2.unsqueeze(0)

                outs.append(h1)
                capt.append(z1_1)

                h = torch.cat((h1, h2), dim = -1)
            else:
                z1 = self.capture(x[t])
                z2 = self.predict(h[0], a[t])

                h = (1-g[t]) * z1 + g[t] * z2
                h = h.unsqueeze(0)

                outs.append(h)
                capt.append(z1)
            if full_hidden:
                hs.append(h)

        if full_hidden:
            return torch.cat(outs), h, torch.cat(capt), torch.cat(hs)
        else:
            return torch.cat(outs), h, torch.cat(capt)


class OpsBase(NNBase):
    def __init__(self, num_inputs, action_space, is_leaf,
                recurrent=False, hidden_size=64, partial_obs=False,
                gate_input='obs', persistent=False, mode='linear',
                resolution_scale=1.0, pred_mode='pred_model', fixed_probability=None):
        super(OpsBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.is_leaf = is_leaf
        self.gate_input = gate_input
        self.hidden_size = hidden_size
        self.persistent = persistent
        self.fixed_probability = fixed_probability
        if self.fixed_probability != None:
            self.bernoulli = torch.distributions.Bernoulli(torch.tensor([self.fixed_probability])) #chance of getting 1, i.e. predict
        self.action_space = action_space
        self.discrete_action = (action_space.__class__.__name__ == "Discrete")

        if mode == 'cnn':
            self.cnn = self.make_cnn(in_dim=num_inputs, out_dim=hidden_size)
            num_inputs = hidden_size


        if is_leaf:
            if self.discrete_action:
                self.act_emb = nn.Embedding(action_space.n + 1, hidden_size, padding_idx=0)
            # else:
            #     self.act_emb = nn.Sequential(
            #         init_(nn.Linear(action_space.shape[0], hidden_size)), nn.ReLU())

            self.cell = OpsCell(num_inputs, act_dim=hidden_size, hidden_size=hidden_size, persistent=persistent, pred_mode=pred_mode)
            self.actor = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        else:
            if self.gate_input == 'hid':
                if self.persistent:
                    num_inputs = 2 * hidden_size
                else:
                    num_inputs = hidden_size

            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),)

            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def make_cnn(self, in_dim, out_dim):
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                    constant_(x, 0), nn.init.calculate_gain('relu'))

        C, H, W = in_dim
        # For 80x60 input
        # assert np.prod(in_dim[1:]) == 80 * 60
        out_shape = (((((H-4-1)//2+1-4-1)//2+1)-3-1)//2+1) * (((((W-4-1)//2+1-4-1)//2+1)-3-1)//2+1)
        encoder = nn.Sequential(
            init_(nn.Conv2d(in_dim[0], 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=5, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            init_(nn.Conv2d(32, 32, kernel_size=4, stride=2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            Flatten(),

            init_(nn.Linear(32 * out_shape, out_dim)),
            nn.ReLU()
        )


        return encoder

    def forward(self, inputs, rnn_hxs, masks, info=None):
        x = inputs

        if self.is_leaf:
            if x.ndim > 3:
                x = x/255
                x = self.cnn(x)
                # if not self.is_leaf:
                #     x = x.detach()

        if info is not None and info.numel() > 0: #2
            assert len(info.shape) == 2
            g, a = info[:, 0].unsqueeze(dim=-1), info[:, 1:]
            if self.fixed_probability != None:
                g = torch.cat([self.bernoulli.sample() for _ in range(g.shape[0])]).unsqueeze(dim=-1).to(g.get_device())
            if self.discrete_action:
                # consider embedding all observations and actions before passing to gru...
                a = self.act_emb(a.squeeze(-1).long())

            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks, g, a)

            hidden_critic = self.critic(x)
            hidden_actor = self.actor(x)
        else: #1
            hidden_critic = self.critic(rnn_hxs)
            hidden_actor = self.actor(rnn_hxs)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs, x



    def _forward_gru(self, x, hxs, masks, gate, action):
        if x.size(0) == hxs.size(0):
            x, hxs, capt = self.cell(
                x.unsqueeze(0),
                (hxs * masks).unsqueeze(0),
                (gate * masks).unsqueeze(0),
                (action * masks).unsqueeze(0))

            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
            capt = capt.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N) #128

            # unflatten
            x = x.view(T, N, x.size(1))
            gate = gate.view(T, N, gate.size(1))
            action = action.view(T, N, action.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            capts = []
            # hxs: [1, 4, 256]
            # masks: [512, 4]
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs, capt = self.cell(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1), # [1, 4, 256]
                    gate[start_idx:end_idx] * masks[start_idx:end_idx, ..., None],
                    action[start_idx:end_idx] * masks[start_idx:end_idx, ..., None])

                outputs.append(rnn_scores)
                capts.append(capt)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            capts = torch.cat(capts, dim=0)

            # flatten
            x = x.view(T * N, -1)
            capts = capts.squeeze(0)
            hxs = hxs.squeeze(0)

        # TODO Use capts
        return x, hxs
