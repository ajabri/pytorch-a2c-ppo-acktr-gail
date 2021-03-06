import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

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

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

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

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

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
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1),
                    )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class Identity(nn.Module):
    def forward(self, x):
        return x


class OpsPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, is_leaf, base=None, base_kwargs=None):
        super(OpsPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

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

        # self.gate_dist = Categorical(self.base.output_size, 3)

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
        value, actor_features, rnn_hxs, all_hxs = self.base(inputs, rnn_hxs, masks, info=info)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, info=None):
        value, _, _, _ = self.base(inputs, rnn_hxs, masks, info=info)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, info=None):
        value, actor_features, rnn_hxs, all_hxs = self.base(inputs, rnn_hxs, masks, info=info)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, all_hxs


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        

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

    

class OpenLoop(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super(OpenLoop, self).__init__()

        self.model = nn.Sequential(
            init_(nn.Linear(act_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        )
        self.pos_encoder = PositionalEncoding(obs_dim, max_len=100)
        self.blind_count = None

    def forward(self, x, g, a):
        reset_count = g<=1
        if self.blind_count is None:
            self.blind_count = reset_count * 0
        self.blind_count *= reset_count

        x = self.pos_encoder(x)
        z = self.model(torch.cat([x, a], dim=-1))

        self.blind_count += 1

        return z


class OpsCell(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super(OpsCell, self).__init__()

        # observation model
        self.capture = nn.Sequential(
            init_(nn.Linear(obs_dim, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh()
        )
        
        # forward model
        self.predict = Dynamics(obs_dim=hidden_size, act_dim=act_dim, hidden_size=hidden_size)
        self.openloop = OpenLoop(obs_dim=hidden_size, act_dim=act_dim, hidden_size=hidden_size)

        for name, param in self.predict.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, h, g, a):
        # B x T x I
        # B x T x H
        T, B, H = x.shape

        outs = []
        capt = []

        reset_count = g <= 1

        for t in range(g.shape[0]):
            z1 = self.capture(x[t])
            z2 = self.predict(h[0], a[t])
            z3 = self.openloop(g[t], a[t])

            h = (1-g[t]) * z1 + g[t] * z2
            h = h.unsqueeze(0)

            outs.append(h)
            capt.append(z1)
        
        return torch.cat(outs), h, torch.cat(capt)


class OpsBase(NNBase):
    def __init__(self, num_inputs, action_space, is_leaf,
        recurrent=False, hidden_size=64, partial_obs=False, gate_input='obs'):
        super(OpsBase, self).__init__(recurrent, num_inputs, hidden_size)

        # if recurrent:
        #     num_inputs = hidden_size

        if partial_obs:
            self.cnn = self.make_cnn(in_dim=num_inputs, out_dim=hidden_size)
            num_inputs = hidden_size

        self.is_leaf = is_leaf
        self.gate_input = gate_input
                               
        if is_leaf:
            self.act_emb = nn.Embedding(action_space.n + 1, hidden_size, padding_idx=0)
            self.cell = OpsCell(num_inputs, act_dim=hidden_size, hidden_size=hidden_size)
        
        if is_leaf:
            self.actor = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

            self.critic = nn.Sequential(
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        else:
            self.actor = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),)

            self.critic = nn.Sequential(
                init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
                init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        
        self.train()

    def make_cnn(self, in_dim, out_dim):
        import kornia
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                    constant_(x, 0), nn.init.calculate_gain('relu'))

        encoder = nn.Sequential(
            # kornia.color.Normalize(
            #     mean=torch.Tensor([[0,0,0]]),
            #     std=torch.Tensor([[128, 128, 128]])
            # ),
            init_(nn.Conv2d(in_dim, 32, 3, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 3, stride=1)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 2 * 2, out_dim)), nn.ReLU())

        return encoder

    def forward(self, inputs, rnn_hxs, masks, info=None):
        x = inputs
        
        # import pdb; pdb.set_trace()
        if x.ndim > 3:
            x /= 255
            x = self.cnn(x)

        if info is not None and info.numel() > 0:
            g, a = torch.split(info, 1, dim=-1)

            # consider embedding all observations and actions before passing to gru...
            a = self.act_emb(a.squeeze(-1).long())
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks, g, a)
        else:
            if self.gate_input == 'hid':
                x = rnn_hxs
            elif not self.gate_input == 'obs':
                assert False, 'invalid gate iput'

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

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
            T = int(x.size(0) / N)

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
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                # import pdb; pdb.set_trace()

                rnn_scores, hxs, capt = self.cell(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1), 
                    gate[start_idx:end_idx] * masks[start_idx:end_idx, ..., None],
                    action[start_idx:end_idx] * masks[start_idx:end_idx, ..., None],
                    )

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