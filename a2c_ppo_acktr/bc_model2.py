import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from evaluation import *
from collections import deque
from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from a2c_ppo_acktr.utils import get_vec_normalize

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

class Model(torch.nn.Module):
    def __init__(self, obs_dim, action_space, hidden_size, discrete_action=False):
        super(Model, self).__init__()
        self.discrete_action = discrete_action

        if discrete_action:
            num_outputs = action_space.n
            self.dist = Categorical(hidden_size, num_outputs)
        else:
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(hidden_size, num_outputs)

        if discrete_action:
            self.act_emb = nn.Embedding(action_space.n + 1, hidden_size, padding_idx=0)
            self.act_dim = 1
        else:
            self.act_emb = nn.Sequential(
                init_(nn.Linear(action_space.shape[0], hidden_size)), nn.ReLU())
            self.act_dim = action_space.shape[0]

        self.capture = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            )

        self.predict = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            )
        self.hidden_size = hidden_size
        self.out = nn.Sequential(
            # init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

    def forward(self, x, decision=None, memory=None, deterministic=False):
        if decision == None:
            decision = torch.zeros((x.shape[0], 1))
        if memory == None:
            pred = 0
        else:
            last_a, rnn_hxs = memory[:, :self.act_dim], memory[:, self.act_dim:]
            if self.discrete_action:
                last_a = last_a.reshape((-1))
            last_a = self.act_emb((last_a+1))
            memory = torch.cat((last_a, rnn_hxs), dim=-1)
            pred = self.predict(memory)

        capt = self.capture(x)
        feature = (1 - decision) * capt + decision * pred
        actor_features = self.out(feature)
        dist = self.dist(actor_features) #[16, 64]

        if deterministic:
            action = dist.mode() #[16, 3]
        else:
            action = dist.sample()

        return action, feature

    def pred_loss(self, x, memory):
        capt = self.capture(x)
        pred = self.predict(memory)
        return ((capt-pred)**2).mean()


class Ops:
    def __init__(self, agent, policy2, rollouts, device='cpu', **kwargs):
        self.device = device
        self.agent = agent
        self.policy1 = agent.actor_critic
        self.policy2 = policy2
        self.rollouts = rollouts
        self.gamma = kwargs['gamma']
        self.gae_lambda = kwargs['gae_lambda']

    def train_policy1(self, rollout_num):
        with torch.no_grad():
            next_value = self.policy1.get_value(
                self.rollouts.obs[-1].to(self.device), self.rollouts.recurrent_hidden_states[-1],
                self.rollouts.masks[-1], info=None).detach()

        self.rollouts.compute_returns(next_value, True, self.gamma,
                                self.gae_lambda, True)

        value_loss, action_loss, dist_entropy, _ = self.agent.update(self.rollouts,
            pred_loss=False, require_memory=True, num_processes=rollout_num,
            device=self.device)

        self.rollouts.after_update()

        return value_loss, action_loss, dist_entropy

    def evaluate(self, envs, log_info, env_name, num_processes, num_steps, ops=True, seed=1, async_params=[False, 1, 1]):
        episode_rewards = deque(maxlen=10)
        obs = self.rollouts.obs[0].to(self.device)
        memory = torch.zeros((num_processes, self.policy2.model.hidden_size+self.policy2.model.act_dim), device=self.device)
        recurrent_hidden_state = torch.zeros((num_processes, self.policy2.model.hidden_size), device=self.device)
        mask = torch.zeros((num_processes, 1), device=self.device)
        for _ in range(num_steps):
            with torch.no_grad():
                if ops:
                    value, decision, action_log_prob, _ = self.policy1.act(obs, recurrent_hidden_state, mask)
                    # decision = torch.randint(0, 2, (num_processes, 1))
                else:
                    decision = torch.zeros((num_processes, 1))

                action, recurrent_hidden_state = self.policy2.model(obs, decision=decision, memory=memory)

            real_act = torch.cat((decision.float(), action), dim=-1)
            obs, reward, done, infos = envs.step(real_act)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            if ops:
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]).to(self.device)
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos]).to(self.device)
                memory=torch.cat((action, recurrent_hidden_state), dim=-1)

                self.rollouts.insert(obs, recurrent_hidden_state, decision, action_log_prob,
                                     value, reward, masks, bad_masks, infos=None)
        if ops:
            value_loss1, action_loss1, dist_entropy1 = self.train_policy1(num_processes)
            log_info['value_loss1'] = value_loss1
            log_info['action_loss1'] = action_loss1
            log_info['dist_entropy1'] = dist_entropy1
            log_info['mean_gt'] = self.rollouts.actions.float().mean().item()
        log_info['student_mean_return'] = np.mean(episode_rewards)
        log_info['student_median_return'] = np.median(episode_rewards)
        log_info['student_max_return'] = np.max(episode_rewards)
        log_info['student_min_return'] = np.min(episode_rewards)




class ImitateTorch:
    def __init__(self, obs_dim, action_space, hidden_size, device='cpu'):
        self.device = device
        self.obs_dim = obs_dim
        discrete_action = action_space.__class__.__name__ == "Discrete"
        self.model = Model(obs_dim, action_space, hidden_size, discrete_action=discrete_action).to(device)

    def train(self, X, Y, last_Y, epochs=1, ops=True, lr=0.01):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr = lr)
        num_steps, num_processes, _ = X.shape
        X, Y, last_Y = X.reshape((num_steps*num_processes, -1)), Y.reshape((num_steps*num_processes, -1)), last_Y.reshape((num_steps*num_processes, -1))
        X, Y, last_Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float(), torch.from_numpy(last_Y).float()
        dataset = torch.utils.data.TensorDataset(X, Y, last_Y)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in (range(epochs)):
            running_loss = 0.0
            for i, data in enumerate(data_loader, 0):

                inputs, labels, memory = data
                inputs, labels, memory = inputs.to(self.device), labels.to(self.device), memory.to(self.device)
                decision = memory[:, 0].reshape((-1, 1))
                if not ops:
                    assert decision.sum() == 0
                memory = memory[:, 1:]
                optimizer.zero_grad()
                action, features = self.model(inputs, decision=decision, memory=memory, deterministic=True)
                # if not ops:
                #     loss = criterion(action, labels)
                # else:
                #     loss = criterion(action, labels) + 0.1* self.model.pred_loss(inputs, memory)
                loss = criterion(action, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print('[%d] loss: %.6f' % (epoch + 1, running_loss / (i+1)))


    def __call__(self, obs, decision=None, memory=None):
        output, feature = self.model(obs, decision, memory)
        return output.detach().to('cpu').numpy(), feature

class DaggerPolicy:
    def __init__(self, env, student, expert, policy1, obs_dim, act_dim, capacity=(1, 10000)):
        self.num_steps, self.num_processes = capacity
        self.student = student
        self.expert = expert
        self.policy1 = policy1
        self.fraction_assist = 1.
        self.next_idx = 0
        self.size = 0

        self.obs_data = np.empty([self.num_steps, self.num_processes, obs_dim])
        self.act_data = np.empty([self.num_steps, self.num_processes, act_dim])
        self.info = np.zeros([self.num_steps, self.num_processes, act_dim+student.model.hidden_size+1])

    def __call__(self, obs, memory, done, last_a, ops=True, device='cpu', info=None):
        with torch.no_grad():
            if ops:
                _, decision, _, _ = self.policy1.act(obs, memory, done)
            else:
                decision = torch.zeros((obs.shape[0], 1)).to(device)

            _, expert_action, _, recurrent_hidden_states = self.expert.act(obs, memory, done, info=info)
            _, memory = self.student(obs, decision=decision, memory=torch.cat((last_a, memory), dim=-1))

        expert_action = expert_action.numpy()
        info = np.concatenate((decision.numpy(), expert_action, memory.numpy()), axis = -1)
        num_processes = obs.shape[0]
        self.obs_data[self.next_idx] = obs.numpy()
        self.act_data[self.next_idx] = expert_action
        self.next_idx = (self.next_idx+1) % self.num_steps
        if self.next_idx != 0:
            self.info[self.next_idx] = info
        self.size = min(self.size+1, self.num_steps)
        return expert_action, memory, recurrent_hidden_states


def get_data(policy_fn, num_steps, num_processes, action_dim,
             device='cpu', hidden_size=32, ops=True, info=None, env_info=None, save_path=''):

    env_name, seed, num_processes, gamma, _, _, _, async_params = env_info
    envs = make_vec_envs(env_name, seed, num_processes, gamma, '', device, False,
                         async_params=async_params)

    _, ob_rms = torch.load(save_path)
    vec_norm = get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    obs = envs.reset()
    episode_rewards = deque(maxlen=10)
    rnn_hxs = torch.zeros((num_processes, hidden_size)).to(device)
    last_a = torch.zeros((num_processes, action_dim)).to(device)
    action1 = torch.zeros((num_processes, 1)).to(device)
    dones = torch.zeros((num_processes, 1)).to(device)
    always_observe = np.zeros((num_processes, 1)).astype(int)

    for step in range(num_steps):
        action, memory, rnn_hxs = policy_fn(obs, rnn_hxs, dones.reshape((-1, 1)), last_a, ops=ops, info=info)
        # synchronous environment, not necessary to use real decision
        real_act = torch.from_numpy(np.concatenate((always_observe, action), axis = -1)).to(device)
        obs, r, dones, infos = envs.step(real_act)
        last_a = torch.from_numpy(action).float()
        info =[torch.cat([action1, last_a], dim=1), info[-1]]

        for i in infos:
            if 'episode' in i.keys():
                episode_rewards.append(i['episode']['r'])

    log_info = {'expert_mean_return': np.mean(episode_rewards),
                'expert_median_return': np.median(episode_rewards),
                'expert_max_return': np.max(episode_rewards),
                'expert_min_return': np.min(episode_rewards),
                }

    envs.close()
    return log_info
