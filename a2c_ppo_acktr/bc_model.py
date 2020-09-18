import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from evaluation import *
from a2c_ppo_acktr.utils import init

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

class Model(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super(Model, self).__init__()

        self.capture = nn.Sequential(
            init_(nn.Linear(obs_dim, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            )

        self.predict = nn.Sequential(
            init_(nn.Linear(hidden_size+act_dim, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(),
            )
        self.hidden_size = hidden_size
        self.out = nn.Linear(hidden_size, act_dim)

    def forward(self, x, decision=None, memory=None):
        if decision == None:
            decision = torch.zeros((x.shape[0], 1))
        if memory == None:
            pred = 0
        else:
            pred = self.predict(memory)

        capt = self.capture(x)
        feature = (1 - decision) * capt + decision * pred
        action = self.out(feature)
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


    def evaluate(self, env, log_info, env_name, rollout_num, ops=True, async_params=None, seed=1):
        returns = []
        all_env_infos = []
        all_decisions = []

        for epi in range(rollout_num):
            obs = env.reset()
            self.rollouts.obs[0][epi].copy_(torch.from_numpy(obs).float())
            recurrent_hidden_state = torch.zeros((1, self.policy2.model.hidden_size), device=self.device)
            mask = torch.zeros((1, 1), device=self.device)
            done = False
            step = 0
            totalr = 0
            path = {'env_infos': {'goal_achieved': []}}
            record_success = True
            # while step < env._horizon and done == False:
            while step < env._horizon:
                with torch.no_grad():
                    action, feature = self.policy2.model(torch.from_numpy(obs).float().reshape((1, -1)))
                    action = action.numpy()
                    if ops:

                        # decision = torch.randint(0, 2, (1, 1))
                        value, decision, action_log_prob, _ = self.policy1.act(
                                            torch.from_numpy(env._obfilt(obs)).float().reshape((1, -1)), recurrent_hidden_state, mask)
                        # decision = torch.randint(0, 5, (1, 1))
                        # decision = decision<3
                        all_decisions.append(decision)
                    else:
                        decision = torch.zeros((1))

                real_act = np.concatenate((decision.numpy().reshape((-1)), action.reshape((-1))))
                obs, r, done, info = env.step(real_act)
                step += 1
                if record_success:
                    totalr += r
                    path['env_infos']['goal_achieved'].extend(info['goal_achieved'])
                if ops:
                    mask = torch.FloatTensor([0.0]).to(self.device) if done else torch.FloatTensor([1.0]).to(self.device)
                    self.rollouts.insert_single(torch.from_numpy(env._obfilt(obs)).float(), feature.reshape((-1)), decision.reshape((-1)),
                                    action_log_prob.reshape((-1)), value.reshape((-1)), torch.FloatTensor([r]).reshape((-1)),
                                    masks = mask, bad_masks = torch.FloatTensor([0.0]).to(self.device), idx=epi)

                recurrent_hidden_state = feature.reshape((1, -1))
                mask = mask.reshape((1, -1))
                if done:
                    obs = env.reset()
                    mask = torch.zeros((1, 1), device=self.device)
                    record_success = False
            returns.append(totalr)
            all_env_infos.append(path)

        if ops:
            value_loss1, action_loss1, dist_entropy1 = self.train_policy1(rollout_num)
            log_info['value_loss1'] = value_loss1
            log_info['action_loss1'] = action_loss1
            log_info['dist_entropy1'] = dist_entropy1
            log_info['mean_gt'] = np.array(all_decisions).mean()
        print([len(all_env_infos[i]['env_infos']['goal_achieved']) for i in range(40)])
        success_rate = evaluate_success(env_name, all_env_infos)
        log_info['student_mean_return'] = np.mean(returns)
        log_info['student_std_return'] = np.std(returns)
        log_info['student_success_rate'] = success_rate




class ImitateTorch:
    def __init__(self, obs_dim, act_dim, hidden_size, device='cpu'):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.model = Model(obs_dim, act_dim, hidden_size).to(device)

    def train(self, X, Y, last_Y, epochs=1, ops=True, lr=0.01):
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr = lr)
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
                action, features = self.model(inputs, decision=decision, memory=memory)
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
        obs_tensor = torch.from_numpy(obs).to(self.device)
        output, feature = self.model(obs_tensor, decision, memory)
        return output.detach().to('cpu').numpy(), feature

class DaggerPolicy:
    def __init__(self, env, student, expert, policy1, obs_dim, act_dim, capacity=10000):
        self.CAPACITY = capacity
        self.student = student
        self.expert = expert
        self.policy1 = policy1
        self.fraction_assist = 1.
        self.next_idx = 0
        self.size = 0

        self.obs_data = np.empty([self.CAPACITY, obs_dim])
        self.act_data = np.empty([self.CAPACITY, act_dim])
        self.info = np.empty([self.CAPACITY, act_dim+student.model.hidden_size+1])

    def __call__(self, obs, memory, done, last_a, ops=True, device='cpu'):
        expert_action = self.expert.get_action(obs)[1]['mean']
        self.obs_data[self.next_idx] = np.float32(obs)
        self.act_data[self.next_idx] = np.float32(expert_action)
        with torch.no_grad():
            if ops:
                _, decision, _, _ = self.policy1.act(torch.from_numpy(obs).float().reshape((1, -1)), memory, done)
                # decision = torch.randint(0, 2, (1, 1)).to(device)
            else:
                decision = torch.zeros((1, 1)).to(device)

            student_action, memory = self.student(obs.reshape((1, -1)), decision=decision, memory=torch.cat((last_a, memory), dim=-1))
            memory = memory.data.numpy()

        info = np.concatenate((decision.reshape((1, -1)), student_action.reshape((1, -1)), memory.reshape((1, -1))), axis = -1)
        self.info[(self.next_idx+1) % self.CAPACITY] = info
        self.next_idx = (self.next_idx+1) % self.CAPACITY
        self.size = min(self.size+1, self.CAPACITY)
        return expert_action, memory


def get_data(env, policy_fn, num_rollouts, env_name, render=False, device='cpu', hidden_size=32, ops=True):
    returns = []
    all_paths = []
    for _ in range(num_rollouts):
        obs = env.reset()
        memory = torch.zeros((1, hidden_size)).to(device)
        last_a = torch.zeros((1, env.env._action_dim)).to(device)
        done = False
        step = 0
        totalr = 0
        path = {'env_infos': {'goal_achieved': []}}
        # while step < env._horizon and done == False:
        while not done:
            action, memory = policy_fn(np.float32(obs), memory.reshape((1, -1)), done, last_a.reshape((1, -1)), ops=ops)
            mask = np.zeros((1))
            real_act = np.concatenate((mask, action))
            obs, r, done, info = env.step(real_act)
            totalr += r
            path['env_infos']['goal_achieved'].append(info['goal_achieved'])
            memory = torch.from_numpy(memory).to(device)
            last_a = torch.from_numpy(action).to(device)
        returns.append(totalr)
        all_paths.append(path)

    success_rate = evaluate_success(env_name, all_paths)

    log_info = {}
    log_info['mean_return'] = np.mean(returns)
    log_info['std_return'] = np.std(returns)
    log_info['success_rate'] = success_rate

    return None, log_info

#
#
# def evaluate_policy(env,
#                     policy,
#                     num_episodes=5,
#                     horizon=None,
#                     gamma=1,
#                     visual=False,
#                     percentile=[],
#                     get_full_dist=False,
#                     mean_action=False,
#                     init_env_state=None,
#                     terminate_at_done=True,
#                     seed=123,
#                     device='cpu'):
#
#     env.set_seed(seed)
#     horizon = env._horizon if horizon is None else horizon
#     mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
#     ep_returns = np.zeros(num_episodes)
#
#     for ep in range(num_episodes):
#         env.reset()
#         if init_env_state is not None:
#             env.set_env_state(init_env_state)
#         t, done = 0, False
#         while t < horizon and (done == False or terminate_at_done == False):
#             env.render() if visual is True else None
#             o = env.get_obs().reshape((1, -1))
#             act = policy(torch.from_numpy(o).float().to(device))[0].detach().numpy().reshape((-1))
#             mask = np.zeros((1))
#             real_act = np.concatenate((mask, act))
#             o, r, done, _ = env.step(real_act)
#             ep_returns[ep] += (gamma ** t) * r
#             t += 1
#
#     mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
#     min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
#     base_stats = [mean_eval, std, min_eval, max_eval]
#
#     percentile_stats = []
#     for p in percentile:
#         percentile_stats.append(np.percentile(ep_returns, p))
#
#     full_dist = ep_returns if get_full_dist is True else None
#
#     return [base_stats, percentile_stats, full_dist][0][0]
#
#
# def behavior_cloning(student, demo_paths, test_e, expert_e, device='cpu'):
#     bc_optimizer = optim.Adam(student.model.parameters(), lr=1e-3)
#     observations = np.concatenate([demo_path['observations'] for demo_path in demo_paths], axis = 0)
#     actions = np.concatenate([demo_path['actions'] for demo_path in demo_paths], axis = 0)
#     num_samples = observations.shape[0]
#     mb_size = 64
#     for _ in range(5):
#         for _ in range(num_samples//mb_size):
#             rand_idx = np.random.choice(num_samples, size=mb_size)
#             bc_optimizer.zero_grad()
#             obs = torch.from_numpy(observations[rand_idx]).to(device)
#             gt_act = torch.from_numpy(actions[rand_idx]).to(device)
#             act = student.model(obs.float())
#             loss = ((act[0] - gt_act.detach())**2).mean()
#             loss.backward()
#             bc_optimizer.step()
#         print(loss.item())
#     score = evaluate_policy(test_e, student.model, num_episodes=25, mean_action=True, device=device)
#     print("Score with behavior cloning in Asynchronous case =", score)
#
#     score = evaluate_policy(expert_e, student.model, num_episodes=25, mean_action=True, device=device)
#     print("Score with behavior cloning =", score)
