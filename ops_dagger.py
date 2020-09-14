from pdb import set_trace as st
import copy
import glob
import os
import time
from collections import deque
import warnings
warnings.simplefilter("ignore", UserWarning)

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from args import get_args
# from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from torch.autograd import Variable
from a2c_ppo_acktr.model import OpsPolicy
from a2c_ppo_acktr.storage import RolloutStorage
import time
import torch.optim as optim
import wandb
from a2c_ppo_acktr import logging
import pickle

import json
from evaluation import *
from experiment_utils.run_sweep import run_sweep
from a2c_ppo_acktr.wrappers import *


INSTANCE_TYPE = 'c4.4xlarge'
EXP_NAME = 'async/relocate'


class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)
device = 'cpu'

def evaluate_policy(env,
                    policy,
                    num_episodes=5,
                    horizon=None,
                    gamma=1,
                    visual=False,
                    percentile=[],
                    get_full_dist=False,
                    mean_action=False,
                    init_env_state=None,
                    terminate_at_done=True,
                    seed=123,
                    device='cpu'):

    env.set_seed(seed)
    horizon = env.env._horizon if horizon is None else horizon
    mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
    ep_returns = np.zeros(num_episodes)

    for ep in range(num_episodes):
        env.reset()
        if init_env_state is not None:
            env.set_env_state(init_env_state)
        t, done = 0, False
        while t < horizon and (done == False or terminate_at_done == False):
            env.render() if visual is True else None
            o = env.get_obs().reshape((1, -1))
            act = policy(torch.from_numpy(o).float().to(device)).detach().numpy().reshape((-1))
            mask = np.zeros((1))
            real_act = np.concatenate((mask, act))
            o, r, done, _ = env.step(real_act)
            ep_returns[ep] += (gamma ** t) * r
            t += 1

    mean_eval, std = np.mean(ep_returns), np.std(ep_returns)
    min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
    base_stats = [mean_eval, std, min_eval, max_eval]

    percentile_stats = []
    for p in percentile:
        percentile_stats.append(np.percentile(ep_returns, p))

    full_dist = ep_returns if get_full_dist is True else None

    return [base_stats, percentile_stats, full_dist][0][0]


class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, output_size))

    def forward(self, x):
        return self.model(x)

def main(kwargs):
    wandb.init(project=kwargs['proj_name'], config=kwargs)
    torch.manual_seed(kwargs['seed'])

    class ImitateTorch:
        def __init__(self, env, obs_dim, act_dim):
            self.model = Model(obs_dim, act_dim).to(device)

        def train(self, X, Y, epochs=1):
            criterion = nn.MSELoss()
            optimizer = optim.SGD(self.model.parameters(), lr = 0.01)
            X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
            dataset = torch.utils.data.TensorDataset(X, Y)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

            for epoch in (range(epochs)):

                running_loss = 0.0
                for i, data in enumerate(data_loader, 0):

                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                print('[%d] loss: %.6f' % (epoch + 1, running_loss / (i+1)))

        def evaluate(self, env, log_info):
            returns = []
            all_paths = []
            num_rollouts = 40

            for _ in range(num_rollouts):
                obs = env.reset()
                done = False
                step = 0
                totalr = 0
                path = {}
                path['env_infos'] = {}
                path['env_infos']['goal_achieved'] = []
                while step < env.env._horizon and done == False:
                    action = self.model(torch.from_numpy(obs).float()).detach().numpy()
                    mask = np.zeros((1))
                    real_act = np.concatenate((mask, action))
                    obs, r, done, info = env.step(real_act)
                    totalr += r
                    step += 1
                    path['env_infos']['goal_achieved'].append(info['goal_achieved'])
                returns.append(totalr)
                all_paths.append(path)

            success_rate = evaluate_success(kwargs['env_name'], all_paths)
            print('student mean return', np.mean(returns))
            print('student std of return', np.std(returns))
            print('student success_rate', success_rate)
            log_info['student_mean_return'] = np.mean(returns)
            log_info['student_std_return'] = np.std(returns)
            log_info['student_success_rate'] = success_rate


        def __call__(self, obs):
            obs_tensor = torch.from_numpy(obs).to(device)
            output = self.model(obs_tensor)
            return output.detach().to('cpu').numpy()


    class DaggerPolicy:
        def __init__(self, env, student, expert, obs_dim, act_dim):
            self.CAPACITY = 50000
            self.student = student
            self.expert = expert
            self.fraction_assist = 1.
            self.next_idx = 0
            self.size = 0

            self.obs_data = np.empty([self.CAPACITY, obs_dim])
            self.act_data = np.empty([self.CAPACITY, act_dim])

        def __call__(self, obs):
            expert_action = self.expert.get_action(obs)[1]['mean']
            student_action = self.student(obs).reshape((-1))
            self.obs_data[self.next_idx] = np.float32(obs)
            self.act_data[self.next_idx] = np.float32(expert_action)
            self.next_idx = (self.next_idx+1) % self.CAPACITY
            self.size = min(self.size+1, self.CAPACITY)
            if np.random.random()<self.fraction_assist:
                return expert_action
            else:
                return student_action

        def expert_data(self):
            return(self.obs_data[:self.size], self.act_data[:self.size])


    def get_data(env, policy_fn, num_rollouts, render=False):
        returns = []
        observations = []
        actions = []
        all_paths = []
        for _ in range(num_rollouts):
            obs = env.reset()
            done = False
            step = 0
            totalr = 0
            path = {}
            path['env_infos'] = {}
            path['env_infos']['goal_achieved'] = []
            while step < e.env._horizon and done == False:
            # for _ in range(e._horizon):
                action = policy_fn(np.float32(obs))
                observations.append(obs)
                actions.append(action)
                mask = np.zeros((1))
                real_act = np.concatenate((mask, action))
                obs, r, done, info = env.step(real_act)
                totalr += r
                path['env_infos']['goal_achieved'].append(info['goal_achieved'])
            returns.append(totalr)
            all_paths.append(path)

        success_rate = evaluate_success(kwargs['env_name'], all_paths)

        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        print('success_rate', success_rate)
        log_info = {}
        log_info['mean_return'] = np.mean(returns)
        log_info['std_return'] = np.std(returns)
        log_info['success_rate'] = success_rate

        expert_data = {'observations': np.array(observations),
                         'actions': np.array(actions)}

        return (np.array(observations), np.array(actions)), log_info



    def dagger(env, student, expert, obs_dim, act_dim):
        dagger_policy = DaggerPolicy(env, student, expert, obs_dim, act_dim)
        num_rollouts=200

        for i in range(200):
            if i==0:
                epochs=4
            else:
                epochs = 4
                # dagger_policy.fraction_assist -= 0.01
            _, log_info = get_data(env, dagger_policy, num_rollouts=num_rollouts)
            student.train(dagger_policy.obs_data, dagger_policy.act_data, epochs=epochs)
            student.evaluate(env, log_info)
            wandb.log(log_info)



    from mjrl.utils.gym_env import GymEnv
    import mj_envs
    e = GymEnv(kwargs['env_name'])
    obs_interval, predict_interval, no_op = kwargs['obs_interval'], kwargs['predict_interval'], kwargs['no_op']
    # which is actually True in this case
    assert no_op == False
    e = AsyncWrapper(e, obs_interval=obs_interval, predict_interval=predict_interval, no_op=no_op, no_op_action = None)

    demo_file = '/home/vioichigo/ops/hand_dapg/dapg/demonstrations/' + kwargs['env_name'] + '_demos.pickle'
    demo_paths = pickle.load(open(demo_file, 'rb'))

    obs_dim, act_dim = e.env._observation_dim, e.env._action_dim
    student = ImitateTorch(e, obs_dim, act_dim)


    bc_optimizer = optim.Adam(student.model.parameters(), lr=1e-3)
    observations = np.concatenate([demo_path['observations'] for demo_path in demo_paths], axis = 0)
    actions = np.concatenate([demo_path['actions'] for demo_path in demo_paths], axis = 0)
    num_samples = observations.shape[0]
    mb_size = 64
    for _ in range(5):
        for _ in range(num_samples//mb_size):
            rand_idx = np.random.choice(num_samples, size=mb_size)
            bc_optimizer.zero_grad()
            obs = torch.from_numpy(observations[rand_idx]).to(device)
            gt_act = torch.from_numpy(actions[rand_idx]).to(device)
            act = student.model(obs.float())
            loss = ((act - gt_act.detach())**2).mean()
            loss.backward()
            bc_optimizer.step()
        print(loss.item())

    score = evaluate_policy(e, student.model, num_episodes=25, mean_action=True, device=device)
    print("Score with behavior cloning =", score)

    expert = pickle.load(open('/home/vioichigo/ops/hand_dapg/dapg/policies/' + kwargs['env_name'] + '.pickle', 'rb'))

    dagger(e, student, expert, obs_dim, act_dim)



# NOTE: 200 paths, episode length 100

if __name__ == "__main__":
    sweep_params = {
        'algo': 'ppo',
        'seed': 111,
        'env_name': 'hammer-v0',

        'use_gae': True,
        'lr': 2.5e-4,
        'clip_param': 0.2,
        'value_loss_coef': 0.5,
        'num_rollouts': 200,
        'num_processes': 1,
        'num_mini_batch': 1,
        'log_interval': 1,
        'use_linear_lr_decay': True,
        'entropy_coef': 0.005,
        'num_env_steps': 50000000,
        'bonus1': 0,
        'cuda': False,
        'proj_name': 'dagger2',
        # 'proj_name': 'debug',
        'note': '',
        'debug': False,
        'gate_input': 'hid', #'obs' | 'hid'
        'persistent': False,
        'hidden_size': 128,
        'always_zero': False,
        'pred_loss': False,
        'image_stack': False,
        'save_dir': '',
        'pred_mode': 'pred_model', #'pred_model' | 'pos_enc'
        'no_bonus': 0,
        'fixed_probability': None,
        'obs_interval': 1,
        'predict_interval': 1,
        'no_op': False,
        'scale': 1,
        'ppo_epoch': 10,
        }

    args = get_args()
    for arg in vars(args):
        sweep_params[arg] = getattr(args, arg)

    main(sweep_params)
