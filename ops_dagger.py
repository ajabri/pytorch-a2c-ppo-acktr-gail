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

from a2c_ppo_acktr import algo, utils
from args import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import OpsPolicy
from a2c_ppo_acktr.storage import RolloutStorage
import time
import torch.optim as optim
import wandb
from a2c_ppo_acktr import logging
import pickle
import math

import json
from evaluation import *
from experiment_utils.run_sweep import run_sweep
from a2c_ppo_acktr.wrappers import *
from a2c_ppo_acktr.bc_model import *


INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'async/pen2'


class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)

device = 'cpu'
def main(**kwargs):
    from a2c_ppo_acktr.arguments import get_args
    args = get_args()

    for arg in vars(args):
        if arg not in kwargs:
            kwargs[arg] = getattr(args, arg)
    assert kwargs['no_op'] == False
    assert kwargs['extend_horizon'] == False
    from mjrl.utils.gym_env import GymEnv
    import sys
    sys.path.append('/root/code/mj_envs')
    import mj_envs
    wandb.init(project=kwargs['proj_name'], config=kwargs)
    torch.manual_seed(kwargs['seed'])

    e = GymEnv(kwargs['env_name'])
    obs_interval, predict_interval, no_op = kwargs['obs_interval'], kwargs['predict_interval'], kwargs['no_op']
    async_params=[obs_interval, predict_interval, no_op]
    test_e = HandNormAsyncWrapper(e, obs_interval=obs_interval, predict_interval=predict_interval,
                             no_op=no_op, no_op_action = None, gamma=kwargs['gamma'])
    expert_e = HandAsyncWrapper(e, obs_interval=1, predict_interval=1, no_op=no_op, no_op_action = None)
    expert_e._horizon = e._horizon
    test_e._horizon = e._horizon

    demo_file = '/home/vioichigo/pytorch-a2c-ppo-acktr-gail/demonstrations/' + kwargs['env_name'] + '_demos.pickle'
    demo_paths = pickle.load(open(demo_file, 'rb'))

    obs_dim, act_dim = e._observation_dim, e._action_dim

    expert = pickle.load(open('/home/vioichigo/pytorch-a2c-ppo-acktr-gail/policies/' + kwargs['env_name'] + '.pickle', 'rb'))
    student = ImitateTorch(obs_dim, act_dim, hidden_size=kwargs['hidden_size'], device=device)

    actor_critic = OpsPolicy(
        e.observation_space.shape, gym.spaces.Discrete(2),
        is_leaf=False,
        base_kwargs=dict(
            recurrent=True,
            gate_input='hid',
            hidden_size=kwargs['hidden_size'],
            ))

    agent = algo.PPO(
        actor_critic,
        kwargs['clip_param'],
        kwargs['ppo_epoch'],
        kwargs['num_mini_batch'],
        kwargs['value_loss_coef'],
        kwargs['entropy_coef'],
        lr=kwargs['lr'],
        eps=kwargs['eps'],
        max_grad_norm=kwargs['max_grad_norm'])

    rollouts = RolloutStorage(expert_e._horizon, kwargs['rollout_num'], e.observation_space.shape, e.action_space,
                                kwargs['hidden_size'], info_size=0, action_shape=1)

    if kwargs['bc']:
        behavior_cloning(student, demo_paths, test_e, expert_e, device=device)
    ops = Ops(agent, student, rollouts, **kwargs)

    def dagger(env, test_env, student, expert, obs_dim, act_dim):
        dagger_policy = DaggerPolicy(env, student, expert, actor_critic, obs_dim, act_dim, capacity=kwargs['rollout_num']*e._horizon)

        for i in range(2000):
            epochs = 4
            _, log_info = get_data(env, dagger_policy, num_rollouts=kwargs['rollout_num'], env_name=kwargs['env_name'],
                                   device=device, hidden_size=kwargs['hidden_size'], pred_method=kwargs['decision'])
            student.train(dagger_policy.obs_data, dagger_policy.act_data, dagger_policy.info, epochs=epochs, pred_method=kwargs['decision'], lr=kwargs['bc_lr'])
            ops.evaluate(test_env, log_info, kwargs['env_name'], rollout_num=kwargs['rollout_num'],
                            pred_method=kwargs['decision'], async_params=async_params, seed=kwargs['seed'])
            wandb.log(log_info)
            print(i, log_info)

    dagger(expert_e, test_e, student, expert, obs_dim, act_dim)



# NOTE: 200 paths, episode length 100

if __name__ == "__main__":
    # run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)

    # sweep_params = {
    #     'algo': 'dagger',
    #     'seed': 111,
    #     'env_name': 'pen-v0',
    #
    #     'cuda': False,
    #     # 'proj_name': 'dagger',
    #     'proj_name': 'debug',
    #     'note': '',
    #     'debug': False,
    #     'hidden_size': 32,
    #     'gate_input': 'hid', #'obs' | 'hid'
    #     'persistent': False,
    #     'pred_loss': False,
    #     'obs_interval': 2,
    #     'predict_interval': 1,
    #     'no_op': False,
    #     'extend_horizon': False,
    #     'decision': 'ops', #'fixed', 'ops', 'none', 'random'
    #     'clip_param': 0.1,
    #     'entropy_coef': 5e-3,
    #     'lr': 2.5e-2,
    #     'bc_lr': 1e-2,
    #     'rollout_num': 40,
    #     'num_mini_batch': 4,
    #     'bc': False,
    #     }

    sweep_params = {
        'algo': ['dagger'],
        'seed': [111, 123],
        'env_name': ['pen-v0'],

        'cuda': [False],
        'proj_name': ['dagger12'],
        # 'proj_name': ['debug'],
        'note': ['single'],
        'debug': [False],
        'hidden_size': [32],
        'gate_input': ['hid'], #'obs' | 'hid'
        'persistent': [False],
        'pred_loss': [False],
        'obs_interval': [3, 5, 10, 15, 20],
        'predict_interval': [1],
        'no_op': [False],
        'extend_horizon': [False],
        'decision': ['fixed', 'ops', 'none', 'random'], #'fixed', 'ops', 'none', 'random'
        'clip_param': [0.1],
        'entropy_coef': [5e-3],
        'lr': [2.5e-2],
        'bc_lr': [1e-2],
        'rollout_num': [40],
        'num_mini_batch': [8],
        'bc': [False],
        }

    run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)

    # args = get_args()
    # for arg in vars(args):
    #     sweep_params[arg] = getattr(args, arg)

    # main(**sweep_params)
