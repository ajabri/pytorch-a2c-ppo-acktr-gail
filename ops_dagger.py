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
EXP_NAME = 'async/relocate'


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
    test_e = AsyncWrapper(e, obs_interval=obs_interval, predict_interval=predict_interval, no_op=no_op, no_op_action = None)
    expert_e = AsyncWrapper(e, obs_interval=1, predict_interval=1, no_op=no_op, no_op_action = None)
    test_e._horizon = math.ceil(e._horizon / obs_interval)
    expert_e._horizon = e._horizon

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
        dagger_policy = DaggerPolicy(env, student, expert, actor_critic, obs_dim, act_dim)
        num_rollouts=20

        for i in range(5000):
            epochs = 4
                # dagger_policy.fraction_assist -= 0.01
            _, log_info = get_data(env, dagger_policy, num_rollouts=num_rollouts, env_name=kwargs['env_name'],
                                   device=device, hidden_size=kwargs['hidden_size'], ops=kwargs['ops'])
            student.train(dagger_policy.obs_data, dagger_policy.act_data, dagger_policy.info, epochs=epochs, ops=kwargs['ops'])
            ops.evaluate(test_env, log_info, kwargs['env_name'], rollout_num=kwargs['rollout_num'], ops=kwargs['ops'])
            wandb.log(log_info)
            print(i, log_info)

    dagger(expert_e, test_e, student, expert, obs_dim, act_dim)





# NOTE: 200 paths, episode length 100

if __name__ == "__main__":
    # sweep_params = {
    #     'algo': ['dagger'],
    #     'seed': [111, 123],
    #     'env_name': ['hammer-v0', 'pen-v0', 'door-v0', 'relocate-v0'],
    #
    #     'cuda': [False],
    #     'proj_name': ['dagger3'],
    #     # 'proj_name': ['debug'],
    #     'note': [''],
    #     'debug': [False],
    #     'gate_input': ['hid'], #'obs' | 'hid'
    #     'persistent': [False],
    #     'hidden_size': [32],
    #     'pred_loss': [False],
    #     'obs_interval': [1, 2, 5, 10],
    #     'predict_interval': [1],
    #     'no_op': [False],
    #     'extend_horizon': [False],
    #     }
    #
    # run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)

    sweep_params = {
        'algo': 'dagger',
        'seed': 111,
        'env_name': 'pen-v0',

        'cuda': False,
        # 'proj_name': 'dagger3',
        'proj_name': 'debug',
        'note': '',
        'debug': False,
        'hidden_size': 32,
        'gate_input': 'hid', #'obs' | 'hid'
        'persistent': False,
        'hidden_size': 32,
        'pred_loss': False,
        'obs_interval': 10,
        'predict_interval': 1,
        'no_op': False,
        'extend_horizon': False,
        'ops': True,
        'clip_param': 0.1,
        'num_mini_batch': 1,
        'entropy_coef': 5e-3,
        'lr': 2.5e-2,
        'rollout_num': 20,
        'num_mini_batch': 4,
        'bc': False,
        }


    # args = get_args()
    # for arg in vars(args):
    #     sweep_params[arg] = getattr(args, arg)

    main(**sweep_params)
