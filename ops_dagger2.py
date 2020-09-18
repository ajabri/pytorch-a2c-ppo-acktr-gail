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
from a2c_ppo_acktr.bc_model2 import *
from a2c_ppo_acktr.utils import get_vec_normalize


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
    wandb.init(project=kwargs['proj_name'], config=kwargs)
    torch.manual_seed(kwargs['seed'])
    torch.set_num_threads(1)
    async_params=[kwargs['obs_interval'], kwargs['predict_interval'], kwargs['no_op']]

    real_envs = make_vec_envs(kwargs['env_name'], kwargs['seed'], kwargs['num_processes'],
                         kwargs['gamma'], '', device, False,
                         # resolution_scale=kwargs['scale'], image_stack=kwargs['image_stack'],
                         async_params=async_params)

    discrete_action = real_envs.action_space.__class__.__name__ == "Discrete"
    obs_dim = real_envs.observation_space.shape[0]
    act_dim = 1 if discrete_action else real_envs.action_space.shape[0]

    save_path = "/home/vioichigo/pytorch-a2c-ppo-acktr-gail/policies/" + kwargs['env_name'] + ".pt"
    expert, ob_rms = torch.load(save_path)

    vec_norm = get_vec_normalize(real_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    # expert = pickle.load(open('/home/vioichigo/pytorch-a2c-ppo-acktr-gail/policies/' + kwargs['env_name'] + '.pickle', 'rb'))
    student = ImitateTorch(obs_dim, real_envs.action_space, hidden_size=kwargs['hidden_size'], device=device)

    actor_critic = OpsPolicy(
        real_envs.observation_space.shape, gym.spaces.Discrete(2),
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

    rollouts = RolloutStorage(kwargs['num_steps'], kwargs['num_processes'],
                              real_envs.observation_space.shape, real_envs.action_space,
                              kwargs['hidden_size'], info_size=0, action_shape=1)
    obs = real_envs.reset()
    rollouts.obs[0].copy_(obs)
    ops = Ops(agent, student, rollouts, **kwargs)
    dagger_policy = DaggerPolicy(real_envs, student, expert, actor_critic, obs_dim,
                                 act_dim, capacity=(kwargs['num_steps'], kwargs['num_processes']))

    step_indices = torch.from_numpy(np.array([0 for _ in range(kwargs['num_processes'])])).to(device)
    action1 = torch.zeros((kwargs['num_processes'], 1)).to(device).long()
    if discrete_action:
        last_action = rollouts.actions[0] + 1
    else:
        last_action = rollouts.actions[0]
        action1 = action1.float()

    for i in range(200):
        epochs = 4
        log_info = get_data(dagger_policy, num_steps=kwargs['num_steps'], num_processes=kwargs['num_processes'],
                            action_dim=act_dim, device=device, hidden_size=kwargs['hidden_size'], ops=kwargs['ops'],
                            info=[torch.cat([action1, last_action], dim=1), step_indices],
                            env_info=[kwargs['env_name'], kwargs['seed'], kwargs['num_processes'],
                                     kwargs['gamma'], '', device, False, [1, 1, kwargs['no_op']]], save_path=save_path)
        student.train(dagger_policy.obs_data, dagger_policy.act_data, dagger_policy.info,
                      epochs=epochs, ops=kwargs['ops'], lr=kwargs['bc_lr'])

        ops.evaluate(real_envs, log_info, kwargs['env_name'], num_processes=kwargs['num_processes'],
                     num_steps=kwargs['num_steps'], ops=kwargs['ops'], async_params=async_params, seed=kwargs['seed'])
        wandb.log(log_info)
        print(i, log_info)






# NOTE: 200 paths, episode length 100

if __name__ == "__main__":
    # run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)

    sweep_params = {
        'algo': 'dagger',
        'seed': 111,
        'env_name': 'InvertedPendulum-v2',

        'cuda': False,
        # 'proj_name': 'dagger',
        'proj_name': 'debug',
        'note': '',
        'debug': False,
        'hidden_size': 64,
        'num_steps': 2048,
        'gate_input': 'hid', #'obs' | 'hid'
        'persistent': False,
        'pred_loss': False,
        'obs_interval': 1,
        'predict_interval': 1,
        'no_op': False,
        'ops': True,
        'clip_param': 0.1,
        'entropy_coef': 5e-3,
        'lr': 2.5e-2,
        'bc_lr': 1e-3,
        'num_processes': 4,
        'num_mini_batch': 2,
        }


    # args = get_args()
    # for arg in vars(args):
    #     sweep_params[arg] = getattr(args, arg)

    main(**sweep_params)
