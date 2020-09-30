import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from pdb import set_trace as st
import wandb
from experiment_utils.run_sweep import run_sweep

from evaluation import evaluate

INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'async/mujoco-test-wrapper'

def main(**kwargs):
    args = get_args()
    for arg in vars(args):
        if arg not in kwargs:
            kwargs[arg] = getattr(args, arg)

    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed_all(kwargs['seed'])
    # wandb.init(project='atari-base', config = args)
    wandb.init(project=kwargs['proj_name'], config = kwargs)

    # if kwargs['cuda'] and torch.cuda.is_available() and kwargs['cuda_deterministic']:
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

    log_dir = os.getcwd() + '/data/' + EXP_NAME
    eval_log_dir = log_dir + "/eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:6" if kwargs['cuda'] else "cpu")
    async_params = [kwargs['obs_interval'], kwargs['pred_interval']]

    envs = make_vec_envs(kwargs['env_name'], kwargs['seed'], kwargs['num_processes'],
                         kwargs['gamma'], kwargs['log_dir'], device, False,
                         async_params=async_params)
    discrete_action = envs.action_space.__class__.__name__ == "Discrete"

    def make_agent(is_leaf=True):
        if is_leaf:
            obs_shape = envs.observation_space.shape
            action_shape = envs.action_space
            act_dim = envs.action_space.n if discrete_action else envs.observation_space.shape[0]

        else:
            obs_shape = (kwargs['hidden_size'],)
            action_shape = gym.spaces.Discrete(2)
            act_dim= None

        actor_critic = Policy(
            obs_shape,
            action_shape,
            base_kwargs={'recurrent': kwargs['recurrent_policy'],
                         'hidden_size': kwargs['hidden_size'],
                         'is_leaf': is_leaf,
                         'ops': kwargs['ops'] if is_leaf else False,
                         'act_dim': act_dim,
                         'discrete_action': discrete_action})
        actor_critic.to(device)

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

        return actor_critic, agent

    if kwargs['ops']:
        root = make_agent(is_leaf=False)
        leaf = make_agent(is_leaf=True)
        actor_critic, agent = list(zip(root, leaf))
    else:
        actor_critic, agent = make_agent(is_leaf=True)

    rollouts = RolloutStorage(kwargs['num_steps'], kwargs['num_processes'],
                              envs.observation_space.shape, envs.action_space,
                              kwargs['hidden_size'],
                              device=device
                              )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        kwargs['num_env_steps']) // kwargs['num_steps'] // kwargs['num_processes']

    def act(actor_critic, step, info=None):
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                rollouts.masks[step], info=info)
        return value, action, action_log_prob, recurrent_hidden_states

    def get_next_value(actor_critic):
        info = torch.cat((rollouts.infos[-1], rollouts.decisions[-1]), dim=-1).to(device)
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1], info=info).detach()
        return next_value

    def get_last_action(step):
        last_action = rollouts.actions[step-1]
        if not discrete_action:
            last_action = last_action.float()
        return last_action


    for j in range(num_updates):

        if kwargs['use_linear_lr_decay']:
            # decrease learning rate linearly
            if kwargs['ops']:
                [utils.update_linear_schedule(
                    a.optimizer, j, num_updates,
                    kwargs['lr']) for a in agent]
            else:
                utils.update_linear_schedule(
                    agent.optimizer, j, num_updates,
                    kwargs['lr'])

        for step in range(kwargs['num_steps']):
            last_action = get_last_action(step)
            # Sample actions
            if kwargs['ops']:
                value1, decisions, action_log_prob1, _ = act(actor_critic[0], step)
                value2, action, action_log_prob2, recurrent_hidden_states = act(actor_critic[1], step,
                                                        info=torch.cat([decisions, last_action], dim=1))
            else:
                decisions = torch.zeros((kwargs['num_processes'], 1)).to(device).long()
                value, action, action_log_prob, recurrent_hidden_states = act(actor_critic, step)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(torch.cat((decisions, action), dim=-1))

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            if kwargs['ops']:
                rollouts.insert(obs, recurrent_hidden_states, action,
                                [action_log_prob1, action_log_prob2], [value1, value1], reward, masks, bad_masks,
                                infos=last_action, decisions=decisions, ops=True)
            else:
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks, decisions=decisions, infos=None)

        if kwargs['ops']:
            next_value = [get_next_value(actor_critic[0]), get_next_value(actor_critic[1])]
        else:
            next_value = get_next_value(actor_critic)

        rollouts.compute_returns(next_value, kwargs['use_gae'], kwargs['gamma'],
                                 kwargs['gae_lambda'], kwargs['use_proper_time_limits'], kwargs['ops'])

        if kwargs['ops']:
            value_loss1, action_loss1, dist_entropy1 = agent[0].update(rollouts, is_leaf=False)
            value_loss2, action_loss2, dist_entropy2 = agent[1].update(rollouts, is_leaf=True)
        else:
            value_loss2, action_loss2, dist_entropy2 = agent.update(rollouts)

        rollouts.after_update()

        log_dict = {}

        # if (j % kwargs['save_interval'] == 0 or j == num_updates - 1) and log_dir != "":
        #     save_path = log_dir
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass
        #     if kwargs['ops']:
        #         torch.save([
        #             actor_critic,
        #             getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #         ], os.path.join(save_path, kwargs['env_name'] + str(j) +".pt"))
        #     else:
        #         torch.save([
        #             actor_critic[0],
        #             actor_critic[1],
        #             getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
        #         ], os.path.join(save_path, kwargs['env_name'] + str(j) +".pt"))

        if j % kwargs['log_interval'] == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * kwargs['num_processes'] * kwargs['num_steps']
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))

            log_dict['mean_reward'] = np.mean(episode_rewards)
            log_dict['median_reward'] = np.median(episode_rewards)
            log_dict['max_reward'] = np.max(episode_rewards)
            log_dict['min_reward'] = np.min(episode_rewards)
            log_dict['value_loss2'] = value_loss2
            log_dict['action_loss2'] = action_loss2
            log_dict['dist_entropy2'] = dist_entropy2
            log_dict['mean_gt'] = rollouts.decisions.float().mean().item()
            if kwargs['ops']:
                log_dict['value_loss1'] = value_loss1
                log_dict['action_loss1'] = action_loss1
                log_dict['dist_entropy1'] = dist_entropy1

        if (kwargs['eval_interval'] is not None and len(episode_rewards) > 1 and j % kwargs['eval_interval'] == 0):
            if len(envs.observation_space.shape) == 3:
                ob_rms = None
            else:
                ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, kwargs['env_name'], kwargs['seed'],
                     kwargs['num_processes'], eval_log_dir, device, log_dict, async_params, j=j, ops=kwargs['ops'],
                     hidden_size=kwargs['hidden_size'], keep_vis=kwargs['keep_vis'])

        if log_dict != {}:
            wandb.log(log_dict)


if __name__ == "__main__":
    sweep_params = {
        'seed': [222, 333],
        'algo': ['ppo'],

        'env_name': ['CartPole-v1'],
        'use_gae': [True],
        'lr': [3e-4],
        'value_loss_coef': [0.5],
        'num_processes': [4],
        'num_steps': [2048],
        'num_mini_batch': [32],
        'log_interval': [1],
        'use_linear_lr_decay': [True],
        'entropy_coef': [0],
        'num_env_steps': [1000000],
        'cuda': [False],
        'proj_name': ['wrapper-debug'],
        'note': [''],
        'hidden_size': [64],
        'ops': [True, False],
        'eval_interval': [5],
        'obs_interval': [0, 1, 2, 3, 5, 8],
        'ppo_epoch': [10],
        'gae_lambda': [0.95],
        'use_proper_time_limits': [True],
        'save_interval': [10],
        'keep_vis': [False],
        }

    run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)
    # xvfb-run -s "-screen 0 1400x900x24"
    # python aws.py --mode ec2 --python_cmd 'xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python'
    # python aws.py --mode local_docker --python_cmd 'xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python'
    # python aws.py --mode local_docker --python_cmd 'xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3' --docker_command "--user vioichigo --workdir /home/vioichigo/code --gpus all"
