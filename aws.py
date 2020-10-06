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
from a2c_ppo_acktr.model import OpsPolicy
from a2c_ppo_acktr.storage import RolloutStorage
from pdb import set_trace as st
import wandb
from experiment_utils.run_sweep import run_sweep

from evaluation import evaluate

INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'async/debug-histogram'

def main(**kwargs):
    args = get_args()
    for arg in vars(args):
        if arg not in kwargs:
            kwargs[arg] = getattr(args, arg)

    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed_all(kwargs['seed'])
    wandb.init(project=kwargs['proj_name'], config = kwargs)
    kwargs['always_zero'] = (kwargs['ops'] == False)

    if kwargs['cuda'] and torch.cuda.is_available() and kwargs['cuda_deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.getcwd() + '/data/' + kwargs['env_name'] + '/' + str(kwargs['bonus1']) + '-' + str(kwargs['no_bonus']) + '-' + str(kwargs['obs_interval'])
    eval_log_dir = log_dir + "/eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:6" if kwargs['cuda'] else "cpu")
    async_params = [kwargs['obs_interval'], 1]

    envs = make_vec_envs(kwargs['env_name'], kwargs['seed'], kwargs['num_processes'],
                         kwargs['gamma'], kwargs['log_dir'], device, False,
                         async_params=async_params, scale=kwargs['scale'])
    discrete_action = envs.action_space.__class__.__name__ == "Discrete"

    def make_agent(is_leaf=True):
        ## AGENT CONSTRUCTION:
        ## Modularize this and allow for cascading (obs dim for child policy should be cat of obs and parents output)
        actor_critic = OpsPolicy(
            envs.observation_space.shape,
            envs.action_space if is_leaf else gym.spaces.Discrete(2),
            is_leaf=is_leaf,
            base_kwargs=dict(
                recurrent=True,
                gate_input='obs' if is_leaf else 'hid',
                hidden_size=kwargs['hidden_size'],
                persistent=kwargs['persistent']),
                )

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

        if discrete_action:
            action_dim = 1
            info_size = 2
        else:
            action_dim = envs.action_space.shape[0] if is_leaf else 1
            info_size = 1+envs.action_space.shape[0]

        if kwargs['persistent']:
            recurrent_hidden_size = actor_critic.recurrent_hidden_state_size * 2
        else:
            recurrent_hidden_size = actor_critic.recurrent_hidden_state_size

        rollouts = RolloutStorage(kwargs['num_steps'], kwargs['num_processes'],
                                    envs.observation_space.shape, envs.action_space,
                                    recurrent_hidden_size,
                                    info_size=info_size if is_leaf else 0, action_shape=action_dim)

        actor_critic.to(device)
        rollouts.to(device)
        return actor_critic, agent, rollouts

    root = make_agent(is_leaf=False)
    leaf = make_agent(is_leaf=True)
    actor_critic, agent, rollouts = list(zip(root, leaf))

    time_start = time.time()
    obs = envs.reset()
    for r in rollouts:
        r.obs[0].copy_(obs)
        r.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        kwargs['num_env_steps']) // kwargs['num_steps'] // kwargs['num_processes']

    img_trajs = [[]]

    def act(i, step, **kwargs):
        # collect rollouts
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic[i].act(
                rollouts[i].obs[step].to(device), rollouts[i].recurrent_hidden_states[step],
                rollouts[i].masks[step], **kwargs)

            return value, action, action_log_prob, recurrent_hidden_states

    for j in range(num_updates):
        start_time = time.time()

        if kwargs['use_linear_lr_decay']:
            # decrease learning rate linearly
            (utils.update_linear_schedule(
                _agent.optimizer, j, num_updates,
                _agent.optimizer.lr if kwargs['algo'] == "acktr" else kwargs['lr'][-1])
                for _agent in agent
            )

        for step in range(kwargs['num_steps']):
            # Sample actions
            value1, action1, action_log_prob1, _ = act(0, step)

            if kwargs['always_zero']:
                action1 = torch.zeros(action1.shape).to(device).long()

            if discrete_action:
                last_action = rollouts[1].actions[step-1] + 1 #to distinguish actual action from default ones
            else:
                last_action = rollouts[1].actions[step-1]
                action1 = action1.float()

            value2, action2, action_log_prob2, recurrent_hidden_states2 = act(1, step,
                info=torch.cat([action1, last_action], dim=1))

            action = action2
            recurrent_hidden_states = recurrent_hidden_states2

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(torch.cat((action1, action2), dim=-1))

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done]).to(device)
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos]).to(device)

            if j < kwargs['no_bonus']:
                scaled_reward = reward
            else:
                scaled_reward = (action1 * kwargs['bonus1']).to(device) + reward.to(device)

            rollouts[0].insert(obs, recurrent_hidden_states, action1,
                            action_log_prob1, value1, scaled_reward, masks, bad_masks,
                            infos=None)

            rollouts[1].insert(obs, recurrent_hidden_states, action2,
                            action_log_prob2, value2, reward, masks, bad_masks,
                            infos=torch.cat([action1, last_action], dim=1))

        print(episode_rewards)
        def update(i, info=None):
            with torch.no_grad():
                next_value = actor_critic[i].get_value(
                    rollouts[i].obs[-1].to(device), rollouts[i].recurrent_hidden_states[-1],
                    rollouts[i].masks[-1], info=info).detach()

            rollouts[i].compute_returns(next_value, kwargs['use_gae'], kwargs['gamma'],
                                    kwargs['gae_lambda'], kwargs['use_proper_time_limits'])

            pred_loss = ((i!=0) and (kwargs['pred_loss']))
            require_memory = ((i==0) and (kwargs['gate_input'] == 'hid')) or (pred_loss and kwargs['persistent'])
            value_loss, action_loss, dist_entropy, pred_err = agent[i].update(rollouts[i],
                pred_loss=pred_loss, require_memory=require_memory, num_processes=kwargs['num_processes'],
                device=device)

            rollouts[i].after_update()

            return value_loss, action_loss, dist_entropy, pred_err

        if j % 2 == 0 or True:
            print("updating agent 0")
            value_loss1, action_loss1, dist_entropy1, pred_err1 = update(0)
        if (j % 2) == 1 or True:
            print("updating agent 1")
            # use updated pi_1
            _, action1, _, _ = actor_critic[0].act(
                    rollouts[0].obs[-1].to(device), rollouts[0].recurrent_hidden_states[-1],
                    rollouts[0].masks[-1])

            if discrete_action:
                value_loss2, action_loss2, dist_entropy2, pred_err2 = update(1,
                    info=torch.cat([action1, rollouts[1].actions[-1]+1 ], dim=-1))
            else:
                value_loss2, action_loss2, dist_entropy2, pred_err2 = update(1,
                    info=torch.cat([action1.float(), rollouts[1].actions[-1]], dim=-1))

        log_dict = {'steps': kwargs['num_steps'] * kwargs['num_processes'] * j, 'update_time': time.time() - start_time}

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
            log_dict['mean_gt'] = rollouts[0].actions.float().mean().item()
            if kwargs['ops']:
                log_dict['value_loss1'] = value_loss1
                log_dict['action_loss1'] = action_loss1
                log_dict['dist_entropy1'] = dist_entropy1

        if (j % kwargs['eval_interval'] == 0 or j == num_updates - 1):
            save_path = log_dir
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            if len(envs.observation_space.shape) == 3:
                torch.save([
                    actor_critic[0],
                    actor_critic[1],
                ], os.path.join(save_path, str(j) + ".pt"))
            else:
                torch.save([
                    actor_critic[0],
                    actor_critic[1],
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], os.path.join(save_path, str(j) + ".pt"))

        if (kwargs['eval_interval'] is not None and len(episode_rewards) > 1 and j % kwargs['eval_interval'] == 0):
            if len(envs.observation_space.shape) == 3:
                ob_rms = None
            else:
                ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, kwargs['env_name'], kwargs['seed'],
                     kwargs['num_processes'], eval_log_dir, device, log_dict, async_params, j=j, ops=kwargs['ops'],
                     hidden_size=kwargs['hidden_size'], keep_vis=kwargs['keep_vis'], persistent=kwargs['persistent'],
                     scale=kwargs['scale'])

        wandb.log(log_dict)

if __name__ == "__main__":
    sweep_params = {
        'seed': [222],
        'algo': ['ppo'],

        # 'env_name': ['CartPole-v1'],
        # 'env_name': ['MiniGrid-Dynamic-Obstacles-5x5-v0', 'MiniGrid-Dynamic-Obstacles-6x6-v0', 'MiniGrid-Dynamic-Obstacles-8x8-v0'],
        # 'env_name': ['MiniGrid-Dynamic-Obstacles-5x5-v0', 'MiniGrid-Dynamic-Obstacles-6x6-v0'],
        # 'env_name': ['VizdoomCorridor-v0'],
        'env_name': ['VizdoomDefendCenter-v0'],
        # 'env_name': ['CarRacing-v0'],

        # 'env_name': ['Hopper-v2', 'Walker2d-v2'],
        # 'env_name': ['Walker2d-v2'],
        'use_gae': [True],
        'lr': [2.5e-4],
        'value_loss_coef': [0.5],
        'num_processes': [16],
        'num_steps': [512],
        'num_mini_batch': [4],
        'log_interval': [1],
        'use_linear_lr_decay': [True],
        'entropy_coef': [0.005],
        'num_env_steps': [50000000],
        'cuda': [True],
        'proj_name': ['async-vizdoom7'],
        'note': [''],
        'hidden_size': [128],
        'bonus1': [0],
        'no_bonus': [0],
        'ops': [False],
        'eval_interval': [10],
        'obs_interval': [0],
        # 'ppo_epoch': [10],
        'gae_lambda': [0.95],
        'use_proper_time_limits': [True],
        # 'save_interval': [10],
        'keep_vis': [True],
        'persistent': [True],
        'pred_loss': [False],
        'scale': [0.25],
        }

    run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)
    # xvfb-run -s "-screen 0 1400x900x24"
    # python aws.py --mode ec2 --python_cmd 'xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python'
    # python aws.py --mode local_docker --python_cmd 'xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python'
    # python aws.py --mode local_docker --python_cmd 'xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3' --docker_command "--user vioichigo --workdir /home/vioichigo/code --gpus all"
