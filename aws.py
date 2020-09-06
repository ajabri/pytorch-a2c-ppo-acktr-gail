from pdb import set_trace as st
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
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import OpsPolicy
from a2c_ppo_acktr.storage import RolloutStorage
import time
import wandb
from a2c_ppo_acktr import logging

import json
from evaluation import *
from experiment_utils.run_sweep import run_sweep

class ClassEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        if callable(o):
            return {'function': o.__name__}
        return json.JSONEncoder.default(self, o)

INSTANCE_TYPE = 'c4.4xlarge'
EXP_NAME = 'async/car3'

def main(**kwargs):
    args = get_args()

    for arg in vars(args):
        if arg not in kwargs:
            kwargs[arg] = getattr(args, arg)

    torch.manual_seed(kwargs['seed'])
    if kwargs['cuda']:
        torch.cuda.manual_seed_all(kwargs['seed'])
    device = torch.device("cuda:5" if kwargs['cuda'] else "cpu")
    if kwargs['cuda'] and torch.cuda.is_available() and kwargs['cuda_deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # log_dir = os.path.expanduser(kwargs['log_dir'])
    exp_dir = os.getcwd() + '/data/' + EXP_NAME
    if os.path.isdir(exp_dir) == False:
        os.makedirs(exp_dir)
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    log_dir = exp_dir + '/train'
    eval_log_dir = exp_dir + '/eval'
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    if not kwargs['debug']:
        wandb.init(project=kwargs['proj_name'], config = kwargs)

    torch.set_num_threads(1)
    envs = make_vec_envs(kwargs['env_name'], kwargs['seed'], kwargs['num_processes'],
                         kwargs['gamma'], log_dir, device, False,
                         resolution_scale=kwargs['scale'], image_stack=kwargs['image_stack'],
                         async_params=[kwargs['obs_interval'], kwargs['predict_interval'], kwargs['no_op']])

    # flip, flip1 = False, False
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
                gate_input='obs' if is_leaf else kwargs['gate_input'],
                hidden_size=kwargs['hidden_size'],
                pred_mode=kwargs['pred_mode'],
                fixed_probability=kwargs['fixed_probability'],
                persistent=kwargs['persistent']),
                )

        if is_leaf:
            lr = kwargs['lr'][1]
        else:
            lr = kwargs['lr'][0]

        entropy_coef = kwargs['entropy_coef']

        if kwargs['algo'] == 'ppo':
            agent = algo.PPO(
                actor_critic,
                kwargs['clip_param'],
                kwargs['ppo_epoch'],
                kwargs['num_mini_batch'],
                kwargs['value_loss_coef'],
                entropy_coef,
                lr=lr,
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

    step_indices = torch.from_numpy(np.array([0 for _ in range(kwargs['num_processes'])])).to(device)

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
                info=[torch.cat([action1, last_action], dim=1), step_indices])

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
                            infos=[torch.cat([action1, last_action], dim=1), step_indices])
            step_indices = torch.from_numpy(np.array([info['step_index'] for info in infos])).to(device)

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
                    info=[torch.cat([action1, rollouts[1].actions[-1]+1 ], dim=-1), step_indices])
            else:
                value_loss2, action_loss2, dist_entropy2, pred_err2 = update(1,
                    info=[torch.cat([action1.float(), rollouts[1].actions[-1]], dim=-1), step_indices])


        # save for every interval-th episode or for the last epoch
        if (j % kwargs['save_interval'] == 0 or j == num_updates - 1) and kwargs['save_dir'] != "":
            save_path = os.path.join(kwargs['save_dir'], kwargs['algo'])
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, kwargs['env_name'] + ".pt"))

        if j % kwargs['log_interval'] == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * kwargs['num_processes'] * kwargs['num_steps']
            end = time.time()
            print(
                "Updates {}/{}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                .format(j, num_updates, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards),
                        ))
            print('update_time:', time.time()-time_start)

            if not kwargs['debug']:
                wandb.log(dict(
                    median_reward=np.median(episode_rewards), mean_reward=np.mean(episode_rewards),
                    min_reward=np.min(episode_rewards), max_reward=np.max(episode_rewards),
                    update_time=time.time()-time_start,
                ))
            time_start = time.time()


            if j % 5 == 0 and kwargs['env_name'].startswith("MiniGrid"):
                data = rollouts[0].obs[:-1]
                gate = rollouts[0].actions.byte().squeeze()

                capt = data[1 - gate]
                pred = data[gate]

                if not kwargs['debug']:
                    # wandb_lunarlander(capt, pred)
                    logging.wandb_minigrid(capt, pred)


            if not kwargs['debug']:
                if (j % 2) == 0:
                    wandb.log(dict(ent1=dist_entropy1, val1=value_loss1, aloss1=action_loss1,))
                    print("ent1 {:.4f}, val1 {:.4f}, loss1 {:.4f}\n".format(
                        dist_entropy1, value_loss1, action_loss1))
                if (j % 2) == 1:
                    wandb.log(dict(ent2=dist_entropy2, val2=value_loss2, aloss2=action_loss2, prederr2=pred_err2))
                    print("ent2 {:.4f}, val2 {:.4f}, loss2 {:.4f}, prederr2 {:.4f}\n".format(
                        dist_entropy2, value_loss2, action_loss2, pred_err2))

                wandb.log(dict(mean_gt=rollouts[0].actions.float().mean().item()))

        if j % kwargs['gif_save_interval'] == 0 and not kwargs['debug'] and (kwargs['env_name'].startswith("Car") or kwargs['env_name'].startswith("Mini")):
            img_list = save_gif(actor_critic, kwargs['env_name'], kwargs['seed'],
                         kwargs['num_processes'], device, j, kwargs['bonus1'], save_dir = eval_log_dir,
                         persistent = kwargs['persistent'], always_zero=kwargs['always_zero'],
                         resolution_scale = kwargs['scale'], image_stack=kwargs['image_stack'],
                         async_params=[kwargs['obs_interval'], kwargs['predict_interval'], kwargs['no_op']])

            if kwargs['env_name'].startswith("Mini"):
                wandb.log({"visualization %s" % j: wandb.Image(img_list[0])})
                wandb.log({"video %s" % j: wandb.Video(img_list[1], fps=4, format="gif")})
                wandb.log(dict(eval_mean_reward=img_list[-2]))
                wandb.log(dict(eval_mean_gt=img_list[-1]))
            else:
                wandb.log({"video %s" % j: wandb.Video(img_list[0], fps=4, format="gif")})
                wandb.log(dict(eval_mean_reward=img_list[-2]))
                wandb.log(dict(eval_mean_gt=img_list[-1]))

        # if not kwargs['debug'] and j % kwargs['gif_save_interval'] == 0:
        #     success_rate, eval_reward = evaluate_actions(actor_critic, kwargs['env_name'], kwargs['seed'],
        #                      kwargs['num_processes'], device, j, kwargs['bonus1'], save_dir = eval_log_dir,
        #                      persistent = kwargs['persistent'], always_zero=kwargs['always_zero'],
        #                      resolution_scale = kwargs['scale'], image_stack=kwargs['image_stack'])
        #     wandb.log(dict(success_rate=success_rate))
        #     wandb.log(dict(eval_mean_reward=eval_reward))


if __name__ == "__main__":
    sweep_params = {
        'algo': 'ppo',
        'seed': 111,
        # 'env_name': 'MiniWorld-YMaze-v0',
        # 'env_name': 'MiniWorld-CollectHealth-v0',
        # 'env_name': 'CarRacing-v0',
        # 'env_name': 'SawyerLift',
        'env_name': 'VizdoomDefendLine-v0',
        # 'env_name': 'VizdoomHealthGathering-v0',
        # 'env_name': 'FetchReach-v1', 'FetchPickAndPlace-v1', 'FetchPush-v1', 'FetchSlide-v1',
        # 'env_name': 'MiniGrid-MultiRoom-N4-S5-v0',
        # 'env_name': 'MiniWorld-FourRooms-v0',

        'use_gae': True,
        'lr': [2.5e-4, 2.5e-4],
        'clip_param': 0.1,
        'value_loss_coef': 0.5,
        'num_processes': 16,
        # 'num_processes': 8,
        'num_steps': 512,
        'num_mini_batch': 4,
        'log_interval': 1,
        'use_linear_lr_decay': True,
        'entropy_coef': 0.005,
        'num_env_steps': 50000000,
        'bonus1': 0,
        'cuda': True,
        # 'proj_name': 'async-maze',
        'proj_name': 'async-vizdoom',
        # 'proj_name': 'async-car',
        # 'proj_name': 'debug',
        'gif_save_interval': 200,
        'note': '',
        'debug': False,
        'gate_input': 'hid', #'obs' | 'hid'
        'persistent': True,
        'scale': 0.25,
        'hidden_size': 128,
        'always_zero': False,
        'pred_loss': False,
        'image_stack': False,
        'save_dir': '',
        'pred_mode': 'pred_model', #'pred_model' | 'pos_enc'
        'no_bonus': 0,
        'fixed_probability': None,
        'obs_interval': 6,
        'predict_interval': 1,
        'no_op': True,
        }

    # run_sweep(main, sweep_params, EXP_NAME, INSTANCE_TYPE)
    main(**sweep_params)

    # python aws.py --mode ec2 --python_cmd 'xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python'
    # python aws.py --mode local_docker --python_cmd 'xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python'
    # python aws.py --mode local_docker --python_cmd 'xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3' --docker_command "--user vioichigo --workdir /home/vioichigo/code --gpus all"
