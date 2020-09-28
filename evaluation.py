import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from pdb import set_trace as st
import time
import cv2
import gym
import wandb

def to_hist(collected_agent_views):
    img_list = []
    for agent_views in collected_agent_views[:4]:
        im = np.max(agent_views, axis=0)
        img_list.append(im)

    H, W, D = img_list[0].shape
    split = int(np.sqrt(len(img_list)))
    vertical = np.concatenate(img_list)
    result = np.concatenate(np.reshape(vertical, [split, H*split, W, D]), axis=1)
    return result

def to_video(list_of_imgs):
    """input: [4, img]"""
    real_imgs = []
    for imgs in list_of_imgs:
        real_imgs.append(imgs)
        H, W, D = imgs[0].shape
        real_imgs.append(np.zeros((10, H, W, D)))
    real_imgs = np.transpose(np.concatenate(real_imgs), (0, 3, 1, 2))
    return real_imgs

def concat_views(real_state, agent_view):
    H, W, D = real_state.shape
    border = np.zeros((H, 3, D))
    return np.concatenate((real_state, border, agent_view), axis=1)

def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, log_dict, async_params=[1, 1], j=0):
    """we also need some visualizations. """
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True, async_params=async_params)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    eval_views, pairs = [[] for _ in range(num_processes)], [[] for _ in range(num_processes)]
    collected_views, collected_pairs = [], []
    max_size = 50

    start = time.time()
    while len(eval_episode_rewards) < 4:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)
        full_obs = eval_envs.full_obs() #[num_processes, ((210, 160, 3), (210, 160, 3))]
        decisions = np.zeros((num_processes, 1))
        for (pair, eval_view, full_ob, decision, d, idx, info) in zip(pairs, eval_views, full_obs, decisions, done, range(num_processes), infos):
            if full_ob[0].shape[0] > max_size:
                scale = 1 / (full_ob[0].shape[0]//max_size)
                state = cv2.resize(full_ob[0], None, fx=scale, fy=scale)
                view = cv2.resize(full_ob[1], None, fx=scale, fy=scale)
            else:
                state, view = full_ob

            real_view = view * (1 - decision) + (view//2) * decision
            if 'episode' in info.keys():
                pair.append(concat_views(state, real_view))
                eval_view.append(real_view)
                eval_episode_rewards.append(info['episode']['r'])

                collected_views.append(eval_view)
                collected_pairs.append(pair)
                pairs[idx] = []
                eval_views[idx] = []
            elif not d:
                pair.append(concat_views(state, real_view))
                eval_view.append(real_view)
            elif d:
                # ignore the episodes that failed
                pairs[idx] = []
                eval_views[idx] = []

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

    eval_envs.close()
    imgs = to_hist(collected_views)
    pairs = to_video(collected_pairs)

    log_dict['eval_len'] = len(eval_episode_rewards)
    log_dict['mean_rew'] = np.mean(eval_episode_rewards)
    log_dict['eval_time'] = time.time() - start
    log_dict["hist " + str(j)] = wandb.Image(imgs)
    log_dict["agent_views " + str(j)] = wandb.Video(pairs, fps=4, format="gif")
