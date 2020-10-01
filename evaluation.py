import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from pdb import set_trace as st
import time
import cv2
import gym
import wandb
import imageio

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
        real_imgs.append(np.zeros((3, H, W, D)))
    real_imgs = np.transpose(np.concatenate(real_imgs), (0, 3, 1, 2))
    # real_imgs = np.concatenate(real_imgs)
    return real_imgs

def concat_views(real_state, agent_view):
    H, W, D = real_state.shape
    border = np.zeros((H, 3, D))
    return np.concatenate((real_state, border, agent_view), axis=1)


def slip_channel(rgb_img, decision, env_name):
    rgb_img2 = rgb_img.copy()
    r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
    if env_name in ['CartPole-v1']:
        indices = np.logical_and(r!=0, np.logical_and(g!=0, b<200))
        ratio = r[indices].reshape((-1, 1))
    elif env_name in ['PongNoFrameSkip-v1']:
        indices = np.logical_and(r>100, np.logical_and(g>100, b>100))
        ratio = r[indices].reshape((-1, 1))
    rgb_img2[indices] = ratio * np.array([0, 0, 1])
    rgb_img[indices] = ratio * np.array([1, 0, 0])
    return rgb_img2 * decision + rgb_img * (1-decision)

# def act(actor_critic, obs, eval_recurrent_hidden_states, eval_masks, info=None):
#     with torch.no_grad():
#         _, action, _, eval_recurrent_hidden_states = actor_critic.act(
#             obs,
#             eval_recurrent_hidden_states,
#             eval_masks,
#             # deterministic=True,
#             deterministic=False,
#             info=info)
#     return action, eval_recurrent_hidden_states

def act(actor_critic, obs, recurrent_hidden_states, masks, **kwargs):
    with torch.no_grad():
        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, **kwargs)

        return action, recurrent_hidden_states

def config_env(env_name):
    if env_name in ['CartPole-v1']:
        max_size = 200
    elif env_name in ['PongNoFrameSkip-v1']:
        max_size = 40
    else:
        return 100
    return max_size


def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, log_dict, async_params=[1, 1], j=0, ops=False, hidden_size=64, keep_vis=True):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, keep_vis,
                              async_params=async_params, keep_vis=keep_vis)

    discrete_action = eval_envs.action_space.__class__.__name__ == "Discrete"

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []
    all_decisions = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(num_processes, hidden_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    if keep_vis:
        eval_views, pairs = [[] for _ in range(num_processes)], [[] for _ in range(num_processes)]
        collected_views, collected_pairs = [], []
    max_size = config_env(env_name)

    last_action = torch.zeros((num_processes, 1)).to(device).long()
    start = time.time()
    while len(eval_episode_rewards) < 4:
        if ops:
            decisions, _ = act(actor_critic[0], obs, eval_recurrent_hidden_states, eval_masks)
        else:
            decisions = torch.zeros((num_processes, 1)).to(device).long()
        action, eval_recurrent_hidden_states = act(actor_critic[1], obs, eval_recurrent_hidden_states, eval_masks,
                                                   info=torch.cat([decisions, last_action], dim=1))
        if discrete_action:
            last_action = action + 1
        else:
            last_action = action

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(torch.cat((decisions, action), dim=-1))
        if decisions.device.type == 'cuda':
            decisions = decisions.cpu().numpy()
        else:
            decisions = decisions.numpy()
        all_decisions.append(decisions)

        if keep_vis:
            full_obs = eval_envs.full_obs() #[num_processes, ((210, 160, 3), (210, 160, 3))]
            for (pair, eval_view, full_ob, decision, d, idx, info) in zip(pairs, eval_views, full_obs, decisions, done, range(num_processes), infos):
                if full_ob[0].shape[0] > max_size:
                    scale = 1 / (full_ob[0].shape[0]//max_size)
                    state = cv2.resize(full_ob[0], None, fx=scale, fy=scale)
                    view = cv2.resize(full_ob[1], None, fx=scale, fy=scale)
                else:
                    state, view = full_ob

                real_view = slip_channel(view.copy(), decision, env_name)
                if 'episode' in info.keys():
                    pair.append(concat_views(state, view * (1 - decision) + (view//2) * decision))
                    eval_view.append(real_view)
                    eval_episode_rewards.append(info['episode']['r'])

                    collected_views.append(eval_view)
                    collected_pairs.append(pair)
                    pairs[idx] = []
                    eval_views[idx] = []
                elif not d:
                    pair.append(concat_views(state, view * (1 - decision) + (view//2) * decision))
                    eval_view.append(real_view)
                elif d:
                    # ignore the episodes that failed
                    pairs[idx] = []
                    eval_views[idx] = []
        else:
            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

    eval_envs.close()
    if keep_vis:
        pairs = to_video(collected_pairs)
        imgs = to_hist(collected_views)
        log_dict["hist " + str(j)] = wandb.Image(imgs)
        log_dict["agent_views " + str(j)] = wandb.Video(pairs, fps=4, format="gif")

    log_dict['eval_len'] = len(eval_episode_rewards)
    log_dict['eval_mean_rew'] = np.mean(eval_episode_rewards)
    log_dict['eval_mean_gt'] = np.mean(all_decisions)
    log_dict['eval_time'] = time.time() - start
