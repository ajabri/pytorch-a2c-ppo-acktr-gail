import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from pdb import set_trace as st
import cv2
import imageio
import os
import gym
from collections import deque


def act(actor_critic, obs, recurrent_hidden_states, masks, **kwargs):
    with torch.no_grad():
        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states,
            masks, **kwargs)

        return value, action, action_log_prob, recurrent_hidden_states

def sample_paths(actor_critic,
             env_name,
             seed,
             device,
             persistent = False,
             always_zero = False,
             resolution_scale = 1,
             image_stack=False,
             async_params=[1, 1, False]):
    num_processes = 20
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, '', device, True, get_pixel = True, resolution_scale = resolution_scale, image_stack=image_stack)

    obs = eval_envs.reset()
    from mjrl.utils.gym_env import GymEnv
    import mj_envs
    horizon = GymEnv(env_name)._horizon
    step_indices = torch.from_numpy(np.array([0 for _ in range(num_processes)])).to(device)
    dicrete_action = (eval_envs.action_space.__class__.__name__ == "Discrete")

    if persistent:
        recurrent_hidden_size = actor_critic[0].recurrent_hidden_state_size * 2
    else:
        recurrent_hidden_size = actor_critic[0].recurrent_hidden_state_size

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, recurrent_hidden_size, device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    all_paths = [{} for _ in range(num_processes)]

    if dicrete_action:
        action2 = torch.zeros((num_processes, 1)).to(device).long()
    else:
        action2 = torch.zeros((num_processes, eval_envs.action_space.shape[0])).to(device).long()

    count = 0
    while count < horizon: #increase it so that now all the episodes termiante
        count += 1
        value1, action1, action_log_prob1, recurrent_hidden_states1 = act(actor_critic[0], obs, eval_recurrent_hidden_states, eval_masks)

        if always_zero:
            action1 = torch.zeros(action1.shape).to(device).long()

        if dicrete_action:
            last_action = 1 + action2
        else:
            last_action = action2.float()
            action1 = action1.float()

        value2, action2, action_log_prob2, recurrent_hidden_states2 = act(actor_critic[1], obs, eval_recurrent_hidden_states, eval_masks,
            info=[torch.cat([action1, last_action], dim=1), step_indices])

        action = action2
        eval_recurrent_hidden_states = recurrent_hidden_states2

        obs, reward, done, infos = eval_envs.step(torch.cat((action1, action), dim=-1))
        step_indices = torch.from_numpy(np.array([info['step_index'] for info in infos])).to(device)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for (idx, info) in enumerate(infos):
            if 'env_infos' in all_paths[idx].keys():
                all_paths[idx]['env_infos']['goal_achieved'].extend(info['goal_achieved'])
            else:
                all_paths[idx]['env_infos'] = {}
                all_paths[idx]['env_infos']['goal_achieved'] = []

            if 'rewards' in all_paths[idx].keys():
                all_paths[idx]['rewards'].append(reward[idx])
            else:
                all_paths[idx]['rewards'] = []

    eval_envs.close()

    return all_paths



def evaluate_actions(actor_critic,
             env_name,
             seed,
             num_processes,
             device,
             epoch,
             bonus1,
             save_dir = './saved',
             tile_size = 1,
             persistent = False,
             always_zero = False,
             resolution_scale = 1,
             image_stack=False):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, '', device, True, get_pixel = True, resolution_scale = resolution_scale, image_stack=image_stack)

    eval_episode_rewards = []
    count = 10
    success_rates = deque(maxlen=count)
    obs = eval_envs.reset()
    step_indices = torch.from_numpy(np.array([0 for _ in range(num_processes)])).to(device)

    if persistent:
        recurrent_hidden_size = actor_critic[0].recurrent_hidden_state_size * 2
    else:
        recurrent_hidden_size = actor_critic[0].recurrent_hidden_state_size

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, recurrent_hidden_size, device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    all_paths = [[] for _ in range(num_processes)]

    dicrete_action = (eval_envs.action_space.__class__.__name__ == "Discrete")
    while len(eval_episode_rewards) < count: #increase it so that now all the episodes termiante
        value1, action1, action_log_prob1, recurrent_hidden_states1 = act(actor_critic[0], obs, eval_recurrent_hidden_states, eval_masks)

        if always_zero:
            action1 = torch.zeros(action1.shape).to(device).long()

        if len(eval_episode_rewards) == 0:
            if dicrete_action:
                action2 = torch.zeros(action1.shape).to(device).long()
            else:
                action2 = torch.zeros((num_processes, eval_envs.action_space.shape[0])).to(device).long()

        if dicrete_action:
            last_action = 1 + action2
        else:
            last_action = action2.float()
            action1 = action1.float()

        value2, action2, action_log_prob2, recurrent_hidden_states2 = act(actor_critic[1], obs, eval_recurrent_hidden_states, eval_masks,
            info=[torch.cat([action1, last_action], dim=1), step_indices])

        action = action2
        eval_recurrent_hidden_states = recurrent_hidden_states2

        obs, reward, done, infos = eval_envs.step(action)
        step_indices = torch.from_numpy(np.array([info['step_index'] for info in infos])).to(device)

        if action1.shape[-1] > 1:
            gate = action1[:, 0].byte().squeeze()
        else:
            gate = action1.byte().squeeze()

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
            if is_success in info.keys():
                success_rates.append(info['is_success'])
    eval_envs.close()

    return np.mean(success_rates), np.mean(eval_episode_rewards)



def save_gif(actor_critic,
             env_name,
             seed,
             num_processes,
             device,
             epoch,
             bonus1,
             save_dir = './saved',
             tile_size = 1,
             persistent = False,
             always_zero = False,
             resolution_scale = 1,
             image_stack=False,
             async_params=[1, 1, False]):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, '', device, True, get_pixel = True, resolution_scale = resolution_scale, image_stack=image_stack)

    eval_episode_rewards = []
    obs = eval_envs.reset()
    _, C, H, W = obs.shape
    step_indices = torch.from_numpy(np.array([0 for _ in range(num_processes)])).to(device)

    if persistent:
        recurrent_hidden_size = actor_critic[0].recurrent_hidden_state_size * 2
    else:
        recurrent_hidden_size = actor_critic[0].recurrent_hidden_state_size

    eval_recurrent_hidden_states = torch.zeros(
        num_processes, recurrent_hidden_size, device=device)

    eval_masks = torch.zeros(num_processes, 1, device=device)

    all_paths = [[] for _ in range(num_processes)]

    if env_name.startswith("MiniWorld"):
        all_top_views = [[] for _ in range(num_processes)]

    all_dones = [[] for _ in range(num_processes)]

    dicrete_action = (eval_envs.action_space.__class__.__name__ == "Discrete")
    while len(eval_episode_rewards) < 16: #increase it so that now all the episodes termiante
        value1, action1, action_log_prob1, recurrent_hidden_states1 = act(actor_critic[0], obs, eval_recurrent_hidden_states, eval_masks)

        if always_zero:
            action1 = torch.zeros(action1.shape).to(device).long()

        if len(eval_episode_rewards) == 0:
            if dicrete_action:
                action2 = torch.zeros(action1.shape).to(device).long()
            else:
                action2 = torch.zeros((num_processes, eval_envs.action_space.shape[0])).to(device).long()

        if dicrete_action:
            last_action = 1 + action2
        else:
            last_action = action2.float()
            action1 = action1.float()

        value2, action2, action_log_prob2, recurrent_hidden_states2 = act(actor_critic[1], obs, eval_recurrent_hidden_states, eval_masks,
            info=[torch.cat([action1, last_action], dim=1), step_indices])

        action = action2
        eval_recurrent_hidden_states = recurrent_hidden_states2

        obs, reward, done, infos = eval_envs.step(torch.cat((action1, action), dim=-1))
        step_indices = torch.from_numpy(np.array([info['step_index'] for info in infos])).to(device)

        imgs = np.array(eval_envs.full_obs())
        if action1.shape[-1] > 1:
            gate = action1[:, 0].byte().squeeze()
        else:
            gate = action1.byte().squeeze()
        if gate.is_cuda:
            gate = gate.reshape((-1, 1, 1, 1)).cpu().data.numpy()
        else:
            gate = gate.reshape((-1, 1, 1, 1)).data.numpy()

        if env_name.startswith("MiniWorld"):
            # print(obs.shape, imgs.shape) [16, 12, 60, 80] (16, 60, 80, 3)
            colored_obs = imgs[:, 2] * gate + imgs[:, 3] * (1-gate) #this is a top down view
            imgs = imgs[:, 0] * gate + imgs[:, 1] * (1-gate) #this is a top down view

            for (ob, im, paths, d, dones, top_view) in zip(colored_obs, imgs, all_paths, done, all_dones, all_top_views):
                paths.append(ob[:, :, :3])
                dones.append(d)
                top_view.append(im)
        else:
            imgs = imgs[:, 0] * gate + imgs[:, 1] * (1-gate)
            for (im, paths, d, dones) in zip(imgs, all_paths, done, all_dones):
                paths.append(im[:, :, :3])
                dones.append(d)


        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    if env_name.startswith("MiniGrid-MultiRoom"):
        all_dones = np.array(all_dones)
    elif env_name.startswith("MiniWorld"):
        all_dones = np.array(all_dones)
        rows, columns = np.where(all_dones == True)
        total = []
        epi_length = all_dones.shape[1]
        total_for_img = []
        """only save the episodes that terminate. """
        # it's possible that there are two or none-dones

        for i in range(num_processes):
            if i in rows: #if at least one of the episodes in this process terminates
                idx = np.where(rows == i)[0][0]
                r, done = i, columns[idx]
                path = np.array(all_paths[r])
                path[done+1:] = 0
                total.append(path[:done+1])
                path = np.array(all_top_views[r])
                path[done:] = 0
                total_for_img.append(path)
            else:
                total_for_img.append(all_top_views[i])


        all_paths = np.array(total_for_img)
        # change the color of the starting point
        r, g, b = all_paths[:, :, :, :, 0], all_paths[:, :, :, :, 1], all_paths[:, :, :, :, 2]
        # get all the agent positions
        indices = np.logical_or(np.logical_and(r!=0, np.logical_and(g==0, b==0)), np.logical_and(b!=0, np.logical_and(g==0, r==0)))
        # only choose the starting point
        indices[:, 1:] = False
        ratio = r[indices].reshape((-1, 1)) + b[indices].reshape((-1, 1))
        all_paths[indices] += ratio * np.array([0, 1, 0]).astype(all_paths.dtype)
        img_list = np.max(all_paths, axis = 1)

        num_processes, H, W, D = img_list.shape
        num = 4
        img_list = img_list.reshape((num_processes//num, num, H, W, D))
        img_list = np.transpose(img_list, (0, 2, 1, 3, 4))
        img_list = np.clip(img_list.reshape((num*H, num*W, D)), 0, 255)
    else:
        all_dones = np.array(all_dones)
        rows, columns = np.where(all_dones == True)
        total = []
        epi_length = all_dones.shape[1]
        total_for_img = []

        for i in range(num_processes):
            if i in rows: #if at least one of the episodes in this process terminates
                idx = np.where(rows == i)[0][0]
                r, done = i, columns[idx]
                path = np.array(all_paths[r])
                total.append(path[:done+1])
            else:
                total.append(np.array(all_paths[i]))
            _, D, H, W = obs.shape
            total.append(np.zeros((1, H, W, D)))

    if env_name.startswith("Mini"):
        all_paths = np.array(total_for_img)
        # change the color of the starting point
        r, g, b = all_paths[:, :, :, :, 0], all_paths[:, :, :, :, 1], all_paths[:, :, :, :, 2]
        # get all the agent positions
        indices = np.logical_or(np.logical_and(r!=0, np.logical_and(g==0, b==0)), np.logical_and(b!=0, np.logical_and(g==0, r==0)))
        # only choose the starting point
        indices[:, 1:] = False
        ratio = r[indices].reshape((-1, 1)) + b[indices].reshape((-1, 1))
        all_paths[indices] += ratio * np.array([0, 1, 0]).astype(all_paths.dtype)
        img_list = np.max(all_paths, axis = 1)

        num_processes, H, W, D = img_list.shape
        num = 4
        img_list = img_list.reshape((num_processes//num, num, H, W, D))
        img_list = np.transpose(img_list, (0, 2, 1, 3, 4))
        img_list = np.clip(img_list.reshape((num*H, num*W, D)), 0, 255)

    total = np.clip(np.concatenate(total), 0, 255)
    total = np.transpose(total, (0, 3, 1, 2))

    eval_envs.close()
    if env_name.startswith("Mini"):
        return img_list, total, np.mean(eval_episode_rewards), action1.float().mean().item()
    else:
        return total, np.mean(eval_episode_rewards), action1.float().mean().item()


def evaluate_success(env_id, paths):
    num_success = 0
    num_paths = len(paths)
    if env_id.startswith("pen"):
        critiria = 20
    else:
        critiria = 25  # success if object close to target for 25 steps

    for path in paths:
        if np.sum(path['env_infos']['goal_achieved']) > critiria:
            num_success += 1
    success_percentage = num_success*100.0/num_paths
    return success_percentage

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
    horizon = env._horizon if horizon is None else horizon
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
            _, act, _, _ = policy.act(torch.from_numpy(o).float().to(device), rnn_hxs=None, masks=torch.zeros((1)).to(device).float(), deterministic=mean_action)
            o, r, done, _ = env.step(act.reshape((-1)).data.numpy())
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
