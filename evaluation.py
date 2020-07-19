import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
from pdb import set_trace as st
import cv2
import imageio
import os
import gym


def act(actor_critic, obs, recurrent_hidden_states, masks, **kwargs):
    with torch.no_grad():
        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states,
            masks, **kwargs)

        return value, action, action_log_prob, recurrent_hidden_states


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
             resolution_scale = 1,):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, '', device, True, get_pixel = True, resolution_scale = resolution_scale)

    eval_episode_rewards = []
    obs = eval_envs.reset()
    _, C, H, W = obs.shape

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
    while len(eval_episode_rewards) < 10:
        # print(len(eval_episode_rewards))
        value1, action1, action_log_prob1, recurrent_hidden_states1 = act(actor_critic[0], obs, eval_recurrent_hidden_states, eval_masks)

        # TODO make sure the last index of actions is the right hting to do
        if always_zero:
            action1 = torch.zeros(action1.shape).to(device).long()

        if len(eval_episode_rewards) == 0:
            action2 = torch.zeros(action1.shape).to(device).long()

        if dicrete_action:
            last_action = 1 + action2
        else:
            last_action = action2

        value2, action2, action_log_prob2, recurrent_hidden_states2 = act(actor_critic[1], obs, eval_recurrent_hidden_states, eval_masks,
            info=torch.cat([action1, last_action], dim=1))

        action = action2
        eval_recurrent_hidden_states = recurrent_hidden_states2

        obs, reward, done, infos = eval_envs.step(action)

        imgs = np.array(eval_envs.full_obs())
        gate = action1.byte().squeeze()
        if gate.is_cuda:
            gate = gate.reshape((-1, 1, 1, 1)).cpu().data.numpy()
        else:
            gate = gate.reshape((-1, 1, 1, 1)).data.numpy()
        imgs = imgs[:, 0] * gate + imgs[:, 1] * (1-gate)

        if env_name.startswith("MiniWorld"):
            # print(obs.shape, imgs.shape) [16, 12, 60, 80] (16, 60, 80, 3)
            for (ob, im, paths, d, dones, top_view) in zip(obs, imgs, all_paths, done, all_dones, all_top_views):
                if ob.is_cuda:
                    paths.append(ob[:3].cpu().detach().numpy())
                else:
                    paths.append(ob[:3].detach().numpy())
                dones.append(d)
                top_view.append(im)
        else:
            for (im, paths, d, dones) in zip(imgs, all_paths, done, all_dones):
                paths.append(im)
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
        rows, columns = np.where(all_dones=True)
        total = []
        epi_length = all_dones.shape[1]
        total_for_img = []
        """only save the episodes that terminate. """
        # it's possible that there are two or none-dones
        row_record = []
        for (r, c) in zip(rows, columns):
            if r not in row_record:
                row_record.append(r)
                path = np.array(all_paths[r])
                done = c
                path[done:] = 0
                total_for_img.append(path)
                total.append(path[:done+1])

        if len(total_for_img) < num_processes:
            for _ in range(num_processes - len(total_for_img)):
                total_for_img.append(np.zeros((epi_length, 200, 200, 3)))

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

        # print([t.shape for t in total])
        total = np.concatenate(total)
        dir_name = save_dir
        if os.path.isdir(dir_name) == False:
            os.makedirs(dir_name)
    elif env_name.startswith("MiniWorld"):
        all_dones = np.array(all_dones)
        rows, columns = np.where(all_dones == True)
        total = []
        epi_length = all_dones.shape[1]
        total_for_img = []
        """only save the episodes that terminate. """
        # it's possible that there are two or none-dones
        row_record = []

        for i in range(num_processes):
            row_record.append(i)
            if i in rows: #if at least one of the episodes in this process terminates
                idx = np.where(rows == i)[0][0]
                r, done = i, columns[idx]
                path = np.array(all_paths[r])
                path[done+1:] = 0
                total.append(path[:done+1])
                # total.append(path[:done])
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

        # print([t.shape for t in total])
        total = np.concatenate(total)
        total = np.transpose(total, (0, 2, 3, 1))
        dir_name = save_dir
        if os.path.isdir(dir_name) == False:
            os.makedirs(dir_name)

    print(img_list.shape, total[0].shape, len(total))
    imageio.mimsave(dir_name + '/bouns-' + str(bonus1) + '-epoch-'+ str(epoch) + '-seed-'+ str(seed) + '.gif', total, duration=0.5)
    # if np.sum(columns) != 16 * 39:
    # cv2.imwrite(dir_name + '/img.png', img_list)
    # print("img saved to", dir_name + '/img.png')
    print(".GIF files saved to", dir_name)

    eval_envs.close()
    return img_list
