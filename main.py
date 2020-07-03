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
from a2c_ppo_acktr.model import Policy, OpsPolicy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

import wandb
from a2c_ppo_acktr import logging

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    if not args.debug:
        wandb.init(project="ops")

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    # render_func = envs.venv.venv.get_images_single
    flip,flip1 = False, False
    
    def make_agent(is_leaf=True):
        ## AGENT CONSTRUCTION:
        ## Modularize this and allow for cascading (obs dim for child policy should be cat of obs and parents output)
        actor_critic = OpsPolicy(
            envs.observation_space.shape,
            envs.action_space if is_leaf else gym.spaces.Discrete(2),
            is_leaf=is_leaf,
            base_kwargs=dict(
                recurrent=True,
                partial_obs=args.partial_obs,
                gate_input=args.gate_input)
                )
                
        actor_critic.to(device)
        # wandb.watch(actor_critic.base)

        if args.algo == 'a2c':
            agent = algo.A2C_ACKTR(
                actor_critic,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm)
        elif args.algo == 'ppo':
            agent = algo.PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)
        elif args.algo == 'acktr':
            agent = algo.A2C_ACKTR(
                actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size,
                                info_size=2 if is_leaf else 0)

        return actor_critic, agent, rollouts
    
    root = make_agent(is_leaf=False)
    leaf = make_agent(is_leaf=True)
    actor_critic, agent, rollouts = list(zip(root, leaf))

    obs = envs.reset()
    for r in rollouts:
        r.obs[0].copy_(obs)
        r.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    img_trajs = [[]]

    def act(i, step, **kwargs):
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic[i].act(
                rollouts[i].obs[step], rollouts[i].recurrent_hidden_states[step],
                rollouts[i].masks[step], **kwargs)
            
            return value, action, action_log_prob, recurrent_hidden_states

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            (utils.update_linear_schedule(
                _agent.optimizer, j, num_updates,
                _agent.optimizer.lr if args.algo == "acktr" else args.lr)
                for _agent in agent
            )

        for step in range(args.num_steps):
            # Sample actions
            value1, action1, action_log_prob1, recurrent_hidden_states1 = act(0, step)

            if np.random.random() > 0.9:
                print(action1.numpy().tolist())
            # TODO make sure the last index of actions is the right hting to do
            last_action = 1 + rollouts[1].actions[step-1]

            # import pdb; pdb.set_trace()
            value2, action2, action_log_prob2, recurrent_hidden_states2 = act(1, step,
                info=torch.cat([action1, last_action], dim=1))

            action = action2
            recurrent_hidden_states = recurrent_hidden_states2

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])


            int_rew = action1 * args.bonus1

            # print(reward, int_rew)
            # reward

            rollouts[0].insert(obs, recurrent_hidden_states, action1,
                            action_log_prob1, value1, reward + int_rew, masks, bad_masks,
                            infos=None)

            rollouts[1].insert(obs, recurrent_hidden_states, action2,
                            action_log_prob2, value2, reward, masks, bad_masks, 
                            infos=torch.cat([action1, last_action], dim=1))

            # if j % 100 == 0 or flip:
            #     flip = True

            #     if masks[0].item() < 1 or flip1:
            #         flip1 = True

            #         if len(img_trajs) > 5:
            #             mean = lambda x: sum(x)/(len(x)+1)
            #             flat = lambda x: [xxx for xx in x for xxx in xx]
            #             norm = lambda x: (x-x.min()) / (x-x.min()).max()

            #             i1 = [[ii[0] for ii in traj if ii[1] == 0] for traj in img_trajs]
            #             i2 = [[ii[0] for ii in traj if ii[1] == 1] for traj in img_trajs]

            #             i1, i2 = flat(i1), flat(i2)
            #             i1, i2 = mean(i1) * 255.0, mean(i2) * 255.0

            #             # i1 = mean([mean(l) for l in i1])*255.0
            #             # i2 = mean([mean(l) for l in i2])*255.0                        
            #             # import pdb; pdb.set_trace()

            #             img = np.concatenate([norm(ii) for ii in (i1, i2) if not isinstance(ii, float)], axis=1)

            #             wandb.log({"%s" % j: [wandb.Image(img, caption="capture - predict")]})

            #             img_trajs = [[]]
            #             flip, flip1 = False, False

            #         if masks[0].item() < 1 and len(img_trajs[-1]) > 0:
            #             img_trajs.append([])

            #         imgs = render_func('rgb_array')
            #         img_trajs[-1].append((imgs, action1[0].item()))

    

        def update(i, info=None):
            with torch.no_grad():
                next_value = actor_critic[i].get_value(
                    rollouts[i].obs[-1], rollouts[i].recurrent_hidden_states[-1],
                    rollouts[i].masks[-1], info=info).detach()

            rollouts[i].compute_returns(next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy, pred_err = agent[i].update(rollouts[i],
                pred_loss=i!=0)

            rollouts[i].after_update()

            return value_loss, action_loss, dist_entropy, pred_err
        
        if j % 2 == 0 or True:
            value_loss1, action_loss1, dist_entropy1, pred_err1 = update(0)
        if (j % 2) == 1 or True:
            _, action1, _, _ = actor_critic[0].act(
                    rollouts[0].obs[-1], rollouts[0].recurrent_hidden_states[-1],
                    rollouts[0].masks[-1])
            
            value_loss2, action_loss2, dist_entropy2, pred_err2 = update(1,
                info=torch.cat([action1, rollouts[1].actions[-1]+1 ], dim=-1))

        # value_loss, action_loss, dist_entropy = list(zip((update(i) for i in range(len(agent)))))



        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards),
                        ))

            if not args.debug:
                wandb.log(dict(
                    median_reward=np.median(episode_rewards), mean_reward=np.mean(episode_rewards),
                    min_reward=np.min(episode_rewards), max_reward=np.max(episode_rewards),
                ))


            if j % 5 == 0:


                data = rollouts[0].obs[:-1]
                gate = rollouts[0].actions.byte().squeeze()

                capt = data[1 - gate]
                # x1, v1, ang1, angv1 = make_histograms(capt.numpy())
                pred = data[gate]
                # x2, v2, ang2, angv2 = make_histograms(pred.numpy())

                #########################
                # if j%50 == 0 and not args.debug:
                #     from sklearn.manifold import TSNE
                #     from matplotlib import pyplot as plt
                #     from matplotlib import cm

                #     comb = torch.cat([capt, pred], dim=0)

                #     xx = TSNE(n_components=2).fit_transform(comb)
                #     cc = np.array([0]*capt.shape[0] + [9]*pred.shape[0])

                #     plt.scatter(xx[:, 0], xx[:, 1],
                #         c=cc, cmap=plt.cm.get_cmap("jet", 10),
                #         alpha=0.9, s=50)
                #     wandb.log({
                #         "tsne %s" % j: plt,
                #     })
                # plt.colorbar(ticks=range(2))
                # plt.clim(-0.5, 9.5)
                #########################

                if not args.debug:
                    # wandb_lunarlander(capt, pred)
                    logging.wandb_minigrid(capt, pred)


            if not args.debug:
                if (j % 2) == 0 or True:
                    wandb.log(dict(ent1=dist_entropy1, val1=value_loss1, aloss1=action_loss1,))
                    print("ent1 {:.4f}, val1 {:.4f}, loss1 {:.4f}\n".format(
                        dist_entropy1, value_loss1, action_loss1))
                if (j % 2) == 1 or True:
                    wandb.log(dict(ent2=dist_entropy2, val2=value_loss2, aloss2=action_loss2, prederr2=pred_err2))
                    print("ent2 {:.4f}, val2 {:.4f}, loss2 {:.4f}, prederr2 {:.4f}\n".format(
                        dist_entropy2, value_loss2, action_loss2, pred_err2))

                wandb.log(dict(mean_gt=rollouts[0].actions.float().mean().item()))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    main()
