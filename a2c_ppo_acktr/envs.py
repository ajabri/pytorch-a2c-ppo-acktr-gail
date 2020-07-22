import os

import gym
import gym_minigrid
import numpy as np
import torch
import cv2
from gym.spaces.box import Box
from pdb import set_trace as st

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

from gym.wrappers import FlattenObservation

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

from gym.spaces import Box
from gym import ObservationWrapper
class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image. """

    def __init__(self, env, prop=1):
        super(ResizeObservation, self).__init__(env)
        H, W, _ = env.observation_space.shape
        self.H, self.W = int(H*prop), int(W*prop)
        self.observation_space = Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8)
        self.prop = prop

    def observation(self, observation):
        # observation = cv2.resize(observation, (self.H, self.W, 3), interpolation=cv2.INTER_AREA)
        observation = cv2.resize(observation, None, fx=self.prop, fy=self.prop)
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation

    def full_obs(self):
        rgb_img = self.render()
        rgb_img2 = rgb_img.copy() #200x200x3
        rgb_img2[:, :, -1] += 122
        return rgb_img2, rgb_img


class FullyObsWrapper(ImgObsWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env, tile_size = 8):
        super().__init__(env)
        self.tile_size = tile_size

    def full_obs(self):
        env = self.unwrapped
        rgb_img = env.render(
                    mode='rgb_array',
                    highlight=False,
                    tile_size=self.tile_size
                )

        rgb_img2 = rgb_img.copy() #200x200x3
        r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
        indices = np.logical_and(r!=0, np.logical_and(g==0, b==0))
        ratio = r[indices].reshape((-1, 1))
        rgb_img2[indices] = ratio * np.array([0, 0, 1])
        # RED: observe
        # BLUE: predict
        return rgb_img2, rgb_img
#
class MiniWorldWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, resolution_scale=1):
        if resolution_scale == .5:
            super().__init__(env, obs_height=30, obs_width=40)
        else:
            super().__init__(env)

    def observation(self, observation):
        return observation

    def full_obs(self):
        top_down_view = env.render_top_view()
        top_down_view2 = top_down_view.copy()
        r, g, b = top_down_view[:, :, 0], top_down_view[:, :, 1], top_down_view[:, :, 2]
        indices = np.logical_and(r!=0, np.logical_and(g==0, b==0))
        ratio = 255
        top_down_view2[indices] = ratio * np.array([0, 0, 1])
        top_down_view[indices] = ratio * np.array([1, 0, 0])

        obs = env.render_obs()
        obs2 = obs.copy()
        obs2[:, :, -1] += 125
        return top_down_view2, top_down_view, obs2, obs


def make_env(env_id, seed, rank, log_dir, allow_early_resets, get_pixel = False, resolution_scale = 1.):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif env_id.startswith("MiniGrid"):
            env = gym.make(env_id)
            if get_pixel:
                env = FullyObsWrapper(env)
            else:
                # env = RGBImgPartialObsWrapper(env, tile_size = 1)
                env = ImgObsWrapper(env)
        elif env_id.startswith("MiniWorld"):
            import gym_miniworld
            # from gym_miniworld.miniworld import MiniWorldEnv, Room
            # from gym_miniworld.envs.ymaze import YMaze
            # from gym_miniworld.envs.fourrooms import FourRooms
            # class YMazeNew(YMaze):
            #     def __init__(self):
            #         # super().__init__(obs_height=15, obs_width=20)
            #         if resolution_scale == .5:
            #             super().__init__(obs_height=30, obs_width=40)
            #         else:
            #             super().__init__()
            #
            #     def full_obs(self):
            #         """
            #         actually just a change of view, change it in the future"""
            #         obs = env.render_top_view()
            #         obs2 = obs.copy()
            #         r, g, b = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2]
            #         indices = np.logical_and(r!=0, np.logical_and(g==0, b==0))
            #         # ratio = r[indices].reshape((-1, 1))
            #         ratio = 255
            #         obs2[indices] = ratio * np.array([0, 0, 1])
            #         obs[indices] = ratio * np.array([1, 0, 0])
            #         return obs2, obs
            #         # # RED: observe
            #         # # GREEN: predict
            #
            # class FourRoomsNew(FourRooms):
            #     def __init__(self):
            #         if resolution_scale == .5:
            #             super().__init__(obs_height=30, obs_width=40)
            #         else:
            #             super().__init__()
            #
            #     def full_obs(self):
            #         """
            #         actually just a change of view, change it in the future"""
            #         obs = env.render_top_view()
            #         obs2 = obs.copy()
            #         r, g, b = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2]
            #         indices = np.logical_and(r!=0, np.logical_and(g==0, b==0))
            #         ratio = 255
            #         obs2[indices] = ratio * np.array([0, 0, 1])
            #         obs[indices] = ratio * np.array([1, 0, 0])
            #         return obs2, obs
            #
            #
            # # env = YMazeNew()
            # env = FourRoomsNew()
            env = gym.make(env_id)
            env = MiniWorldWrapper(env)
        else:
            env = gym.make(env_id)
            env = ResizeObservation(env, prop=resolution_scale)


        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        # is_minigrid = 'minigrid' in env_id.lower()

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        elif env_id.startswith("MiniGrid"):
            env = FlattenObservation(env)
            # env = TransposeImage(env, op=[2, 0, 1])
            return env
        elif env_id.startswith("MiniWorld"):
            env = TransposeImage(env, op=[2, 0, 1])
            return env
        elif len(env.observation_space.shape) == 3:
            env = TransposeImage(env, op=[2, 0, 1])
            return env
            # raise NotImplementedError(
            #     "CNN models work only for atari,\n"
            #     "please use a custom wrapper for a custom pixel input env.\n"
            #     "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None,
                  get_pixel=False,
                  resolution_scale=1.):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, get_pixel=get_pixel, resolution_scale=resolution_scale)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='spawn')
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    # For miniworld and minigrid environments, the input dimension is small enough to only use
    # elif len(envs.observation_space.shape) == 3 and not env_name.startswith("Mini"):
    #     envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
