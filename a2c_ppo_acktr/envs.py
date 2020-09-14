import os

import gym
from pdb import set_trace as st
import gym_minigrid
import numpy as np
import torch
import cv2
from gym.spaces.box import Box
import sys
sys.path.append('/home/vioichigo/baselines')
from gym.envs.registration import EnvSpec
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

try:
    import robosuite as suite
except ImportError:
    pass

try:
    import vizdoomgym
except ImportError:
    pass

from gym.wrappers import FlattenObservation
from a2c_ppo_acktr.wrappers import *

def make_env(env_id, seed, rank, log_dir, allow_early_resets, get_pixel = False, resolution_scale = 1., async_params=[1, 1, False]):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        elif env_id.startswith("MiniGrid"):
            env = gym.make(env_id)
            if get_pixel:
                env = TiledObsWrapper(env)
            else:
                # env = RGBImgPartialObsWrapper(env, tile_size = 1)
                env = ImgObsWrapper(env)
        elif env_id.startswith("MiniWorld"):
            no_op_action = 7
            import pyglet
            try:
                import gym_miniworld
                from gym_miniworld.miniworld import MiniWorldEnv, Room
                from gym_miniworld.envs.ymaze import YMaze
                from gym_miniworld.envs.collecthealth import CollectHealth
                from gym_miniworld.envs.fourrooms import FourRooms
            except pyglet.canvas.xlib.NoSuchDisplayException:
                pass

            class YMazeNew(YMaze):
                def __init__(self, resolution_scale=1.):
                    obs_height, obs_width = 60, 80
                    real_obs_height, real_obs_width = int(resolution_scale*obs_height), int(resolution_scale*obs_width)
                    super().__init__(obs_height=real_obs_height, obs_width=real_obs_width)

                def full_obs(self):
                    """
                    actually just a change of view, change it in the future"""
                    top_down_view = self.render_top_view()
                    top_down_view2 = top_down_view.copy()
                    r, g, b = top_down_view[:, :, 0], top_down_view[:, :, 1], top_down_view[:, :, 2]
                    indices = np.logical_and(r!=0, np.logical_and(g==0, b==0))
                    top_down_view2[indices] = np.array([0, 0, 255])
                    top_down_view[indices] = np.array([255, 0, 0])
                    obs = self.render_obs()
                    obs2 = obs.copy()
                    obs2 = obs2//4
                    return top_down_view2, top_down_view, obs2, obs
            class CollectHealthNew(CollectHealth):
                def __init__(self, resolution_scale=1.):
                    obs_height, obs_width = 60, 80
                    real_obs_height, real_obs_width = int(resolution_scale*obs_height), int(resolution_scale*obs_width)
                    super().__init__(obs_height=real_obs_height, obs_width=real_obs_width)

                def full_obs(self):
                    """
                    actually just a change of view, change it in the future"""
                    top_down_view = self.render_top_view()
                    top_down_view2 = top_down_view.copy()
                    r, g, b = top_down_view[:, :, 0], top_down_view[:, :, 1], top_down_view[:, :, 2]
                    indices = np.logical_and(r!=0, np.logical_and(g==0, b==0))
                    top_down_view2[indices] = np.array([0, 0, 255])
                    top_down_view[indices] = np.array([255, 0, 0])
                    obs = self.render_obs()
                    obs2 = obs.copy()
                    obs2 = obs2//4
                    return top_down_view2, top_down_view, obs2, obs

            class FourRoomsNew(FourRooms):
                def __init__(self, resolution_scale=1.):
                    obs_height, obs_width = 60, 80
                    real_obs_height, real_obs_width = int(resolution_scale*obs_height), int(resolution_scale*obs_width)
                    super().__init__(obs_height=real_obs_height, obs_width=real_obs_width)

                def full_obs(self):
                    top_down_view = self.render_top_view()
                    top_down_view2 = top_down_view.copy()
                    r, g, b = top_down_view[:, :, 0], top_down_view[:, :, 1], top_down_view[:, :, 2]
                    indices = np.logical_and(r!=0, np.logical_and(g==0, b==0))
                    # ratio = r[indices].reshape((-1, 1))
                    # ratio = 255
                    top_down_view2[indices] = np.array([0, 0, 255])
                    # top_down_view2[indices] = np.array([91, 127, 228])
                    # top_down_view[indices] = np.array([255, 0, 0])
                    top_down_view[indices] = np.array([255, 0, 0])
                    # top_down_view2[indices] = np.array([49, 154, 87])
                    # # ratio * np.array([0, 0, 1])
                    # top_down_view[indices] = np.array([226, 34, 92])
                    # ratio * np.array([1, 0, 0])
                    obs = self.render_obs()
                    obs2 = obs.copy()
                    # r, g, b = obs2[:, :, 0], obs2[:, :, 1], obs2[:, :, 2]
                    # indices = np.logical_and(r<200, np.logical_and(g<200, b<200))
                    # obs2[indices] = obs2[indices] + np.array([0, 0, 100])
                    # [:, :, -1] += 125
                    obs2 = obs2//4
                    return top_down_view2, top_down_view, obs2, obs

            if env_id.startswith("MiniWorld-FourRooms"):
                env = FourRoomsNew(resolution_scale=resolution_scale)
            elif env_id.startswith("MiniWorld-YMaze"):
                env = YMazeNew(resolution_scale=resolution_scale)
            elif env_id.startswith("MiniWorld-CollectHealth"):
                env = CollectHealthNew(resolution_scale=resolution_scale)
            else:
                raise NotImplementedError("resolution needs to be changed.")
        elif env_id.startswith("Sawyer"):
            env = suite.make(
                    env_id,
                    has_renderer=False,          # no on-screen renderer
                    has_offscreen_renderer=True, # off-screen renderer is required for camera observations
                    ignore_done=False,            # (optional) never terminates episode
                    use_camera_obs=True,         # use camera observations
                    camera_height=84,            # set camera height
                    camera_width=84,             # set camera width
                    camera_name='agentview',     # use "agentview" camera
                    use_object_obs=False,        # no object feature when training on pixels
                    reward_shaping=True          # (optional) using a shaping reward
                )
            no_op_action = np.zeros(env.dof)
            env.spec = EnvSpec(env_id + '-v0')

            env = RobotSuiteWrapper(env)
            env = ResizeObservation(env, prop=resolution_scale)

        elif env_id.startswith("Vizdoom"):
            env = gym.make(env_id)
            no_op_action = None
            env = ResizeObservation(env, prop=resolution_scale)

        elif env_id in ['relocate-v0', 'pen-v0', 'hammer-v0', 'door-v0']:
            try:
                from mjrl.utils.gym_env import GymEnv
                import mj_envs
            except ImportError:
                pass
            env = GymEnv(env_id)
            env.spec = EnvSpec(env_id)
            env = HandWrapper(env)
            no_op_action = 0

        else:
            env = gym.make(env_id)
            no_op_action = 0
            env = ResizeObservation(env, prop=resolution_scale)

        obs_interval, predict_interval, no_op = async_params
        env = AsyncWrapper(env, obs_interval=obs_interval, predict_interval=predict_interval, no_op=no_op, no_op_action = no_op_action)


        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)
        env = TimeStepCounter(env)

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

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if obs_shape != None and len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
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
                  resolution_scale=1.,
                  image_stack=False,
                  async_params=[1, 1, False]):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, get_pixel=get_pixel, resolution_scale=resolution_scale, async_params=async_params)
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
    elif len(envs.observation_space.shape) == 3 and image_stack:
        envs = VecPyTorchFrameStack(envs, 4, device)

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
