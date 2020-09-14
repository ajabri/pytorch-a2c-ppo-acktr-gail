import gym
from gym.wrappers import FlattenObservation
import cv2
from gym.spaces.box import Box
from pdb import set_trace as st
from gym import ObservationWrapper
import numpy as np
from gym import error, spaces
from gym.utils import seeding


class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

class ObservationOnlyWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = env.observation_space['observation'].shape[0] + env.observation_space['achieved_goal'].shape[0] + env.observation_space['desired_goal'].shape[0]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(shape,), dtype='float32')

    def observation(self, obs):
        o = obs['observation']
        a = obs['achieved_goal']
        d = obs['desired_goal']
        # print(o.shape, a.shape, d.shape)
        concatenated = np.concatenate((o, a, d))
        return concatenated
        # return obs['observation']

class TiledObsWrapper(ImgObsWrapper):
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


class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image. """

    def __init__(self, env, prop=1):
        super(ResizeObservation, self).__init__(env)
        self.env = env
        H, W, _ = env.observation_space.shape
        self.H, self.W = int(H*prop), int(W*prop)
        self.observation_space = Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8)
        self.prop = prop

    def observation(self, observation):
        # observation = cv2.resize(observation, (self.H, self.W))
        observation = cv2.resize(observation, None, fx=self.prop, fy=self.prop)
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation

    def full_obs(self):
        rgb_img = self.observation(self.state)
        # print(rgb_img.observation)
        rgb_img2 = rgb_img.copy() #200x200x3
        rgb_img2 = rgb_img2//4
        return rgb_img2, rgb_img

class TimeStepCounter(gym.Wrapper):
    def __init__(self, env):
        super(TimeStepCounter, self).__init__(env)

    def reset(self):
        self.step_index = 0
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        info['step_index'] = self.step_index
        self.step_index += 1
        return obs, rew, done, info



class AsyncWrapper(gym.Wrapper):
    def __init__(self, env, obs_interval, predict_interval, no_op=False, no_op_action=None):
        # missing variables below
        env.reward_range = (-float('inf'), float('inf'))
        env.metadata = None
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.obs_interval = obs_interval
        self.predict_interval = predict_interval
        assert self.predict_interval == 1 #remove and fix it in the future
        self.no_op_action = no_op_action
        self.no_op = no_op
        self.which_obs = 'first' #TODO: implement it.

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        predict, action = action[0], action[1:]
        dones = []
        goal_achieved = []
        if not predict:
            curr_action = self.no_op_action if self.no_op else action
            # time taken to process the last observation. While processing the images, do some no_op
            for _ in range(self.obs_interval - 1):
                _, _, done, info = self.env.step(curr_action)
                if 'goal_achieved' in info.keys():
                    goal_achieved.append(info['goal_achieved'])
                dones.append(done)
        obs, reward, done, info = self.env.step(action)
        dones.append(done)
        if 'goal_achieved' in info.keys():
            goal_achieved.append(info['goal_achieved'])
        done = True in dones
        info['goal_achieved'] = goal_achieved
        return obs, reward, done, info


class RobotSuiteWrapper(gym.Wrapper):
    def __init__(self, env):
        # missing variables below
        env.reward_range = (-float('inf'), float('inf'))
        env.metadata = None
        env.action_space = spaces.Box(-np.inf, np.inf, shape=(env.dof,), dtype='float32')
        env.observation_space = spaces.Box(-np.inf, np.inf, shape=(env.camera_height, env.camera_width, 3), dtype='float32')
        gym.Wrapper.__init__(self, env)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        obs = self.env.reset()
        return obs['image']

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs['image'], reward, done, info


class HandWrapper(gym.Wrapper):
    def __init__(self, env):
        # missing variables below
        env.reward_range = (-float('inf'), float('inf'))
        env.metadata = None
        gym.Wrapper.__init__(self, env)

    def reset(self):
        return self.env.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def close(self):
        pass
