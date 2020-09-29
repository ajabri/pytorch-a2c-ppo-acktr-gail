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
        concatenated = np.concatenate((o, a, d))
        return concatenated

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




class AsyncWrapper(gym.Wrapper):
    def __init__(self, env, obs_interval, predict_interval):
        if not 'reward_range' in env.__dict__:
            env.reward_range = (-float('inf'), float('inf'))
        if not 'metadata' in env.__dict__:
            env.metadata = None
        """ pending obs: the state at the step when the agent request an observation
            last obs: the most recent observation the agent got
            real state: the real current state
            NOTE: this wrappe currently only works for those environments
                  whose observations are imgs
            TODO: call render to save imgs for other environments.
        """
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.obs_interval = obs_interval
        self.predict_interval = predict_interval

    def reset(self):
        obs = self.env.reset()
        self.real_state = obs.copy()
        # directly request observation at the first step
        self.step_count = 0
        if self.obs_interval == 0:
            self.last_obs = obs.copy()
        else:
            self.pending_obs = obs.copy()
            self.last_obs = np.zeros_like(obs)
        return self.last_obs

    def step(self, action):
        predict, action = action[0], action[1:]
        if len(action) == 1:
            action = action[0]

        obs, reward, done, info = self.env.step(action)
        self.real_state = obs.copy()
        if self.obs_interval == 0:
            self.last_obs = obs.copy()
        elif self.step_count == -1:
            if not predict:
                self.step_count = 0
                self.pending_obs = obs
        else:
            self.step_count = (self.step_count + 1) % self.obs_interval
            if self.step_count == 0:
                self.last_obs = self.pending_obs
                if predict:
                    self.step_count = -1
                else:
                    self.pending_obs = obs

        return self.last_obs, reward, done, info

    def full_obs(self):
        return self.real_state, self.last_obs
