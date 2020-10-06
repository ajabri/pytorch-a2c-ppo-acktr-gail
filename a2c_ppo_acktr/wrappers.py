import gym
# from gym.wrappers import FlattenObservation
import cv2
from gym.spaces.box import Box
from pdb import set_trace as st
from gym import ObservationWrapper
import numpy as np
from gym import error, spaces
from gym.utils import seeding


class MinigridWrapper(gym.core.ObservationWrapper):
    def __init__(self, env, tile_size = 8):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']
        self.tile_size = tile_size

    def observation(self, obs):
        obs = obs['image']
        return obs

class DynamicObsWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env

    def step(self, act):
        obs, reward, done, info = self.env.step(act)
        dist_to_obstacle = []
        for i in range(len(self.env.obstacles)):
            dist_to_obstacle.append(((self.env.agent_pos - self.env.obstacles[i].cur_pos)**2).sum())
        info['dist_to_obstacle'] = dist_to_obstacle
        return obs, reward, done, info

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



class AsyncWrapper(gym.Wrapper):
    def __init__(self, env, env_id, obs_interval, predict_interval, keep_vis=False):
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
        self.env_id = env_id
        self.obs_interval = obs_interval
        self.predict_interval = predict_interval
        self.keep_vis = keep_vis

    def get_vis(self, obs):
        if self.env_id.startswith("MiniGrid"):
            rgb_img = self.unwrapped.render(
                        mode='rgb_array',
                        highlight=True,
                        tile_size=16
                    ).copy()
            return rgb_img
        elif len(obs.shape) == 3:
            return obs.copy()
        else:
            return self.render(mode='rgb_array').copy()

    def reset(self):
        obs = self.env.reset()
        # directly request observation at the first step
        self.step_count = 0
        if self.obs_interval == 0:
            self.last_obs = obs.copy()
        else:
            self.pending_obs = obs.copy()
            self.last_obs = np.zeros_like(obs)

        if self.keep_vis:
            self.real_vis = self.get_vis(obs)
            if self.obs_interval == 0:
                self.last_vis = self.real_vis.copy()
            else:
                self.pending_vis = self.real_vis.copy()
                self.last_vis = np.zeros_like(self.pending_vis) #no memory

        return self.last_obs

    def step(self, action):
        predict, action = action[0], action[1:]
        if len(action) == 1:
            action = action[0]

        obs, reward, done, info = self.env.step(action)
        if self.obs_interval == 0:
            self.last_obs = obs.copy()
        elif self.step_count == -1:
            if not predict:
                self.step_count = 0
                self.pending_obs = obs.copy()
        else:
            self.step_count = (self.step_count + 1) % self.obs_interval
            if self.step_count == 0:
                self.last_obs = self.pending_obs
                if predict:
                    self.step_count = -1
                else:
                    self.pending_obs = obs.copy()

        if self.keep_vis:
            self.real_vis = self.get_vis(obs)
            if self.obs_interval == 0:
                self.last_vis = self.real_vis.copy()
            elif self.step_count == -1:
                if not predict:
                    self.pending_vis = self.real_vis.copy()
            elif self.step_count == 0:
                    self.last_vis = self.pending_vis
                    if not predict:
                        self.pending_vis = self.real_vis.copy()

        return self.last_obs, reward, done, info

    def full_obs(self):
        assert self.keep_vis
        return self.real_vis, self.last_vis
