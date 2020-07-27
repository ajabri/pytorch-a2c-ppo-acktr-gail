import gym
from gym.wrappers import FlattenObservation
import cv2
from gym.spaces.box import Box
from pdb import set_trace as st
from gym import ObservationWrapper
import numpy as np

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']


class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image. """

    def __init__(self, env, prop=1):
        super(ResizeObservation, self).__init__(env)
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
        rgb_img2[:, :, -1] += 100
        return rgb_img2, rgb_img


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

# class MiniWorldWrapper(gym.core.ObservationWrapper):
#     def __init__(self, env, resolution_scale=1):
#         obs_height, obs_width = 60, 80
#         real_obs_height, real_obs_width = int(resolution_scale*obs_height), int(resolution_scale*obs_width)
#         super().__init__(env, obs_height=real_obs_height, obs_width=real_obs_width)
#
#     def observation(self, observation):
#         return observation
#
#     def full_obs(self):
#         top_down_view = env.render_top_view()
#         top_down_view2 = top_down_view.copy()
#         r, g, b = top_down_view[:, :, 0], top_down_view[:, :, 1], top_down_view[:, :, 2]
#         indices = np.logical_and(r!=0, np.logical_and(g==0, b==0))
#         ratio = 255
#         top_down_view2[indices] = ratio * np.array([0, 0, 1])
#         top_down_view[indices] = ratio * np.array([1, 0, 0])
#
#         obs = env.render_obs()
#         obs2 = obs.copy()
#         obs2[:, :, -1] += 125
#         return top_down_view2, top_down_view, obs2, obs
import pyglet
try:
    import gym_miniworld
    from gym_miniworld.miniworld import MiniWorldEnv, Room
    from gym_miniworld.envs.ymaze import YMaze
    from gym_miniworld.envs.fourrooms import FourRooms
except pyglet.canvas.xlib.NoSuchDisplayException:
    pass


class YMazeNew(YMaze):
    def __init__(self, resolution_scale=1.):
        obs_height, obs_width = 60, 80
        real_obs_height, real_obs_width = int(resolution_scale*obs_height), int(resolution_scale*obs_width)
        super().__init__(obs_height=real_obs_height, obs_width=real_obs_width)

    # def step(self, action):
    #     obs, reward, done, info = super().step(action)
    #
    #     if self.near(self.box):
    #         reward += self._reward()
    #         done = True
    #
    #     return obs, reward, done, info

    def full_obs(self):
        """
        actually just a change of view, change it in the future"""
        top_down_view = self.render_top_view()
        top_down_view2 = top_down_view.copy()
        r, g, b = top_down_view[:, :, 0], top_down_view[:, :, 1], top_down_view[:, :, 2]
        indices = np.logical_and(r!=0, np.logical_and(g==0, b==0))
        # ratio = r[indices].reshape((-1, 1))
        ratio = 255
        top_down_view2[indices] = ratio * np.array([0, 0, 1])
        top_down_view[indices] = ratio * np.array([1, 0, 0])
        obs = self.render_obs()
        obs2 = obs.copy()
        r, g, b = obs2[:, :, 0], obs2[:, :, 1], obs2[:, :, 2]
        indices = np.logical_and(r<200, np.logical_and(g<200, b<200))
        obs2[indices] = obs2[indices] + np.array([0, 0, 100])
        # [:, :, -1] += 125
        return top_down_view2, top_down_view, obs2, obs
        # # RED: observe
        # # GREEN: predict

class FourRoomsNew(FourRooms):
    def __init__(self, resolution_scale=1.):
        obs_height, obs_width = 60, 80
        real_obs_height, real_obs_width = int(resolution_scale*obs_height), int(resolution_scale*obs_width)
        super().__init__(obs_height=real_obs_height, obs_width=real_obs_width)

    # def step(self, action):
    #     obs, reward, done, info = super().step(action)
    #
    #     # define a dense reward for this environment
    #     if self.near(self.box):
    #     #     reward += self._reward()
    #         done = True
    #         # reward += 200
    #     ent0, ent1 = self.box, self.agent
    #     reward -= (np.abs(ent0.pos[0] - ent1.pos[0]) + np.abs(ent0.pos[1] - ent1.pos[1]))
    #
    #     # dist = np.linalg.norm(ent0.pos - ent1.pos)
    #     # return dist < ent0.radius + ent1.radius + 1.1 * self.max_forward_step
    #
    #     return obs, reward, done, info

    def full_obs(self):
        top_down_view = self.render_top_view()
        top_down_view2 = top_down_view.copy()
        r, g, b = top_down_view[:, :, 0], top_down_view[:, :, 1], top_down_view[:, :, 2]
        indices = np.logical_and(r!=0, np.logical_and(g==0, b==0))
        # ratio = r[indices].reshape((-1, 1))
        # ratio = 255
        top_down_view2[indices] = np.array([0, 0, 255])
        top_down_view[indices] = np.array([255, 0, 0])
        # top_down_view2[indices] = np.array([49, 154, 87])
        # # ratio * np.array([0, 0, 1])
        # top_down_view[indices] = np.array([226, 34, 92])
        # ratio * np.array([1, 0, 0])
        obs = self.render_obs()
        obs2 = obs.copy()
        r, g, b = obs2[:, :, 0], obs2[:, :, 1], obs2[:, :, 2]
        indices = np.logical_and(r<200, np.logical_and(g<200, b<200))
        obs2[indices] = obs2[indices] + np.array([0, 0, 100])
        # [:, :, -1] += 125
        return top_down_view2, top_down_view, obs2, obs
