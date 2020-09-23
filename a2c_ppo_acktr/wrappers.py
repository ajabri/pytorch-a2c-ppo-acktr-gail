import gym
# from gym.wrappers import FlattenObservation
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
    def __init__(self, env, obs_interval, predict_interval, no_op=False, no_op_action=None, record_imgs=False):
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
        self.action_dim = self.env.action_space.shape
        self.img_list = []
        self.record_imgs = record_imgs
        if self.action_dim == ():
            self.last_action = 0
        else:
            self.last_action = np.zeros(self.env.action_space.shape[0])

    def reset(self):
        obs = self.env.reset()
        rgb_img = self.env.render(mode='rgb_array')
        if self.record_imgs:
            self.img_list = [rgb_img]
        if self.action_dim == ():
            self.last_action = 0
        else:
            self.last_action = np.zeros(self.env.action_space.shape[0])
        return obs

    def step(self, action):
        # self.img_list = []
        predict, action = action[0], action[1:]
        infos, rewards, actions, dones = {}, 0, [], []
        if not predict:
            no_op = self.no_op_action if self.no_op else self.last_action
            for _ in range(self.obs_interval - 1):
                actions.append(no_op)
        actions.append(action)
        self.last_action = action
        for action in actions:
            if self.action_dim == ():
                action = action[0]
            obs, reward, done, info = self.env.step(action)
            # if self.record_imgs:
            #     self.img_list.append(self.env.render(mode='rgb_array'))
            rewards += reward
            for k in info.keys():
                if k in infos:
                    infos[k].append(info[k])
                else:
                    infos[k] = [info[k]]
            if done:
                break
        # if self.record_imgs:
        #     if len(self.img_list) > 1:
        #         self.img_list = [i//2 for i in self.img_list[:-1]] + self.img_list[-1]
        return obs, rewards, done, infos

    def full_obs(self):
        rgb_img, rgb_img2 = [], []
        for img in self.img_list:
            img2 = img.copy() #200x200x3
            r, g, b = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
            indices = np.logical_and(r<155, np.logical_and(g<155, b<155))
            img2[indices] = img2[indices] + np.array([0, 0, 100])
            rgb_img.append(img)
            rgb_img2.append(img2)
        return rgb_img2, rgb_img



class HandAsyncWrapper(gym.Wrapper):
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
        self.action_dim = self.env.action_space.shape

    def reset(self):
        obs = self.env.reset()
        self.no_op_counter = 0
        self.last_obs = obs
        return obs

    def step(self, action):
        _, action = action[0], action[1:] #predict is not used for now
        self.no_op_counter = (self.no_op_counter + 1) % self.obs_interval
        if self.action_dim == ():
            action = action[0]
        obs, reward, done, info = self.env.step(action)
        if self.no_op_counter % self.obs_interval == 0:
            self.last_obs = obs
        info['valid_obs'] = (self.no_op_counter % self.obs_interval == 0)
        return self.last_obs, reward, done, info

    def full_obs(self):
        rgb_img = self.env.render(mode='rgb_array')
        rgb_img2 = rgb_img.copy() #200x200x3
        rgb_img2 = rgb_img2//4
        return rgb_img2, rgb_img

class HandNormAsyncWrapper(gym.Wrapper):
    def __init__(self, env, obs_interval, predict_interval, no_op=False,
                 no_op_action=None, gamma=None, stop_at_done=False):
        # missing variables below
        env.reward_range = (-float('inf'), float('inf'))
        env.metadata = None
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.obs_interval = obs_interval
        self.predict_interval = predict_interval
        assert self.predict_interval == 1 #remove and fix it in the future
        from baselines.common.running_mean_std import RunningMeanStd
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.ret_rms = RunningMeanStd(shape=())
        self.clipob = 10
        self.cliprew = 10
        self.ret = 0
        self.gamma = gamma
        self.epsilon = 1e-8
        self.stop_at_done = stop_at_done

    def reset(self):
        obs = self.env.reset()
        self.no_op_counter = 0
        self.last_obs = obs
        return obs

    def step(self, action):
        self.no_op_counter = (self.no_op_counter + 1) % self.obs_interval
        _, action = action[0], action[1:]
        obs, reward, done, info = self.env.step(action)
        self.ret = self.ret * self.gamma + reward
        self.ret_rms.update(np.array([self.ret]))
        self.ob_rms.update(np.expand_dims(self.last_obs, axis=0))
        reward = np.clip(reward / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        if done:
            self.ret = 0.
        if self.no_op_counter % self.obs_interval == 0:
            self.last_obs = obs
        info['valid_obs'] = (self.no_op_counter % self.obs_interval == 0)
        return self.last_obs, reward, done, info

    def _obfilt(self, obs):
        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        return obs
#
# class RobotSuiteWrapper(gym.Wrapper):
#     def __init__(self, env):
#         # missing variables below
#         env.reward_range = (-float('inf'), float('inf'))
#         env.metadata = None
#         env.action_space = spaces.Box(-np.inf, np.inf, shape=(env.dof,), dtype='float32')
#         env.observation_space = spaces.Box(-np.inf, np.inf, shape=(env.camera_height, env.camera_width, 3), dtype='float32')
#         gym.Wrapper.__init__(self, env)
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#
#     def reset(self):
#         obs = self.env.reset()
#         return obs['image']
#
#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         return obs['image'], reward, done, info
#
#
# class HandWrapper(gym.Wrapper):
#     def __init__(self, env):
#         # missing variables below
#         env.reward_range = (-float('inf'), float('inf'))
#         env.metadata = None
#         gym.Wrapper.__init__(self, env)
#
#     def reset(self):
#         return self.env.reset()
#
#     def seed(self, seed=None):
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]
#
#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         return obs, reward, done, info
#
#     def close(self):
#         pass
