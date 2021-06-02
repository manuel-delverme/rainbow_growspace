"""
From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
"""

import cv2
import gym
import numpy as np
from gym.spaces.box import Box


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None, stack_size=4):
        super(WrapPyTorch, self).__init__(env)
        width, height, features = env.observation_space.shape
        self.observation_space = Box(
            env.observation_space.low.transpose(2, 0, 1),
            env.observation_space.high.transpose(2, 0, 1),
            [features, width, height], dtype=env.observation_space.dtype)
        self.stack_size = stack_size
        self.current_obs = np.zeros((self.stack_size, width, height, features))

    def observation(self, observation):
        x = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        y = np.expand_dims(x, 0)
        if self.stack_size > 1:
            self.current_obs[:-1, :] = self.current_obs[1:, :]
        self.current_obs[-1, :] = y
        return self.current_obs.transpose((0, 3, 1, 2))
