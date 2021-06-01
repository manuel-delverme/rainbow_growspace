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
        self.observation_space = Box(0, 255, [3, 84, 84], dtype=np.uint8)
        self.stack_size = stack_size
        self.current_obs = np.zeros((self.stack_size, *env.observation_space.shape))

    def observation(self, observation):
        x = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_AREA)
        y = np.expand_dims(x, 0)
        if self.stack_size > 1:
            self.current_obs[:-1, :] = self.current_obs[1:, :]
        self.current_obs[-1, :] = y
        return self.current_obs.transpose(0, 3, 1, 2)
