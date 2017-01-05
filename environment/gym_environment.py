# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import gym

from constants import ENV_NAME
from environment import environment

class GymEnvironment(environment.Environment):
  @staticmethod
  def get_action_size():
    env = gym.make(ENV_NAME)
    return env.action_space.n
  
  def __init__(self, display=False, frame_skip=4, no_op_max=30):
    environment.Environment.__init__(self)
    
    self._display = display
    self._frame_skip = frame_skip
    if self._frame_skip < 1:
      self._frame_skip = 1
    self._no_op_max = no_op_max
    
    self.env = gym.make(ENV_NAME)
    self.reset()

  def reset(self):
    observation = self.env.reset()
    self.last_state = self._preprocess_frame(observation)
    self.last_action = 0
    self.last_reward = 0
    
    # randomize initial state
    if self._no_op_max > 0:
      no_op = np.random.randint(0, self._no_op_max + 1)
      for _ in range(no_op):
        observation, _, _, _ = self.env.step(0)
      if no_op > 0:
        self.last_state = self._preprocess_frame(observation)
        
  def _preprocess_frame(self, observation):
    # observation shape = (210, 160, 3)
    observation = observation.astype(np.float32)
    resized_observation = cv2.resize(observation, (84, 84))
    resized_observation = resized_observation / 255.0
    return resized_observation
  
  def _process_frame(self, action):
    reward = 0
    for i in range(self._frame_skip):
      observation, r, terminal, _ = self.env.step(action)
      reward += r
      if terminal:
        break
    state = self._preprocess_frame(observation)
    return state, reward, terminal
  
  def process(self, action):
    if self._display:
      self.env.render()
    
    state, reward, terminal = self._process_frame(action)
    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change
