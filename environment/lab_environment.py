# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import deepmind_lab

from environment import environment
from constants import ENV_NAME


def _action(*entries):
  return np.array(entries, dtype=np.intc)


class LabEnvironment(environment.Environment):
  ACTION_LIST = [
    _action(-20,   0,  0,  0, 0, 0, 0), # look_left
    _action( 20,   0,  0,  0, 0, 0, 0), # look_right
    _action(  0,  10,  0,  0, 0, 0, 0), # look_up
    _action(  0, -10,  0,  0, 0, 0, 0), # look_down
    _action(  0,   0, -1,  0, 0, 0, 0), # strafe_left
    _action(  0,   0,  1,  0, 0, 0, 0), # strafe_right
    _action(  0,   0,  0,  1, 0, 0, 0), # forward
    _action(  0,   0,  0, -1, 0, 0, 0), # backward
    #_action(  0,   0,  0,  0, 1, 0, 0), # fire
    #_action(  0,   0,  0,  0, 0, 1, 0), # jump
    #_action(  0,   0,  0,  0, 0, 0, 1)  # crouch
  ]

  @staticmethod
  def get_action_size():
    return len(LabEnvironment.ACTION_LIST)
  
  def __init__(self):
    environment.Environment.__init__(self)

    level = ENV_NAME

    self._env = deepmind_lab.Lab(
      level,
      ['RGB_INTERLACED'],
      config={
        'fps': str(60),
        'width': str(84),
        'height': str(84)
      })
    self.reset()

  def reset(self):
    self._env.reset()
    obs = self._env.observations()['RGB_INTERLACED']
    self.last_state = self._preprocess_frame(obs)
    self.last_action = 0
    self.last_reward = 0
    
  def _preprocess_frame(self, image):
    image = image.astype(np.float32)
    image = image / 255.0
    return image

  def process(self, action):
    real_action = LabEnvironment.ACTION_LIST[action]
    
    reward = self._env.step(real_action, num_steps=1)
    terminal = not self._env.is_running()

    if not terminal:
      obs = self._env.observations()['RGB_INTERLACED']
      state = self._preprocess_frame(obs)
    else:
      state = self.last_state
    
    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward    
    return state, reward, terminal, pixel_change
