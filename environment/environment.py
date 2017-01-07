# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

from constants import ENV_TYPE

class Environment(object):
  # cached action size
  action_size = -1
  
  @staticmethod
  def create_environment():
    if ENV_TYPE == 'maze':
      from . import maze_environment
      return maze_environment.MazeEnvironment()
    elif ENV_TYPE == 'lab':
      from . import lab_environment
      return lab_environment.LabEnvironment()
    else:
      from . import gym_environment
      return gym_environment.GymEnvironment()
  
  @staticmethod
  def get_action_size():
    if Environment.action_size >= 0:
      return Environment.action_size

    if ENV_TYPE == 'maze':
      from . import maze_environment
      Environment.action_size = \
        maze_environment.MazeEnvironment.get_action_size()
    elif ENV_TYPE == "lab":
      from . import lab_environment
      Environment.action_size = \
        lab_environment.LabEnvironment.get_action_size()
    else:
      from . import gym_environment
      Environment.action_size = \
        gym_environment.GymEnvironment.get_action_size()
    return Environment.action_size

  def __init__(self):
    pass

  def process(self, action):
    pass

  def reset(self):
    pass

  def _subsample(self, a, average_width):
    s = a.shape
    sh = s[0]//average_width, average_width, s[1]//average_width, average_width
    return a.reshape(sh).mean(-1).mean(1)  

  def _calc_pixel_change(self, state, last_state):
    d = np.absolute(state[2:-2,2:-2,:] - last_state[2:-2,2:-2,:])
    # (80,80,3)
    m = np.mean(d, 2)
    c = self._subsample(m, 4)
    return c
