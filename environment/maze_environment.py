# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from environment import environment

class MazeEnvironment(environment.Environment):
  @staticmethod
  def get_action_size():
    return 4
  
  def __init__(self):
    environment.Environment.__init__(self)
    
    self._map_data = \
                     "--+---G" \
                     "--+-+++" \
                     "S-+---+" \
                     "--+++--" \
                     "--+-+--" \
                     "--+----" \
                     "-----++" 
    
    self._setup()
    self.reset()

  def _setup(self):
    image = np.zeros( (84, 84, 3), dtype=float )

    start_pos = (-1, -1)
    goal_pos  = (-1, -1)
  
    for y in range(7):
      for x in range(7):
        p = self._get_pixel(x,y)
        if p == '+':
          self._put_pixel(image, x, y, 0)
        elif p == 'S':
          start_pos = (x, y)
        elif p == 'G':
          goal_pos = (x, y)

    self._maze_image = image
    self._start_pos = start_pos
    self._goal_pos = goal_pos
    
  def reset(self):
    self.x = self._start_pos[0]
    self.y = self._start_pos[1]
    self.last_state = self._get_current_image()
    self.last_action = 0
    self.last_reward = 0    
    
  def _put_pixel(self, image, x, y, channel):
    for i in range(12):
      for j in range(12):
        image[12*y + j, 12*x + i, channel] = 1.0
        
  def _get_pixel(self, x, y):
    data_pos = y * 7 + x
    return self._map_data[data_pos]

  def _is_wall(self, x, y):
    return self._get_pixel(x, y) == '+'

  def _clamp(self, n, minn, maxn):
    if n < minn:
      return minn, True
    elif n > maxn:
      return maxn, True
    return n, False
  
  def _move(self, dx, dy):
    new_x = self.x + dx
    new_y = self.y + dy

    new_x, clamped_x = self._clamp(new_x, 0, 6)
    new_y, clamped_y = self._clamp(new_y, 0, 6)

    hit_wall = False

    if self._is_wall(new_x, new_y):
      new_x = self.x
      new_y = self.y

    hit = clamped_x or clamped_y or hit_wall
    return new_x, new_y, hit

  def _get_current_image(self):
    image = np.array(self._maze_image)
    self._put_pixel(image, self.x, self.y, 1)
    return image

  def process(self, action):
    dx = 0
    dy = 0
    if action == 0: # UP
      dy = -1
    if action == 1: # DOWN
      dy = 1
    if action == 2: # LEFT
      dx = -1
    if action == 3: # RIGHT
      dx = 1

    self.x, self.y, hit = self._move(dx, dy)

    image = self._get_current_image()
    
    terminal = (self.x == self._goal_pos[0] and
                self.y == self._goal_pos[1])

    if terminal:
      reward = 1
    elif hit:
      reward = -1
    else:
      reward = 0

    pixel_change = self._calc_pixel_change(image, self.last_state)
    self.last_state = image
    self.last_action = action
    self.last_reward = reward
    return image, reward, terminal, pixel_change
