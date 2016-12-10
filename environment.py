# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import gym
import cv2

from constants import GYM_ENV

class Environment(object):
  # cached action size
  action_size = -1
  
  @staticmethod
  def create_environment():
    if GYM_ENV == "DebugMaze":
      return MazeEnvironment()
    else:
      return GameEnvironment()
  
  @staticmethod
  def get_action_size():
    if Environment.action_size >= 0:
      return Environment.action_size

    if GYM_ENV == "DebugMaze":
      Environment.action_size = 4
      return 4
    else:
      env = gym.make(GYM_ENV)
      Environment.action_size = env.action_space.n
      print("intialize action size={0}".format(Environment.action_size))
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
    """
    # (80,80)
    c = np.zeros([20,20], dtype=np.float32)
    for x in range(80):
      for y in range(80):
        c[x // 4, y // 4] += m[x,y]
    c = c / 16.0
    """
    c = self._subsample(m, 4)
    return c
  

  
class GameEnvironment(Environment):
  def __init__(self, display=False, frame_skip=4, no_op_max=30):
    Environment.__init__(self)
    
    self._display = display
    self._frame_skip = frame_skip
    if self._frame_skip < 1:
      self._frame_skip = 1
    self._no_op_max = no_op_max
    
    self.env = gym.make(GYM_ENV)
    self.reset()

    
  def reset(self):
    observation = self.env.reset()
    self.last_state = self._preprocess_frame(observation)
    
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
    return state, reward, terminal, pixel_change


"""
    + + W + + + G
    + + W + W W W
    S + W + + + W
    + + W W W + +
    + + W + W + +
    + + W + + + +
    + + + + + W W
"""

class MazeEnvironment(Environment):
  def __init__(self):
    Environment.__init__(self)
    
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
    return image, reward, terminal, pixel_change

    
