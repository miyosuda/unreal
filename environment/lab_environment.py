# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import cv2
import numpy as np
import deepmind_lab

from environment import environment
from constants import ENV_NAME

COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2

def worker(conn):
  env = gym.make(ENV_NAME)
  conn.send(0)
  
  while True:
    command, arg = conn.recv()

    if command == COMMAND_RESET:
      env.reset()
      obs = env.observations()['RGB_INTERLACED']
      conn.send(obs)
    elif command == COMMAND_ACTION:
      reward = env.step(arg, num_steps=1)
      terminal = not env.is_running()
      if not terminal:
        obs = env.observations()['RGB_INTERLACED']
      else:
        obs = 0
      conn.send([obs, reward, terminal])
    elif command == COMMAND_TERMINATE:
      break
    else:
      print("bad command: {}".format(command))


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
    
    self.conn, child_conn = Pipe()
    self.proc = Process(target=worker, args=(child_conn,))
    self.proc.start()
    
    handshake = self.conn.recv()
    print("handshake={}".format(handshake)) #..
    
    self.reset()

  def reset(self):
    self.conn.send([COMMAND_RESET, 0])
    obs = self.conn.recv()
    
    self.last_state = self._preprocess_frame(obs)
    self.last_action = 0
    self.last_reward = 0
    
  def _preprocess_frame(self, image):
    image = image.astype(np.float32)
    image = image / 255.0
    return image

  def process(self, action):
    real_action = LabEnvironment.ACTION_LIST[action]

    self.conn.send([COMMAND_ACTION, real_action])
    obs, reward, terminal = self.conn.recv()

    if not terminal:
      state = self._preprocess_frame(obs)
    else:
      state = self.last_state
    
    pixel_change = self._calc_pixel_change(state, self.last_state)
    self.last_state = state
    self.last_action = action
    self.last_reward = reward
    return state, reward, terminal, pixel_change
