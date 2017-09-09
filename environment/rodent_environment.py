# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Pipe
import numpy as np
import os
import rodent

from environment import environment

MAX_STEP_NUM = 60 * 30

class RodentSeekAvoidEnvironment(object):
  ACTION_LIST = [
    [-20,   0,   0], # look_left
    [ 20,   0,   0], # look_right
    [  0,   -1,  0], # strafe_left
    [  0,    1,  0], # strafe_right
    [  0,    0,  1], # forward
    [  0,    0, -1], # backward
  ]

  def __init__(self):
    floor_texture_path = os.path.dirname(os.path.abspath(__file__)) + "/data/floor0.png"
    self.env = rodent.Environment(width=84, height=84,
                                  floor_size=[60,60],
                                  floor_texture_path=floor_texture_path)
    self._prepare_wall()
    
    self.plus_obj_ids_set = set()
    self.minus_obj_ids_set = set()

    self.reset()

  def _prepare_wall(self):
    wall_distance = 30.0

    wall_texture_path = os.path.dirname(os.path.abspath(__file__)) + "/data/wall0.png"
    
    # -Z
    self.env.add_box(texture_path=wall_texture_path,
                     half_extent=[wall_distance, 1.0, 1.0],
                     pos=[0.0, 1.0, -wall_distance],
                     rot=0.0,
                     detect_collision=False)
    # +Z
    self.env.add_box(texture_path=wall_texture_path,
                     half_extent=[wall_distance, 1.0, 1.0],
                     pos=[0.0, 1.0, wall_distance],
                     rot=0.0,
                     detect_collision=False)
    # -X
    self.env.add_box(texture_path=wall_texture_path,
                     half_extent=[1.0, 1.0, wall_distance],
                     pos=[-wall_distance, 1.0, 0.0],
                     rot=0.0,
                     detect_collision=False)
    # +X
    self.env.add_box(texture_path=wall_texture_path,
                     half_extent=[1.0, 1.0, wall_distance],
                     pos=[wall_distance, 1.0, 0.0],
                     rot=0.0,
                     detect_collision=False)

  def _locate_plus_reward_obj(self, x, z, rot):
    model_path = os.path.dirname(os.path.abspath(__file__)) + "/data/apple0.obj"
    pos_scale = 0.075
    pos = [x * pos_scale, 0.0, z * pos_scale]
    obj_id = self.env.add_model(path=model_path,
                                scale=[1.0, 1.0, 1.0],
                                pos=pos,
                                rot=rot,
                                detect_collision=True)
    self.plus_obj_ids_set.add(obj_id)

  def _locate_minus_reward_obj(self, x, z, rot):
    model_path = os.path.dirname(os.path.abspath(__file__)) + "/data/lemon0.obj"
    pos_scale = 0.075
    pos = [x * pos_scale, 0.0, z * pos_scale]    
    obj_id = self.env.add_model(path=model_path,
                                scale=[1.0, 1.0, 1.0],
                                pos=pos,
                                rot=rot,
                                detect_collision=True)
    self.minus_obj_ids_set.add(obj_id)
  
  def reset(self):
    # Clear remaining reward objects
    self._clear_objects()

    # Add rewards
    self._locate_plus_reward_obj(x=96, z=0, rot=0.625)
    self._locate_plus_reward_obj(x=192, z=-112, rot=0.375)
    self._locate_plus_reward_obj(x=-128, z=-32, rot=0.0)
    self._locate_plus_reward_obj(x=-144, z=184, rot=0.0)
    self._locate_plus_reward_obj(x=176, z=208, rot=0.75)
    self._locate_plus_reward_obj(x=160, z=104, rot=0.0)
    self._locate_plus_reward_obj(x=80, z=192, rot=0.5)
    self._locate_plus_reward_obj(x=-120, z=-160, rot=0.375)
    self._locate_plus_reward_obj(x=-248, z=80, rot=0.5)
    self._locate_plus_reward_obj(x=96, z=-184, rot=0.0)
    self._locate_plus_reward_obj(x=-64, z=272, rot=0.125)
    self._locate_plus_reward_obj(x=288, z=-88, rot=0.875)
    self._locate_plus_reward_obj(x=-312, z=-96, rot=0.125)
    self._locate_plus_reward_obj(x=-256, z=-312, rot=0.875)
    self._locate_plus_reward_obj(x=240, z=232, rot=0.0)

    self._locate_minus_reward_obj(x=-184, z=-232, rot=0.0)
    self._locate_minus_reward_obj(x=104, z=-80, rot=0.0)
    self._locate_minus_reward_obj(x=200, z=40, rot=0.25)
    self._locate_minus_reward_obj(x=-240, z=-8, rot=0.625)
    self._locate_minus_reward_obj(x=-48, z=152, rot=0.0)
    self._locate_minus_reward_obj(x=48, z=-296, rot=0.875)
    self._locate_minus_reward_obj(x=-248, z=216, rot=0.375)
    
    # Locate agent to default position
    self.env.locate_agent(pos=[0,0,0],
                          rot=0.0)

    self.step_num = 0
    obs = self.env.step(action=[0,0,0], num_steps=1)
    screen = obs["screen"]
    return screen

  def _clear_objects(self):
    for id in self.plus_obj_ids_set:
      self.env.remove_obj(id)
    for id in self.minus_obj_ids_set:
      self.env.remove_obj(id)
      
    self.plus_obj_ids_set = set()
    self.minus_obj_ids_set = set()

  def step(self, action, num_steps):
    real_action = RodentSeekAvoidEnvironment.ACTION_LIST[action]

    obs = self.env.step(action=real_action, num_steps=num_steps)
    self.step_num += num_steps
    
    screen = obs["screen"]
    collided = obs["collided"]

    reward = 0
    if len(collided) != 0:
      for id in collided:
        if id in self.plus_obj_ids_set:
          reward += 1
          self.plus_obj_ids_set.remove(id)          
        elif id in self.minus_obj_ids_set:
          reward -= 1
          self.minus_obj_ids_set.remove(id)
        self.env.remove_obj(id)

    
    is_empty = len(self.plus_obj_ids_set) == 0
    time_over = self.step_num >= MAX_STEP_NUM
    terminal = is_empty or time_over
    
    return screen, reward, terminal

  def close(self):
    del self.env



COMMAND_RESET     = 0
COMMAND_ACTION    = 1
COMMAND_TERMINATE = 2

def worker(conn):
  env = RodentSeekAvoidEnvironment()
  conn.send(0)
  
  while True:
    command, arg = conn.recv()

    if command == COMMAND_RESET:
      obs = env.reset()
      conn.send(obs)
    elif command == COMMAND_ACTION:
      obs, reward, terminal = env.step(arg, num_steps=1)
      conn.send([obs, reward, terminal])
    elif command == COMMAND_TERMINATE:
      break
    else:
      print("bad command: {}".format(command))
  env.close()
  conn.send(0)
  conn.close()

class RodentEnvironment(environment.Environment):
  @staticmethod
  def get_action_size():
    return len(RodentSeekAvoidEnvironment.ACTION_LIST)
  
  def __init__(self):
    environment.Environment.__init__(self)
    
    self.conn, child_conn = Pipe()
    self.proc = Process(target=worker, args=(child_conn,))
    self.proc.start()
    self.conn.recv()
    self.reset()

  def reset(self):
    self.conn.send([COMMAND_RESET, 0])
    obs = self.conn.recv()
    
    self.last_state = self._preprocess_frame(obs)
    self.last_action = 0
    self.last_reward = 0

  def stop(self):
    self.conn.send([COMMAND_TERMINATE, 0])
    ret = self.conn.recv()
    self.conn.close()
    self.proc.join()
    print("rodent environment stopped")
    
  def _preprocess_frame(self, image):
    image = image.astype(np.float32)
    image = image / 255.0
    return image

  def process(self, action):
    self.conn.send([COMMAND_ACTION, action])
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
