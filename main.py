# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import threading
import numpy as np

import signal
import random
import math
import os
import time

from environment.environment import Environment
from model.model import UnrealModel
from train.trainer import Trainer
from train.rmsprop_applier import RMSPropApplier
from constants import *

def log_uniform(lo, hi, rate):
  log_lo = math.log(lo)
  log_hi = math.log(hi)
  v = log_lo * (1-rate) + log_hi * rate
  return math.exp(v)

device = "/cpu:0"
if USE_GPU:
  device = "/gpu:0"

initial_learning_rate = log_uniform(INITIAL_ALPHA_LOW,
                                    INITIAL_ALPHA_HIGH,
                                    INITIAL_ALPHA_LOG_RATE)

global_t = 0

stop_requested = False
terminate_reqested = False

action_size = Environment.get_action_size()
global_network = UnrealModel(action_size, -1, device)

trainers = []

learning_rate_input = tf.placeholder("float")

grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                              decay = RMSP_ALPHA,
                              momentum = 0.0,
                              epsilon = RMSP_EPSILON,
                              clip_norm = GRAD_NORM_CLIP,
                              device = device)

for i in range(PARALLEL_SIZE):
  trainer = Trainer(i,
                    global_network,
                    initial_learning_rate,
                    learning_rate_input,
                    grad_applier,
                    MAX_TIME_STEP,
                    device = device)
  trainers.append(trainer)

# prepare session
sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                        allow_soft_placement=True))

init = tf.global_variables_initializer()
sess.run(init)

# summary for tensorboard
score_input = tf.placeholder(tf.int32)
tf.summary.scalar("score", score_input)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

# init or load checkpoint with saver
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(sess, checkpoint.model_checkpoint_path)
  print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  tokens = checkpoint.model_checkpoint_path.split("-")
  # set global step
  global_t = int(tokens[1])
  print(">>> global step set: ", global_t)
  # set wall time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'r') as f:
    wall_t = float(f.read())

  next_save_steps = (global_t + SAVE_INTERVAL_STEP) // SAVE_INTERVAL_STEP * SAVE_INTERVAL_STEP
    
else:
  print("Could not find old checkpoint")
  # set wall time
  wall_t = 0.0

  next_save_steps = SAVE_INTERVAL_STEP

  
def save(current_global_step):
  """ Save checkpoint. 
  
  Called from therad-0.
  """
  global next_save_steps
  global train_threads
  global trainers
  global saver
  global stop_requested

  stop_requested = True

  # Wait for all other threads to stop
  for (i, t) in enumerate(train_threads):
    if i != 0:
      t.join()

  # Save
  if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)

  # Write wall time
  wall_t = time.time() - start_time
  wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t)
  with open(wall_t_fname, 'w') as f:
    f.write(str(wall_t))

  print('Start saving.')
  saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)
  print('End saving.')  
  
  stop_requested = False
  next_save_steps += SAVE_INTERVAL_STEP

  # Restart other threads
  for i in range(PARALLEL_SIZE):
    if i != 0:
      thread = threading.Thread(target=train_function, args=(i,))
      train_threads[i] = thread
      thread.start()


def train_function(parallel_index):
  """ Train each environment. """
  
  global global_t
  
  trainer = trainers[parallel_index]
  # set start_time
  start_time = time.time() - wall_t
  trainer.set_start_time(start_time)

  while True:
    if stop_requested:
      break
    if terminate_reqested:
      break
    if global_t > MAX_TIME_STEP:
      break
    if parallel_index == 0 and global_t > next_save_steps:
      # Save checkpoint
      save(global_t)

    diff_global_t = trainer.process(sess, global_t, summary_writer,
                                    summary_op, score_input)
    global_t += diff_global_t
    
    
def signal_handler(signal, frame):
  global terminate_reqested
  print('You pressed Ctrl+C!')
  terminate_reqested = True

train_threads = []
for i in range(PARALLEL_SIZE):
  train_threads.append(threading.Thread(target=train_function, args=(i,)))
  
signal.signal(signal.SIGINT, signal_handler)

# set start time
start_time = time.time() - wall_t

for t in train_threads:
  t.start()

print('Press Ctrl+C to stop')
signal.pause()
