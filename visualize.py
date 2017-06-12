# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt

from environment.environment import Environment
from model.model import UnrealModel
from options import get_options


# get command line args
flags = get_options("visualize")


def main(args):
  action_size = Environment.get_action_size(flags.env_type, flags.env_name)
  global_network = UnrealModel(action_size, -1,
                               flags.use_pixel_change,
                               flags.use_value_replay,
                               flags.use_reward_prediction,
                               0.0,
                               0.0,
                               "/cpu:0") # use CPU for weight visualize tool
  
  sess = tf.Session()
  
  init = tf.global_variables_initializer()
  sess.run(init)
  
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
  else:
    print("Could not find old checkpoint")
  
  vars = {}
  var_list = global_network.get_vars()
  for v in var_list:
    vars[v.name] = v
  
  W_conv1 = sess.run(vars['net_-1/base_conv/W_base_conv1:0'])
  
  # show graph of W_conv1
  fig, axes = plt.subplots(3, 16, figsize=(12, 6),
               subplot_kw={'xticks': [], 'yticks': []})
  fig.subplots_adjust(hspace=0.1, wspace=0.1)
  
  for ax,i in zip(axes.flat, range(3*16)):
    inch = i//16
    outch = i%16
    img = W_conv1[:,:,inch,outch]
    ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title(str(inch) + "," + str(outch))
  
  plt.show()

if __name__ == '__main__':
  tf.app.run()
