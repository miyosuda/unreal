# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt

from environment.environment import Environment
from model.model import UnrealModel
from constants import *


# use CPU for weight visualize tool
device = "/cpu:0"

action_size = Environment.get_action_size()
global_network = UnrealModel(action_size, -1, device)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
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


"""
net_-1/base_conv/W_base_conv1:0
net_-1/base_conv/b_base_conv1:0
net_-1/base_conv/W_base_conv2:0
net_-1/base_conv/b_base_conv2:0
net_-1/base_lstm/W_base_fc1:0
net_-1/base_lstm/b_base_fc1:0
net_-1/base_lstm/BasicLSTMCell/Linear/Matrix:0
net_-1/base_lstm/BasicLSTMCell/Linear/Bias:0
net_-1/base_policy/W_base_fc_p:0
net_-1/base_policy/b_base_fc_p:0
net_-1/base_value/W_base_fc_v:0
net_-1/base_value/b_base_fc_v:0
net_-1/W_pc_fc1:0
net_-1/b_pc_fc1:0
net_-1/W_pc_deconv_v:0
net_-1/b_pc_deconv_v:0
net_-1/W_pc_deconv_a:0
net_-1/b_pc_deconv_a:0
"""
