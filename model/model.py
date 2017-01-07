# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from constants import *


# weight initialization based on muupan's code
# https://github.com/muupan/async-rl/blob/master/a3c_ale.py
def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


class UnrealModel(object):
  """
  UNREAL algorithm network model.
  """
  def __init__(self,
               action_size,
               thread_index, # -1 for global
               device,
               for_display=False):
    self._device = device
    self._action_size = action_size
    self._thread_index = thread_index
    self._create_network(for_display)
    
    
  def _create_network(self, for_display):
    scope_name = "net_{0}".format(self._thread_index)
    with tf.device(self._device), tf.variable_scope(scope_name) as scope:
      # lstm
      self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(256, state_is_tuple=True)
      
      # [base A3C network]
      self._create_base_network()

      # [Pixel change network]
      if USE_PIXEL_CHANGE:
        self._create_pc_network()
        if for_display:
          self._create_pc_network_for_display()

      # [Value replay network]
      if USE_VALUE_REPLAY:
        self._create_vr_network()

      # [Reawrd prediction network]
      if USE_REWARD_PREDICTION:
        self._create_rp_network()
      
      self.reset_state()

      self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)


  def _create_base_network(self):
    # State (Base image input)
    self.base_input = tf.placeholder("float", [None, 84, 84, 3])

    # Last action and reward
    self.base_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1])
    
    # Conv layers
    base_conv_output = self._base_conv_layers(self.base_input)
    
    # LSTM layer
    self.base_initial_lstm_state0 = tf.placeholder(tf.float32, [1, 256])
    self.base_initial_lstm_state1 = tf.placeholder(tf.float32, [1, 256])
    
    self.base_initial_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(self.base_initial_lstm_state0,
                                                                 self.base_initial_lstm_state1)

    self.base_lstm_outputs, self.base_lstm_state = \
        self._base_lstm_layer(base_conv_output,
                              self.base_last_action_reward_input,
                              self.base_initial_lstm_state)

    self.base_pi = self._base_policy_layer(self.base_lstm_outputs) # policy output
    self.base_v  = self._base_value_layer(self.base_lstm_outputs)  # value output

    
  def _base_conv_layers(self, state_input, reuse=False):
    with tf.variable_scope("base_conv", reuse=reuse) as scope:
      # Weights
      W_conv1, b_conv1 = self._conv_variable([8, 8, 3, 16],  "base_conv1")
      W_conv2, b_conv2 = self._conv_variable([4, 4, 16, 32], "base_conv2")

      # Nodes
      h_conv1 = tf.nn.relu(self._conv2d(state_input, W_conv1, 4) + b_conv1) # stride=4
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1,     W_conv2, 2) + b_conv2) # stride=2
      return h_conv2
  
  
  def _base_lstm_layer(self, conv_output, last_action_reward_input, initial_state_input,
                       reuse=False):
    with tf.variable_scope("base_lstm", reuse=reuse) as scope:
      # Weights
      W_fc1, b_fc1 = self._fc_variable([2592, 256], "base_fc1")

      # Nodes
      conv_output_flat = tf.reshape(conv_output, [-1, 2592])
      # (-1,9,9,32) -> (-1,2592)
      conv_output_fc = tf.nn.relu(tf.matmul(conv_output_flat, W_fc1) + b_fc1)
      # (unroll_step, 256)

      step_size = tf.shape(conv_output_fc)[:1]

      lstm_input = tf.concat(1, [conv_output_fc, last_action_reward_input])
      # (unroll_step, 256+action_size+1)

      lstm_input_reshaped = tf.reshape(lstm_input, [1, -1, 256+self._action_size+1])
      # (1, unroll_step, 256+action_size+1)

      lstm_outputs, lstm_state = tf.nn.dynamic_rnn(self.lstm_cell,
                                                   lstm_input_reshaped,
                                                   initial_state = initial_state_input,
                                                   sequence_length = step_size,
                                                   time_major = False,
                                                   scope = scope)
      
      lstm_outputs = tf.reshape(lstm_outputs, [-1,256])
      #(1,unroll_step,256) for back prop, (1,1,256) for forward prop.
      return lstm_outputs, lstm_state


  def _base_policy_layer(self, lstm_outputs, reuse=False):
    with tf.variable_scope("base_policy", reuse=reuse) as scope:
      # Weight for policy output layer
      W_fc_p, b_fc_p = self._fc_variable([256, self._action_size], "base_fc_p")
      # Policy (output)
      base_pi = tf.nn.softmax(tf.matmul(lstm_outputs, W_fc_p) + b_fc_p)
      return base_pi


  def _base_value_layer(self, lstm_outputs, reuse=False):
    with tf.variable_scope("base_value", reuse=reuse) as scope:
      # Weight for value output layer
      W_fc_v, b_fc_v = self._fc_variable([256, 1], "base_fc_v")
      
      # Value (output)
      v_ = tf.matmul(lstm_outputs, W_fc_v) + b_fc_v
      base_v = tf.reshape( v_, [-1] )
      return base_v


  def _create_pc_network(self):
    # State (Image input) 
    self.pc_input = tf.placeholder("float", [None, 84, 84, 3])

    # Last action and reward
    self.pc_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1])

    # pc conv layers
    pc_conv_output = self._base_conv_layers(self.pc_input, reuse=True)

    # pc lastm layers
    pc_initial_lstm_state = self.lstm_cell.zero_state(1, tf.float32)
    # (Initial state is always resetted.)
    
    pc_lstm_outputs, _ = self._base_lstm_layer(pc_conv_output,
                                               self.pc_last_action_reward_input,
                                               pc_initial_lstm_state,
                                               reuse=True)
    
    self.pc_q, self.pc_q_max = self._pc_deconv_layers(pc_lstm_outputs)

    
  def _create_pc_network_for_display(self):
    self.pc_q_disp, self.pc_q_max_disp = self._pc_deconv_layers(self.base_lstm_outputs, reuse=True)
    
  
  def _pc_deconv_layers(self, lstm_outputs, reuse=False):
    with tf.variable_scope("pc_deconv", reuse=reuse) as scope:    
      # (Spatial map was written as 7x7x32, but here 9x9x32 is used to get 20x20 deconv result?)
      # State (image input for pixel change)
      W_pc_fc1, b_pc_fc1 = self._fc_variable([256, 9*9*32], "pc_fc1")
        
      W_pc_deconv_v, b_pc_deconv_v = self._conv_variable([4, 4, 1, 32],
                                                         "pc_deconv_v", deconv=True)
      W_pc_deconv_a, b_pc_deconv_a = self._conv_variable([4, 4, self._action_size, 32],
                                                         "pc_deconv_a", deconv=True)
      
      h_pc_fc1 = tf.nn.relu(tf.matmul(lstm_outputs, W_pc_fc1) + b_pc_fc1)
      h_pc_fc1_reshaped = tf.reshape(h_pc_fc1, [-1,9,9,32])
      # Dueling network for V and Advantage
      h_pc_deconv_v = tf.nn.relu(self._deconv2d(h_pc_fc1_reshaped,
                                                W_pc_deconv_v, 9, 9, 2) +
                                 b_pc_deconv_v)
      h_pc_deconv_a = tf.nn.relu(self._deconv2d(h_pc_fc1_reshaped,
                                                W_pc_deconv_a, 9, 9, 2) +
                                 b_pc_deconv_a)
      # Advantage mean
      h_pc_deconv_a_mean = tf.reduce_mean(h_pc_deconv_a, reduction_indices=3, keep_dims=True)

      # {Pixel change Q (output)
      pc_q = h_pc_deconv_v + h_pc_deconv_a - h_pc_deconv_a_mean
      #(-1, 20, 20, action_size)

      # Max Q
      pc_q_max = tf.reduce_max(pc_q, reduction_indices=3, keep_dims=False)
      #(-1, 20, 20)

      return pc_q, pc_q_max
    

  def _create_vr_network(self):
    # State (Image input)
    self.vr_input = tf.placeholder("float", [None, 84, 84, 3])

    # Last action and reward
    self.vr_last_action_reward_input = tf.placeholder("float", [None, self._action_size+1])

    # VR conv layers
    vr_conv_output = self._base_conv_layers(self.vr_input, reuse=True)

    # pc lastm layers
    vr_initial_lstm_state = self.lstm_cell.zero_state(1, tf.float32)
    # (Initial state is always resetted.)
    
    vr_lstm_outputs, _ = self._base_lstm_layer(vr_conv_output,
                                               self.vr_last_action_reward_input,
                                               vr_initial_lstm_state,
                                               reuse=True)
    # value output
    self.vr_v  = self._base_value_layer(vr_lstm_outputs, reuse=True)

    
  def _create_rp_network(self):
    self.rp_input = tf.placeholder("float", [3, 84, 84, 3])

    # RP conv layers
    rp_conv_output = self._base_conv_layers(self.rp_input, reuse=True)
    rp_conv_output_rehaped = tf.reshape(rp_conv_output, [1,9*9*32*3])
    
    with tf.variable_scope("rp_fc") as scope:
      # Weights
      W_fc1, b_fc1 = self._fc_variable([9*9*32*3, 3], "rp_fc1")

    # Reawrd prediction class output. (zero, positive, negative)
    self.rp_c = tf.nn.softmax(tf.matmul(rp_conv_output_rehaped, W_fc1) + b_fc1)
    # (1,3)

  def _base_loss(self):
    # [base A3C]
    # Taken action (input for policy)
    self.base_a = tf.placeholder("float", [None, self._action_size])
    
    # Advantage (R-V) (input for policy)
    self.base_adv = tf.placeholder("float", [None])
    
    # Avoid NaN with clipping when value in pi becomes zero
    log_pi = tf.log(tf.clip_by_value(self.base_pi, 1e-20, 1.0))
    
    # Policy entropy
    entropy = -tf.reduce_sum(self.base_pi * log_pi, reduction_indices=1)
    
    # Policy loss (output)
    policy_loss = -tf.reduce_sum( tf.reduce_sum( tf.mul( log_pi, self.base_a ),
                                                 reduction_indices=1 ) *
                                  self.base_adv + entropy * ENTROPY_BETA)
    
    # R (input for value target)
    self.base_r = tf.placeholder("float", [None])
    
    # Value loss (output)
    # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
    value_loss = 0.5 * tf.nn.l2_loss(self.base_r - self.base_v)
    
    base_loss = policy_loss + value_loss
    return base_loss

  
  def _pc_loss(self):
    # [pixel change]
    self.pc_a = tf.placeholder("float", [None, self._action_size])
    pc_a_reshaped = tf.reshape(self.pc_a, [-1, 1, 1, self._action_size])

    # Extract Q for taken action
    pc_qa_ = tf.mul(self.pc_q, pc_a_reshaped)
    pc_qa = tf.reduce_sum(pc_qa_, reduction_indices=3, keep_dims=False)
    # (-1, 20, 20)
      
    # TD target for Q
    self.pc_r = tf.placeholder("float", [None, 20, 20])

    pc_loss = PIXEL_CHANGE_LAMBDA * tf.nn.l2_loss(self.pc_r - pc_qa)
    return pc_loss

  
  def _vr_loss(self):
    # R (input for value)
    self.vr_r = tf.placeholder("float", [None])
    
    # Value loss (output)
    vr_loss = tf.nn.l2_loss(self.vr_r - self.vr_v)
    return vr_loss


  def _rp_loss(self):
    # reward prediction target. one hot vector
    self.rp_c_target = tf.placeholder("float", [1,3])
    
    # Reward prediction loss (output)
    rp_c = tf.clip_by_value(self.rp_c, 1e-20, 1.0)
    rp_loss = -tf.reduce_sum(self.rp_c_target * tf.log(rp_c))
    return rp_loss
    
    
  def prepare_loss(self):
    with tf.device(self._device):
      loss = self._base_loss()
      
      if USE_PIXEL_CHANGE:
        pc_loss = self._pc_loss()
        loss = loss + pc_loss

      if USE_VALUE_REPLAY:
        vr_loss = self._vr_loss()
        loss = loss + vr_loss

      if USE_REWARD_PREDICTION:
        rp_loss = self._rp_loss()
        loss = loss + rp_loss
      
      self.total_loss = loss


  def reset_state(self):
    self.base_lstm_state_out = tf.nn.rnn_cell.LSTMStateTuple(np.zeros([1, 256]),
                                                             np.zeros([1, 256]))

  def run_base_policy_and_value(self, sess, s_t, last_action_reward):
    # This run_base_policy_and_value() is used when forward propagating.
    # so the step size is 1.
    pi_out, v_out, self.base_lstm_state_out = sess.run( [self.base_pi, self.base_v, self.base_lstm_state],
                                                        feed_dict = {self.base_input : [s_t],
                                                                     self.base_last_action_reward_input : [last_action_reward],
                                                                     self.base_initial_lstm_state0 : self.base_lstm_state_out[0],
                                                                     self.base_initial_lstm_state1 : self.base_lstm_state_out[1]} )
    # pi_out: (1,3), v_out: (1)
    return (pi_out[0], v_out[0])

  
  def run_base_policy_value_pc_q(self, sess, s_t, last_action_reward):
    # For display tool.
    pi_out, v_out, self.base_lstm_state_out, q_disp_out, q_max_disp_out = \
        sess.run( [self.base_pi, self.base_v, self.base_lstm_state, self.pc_q_disp, self.pc_q_max_disp],
                  feed_dict = {self.base_input : [s_t],
                               self.base_last_action_reward_input : [last_action_reward],
                               self.base_initial_lstm_state0 : self.base_lstm_state_out[0],
                               self.base_initial_lstm_state1 : self.base_lstm_state_out[1]} )
    
    # pi_out: (1,3), v_out: (1), q_disp_out(1,20,20, action_size)
    return (pi_out[0], v_out[0], q_disp_out[0])

  
  def run_base_value(self, sess, s_t, last_action_reward):
    # This run_bae_value() is used for calculating V for bootstrapping at the 
    # end of LOCAL_T_MAX time step sequence.
    # When next sequcen starts, V will be calculated again with the same state using updated network weights,
    # so we don't update LSTM state here.
    v_out, _ = sess.run( [self.base_v, self.base_lstm_state],
                         feed_dict = {self.base_input : [s_t],
                                      self.base_last_action_reward_input : [last_action_reward],
                                      self.base_initial_lstm_state0 : self.base_lstm_state_out[0],
                                      self.base_initial_lstm_state1 : self.base_lstm_state_out[1]} )
    return v_out[0]

  
  def run_pc_q_max(self, sess, s_t, last_action_reward):
    q_max_out = sess.run( self.pc_q_max,
                          feed_dict = {self.pc_input : [s_t],
                                       self.pc_last_action_reward_input : [last_action_reward]} )
    return q_max_out[0]

  
  def run_vr_value(self, sess, s_t, last_action_reward):
    vr_v_out = sess.run( self.vr_v,
                         feed_dict = {self.vr_input : [s_t],
                                      self.vr_last_action_reward_input : [last_action_reward]} )
    return vr_v_out[0]

  
  def run_rp_c(self, sess, s_t):
    # For display tool
    rp_c_out = sess.run( self.rp_c,
                         feed_dict = {self.rp_input : s_t} )
    return rp_c_out[0]

  
  def get_vars(self):
    return self.variables
  

  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "UnrealModel",[]) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)
      

  def _fc_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
    return weight, bias

  
  def _conv_variable(self, weight_shape, name, deconv=False):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
      input_channels  = weight_shape[3]
      output_channels = weight_shape[2]
    else:
      input_channels  = weight_shape[2]
      output_channels = weight_shape[3]
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
    bias   = tf.get_variable(name_b, bias_shape,
                             initializer=conv_initializer(w, h, input_channels))
    return weight, bias

  
  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")


  def _get2d_deconv_output_size(self,
                                input_height, input_width,
                                filter_height, filter_width,
                                stride, padding_type):
    if padding_type == 'VALID':
      out_height = (input_height - 1) * stride + filter_height
      out_width  = (input_width  - 1) * stride + filter_width
      
    elif padding_type == 'SAME':
      out_height = input_height * row_stride
      out_width  = input_width  * col_stride
    
    return out_height, out_width


  def _deconv2d(self, x, W, input_width, input_height, stride):
    filter_height = W.get_shape()[0].value
    filter_width  = W.get_shape()[1].value
    out_channel   = W.get_shape()[2].value
    
    out_height, out_width = self._get2d_deconv_output_size(input_height,
                                                           input_width,
                                                           filter_height,
                                                           filter_width,
                                                           stride,
                                                           'VALID')
    batch_size = tf.shape(x)[0]
    output_shape = tf.pack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='VALID')
