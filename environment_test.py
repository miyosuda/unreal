# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import scipy.misc

from environment import Environment

class TestGameEnvironment(unittest.TestCase):
  def test_process(self):
    environment = Environment.create_environment()
    action_size = Environment.get_action_size()

    print("action_size={}".format(action_size))

    for i in range(3):
      state, reward, terminal, pixel_change = environment.process(0)

      self.assertTrue( state.shape == (84,84,3) )
      self.assertTrue( environment.last_state.shape == (84,84,3) )
      self.assertTrue( pixel_change.shape == (20,20) )

      scipy.misc.imsave("debug_state{0}.png".format(i), state)
      scipy.misc.imsave("debug_pc{0}.png".format(i), pixel_change)
      

if __name__ == '__main__':
  unittest.main()
