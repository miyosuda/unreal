# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
#import scipy.misc

from environment.environment import Environment


class TestEnvironment(unittest.TestCase):
  def test_process(self):
    environment = Environment.create_environment()
    action_size = Environment.get_action_size()

    for i in range(3):
      state, reward, terminal, pixel_change = environment.process(0)

      # Check shape
      self.assertTrue( state.shape == (84,84,3) )
      self.assertTrue( environment.last_state.shape == (84,84,3) )
      self.assertTrue( pixel_change.shape == (20,20) )

      # state and pixel_change value range should be [0,1]
      self.assertTrue( np.amax(state) <= 1.0 )
      self.assertTrue( np.amin(state) >= 0.0 )
      self.assertTrue( np.amax(pixel_change) <= 1.0 )
      self.assertTrue( np.amin(pixel_change) >= 0.0 )

      #scipy.misc.imsave("debug_state{0}.png".format(i), state)
      #scipy.misc.imsave("debug_pc{0}.png".format(i), pixel_change)
      

if __name__ == '__main__':
  unittest.main()
