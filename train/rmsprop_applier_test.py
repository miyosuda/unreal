# -*- coding: utf-8 -*-

import numpy as np
import math
import tensorflow as tf
import rmsprop_applier

class TestRMSPropApplier(tf.test.TestCase):
  def test_apply(self):
    with self.test_session():
      var = tf.Variable([1.0, 2.0])
      
      grad0 = tf.Variable([2.0, 4.0])
      grad1 = tf.Variable([3.0, 6.0])
      
      opt = rmsprop_applier.RMSPropApplier(learning_rate=2.0,
                                           decay=0.9,
                                           momentum=0.0,
                                           epsilon=1.0)
      
      apply_gradient0 = opt._apply_gradients([var], [grad0])
      apply_gradient1 = opt._apply_gradients([var], [grad1])

      tf.global_variables_initializer().run()

      # apply grad0
      apply_gradient0.run()

      ms_x = 1.0
      ms_y = 1.0

      x = 1.0
      y = 2.0
      dx = 2.0
      dy = 4.0
      ms_x = ms_x + (dx * dx - ms_x) * (1.0 - 0.9)
      ms_y = ms_y + (dy * dy - ms_y) * (1.0 - 0.9)
      x = x - (2.0 * dx / math.sqrt(ms_x+1.0))
      y = y - (2.0 * dy / math.sqrt(ms_y+1.0))

      self.assertAllClose(np.array([x, y]), var.eval())

      # apply grad1
      apply_gradient1.run()

      dx = 3.0
      dy = 6.0
      ms_x = ms_x + (dx * dx - ms_x) * (1.0 - 0.9)
      ms_y = ms_y + (dy * dy - ms_y) * (1.0 - 0.9)
      x = x - (2.0 * dx / math.sqrt(ms_x+1.0))
      y = y - (2.0 * dy / math.sqrt(ms_y+1.0))
      
      self.assertAllClose(np.array([x, y]), var.eval())
      
if __name__ == "__main__":
  tf.test.main()
