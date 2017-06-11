# -*- coding: utf-8 -*-
import numpy as np
import math
import tensorflow as tf
from model import UnrealModel

class TestUnrealModel(tf.test.TestCase):
  def test_unreal_variable_size(self):
    """ Check total variable size with all options ON """
    use_pixel_change = True
    use_value_replay = True
    use_reward_prediction = True

    # base: conv=4, fc=2, lstm=2, policy_fc=2, value_fc=2
    # pc:   fc=2, deconv_v=2, deconv_a=2
    # rp:   fc=2
    self.check_model_var_size( use_pixel_change,
                               use_value_replay,
                               use_reward_prediction,
                               20 )

  def test_pc_variable_size(self):
    """ Check total variable size with only pixel change ON """
    use_pixel_change = True
    use_value_replay = False
    use_reward_prediction = False

    # base: conv=4, fc=2, lstm=2, policy_fc=2, value_fc=2
    # pc:   fc=2, deconv_v=2, deconv_a=2
    self.check_model_var_size( use_pixel_change,
                               use_value_replay,
                               use_reward_prediction,
                               18 )

  def test_vr_variable_size(self):
    """ Check total variable size with only value funciton replay ON """
    use_pixel_change = False
    use_value_replay = True
    use_reward_prediction = False

    # base: conv=4, fc=2, lstm=2, policy_fc=2, value_fc=2
    self.check_model_var_size( use_pixel_change,
                               use_value_replay,
                               use_reward_prediction,
                               12 )

  def test_rp_variable_size(self):
    """ Check total variable size with only reward prediction ON """
    use_pixel_change = False
    use_value_replay = False
    use_reward_prediction = True

    # base: conv=4, fc=2, lstm=2, policy_fc=2, value_fc=2
    # rp:   fc=2
    self.check_model_var_size( use_pixel_change,
                               use_value_replay,
                               use_reward_prediction,
                               14 )
    
  def check_model_var_size(self,
                           use_pixel_change,
                           use_value_replay,
                           use_reward_prediction,
                           var_size):
    """ Check variable size of the model """
    
    model = UnrealModel(1,
                        -1,
                        use_pixel_change,
                        use_value_replay,
                        use_reward_prediction,
                        1.0,
                        1.0,
                        "/cpu:0");
    variables = model.get_vars()
    self.assertEqual( len(variables), var_size )


if __name__ == "__main__":
  tf.test.main()
  
