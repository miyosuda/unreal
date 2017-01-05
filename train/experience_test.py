# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from train.experience import Experience, ExperienceFrame


class TestExperience(unittest.TestCase):
  def _add_frame(self, experice, reward):
    frame = ExperienceFrame(0, reward, 0, False, 0, 0, 0)
    experice.add_frame(frame)
    
  def test_process(self):
    experience = Experience(10)

    for i in range(10):
      if i == 5:
        self._add_frame(experience, 1)
      else:
        self._add_frame(experience, 0)

    self.assertTrue( experience.is_full() )
    self.assertTrue( experience._top_frame_index == 0 )
    
    self._add_frame(experience, 0)

    self.assertTrue( experience._top_frame_index == 1 )

    for i in range(100):
      frames = experience.sample_rp_sequence()
      self.assertTrue( len(frames) == 4 )
      # Reward shold be shewed here.
      #print(frames[3].reward)

if __name__ == '__main__':
  unittest.main()
