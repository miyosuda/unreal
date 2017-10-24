# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import train.experience_test
import train.rmsprop_applier_test
import environment.environment_test
import model.model_test

def get_suite():
  suite = unittest.TestSuite()

  suite.addTest(unittest.makeSuite(train.experience_test.TestExperience))
  suite.addTest(unittest.makeSuite(train.rmsprop_applier_test.TestRMSPropApplier))    
  suite.addTest(unittest.makeSuite(environment.environment_test.TestEnvironment))
  suite.addTest(unittest.makeSuite(model.model_test.TestUnrealModel))
  
  return suite

def main():
  suite = get_suite()
  unittest.TextTestRunner().run(suite)

if __name__ == '__main__':
  main()
