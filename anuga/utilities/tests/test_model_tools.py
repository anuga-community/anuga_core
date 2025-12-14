#!/usr/bin/env python


import unittest
import numpy as num
from numpy.random import uniform, seed

from math import sqrt, pi
from anuga.config import epsilon
from anuga.utilities.model_tools import *
verbose = False


class Test_Model_Tools(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_WCC_2016_blockage_factor(self):

        Wm = [0.3, 0.45, 0.6, 0.9, 1.025, 1.2, 1.5, 1.8, 2.1, 2.4,
              2.7, 3.0, 3.3, 3.6, 4.2, 4.8, 5.4, 6.0, 7.2, 8.0, 10.0]
        Hm = [0.3, 0.45, 0.6, 0.9, 1.025, 1.2,
              1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6]
        Dm = [0.3, 0.45, 0.6, 0.9, 1.025, 1.2,
              1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6]

        Ev = [2, 10, 50]  # Severity of Event
        Sc = ['D', 'R']  # Type of Scenario DESIGN, RISK MAN
        Cul = ['B', 'P']  # Type of Culvert Pipe or Box

        i = 0

        expected = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.35, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.0, 0.35, 0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.35, 0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.15, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.05, 0.05, 0.05, 0.05, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, 0.7, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.7, 0.7, 0.7, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5,
                    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.35, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.6, 0.6, 0.6, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.6, 0.6, 0.6, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.6, 0.6, 0.6, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.6, 0.6, 0.6, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.6, 0.6, 0.6, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.05, 0.6, 0.6, 0.6, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.05, 0.05, 0.05, 0.05, 0.6, 0.6, 0.6, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.05, 0.05, 0.05, 0.05, 0.6, 0.6, 0.6, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.05, 0.05, 0.05, 0.05, 0.6, 0.6, 0.6, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.05, 0.05, 0.05, 0.05, 0.6, 0.6, 0.6, 0.5, 0.5, 0.35, 0.35, 0.35, 0.35, 0.35, 0.05, 0.05, 0.05, 0.05, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.75, 0.75, 0.75, 0.75, 0.75, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.75, 0.75, 0.75, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.65, 0.65, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.65, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.75, 0.75, 0.75, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.75, 0.75, 0.75, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.75, 0.75, 0.75, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.75, 0.75, 0.75, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.75, 0.75, 0.75, 0.65, 0.65, 0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.95, 0.95, 0.95, 0.95, 0.95, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.95, 0.95, 0.95, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.6, 0.6, 0.6, 0.95, 0.95, 0.95, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.95, 0.95, 0.95, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.95, 0.95, 0.95, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.95, 0.95, 0.95, 0.75, 0.75, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.95, 0.95, 0.95, 0.75, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.95, 0.95, 0.95, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.95, 0.95, 0.95, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.95, 0.95, 0.95, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.95, 0.95, 0.95, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.95, 0.95, 0.95, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.15, 0.95, 0.95, 0.95, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.15, 0.15, 0.15, 0.15, 0.95, 0.95, 0.95, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.15, 0.15, 0.15, 0.15, 0.95, 0.95, 0.95, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.15, 0.15, 0.15, 0.15, 0.95, 0.95, 0.95, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.15, 0.15, 0.15, 0.15, 0.95, 0.95, 0.95, 0.75, 0.75, 0.6, 0.6, 0.6, 0.6, 0.6, 0.15, 0.15, 0.15, 0.15, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]

        # Fix up a few individual entries
        expected[1769] = 0.25
        expected[1783] = 0.4
        expected[1797] = 0.5
        expected[1811] = 0.5
        expected[1825] = 0.65
        expected[1839] = 0.75

        for C in Cul:
            for S in Sc:
                for E in Ev:
                    if C == 'P':
                        # PIPE TEST
                        for d in Dm:
                            Structure = [d]
                            if verbose:
                                print(Structure)
                            Scenario, Ev_mag, BF_clss, diag, BF = \
                                get_WCC_2016_Blockage_factor(
                                    Structure, E, S, long_result=True, verbose=verbose)
                            if verbose:
                                print('Scenario: %s  Event Magnitude: %s, Structure Class: %s, Diagonal: %7.3fm Blockage Factor: %7.3f' % (
                                    Scenario, Ev_mag, BF_clss, float(diag), float(BF)))
                                print('expected[%g] = %g ' % (i, expected[i]))
                            assert num.allclose(BF, expected[i])

                            i = i+1

                    elif C == 'B':
                        # BOX TEST
                        for w in Wm:
                            for h in Hm:
                                Structure = [h, w]
                                if verbose:
                                    #print(Structure, end=' ')   # This is the correct Python 3 statement
                                    print(Structure)  # FIXME(Ole): for Python 2 compatibility.

                                Scenario, Ev_mag, BF_clss, diag, BF = \
                                    get_WCC_2016_Blockage_factor(
                                        Structure, E, S, long_result=True, verbose=verbose)
                                if verbose:
                                    print('Scenario: %s  Event Magnitude: %s, Structure Class: %s, Diagonal: %7.3fm Blockage Factor: %7.3f' % (
                                        Scenario, Ev_mag, BF_clss, float(diag), float(BF)))
                                    print('expected[%g] = %g ' %
                                          (i, expected[i]))

                                assert num.allclose(BF, expected[i])

                                i = i+1

    def test_create_culvert_bridge_operator_boyd_pipe(self):
        """
        Test creation of Boyd pipe operator (based on diameter being set)
        """

        import os
        import anuga
        
        
        file_contents = \
"""
#--------------------------
# CULVERT DATA FOR: Nrth Bagot Rd
#--------------------------
#blockage = 0.0
label = 'Nrth_Bagot_6x600'
diameter = 0.6
#width=3.04
#height=2.45
apron = 0.5
enquiry_points = [ [9.,5.], [14.,5.] ]
manning=0.013
invert_elevations=[0.0,0.1]
losses = {'inlet':0.5, 'outlet':1.0, 'bend':0.0, 'grate':0.0, 'pier': 0.0, 'other': 0.0}
#end_points = [[10,5], [12,5]]

exchange_lines = [ [[10, 1], [10, 9]], 
                   [[12, 1], [12, 9]] ]
"""

        culvert_bridge_file = 'test_boyd_pipe.txt'
        with open(culvert_bridge_file, 'w') as f:
            f.write(file_contents)

        # Create domain and add culvert/bridge operator
        # using the culvert_bridge_file
        domain1 = anuga.rectangular_cross_domain(30, 10, 30, 10)
        domain1.set_name('domain1_boyd_pipe')
        Br = anuga.Reflective_boundary(domain1)
        Bd = anuga.Dirichlet_boundary([10, 0, 0])
        domain1.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})
        anuga.Create_culvert_bridge_Operator(domain1, culvert_bridge_file)
        os.remove(culvert_bridge_file)
        domain1.evolve_to_end(0.1)

        # Create domain and add culvert/bridge operator
        # using the local_vars dictionary
        domain2 = anuga.rectangular_cross_domain(30, 10, 30, 10)
        domain2.set_name('domain2_boyd_pipe')
        Br = anuga.Reflective_boundary(domain2)
        Bd = anuga.Dirichlet_boundary([10, 0, 0])
        domain2.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})
        local_vars= {}
        exec(file_contents, {}, local_vars)
        #print(local_vars)
        anuga.Boyd_pipe_operator(domain2, **local_vars)
        domain2.evolve_to_end(0.1)

        # Check that the two domains give identical results
        s1 = domain1.get_quantity('stage').centroid_values
        s2 = domain2.get_quantity('stage').centroid_values
        assert num.allclose(s1, s2)

        try:
            os.remove('domain1_boyd_pipe.sww')
            os.remove('domain2_boyd_pipe.sww')
        except FileNotFoundError:
            pass
        

    def test_create_culvert_bridge_operator_boyd_box(self):
        """
        Test creation of Boyd box operator (based on width and height being set)
        """

        import os
        import anuga
        
        
        file_contents = \
"""
#--------------------------
# CULVERT DATA FOR: Nrth Bagot Rd
#--------------------------
#blockage = 0.0
label = 'Nrth_Bagot_6x600'
#diameter = 0.6
width=3.04
height=2.45
apron = 0.5
enquiry_points = [ [9.,5.], [14.,5.] ]
manning=0.013
invert_elevations=[0.0,0.1]
losses = {'inlet':0.5, 'outlet':1.0, 'bend':0.0, 'grate':0.0, 'pier': 0.0, 'other': 0.0}
#end_points = [[10,5], [12,5]]

exchange_lines = [ [[10, 1], [10, 9]], 
                   [[12, 1], [12, 9]] ]
"""

        culvert_bridge_file = 'test_boyd_box.txt'
        with open(culvert_bridge_file, 'w') as f:
            f.write(file_contents)

        #Create domain and add culvert/bridge operator
        #using the culvert_bridge_file
        domain1 = anuga.rectangular_cross_domain(30, 10, 30, 10)
        domain1.set_name('domain1_boyd_box')
        Br = anuga.Reflective_boundary(domain1)
        Bd = anuga.Dirichlet_boundary([10, 0, 0])
        domain1.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})
        anuga.Create_culvert_bridge_Operator(domain1, culvert_bridge_file)
        os.remove(culvert_bridge_file)
        domain1.evolve_to_end(0.1)

        # Create domain and add culvert/bridge operator
        # using the local_vars dictionary
        domain2 = anuga.rectangular_cross_domain(30, 10, 30, 10)
        domain2.set_name('domain2_boyd_box')
        Br = anuga.Reflective_boundary(domain2)
        Bd = anuga.Dirichlet_boundary([10, 0, 0])
        domain2.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})
        local_vars= {}
        exec(file_contents, {}, local_vars)
        #print(local_vars)
        anuga.Boyd_box_operator(domain2, **local_vars)
        domain2.evolve_to_end(0.1)

        # Check that the two domains give identical results
        s1 = domain1.get_quantity('stage').centroid_values
        s2 = domain2.get_quantity('stage').centroid_values
        assert num.allclose(s1, s2)

        try:    
            os.remove('domain1_boyd_box.sww')
            os.remove('domain2_boyd_box.sww')
        except FileNotFoundError:
            pass
        
    def test_create_culvert_bridge_operator_weir_orifice_trapezoid_operator(self):
        """
        Test creation of Weir_orifice_trapezoid_operator (based on z1 or z2 being set)
        """

        import os
        import anuga
        
        
        file_contents = \
"""
#--------------------------
# CULVERT DATA FOR: Nrth Bagot Rd
#--------------------------
#blockage = 0.0
label = 'Nrth_Bagot_6x600'
#diameter = 0.6
width=3.04
height=2.45
z1 = 1.0
z2 = 0.5
apron = 0.5
enquiry_points = [ [9.,5.], [14.,5.] ]
manning=0.013
invert_elevations=[0.0,0.1]
losses = {'inlet':0.5, 'outlet':1.0, 'bend':0.0, 'grate':0.0, 'pier': 0.0, 'other': 0.0}
#end_points = [[10,5], [12,5]]

exchange_lines = [ [[10, 1], [10, 9]], 
                   [[12, 1], [12, 9]] ]
"""

        culvert_bridge_file = 'test_weir_orifice_trapezoid_operator.txt'
        with open(culvert_bridge_file, 'w') as f:
            f.write(file_contents)

        #Create domain and add culvert/bridge operator
        #using the culvert_bridge_file
        domain1 = anuga.rectangular_cross_domain(30, 10, 30, 10)
        domain1.set_name('domain1_weir_orifice_trapezoid')
        Br = anuga.Reflective_boundary(domain1)
        Bd = anuga.Dirichlet_boundary([10, 0, 0])
        domain1.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})
        anuga.Create_culvert_bridge_Operator(domain1, culvert_bridge_file)
        os.remove(culvert_bridge_file)
        domain1.evolve_to_end(0.1)

        # Create domain and add culvert/bridge operator
        # using the local_vars dictionary
        domain2 = anuga.rectangular_cross_domain(30, 10, 30, 10)
        domain2.set_name('domain2_weir_orifice_trapezoid')
        Br = anuga.Reflective_boundary(domain2)
        Bd = anuga.Dirichlet_boundary([10, 0, 0])
        domain2.set_boundary({'left': Bd, 'right': Br, 'top': Br, 'bottom': Br})
        local_vars= {}
        exec(file_contents, {}, local_vars)
        #print(local_vars)
        anuga.Weir_orifice_trapezoid_operator(domain2, **local_vars)
        domain2.evolve_to_end(0.1)

        # Check that the two domains give identical results
        s1 = domain1.get_quantity('stage').centroid_values
        s2 = domain2.get_quantity('stage').centroid_values
        assert num.allclose(s1, s2)
        
        try:
            os.remove('domain1_weir_orifice_trapezoid.sww')
            os.remove('domain2_weir_orifice_trapezoid.sww')
        except FileNotFoundError:
            pass
        


################################################################################

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_Model_Tools)
    runner = unittest.TextTestRunner()
    runner.run(suite)
