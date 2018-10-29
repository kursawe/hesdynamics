import unittest
import os.path
import sys
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np
from jitcdde import jitcdde,y,t

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
# import hes5
import hes_inference

class TestInference(unittest.TestCase):

    def test_inference(self):
        ## write a test here!
        pass
