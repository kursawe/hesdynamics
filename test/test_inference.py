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
        ## run a sample simulation to generate example protein data
        true_data = hes5.generate_langevin_trajectory(duration = 900)
        
        ## the F constant matrix is left out for now
        protein_at_observation = true_data[0:900:10,(0,2)]
        protein_at_observation[:,1] += np.random.randn(90)
            
        parameters = [720, 10000,5,np.log(2)/30, np.log(2)/90, 1, 1, 29]
         
        ## apply kalman filter to the data
        state_space_mean, state_space_variance = hes_inference.kalman_filter(protein,
                                                                             parameters)
        
        self.assertEqual(state_space_mean.shape[0],90)
        self.assertEqual(state_space_mean.shape[1],3)
        self.assertEqual(state_space_variance.shape[0],180)
        self.assertEqual(state_space_variance.shape[1],180)
        # check dimensionality of state_space_mean and the state_space_variance
        # variance needs to be positive definite and symmetric, maybe include quantitative check
        
        ##plot data together with state-space distribution
        pass