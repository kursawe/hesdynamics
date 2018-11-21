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
import hes5
import hes_inference

class TestInference(unittest.TestCase):

    def test_inference(self):
        ## run a sample simulation to generate example protein data
        true_data = hes5.generate_langevin_trajectory(duration = 900, equilibration_time = 1000)

        ## the F constant matrix is left out for now
        protein_at_observation = true_data[0:900:10,(0,2)]
        protein_at_observation[:,1] += np.random.randn(90)*0
        protein_at_observation[:,1] = np.maximum(protein_at_observation[:,1],0)

        parameters = [10000,5,np.log(2)/30, np.log(2)/90, 1, 1, 29]

        ## apply kalman filter to the data
        state_space_mean, state_space_variance = hes_inference.kalman_filter(protein_at_observation,
                                                                             parameters)

        #self.assertEqual(state_space_mean.shape[0],930)
        #self.assertEqual(state_space_mean.shape[1],3)
        #self.assertEqual(state_space_variance.shape[0],1860)
        #self.assertEqual(state_space_variance.shape[1],1860)
        # check dimensionality of state_space_mean and the state_space_variance
        # variance needs to be positive definite and symmetric, maybe include quantitative check
        ##plot data together with state-space distribution
        number_of_states = state_space_mean.shape[0]

        protein_covariance_matrix = state_space_variance[number_of_states:,number_of_states:]
        protein_variance = np.diagonal(protein_covariance_matrix)
        print(protein_variance)
        protein_error = np.sqrt(protein_variance)

        my_figure = plt.figure()
        plt.scatter(np.arange(0,900,10),protein_at_observation[:,1],marker='o',s=4,c='r',label='observations',zorder=3)
        plt.plot(true_data[:,0],true_data[:,2],label='true protein',zorder=1)
#         plt.plot(state_space_mean[:,0],state_space_mean[:,2],label='inferred protein',zorder=2)
        plt.errorbar(state_space_mean[:,0],state_space_mean[:,2],yerr=protein_error)
        plt.ylim(3000,8000)
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','kalman_test_protein.pdf'))

        mRNA_covariance_matrix = state_space_variance[:number_of_states,:number_of_states]
        mRNA_variance = np.diagonal(mRNA_covariance_matrix)
        print(mRNA_variance)
        mRNA_error = np.sqrt(protein_variance)
        my_figure = plt.figure()
        plt.plot(true_data[:,0],true_data[:,1],label='true mRNA',zorder=1)
#         plt.plot(state_space_mean[:,0],state_space_mean[:,1],label='inferred mRNA',zorder=2)
        plt.errorbar(state_space_mean[:,0],state_space_mean[:,1],yerr=mRNA_error)
        plt.ylim(20,70)
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','kalman_test_mRNA.pdf'))
