import unittest
import os.path
import sys
import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestSimpleHes5Model(unittest.TestCase):
                                 
    def test_generate_single_oscillatory_trajectory(self):
        #First: run the model for 100 minutes
        my_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                         repression_threshold = 100,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         repression_delay = 18.5,
                                                         initial_mRNA = 3,
                                                         initial_protein = 100 )

        #Second, plot the model

        print my_trajectory

        figuresize = (4,2.75)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1], label = 'mRNA', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2]*0.03, label = 'Hes protein', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Scaled expression')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','oscillating_trajectory.pdf'))

    def test_generate_hes5_predicted_trajectory(self):
        #First: run the model for 100 minutes
        my_trajectory = hes5.generate_single_trajectory( duration = 720,
                                                         repression_threshold = 100,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         repression_delay = 29,
                                                         initial_mRNA = 3,
                                                         initial_protein = 100 )

        #Second, plot the model

        print my_trajectory

        figuresize = (4,2.75)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1], label = 'mRNA', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','hes5_trajectory.pdf'))
