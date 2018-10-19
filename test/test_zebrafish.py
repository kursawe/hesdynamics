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
import scipy.optimize

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestZebrafish(unittest.TestCase):

    def xest_generate_single_oscillatory_trajectory(self):
        #First: run the model for 100 minutes
        my_trajectory = hes5.generate_deterministic_goodfellow_trajectory( duration = 720,
                                                                           protein_repression_threshold = 100,
                                                                           miRNA_repression_threshold = 10,
                                                                           upper_mRNA_degradation_rate = 0.03,
                                                                           lower_mRNA_degradation_rate = 0.03,
                                                                           protein_degradation_rate = 0.03,
                                                                           hill_coefficient_protein_on_protein = 5,
                                                                           hill_coefficient_miRNA_on_protein = 5,
                                                                           hill_coefficient_protein_on_miRNA = 5,
                                                                           miRNA_degradation_rate = 0.00001,
                                                                           transcription_delay = 19,
                                                                           initial_mRNA = 3,
                                                                           initial_protein = 100,
                                                                           initial_miRNA = 1)
#                                                          integrator = 'PyDDE',
#                                                          for_negative_times = 'no_negative' )

        #Second, plot the model

        figuresize = (4,2.75)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1], label = 'mRNA', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2]*0.03, label = 'Hes protein', color = 'black', ls = '--')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,3]*0.03, label = 'miRNA', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Scaled expression')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','oscillating_trajectory.pdf'))
        

    def protein_difference_upon_degradation_increase(self,
                                                     repression_threshold,
                                                     hill_coefficient,
                                                     RNA_degradation,
                                                     protein_degradation):
        
        _, mean_protein_before_increase = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                                          hill_coefficient,
                                                                          RNA_degradation,
                                                                          protein_degradation,
                                                                          1.0,
                                                                          1.0)

        _, mean_protein_after_increase = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                                         hill_coefficient,
                                                                         RNA_degradation,
                                                                         protein_degradation,
                                                                         1.0,
                                                                         1.0)
        
        difference = (mean_protein_before_increase - mean_protein_after_increase)/mean_protein_before_increase
        
        return difference
    
    def degradation_constraint_function(self,
                                        repression_threshold,
                                        hill_coefficient,
                                        RNA_degradation,
                                        protein_degradation):
        
        _, steady_protein = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                               hill_coefficient,
                                                               RNA_degradation,
                                                               protein_degradation,
                                                               1.0,
                                                               1.0)

        hill_derivative = ( 1.0/np.power(1.0+np.power(steady_protein/repression_threshold,hill_coefficient),2)*
                            np.power(steady_protein/repression_threshold,hill_coefficient - 1)/repression_threshold )
        
        return hill_derivative - RNA_degradation*protein_degradation  
    
    def delay_constraint_function(self,
                                  repression_threshold,
                                  hill_coefficient,
                                  RNA_degradation,
                                  protein_degradation):
        
        hill_derivative = ( 1.0/np.power(1.0+np.power(steady_protein/repression_threshold,hill_coefficient),2)*
                            np.power(steady_protein/repression_threshold,hill_coefficient - 1)/repression_threshold )

        squared_degradation_difference = protein_degradation*protein_degradation - RNA_degradation*RNA_degradation
        squared_degradation_sum = protein_degradation*protein_degradation + RNA_degradation*RNA_degradation

        omega = np.sqrt(0.5*(np.sqrt(squared_degradation_difference*squared_degradation_difference
                               + 4*derivative_term*hill_derivative) - 
                               squared_degradation_sum))
        arccos_value = np.arccos( ( omega*omega - protein_degradation_rate*mRNA_degradation_rate)/
                                    hill_derivative )
        
        return omega*transcription_delay - arccos_value
    
    def test_maximise_protein_reduction_by_degradation_increase(self):

        degradation_constraint = { 'type' : 'ineq',
                                   'fun' : self.degradation_constraint_function }

        delay_constraint = { 'type' : 'ineq',
                             'fun' : self.delay_constraint_function }
        
        optimize_result = scipy.optimize.minimize(fun = self.protein_difference_upon_degradation_increase, 
                                                  x0 = np.array([1.0,5.0,1.0,1.0]),
                                                  constraints = [degradation_constraint, delay_constraint],
                                                  bounds = [[0.0,np.inf],
                                                            [2.0,6.0],
                                                            [0.0,100.0],
                                                            [0.0,100.0]])
        
        print('the maximal difference we can get is')
        print(optimize_result.x)