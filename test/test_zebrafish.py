import unittest
import os.path
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
import matplotlib.gridspec 
from mpl_toolkits.axes_grid1 import make_axes_locatable
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np
import scipy.optimize
import pandas as pd
import seaborn as sns
import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
try:
    import gpflow
except ImportError:
    print('Could not import gpflow. This may affect GP regression tests.')
from numba import jit, autojit

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

import socket
import multiprocessing as mp
domain_name = socket.getfqdn()
if domain_name == 'jochen-ThinkPad-S1-Yoga-12':
    number_of_available_cores = 2
else:
#     number_of_available_cores = 1
    number_of_available_cores = mp.cpu_count()

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
        

    def protein_difference_upon_degradation_increase(self, parameter_array):

        repression_threshold, hill_coefficient, RNA_degradation, protein_degradation = parameter_array
        
        _, mean_protein_before_increase = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                                          hill_coefficient,
                                                                          RNA_degradation,
                                                                          protein_degradation,
                                                                          1.0,
                                                                          1.0)

        _, mean_protein_after_increase = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                                         hill_coefficient,
                                                                         RNA_degradation*2.0,
                                                                         protein_degradation,
                                                                         1.0,
                                                                         1.0)
        
        difference = -(mean_protein_before_increase - mean_protein_after_increase)/mean_protein_before_increase
        
        print('these parameters are')
        print(parameter_array)
        print('this difference is')
        print(difference)
        
        return difference
    
    def degradation_constraint_function(self,parameter_array):
        
        repression_threshold, hill_coefficient, RNA_degradation, protein_degradation = parameter_array
        _, steady_protein = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                               hill_coefficient,
                                                               RNA_degradation,
                                                               protein_degradation,
                                                               1.0,
                                                               1.0)

        hill_derivative = ( hill_coefficient/np.power(1.0+np.power(steady_protein/repression_threshold,hill_coefficient),2)*
                            np.power(steady_protein/repression_threshold,hill_coefficient - 1)/repression_threshold )
        
#         print 'these parameters are'
#         print parameter_array
#         print 'the degradation constraint is'
#         print hill_derivative - RNA_degradation*protein_degradation
        return hill_derivative - RNA_degradation*protein_degradation  
    
    def delay_constraint_function(self,parameter_array):
        
        repression_threshold, hill_coefficient, RNA_degradation, protein_degradation = parameter_array

        _, steady_protein = hes5.calculate_steady_state_of_ode(repression_threshold,
                                                               hill_coefficient,
                                                               RNA_degradation,
                                                               protein_degradation,
                                                               1.0,
                                                               1.0)

        hill_derivative = ( hill_coefficient/np.power(1.0+np.power(steady_protein/repression_threshold,hill_coefficient),2)*
                            np.power(steady_protein/repression_threshold,hill_coefficient - 1)/repression_threshold )

        squared_degradation_difference = protein_degradation*protein_degradation - RNA_degradation*RNA_degradation
        squared_degradation_sum = protein_degradation*protein_degradation + RNA_degradation*RNA_degradation

        try:
            omega = np.sqrt(0.5*(np.sqrt(squared_degradation_difference*squared_degradation_difference
                               + 4*hill_derivative*hill_derivative) - 
                               squared_degradation_sum))
        except RuntimeWarning:
            return -10.0
        arccos_value = np.arccos( ( omega*omega - protein_degradation*RNA_degradation)/
                                    hill_derivative )
        
#         print 'these parameters are'
#         print parameter_array
#         print 'the degradation constraint is'
#         print hill_derivative - RNA_degradation*protein_degradation
        return omega - arccos_value
    
    def xest_maximise_protein_reduction_by_degradation_increase(self):

        degradation_constraint = { 'type' : 'ineq',
                                   'fun' : self.degradation_constraint_function }

        delay_constraint = { 'type' : 'ineq',
                             'fun' : self.delay_constraint_function }
        
        optimize_result = scipy.optimize.minimize(fun = self.protein_difference_upon_degradation_increase, 
                                                  x0 = np.array([1.0,6.0,0.01,0.01]),
                                                  constraints = [degradation_constraint, delay_constraint],
                                                  bounds = [[0.0001,np.inf],
                                                            [2.0,6.0],
                                                            [0.0,3.0],
                                                            [0.0,3.0]])
        
        print('the maximal difference we can get is')
        print(optimize_result.x)
        
    def test_a_make_abc_samples(self):
        print('starting zebrafish abc')
        ## generate posterior samples
        total_number_of_samples = 2000000
#         total_number_of_samples = 5
#         total_number_of_samples = 100
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.6,60),
                        'translation_rate' : (0.04,40),
                        'repression_threshold' : (0,5000),
                        'time_delay' : (1,30),
                        'hill_coefficient' : (2,6),
                        'protein_degradation_rate' : ( np.log(2)/11.0, np.log(2)/11.0 ),
                        'mRNA_half_life' : ( 1, 11) }

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_zebrafish_delay_large',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'full',
                                                                logarithmic = True )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 7))
        
    def xest_plot_zebrafish_inference(self):
#         option = 'prior'
#         option = 'mean_period_and_coherence'
#         option = 'mean_longer_periods_and_coherence'
#         option = 'mean_and_std'
        option = 'mean_std_period'
#         option = 'coherence_decrease_translation'
#         option = 'coherence_decrease_degradation'
#         option = 'dual_coherence_decrease'
#         option = 'dual_coherence_and_lengthscale_decrease'
#         option = 'mean_std_period_fewer_samples'
#         option = 'mean_std_period_coherence'
#         option = 'weird_decrease'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish_delay')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                       model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                                       model_results[:,0]<1500))  #standard deviation
#                                                        model_results[:,1]>0.05)))  #standard deviation
        elif option == 'prior':
            accepted_indices = range(len(prior_samples))
        elif option == 'coherence':
            accepted_indices = np.where( model_results[:,3]>0.3 )  #standard deviation
        elif option == 'period':
            accepted_indices = np.where( model_results[:,2]<100 )  #standard deviation
        elif option == 'period_and_coherence':
            accepted_indices = np.where( np.logical_and( model_results[:,2]<100,
                                                         model_results[:,3]>0.3 ))  
        elif option == 'mean_period_and_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<8000,
                                        np.logical_and(model_results[:,2]<100,
                                                       model_results[:,3]>0.3))))  
        elif option == 'mean_longer_periods_and_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<8000,
                                        np.logical_and(model_results[:,2]<150,
                                        np.logical_and(model_results[:,3]>0.25,
                                                       model_results[:,3]<0.4)))))
        elif option == 'mean_and_std':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                                       model_results[:,1]>0.05))))
        elif option == 'mean_std_period_fewer_samples':
            accepted_indices = np.where(np.logical_and(model_results[:4000,0]>1000, #protein number
                                        np.logical_and(model_results[:4000,0]<2500,
                                        np.logical_and(model_results[:4000,1]<0.15,
                                        np.logical_and(model_results[:4000,1]>0.05,
                                                       model_results[:4000,2]<150)))))
        elif option == 'mean_std_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
        elif option == 'mean_std_period_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<8000,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                        np.logical_and(model_results[:,3]>0.4,
                                                       model_results[:,2]<150))))))
#                                         np.logical_and(model_results[:,2]<150,
        elif option == 'amplitude_and_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<4000, #protein_number
#                                         np.logical_and(model_results[:,4]>40,
#                                         np.logical_and(model_results[:,4]>60, #mrna number
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,3]>0.15)))) #standard deviation
        elif option == 'deterministic': 
             accepted_indices = np.where(np.logical_and(model_results[:,5]>2000, #protein number
                                         np.logical_and(model_results[:,5]<4000, #protein_number
                                         np.logical_and(model_results[:,9]>40,
                                         np.logical_and(model_results[:,9]<60, #mrna number
                                                        model_results[:,6]>0.05)))))  #standard deviation
        elif option == 'weird_decrease':
            change = 'decreased'
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradationtest')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
#             weird_indices = np.where(results_before_change[:,0]>results_after_change[:,0])
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
        elif option == 'coherence_decrease_degradation':
            change = 'decreased'
#             change = 'increased'
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
        elif option == 'coherence_decrease_translation':
#             change = 'decreased'
            change = 'increased'
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
        elif option == 'dual_coherence_decrease':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_likelihoods_coherence_decrease.npy')
            conditions = np.load(saving_path)
            positive_indices = np.where(conditions>0)
            accepted_indices = (accepted_indices[0][positive_indices],)
        elif option == 'dual_coherence_and_lengthscale_decrease':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_likelihoods.npy')
            conditions = np.load(saving_path)
            positive_indices = np.where(conditions>0)
            accepted_indices = (accepted_indices[0][positive_indices],)
        else:
            ValueError('could not identify posterior option')
#       
        if option not in ['weird_decrease', 'coherence_decrease_degradation',
                          'coherence_decrease_translation']:
            my_posterior_samples = prior_samples[accepted_indices]
        else:
            my_posterior_samples = weird_parameters_before

        print('Number of accepted samples is ')
        print(len(my_posterior_samples))
        print('minimal transcription is')
        print(np.min(my_posterior_samples[:,0]))
        print('and in log space')
        print(np.min(np.log10(my_posterior_samples[:,0])))
        print('minimal translation is')
        print(np.min(my_posterior_samples[:,1]))
        print('and in log space')
        print(np.min(np.log10(my_posterior_samples[:,1])))

        my_posterior_samples[:,2]/=1000

        print(my_posterior_samples.shape)
#         my_pairplot = hes5.plot_posterior_distributions(my_posterior_samples)

        data_frame = pd.DataFrame( data = my_posterior_samples[:,:6],
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e3', 
                                             'Transcription delay',
                                             'Hill coefficient',
                                             'mRNA degradation'])

        ### PAIRGRID
#         my_adjusted_posterior_samples = np.copy(my_posterior_samples)
#         my_adjusted_posterior_samples[:,5] = np.log(2)/my_adjusted_posterior_samples[:,5]
#         my_adjusted_posterior_samples[:,0] = np.log10(my_adjusted_posterior_samples[:,0])
#         my_adjusted_posterior_samples[:,1] = np.log10(my_adjusted_posterior_samples[:,1])
#         new_data_frame = pd.DataFrame( data = my_adjusted_posterior_samples[:,:6],
#                                    columns= ['log10(Transcription rate)', 
#                                              'log10(Translation rate)', 
#                                              'Repression threshold/1e3', 
#                                              'Transcription delay',
#                                              'Hill coefficient',
#                                              'mRNA half life'])
#         my_pairplot = sns.PairGrid(new_data_frame)
# #         my_pairplot = sns.pairplot(new_data_frame)
#         my_pairplot.map_upper(plt.scatter, alpha = 0.02, color = 'black', rasterized = True)
# #         my_pairplot.map_upper(sns.kdeplot,rasterized = True)
#         my_pairplot.map_diag(plt.hist)
#         my_pairplot.map_lower(sns.kdeplot, cmap = 'Reds', rasterized = True)
# #         my_pairplot.axes[-1,0].set_xscale("log")
# #         my_pairplot.axes[-1,1].set_xscale("log")
#         my_pairplot.savefig(os.path.join(os.path.dirname(__file__),
#                                          'output',
#                                          'pairplot_zebrafish_abc_' +  option + '.pdf'))
#         ### END PAIRGRID

        sns.set(font_scale = 1.1, rc = {'ytick.labelsize': 6})
#         font = {'size'   : 28}
#         plt.rc('font', **font)
        my_figure = plt.figure(figsize= (11,3))

        my_figure.add_subplot(161)
#         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(np.log10(0.6),np.log10(60.0),20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Transcription rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                    bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
        plt.gca().set_xlim(-0.5,np.log10(60.0))
        plt.ylabel("Probability", labelpad = 20)
        plt.xlabel("Transcription rate \n [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,1)
#         plt.gca().set_ylim(0,1)
#         plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
        plt.xticks([0,1], [r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(162)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(np.log10(0.04),np.log10(40),20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black',
                                 'alpha' : None},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(-2,1)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([-1,0], [r'$10^{-1}$',r'$10^0$'])
        plt.xlabel("Translation rate \n [1/min]")
        plt.gca().set_ylim(0,1)
#         plt.gca().set_ylim(0,1.0)
#         plt.yticks([])
 
        my_figure.add_subplot(163)
        sns.distplot(data_frame['Repression threshold/1e3'],
                     kde = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e3]")
        plt.gca().set_ylim(0,0.5)
        plt.gca().set_xlim(0,5)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(164))
        time_delay_bins = np.linspace(1,30,11)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                     bins = time_delay_bins)
        plt.gca().set_xlim(1,40)
        plt.gca().set_ylim(0,0.07)
#         plt.gca().set_ylim(0,0.04)
        plt.gca().locator_params(axis='x', tight = True, nbins=5)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xlabel(" Transcription delay \n [min]")
#         plt.yticks([])
 
        plots_to_shift.append(my_figure.add_subplot(165))
        sns.distplot(data_frame['Hill coefficient'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.gca().set_ylim(0,0.4)
        plt.gca().set_xlim(2,6)
        plt.gca().locator_params(axis='x', tight = True, nbins=3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        my_figure.add_subplot(166)
#         translation_rate_bins = np.logspace(0,2.3,20)
#         degradation_rate_bins = np.linspace(np.log(2.0)/15.0,np.log(2)/1.0,20)
#         histogram, bin_edges = np.histogram(data_frame['mRNA degradation'], degradation_rate_bins, 
#                                             density = True)
#         plt.hist(histogram[::-1], np.log(2)/bin_edges[::-1] )

        half_lifes = np.log(2)/data_frame['mRNA degradation']
        print(half_lifes)
        half_life_bins = np.linspace(1,11,20)
#         half_life_histogram, _ = np.histogram(half_lifes, half_life_bins, density = True)
#         print(half_life_histogram)
#         prior_histogram, _ = np.histogram( np.log(2)/prior_samples[:,5], half_life_bins, density = True )
#         corrected_histogram = half_life_histogram/prior_histogram
#         corrected_histogram = half_life_histogram
#         print(corrected_histogram)
#         bin_centres = (half_life_bins[:-1] + half_life_bins[1:])/2
#         width = 0.7*(half_life_bins[1] - half_life_bins[0])
         
#         plt.bar(bin_centres, corrected_histogram, align = 'center' , width = width )
        sns.distplot(half_lifes,
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                    bins = half_life_bins)
#
#         sns.distplot(data_frame['mRNA degradation'],
#                      kde = False,
#                      rug = False,
#                      norm_hist = True,
#                      hist_kws = {'edgecolor' : 'black'},
#                      bins = degradation_rate_bins)
# #         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
#         plt.gca().set_xlim(-2,0)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.xticks([-1,0], [r'$10^{-1}$',r'$10^0$'])
        plt.xlabel("mRNA half-life \n [min]")
#         plt.gca().set_ylim(0,4.0)
#         plt.gca().set_ylim(0,1.0)
#         plt.yticks([])
 
        plt.tight_layout(w_pad = 0.0001)
#         plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inference_for_zebrafish_' + option + '.pdf'))

    def xest_plot_zebrafish_period_distribution(self):
        model_option = 'extrinsic_noise'
        if model_option == 'extrinsic_noise':
#         saving_path = os.path.join(os.path.dirname(__file__), 'output',
#                                     'sampling_results_zebrafish')
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                        'sampling_results_zebrafish_extrinsic_noise')
        else:
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                        'sampling_results_zebrafish')
#
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        option = 'mean_std_period'
        if option == 'mean_std':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                                       model_results[:,1]>0.05))))
        elif option == 'mean_std_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
        else:
            raise ValueError('option not recognised')

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]
#         my_posterior_samples = prior_samples
#         my_model_results = model_results

        sns.set()
        # sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        # font = {'size'   : 28}
        # plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_periods = my_model_results[:,2]
        print('mean is')
#         print(np.mean(all_periods[all_periods<10]))
        print(np.mean(all_periods))
        print('median is')
#         print(np.median(all_periods[all_periods<10]))
        print(np.median(all_periods))
        print('minimum is')
        print(np.min(all_periods))
        period_histogram, bins = np.histogram(all_periods[all_periods<300], bins = 400) 
#         period_histogram, bins = np.histogram(all_periods, bins = 400) 
        maximum_index = np.argmax(period_histogram)
        print('max bin is')
# # # #         print bins[maximum_index]
# # #         print bins[maximum_index+1]
# #         print bins[maximum_index+2]
#         print bins[maximum_index-1]
# #         sns.distplot(all_periods[np.logical_and(all_periods<1000,
#                                                 all_periods>100)],
        sns.distplot(all_periods[all_periods<300],
#         sns.distplot(all_periods,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled period [h]")
#         plt.xlim(0,300)
#         plt.ylim(0,0.2)
#         plt.ylim(0,0.0003)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','zebrafish_period_distribution_' + model_option + '_'+ option + '.pdf'))
 
    def xest_plot_zebrafish_coherence_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        option = 'mean_std'
        if option == 'mean_std':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                                       model_results[:,1]>0.05))))
        elif option == 'mean_std_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
        elif option == 'coherence_decrease_translation':
#             change = 'decreased'
            change = 'increased'
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
            weird_results_before = results_before_change[weird_indices]
            weird_results_after = results_after_change[weird_indices]
        elif option == 'coherence_decrease_degradation':
            change = 'decreased'
#             change = 'increased'
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
            weird_results_before = results_before_change[weird_indices]
            weird_results_after = results_after_change[weird_indices]
        else:
            raise ValueError('option not recognised')
#       
        if option not in ['weird_decrease', 'coherence_decrease_translation',
                          'coherence_decrease_degradation']:
            my_posterior_samples = prior_samples[accepted_indices]
            my_model_results = model_results[accepted_indices]
        else:
            my_posterior_samples = weird_parameters_before
            my_model_results = weird_results_before


#         sns.set()
        # sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        # font = {'size'   : 28}
        # plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_coherences = my_model_results[:,3]
        print('largest coherence is')
        print(np.max(all_coherences))
        sns.distplot(all_coherences,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black',
                                 'alpha' : 1},
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled coherence")
#         plt.xlim(0.2,)
        plt.ylim(0,5)
#         plt.xlim(0,20)
#         plt.ylim(0,0.8)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','zebrafish_coherence_distribution_'+option+'.pdf'))
        
    def xest_plot_zebrafish_cov_distribution(self):
        model_option = 'extrinsic_noise'
        if model_option == 'extrinsic_noise':
#         saving_path = os.path.join(os.path.dirname(__file__), 'output',
#                                     'sampling_results_zebrafish')
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                        'sampling_results_zebrafish_extrinsic_noise')
        else:
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                        'sampling_results_zebrafish')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

#         option = 'mean_period'
        option = 'mean_std'
#         option = 'mean_std_period'
        if option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                                       model_results[:,0]<2500))
        elif option == 'mean_std':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
#                                                        model_results[:,1]<0.15)))
                                        np.logical_and(model_results[:,1]<0.15,
                                                        model_results[:,1]>0.05))))
        elif option == 'mean_std_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
        elif option == 'mean_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                                       model_results[:,2]<150)))
        elif option == 'coherence_decrease_translation':
#             change = 'decreased'
            change = 'increased'
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
            weird_results_before = results_before_change[weird_indices]
            weird_results_after = results_after_change[weird_indices]
        elif option == 'coherence_decrease_degradation':
            change = 'decreased'
#             change = 'increased'
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
            weird_results_before = results_before_change[weird_indices]
            weird_results_after = results_after_change[weird_indices]
        else:
            raise ValueError('option not recognised')
#       
        if option not in ['weird_decrease', 'coherence_decrease_translation',
                          'coherence_decrease_degradation']:
            my_posterior_samples = prior_samples[accepted_indices]
            my_model_results = model_results[accepted_indices]
        else:
            my_posterior_samples = weird_parameters_before
            my_model_results = weird_results_before


#         sns.set()
        # sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        # font = {'size'   : 28}
        # plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_covs = my_model_results[:,1]
        print('largest cov is')
        print(np.max(all_covs))
        sns.distplot(all_covs,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black',
                                 'alpha' : 1},
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("COV")
#         plt.xlim(0.2,)
#         plt.ylim(0,5)
#         plt.xlim(0,20)
#         plt.ylim(0,0.8)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','zebrafish_cov_distribution_' + 
                                 model_option + '_' + option +'.pdf'))
        
    def xest_plot_zebrafish_mrna_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        option = 'coherence_increase_translation'
#         option = 'mean_std_period'
        if option == 'mean_and_std':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<8000,
                                        np.logical_and(model_results[:,1]<0.15,
                                                       model_results[:,1]>0.05))))
        elif option == 'prior':
            accepted_indices = np.where(model_results[:,0]>0) # always true
        elif option == 'mean_std_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<8000,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
        elif option == 'coherence_decrease_translation':
#             change = 'decreased'
            change = 'increased'
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
            weird_results_before = results_before_change[weird_indices]
            weird_results_after = results_after_change[weird_indices]
        elif option == 'coherence_increase_translation':
#             change = 'decreased'
            change = 'increased'
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]<results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
            weird_results_before = results_before_change[weird_indices]
            weird_results_after = results_after_change[weird_indices]
 
        elif option == 'coherence_decrease_degradation':
            change = 'decreased'
#             change = 'increased'
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
            weird_results_before = results_before_change[weird_indices]
            weird_results_after = results_after_change[weird_indices]
        elif option == 'coherence_increase_degradation':
            change = 'decreased'
#             change = 'increased'
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]<results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
            weird_results_before = results_before_change[weird_indices]
            weird_results_after = results_after_change[weird_indices]

        else:
            raise ValueError('option not recognised')
#       
        if option not in ['weird_decrease', 'coherence_decrease_translation',
                          'coherence_decrease_degradation',
                          'coherence_increase_degradation',
                          'coherence_increase_translation']:
            my_posterior_samples = prior_samples[accepted_indices]
            my_model_results = model_results[accepted_indices]
        else:
            my_posterior_samples = weird_parameters_before
            my_model_results = weird_results_before


#         sns.set()
        # sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        # font = {'size'   : 28}
        # plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_mrna = my_model_results[:,4]
        print('largest mrna is')
        print(np.max(all_mrna))
        sns.distplot(all_mrna,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black',
                                 'alpha' : 1},
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("mean mRNA")
#         plt.xlim(0.2,)
#         plt.ylim(0,5)
#         plt.xlim(0,20)
#         plt.ylim(0,0.8)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','zebrafish_mRNA_distribution_'+option+'.pdf'))
 
    def xest_increase_mRNA_degradation(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        option = 'mean_std_period'
        if option == 'mean_and_std':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<8000,
                                        np.logical_and(model_results[:,1]<0.15,
                                                       model_results[:,1]>0.05))))
        elif option == 'mean_std_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
        else:
            raise ValueError('option not recognised')

        my_posterior_samples = prior_samples[accepted_indices]
        old_model_results = model_results[accepted_indices]
        my_posterior_samples_changed_degradation = np.copy(my_posterior_samples)
        my_posterior_samples_changed_degradation[:,5]*=1.5
        new_model_results = hes5.calculate_summary_statistics_at_parameters( my_posterior_samples_changed_degradation, 
                                                                        number_of_traces_per_sample=200 )
        old_lengthscales = hes5.calculate_fluctuation_rates_at_parameters(my_posterior_samples, sampling_duration = 12*60) 
        new_lengthscales = hes5.calculate_fluctuation_rates_at_parameters(my_posterior_samples_changed_degradation, sampling_duration = 12*60)  

        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_increased_degradation')

        np.save(saving_path + '.npy', new_model_results)
        np.save(saving_path + '_parameters.npy', my_posterior_samples_changed_degradation )
        np.save(saving_path + '_old.npy', old_model_results)
        np.save(saving_path + '_parameters_old.npy', my_posterior_samples )
        np.save(saving_path + '_old_lengthscales.npy', old_lengthscales)
        np.save(saving_path + '_new_lengthscales.npy', new_lengthscales)
        
    def xest_b_decrease_mRNA_degradation(self):
        print('changing mrna degradation')
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        option = 'mean_std_period'
        if option == 'mean_and_std':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                                       model_results[:,1]>0.05))))
        elif option == 'mean_std_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
        else:
            raise ValueError('option not recognised')

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples')
        print(len(my_posterior_samples))
        old_model_results = model_results[accepted_indices]
        my_posterior_samples_changed_degradation = np.copy(my_posterior_samples)
        my_posterior_samples_changed_degradation[:,5]*=0.5
        new_model_results = hes5.calculate_summary_statistics_at_parameters( my_posterior_samples_changed_degradation, 
                                                                        number_of_traces_per_sample=200 )
#         old_lengthscales = hes5.calculate_fluctuation_rates_at_parameters(my_posterior_samples, sampling_duration = 12*60) 
#         new_lengthscales = hes5.calculate_fluctuation_rates_at_parameters(my_posterior_samples_changed_degradation, sampling_duration = 12*60)  

        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_decreased_degradation')

        np.save(saving_path + '.npy', new_model_results)
        np.save(saving_path + '_parameters.npy', my_posterior_samples_changed_degradation )
        np.save(saving_path + '_old.npy', old_model_results)
        np.save(saving_path + '_parameters_old.npy', my_posterior_samples )
#         np.save(saving_path + '_old_lengthscales.npy', old_lengthscales)
#         np.save(saving_path + '_new_lengthscales.npy', new_lengthscales)
 
    def xest_c_increase_translation(self):
        print('changing translation')
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        option = 'mean_std_period'
        if option == 'mean_and_std':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                                       model_results[:,1]>0.05))))
        elif option == 'mean_std_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
        else:
            raise ValueError('option not recognised')

        my_posterior_samples = prior_samples[accepted_indices]
        old_model_results = model_results[accepted_indices]
        my_posterior_samples_changed_translation = np.copy(my_posterior_samples)
        my_posterior_samples_changed_translation[:,1]*=1.5

        new_model_results = hes5.calculate_summary_statistics_at_parameters( my_posterior_samples_changed_translation, 
                                                                        number_of_traces_per_sample=200 )
#         old_lengthscales = hes5.calculate_fluctuation_rates_at_parameters(my_posterior_samples, sampling_duration = 12*60) 
#         print('got here too')
#         new_lengthscales = hes5.calculate_fluctuation_rates_at_parameters(my_posterior_samples_changed_translation, sampling_duration = 12*60)  
#         print('got here again')

        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_increased_translation')

        np.save(saving_path + '.npy', new_model_results)
        np.save(saving_path + '_parameters.npy', my_posterior_samples_changed_translation )
        np.save(saving_path + '_old.npy', old_model_results)
        np.save(saving_path + '_parameters_old.npy', my_posterior_samples )
#         np.save(saving_path + '_old_lengthscales.npy', old_lengthscales)
#         np.save(saving_path + '_new_lengthscales.npy', new_lengthscales)
 
    def xest_plot_mrna_change_results(self):
        
        change = 'increased'
#         change = 'decreased'

#         plot_option = 'boxplot'
        plot_option = 'lines'
        
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,2]<150)))))
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]
        
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_complete_matrix.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_all.npy'))
        dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_shifted_more.npy'))

        translation_changes = dual_sweep_results[0,0,:,1]
        degradation_changes = dual_sweep_results[0,:,0,0]
        fluctuation_rates_before = dual_sweep_results[:,9,9,-1]

        total_condition_mask = np.zeros(len(dual_sweep_results))
        list_of_indices = []
        corresponding_proportions = []
        periods_before = []
        periods_after = []
        results_before_change = []
        results_after_change = []
        parameters_before = []
        parameters_after = []
        for translation_index, translation_change in enumerate(translation_changes):
            for degradation_index, degradation_change in enumerate(degradation_changes):
                these_results_after = dual_sweep_results[:, 
                                                         degradation_index, 
                                                         translation_index, 
                                                         :]

                relative_noise_after = ( these_results_after[:,-1]/np.power(these_results_after[:,3]*
                                         these_results_after[:,2],2))
                relative_noise_before = ( my_posterior_results[:,-1]/np.power(my_posterior_results[:,1]*
                                          my_posterior_results[:,0],2))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                                 these_results_after[:,2]>my_posterior_results[:,0]*1.8)
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                                 these_results_after[:,5]<my_posterior_results[:,3]))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]>0.1,
#                                                 these_results_after[:,5]<my_posterior_results[:,3])))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2] >my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5] <my_posterior_results[:,3],
#                                                 these_results_after[:,4] <150)))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2] >my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5] <my_posterior_results[:,3],
#                                                 relative_noise_after>relative_noise_before)))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                 np.logical_and(relative_noise_after>relative_noise_before,
#                                                 these_results_after[:,4]<150))))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.5,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.5,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                                 these_fluctuation_rates_after[:,2]>fluctuation_rates_before)))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                                 these_results_after[:,-1]<fluctuation_rates_before)))
#                 condition_mask = np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                                 these_results_after[:,-1]>fluctuation_rates_before)
                condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
                                np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
                                np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
                                np.logical_and(these_results_after[:,4]<150,
                                                these_results_after[:,-1]>fluctuation_rates_before))))
#                                                 these_fluctuation_rates_after[:,2]>fluctuation_rates_before))))
#                 condition_mask = these_fluctuation_rates_after[:,2]>fluctuation_rates_before
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                 np.logical_and(these_results_after[:,4]<150,
#                                                 these_results_after[:,-1]>fluctuation_rates_before))))
                
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                  np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                  np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                  np.logical_and(relative_noise_after>relative_noise_before,
#                                                 these_results_after[:,4]<150))))

                these_indices = np.where(condition_mask)[0]
                if len(these_indices>0):
                    for item in these_indices:
                        list_of_indices.append(item)
                        corresponding_proportions.append((degradation_change, translation_change))
                        periods_before.append(my_posterior_results[item,2])
                        periods_after.append(these_results_after[item,4])
                        these_parameters_before = my_posterior_samples[item]
                        these_parameters_after = my_posterior_samples[item].copy()
                        these_parameters_after[5]*=these_results_after[item,0]
                        these_parameters_after[1]*=these_results_after[item,1]
                        parameters_before.append(these_parameters_before)
                        parameters_after.append(these_parameters_after)
                        these_specific_results_before = my_posterior_results[item]
                        these_specific_results_after = these_results_after[item,2:]
                        results_before_change.append(these_specific_results_before)
                        results_after_change.append(these_specific_results_after)
 
        results_before_change = np.array(results_before_change)
        results_after_change = np.array(results_after_change)
        parameters_before_change = np.array(parameters_before)
        parameters_after_change = np.array(parameters_after)

#         saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradationtest')
#         saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
#         saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
#         saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation_repeated')
#         saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation_repeated')
#         results_after_change = np.load(saving_path + '.npy')
#         parameters_after_change = np.load(saving_path + '_parameters.npy')
#         results_before_change = np.load(saving_path + '_old.npy')
#         parameters_before_change = np.load(saving_path + '_parameters_old.npy')
#         old_lengthscales = np.zeros(len(parameters_before_change))
#         new_lengthscales = np.zeros(len(parameters_before_change))
#         old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
#         new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        old_lengthscales = results_before_change[-1]
        new_lengthscales = results_after_change[-1]
    
        if False:
            indices = np.where(results_before_change[:,3]>results_after_change[:,3])
            results_after_change = results_after_change[indices]
            parameters_after_change = parameters_after_change[indices]
            results_before_change = results_before_change[indices]
            parameters_before_change = parameters_before_change[indices]
            old_lengthscales = old_lengthscales[indices]
            new_lengthscales = new_lengthscales[indices]
            print('number of remaining parameters')
            print(len(old_lengthscales))
            print('number of increasing lengthscales within there')
            print(np.sum(new_lengthscales>old_lengthscales))
            print(old_lengthscales)
            print(new_lengthscales)
 
        this_figure, axes = plt.subplots(3,3,figsize = (6.5,6.5))

        ## DEGRADATION
        this_axes = axes[0,0]
        if plot_option == 'boxplot':
            this_data_frame = pd.DataFrame(np.column_stack((parameters_before_change[:,5],
                                                           parameters_after_change[:,5])),
                                            columns = ['before','after'])
            this_data_frame.boxplot(ax = axes[0,0])
        else: 
            for parameter_index in range(parameters_before_change.shape[0]):
                this_axes.plot([0,1],
                         [parameters_before_change[parameter_index,5],
                          parameters_after_change[parameter_index,5]],
                         color = 'black',
#                          alpha = 0.005)
                         alpha = 1.0)
                this_axes.set_xticks([0,1])
                this_axes.set_xticklabels(['before','after'])
        this_axes.set_ylabel('mRNA degradation')

        ## EXPRESSION
        this_axes = axes[0,1]
        if plot_option == 'boxplot':
            this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,0],
                                                            results_after_change[:,0])),
                                            columns = ['before','after'])
            this_data_frame.boxplot(ax = axes[0,1])
        else:
            this_axes.set_xticks([0,1])
            this_axes.set_xticklabels(['before','after'])
        total_count = 0
        up_count = 0
        for parameter_index in range(results_before_change.shape[0]):
            total_count+=1
            value_before = results_before_change[parameter_index,0]
            value_after = results_after_change[parameter_index,0]
            if value_before<value_after:
                up_count+=1
                this_color = 'blue'
#                 this_alpha = 0.01
                this_alpha = 1.0
                this_z = 0
            else:
                this_color = 'green'
                this_alpha = 1.0
                this_z = 1
            if plot_option == 'lines':
                this_axes.plot([0,1],
                         [results_before_change[parameter_index,0],
                          results_after_change[parameter_index,0]],
                         color = this_color,
                         alpha = this_alpha)
        this_axes.set_title(r'$P_{up}$=' + '{:.2f}'.format(up_count/total_count))
        this_axes.set_ylabel('Hes expression')

        ## STANDARD DEVIATION
        this_axes = axes[0,2]
        if plot_option == 'boxplot':
            this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,1],
                                                            results_after_change[:,1])),
                                            columns = ['before','after'])
            this_data_frame.boxplot(ax = this_axes)
        else: 
            this_axes.set_xticks([0,1])
            this_axes.set_xticklabels(['before','after'])
        total_count = 0
        up_count = 0
        for parameter_index in range(results_before_change.shape[0]):
            total_count+=1
            value_before = results_before_change[parameter_index,1]
            value_after = results_after_change[parameter_index,1]
            if value_before<value_after:
                up_count +=1
                this_color = 'blue'
#                 this_alpha = 0.01
                this_alpha = 1.0
                this_z = 0
            else:
                this_color = 'green'
#                 this_alpha = 0.1
                this_alpha = 1.0
                this_z = 1
            if plot_option == 'lines':
                this_axes.plot([0,1],
                         [results_before_change[parameter_index,1],
                          results_after_change[parameter_index,1]],
                         color = this_color,
                         alpha = this_alpha)
        this_axes.set_title(r'$P_{up}$=' + '{:.2f}'.format(up_count/total_count))
        this_axes.set_ylabel('Hes std')

        ## PERIOD
        this_axes = axes[1,0]
        if plot_option == 'boxplot':
            this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,2],
                                                            results_after_change[:,2])),
                                            columns = ['before','after'])
            this_data_frame.boxplot(ax = this_axes)
        else: 
            this_axes.set_xticks([0,1])
            this_axes.set_xticklabels(['before','after'])
        total_count = 0
        up_count = 0
        for parameter_index in range(results_before_change.shape[0]):
            total_count+=1
            value_before = results_before_change[parameter_index,2]
            value_after = results_after_change[parameter_index,2]
            if value_before<value_after:
                this_color = 'blue'
                this_alpha = 1.0
                up_count+=1
                this_z = 0
            else:
                this_color = 'green'
                this_alpha = 1.0
                this_z = 1
            if plot_option == 'lines':
                this_axes.plot([0,1],
                         [results_before_change[parameter_index,2],
                          results_after_change[parameter_index,2]],
                         color = this_color,
                         alpha = this_alpha)
        this_axes.set_title(r'$P_{up}$=' + '{:.2f}'.format(up_count/total_count))
        this_axes.set_ylabel('Period')

        ## COHERENCE
        this_axes = axes[1,1]
        if plot_option == 'boxplot':
            this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,3],
                                                            results_after_change[:,3])),
                                            columns = ['before','after'])
            this_data_frame.boxplot(ax = this_axes)
        else: 
            this_axes.set_xticks([0,1])
            this_axes.set_xticklabels(['before','after'])
        total_count = 0
        up_count = 0
        for parameter_index in range(results_before_change.shape[0]):
            total_count+=1
            value_before = results_before_change[parameter_index,3]
            value_after = results_after_change[parameter_index,3]
            if value_before<value_after:
                up_count+=1
                this_color = 'blue'
                this_alpha = 1.0
                this_z = 0
            else:
                this_color = 'green'
                this_alpha = 1.0
                this_z = 1
            if plot_option == 'lines':
                this_axes.plot([0,1],
                         [results_before_change[parameter_index,3],
                          results_after_change[parameter_index,3]],
                         color = this_color,
                         alpha = this_alpha)
        this_axes.set_title(r'$P_{up}$=' + '{:.2f}'.format(up_count/total_count))
        this_axes.set_ylabel('Coherence')

        ## MRNA
        this_axes = axes[1,2]
        if plot_option == 'boxplot':
            this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,4],
                                                            results_after_change[:,4])),
                                            columns = ['before','after'])
            this_data_frame.boxplot(ax = this_axes)
        else: 
            this_axes.set_xticks([0,1])
            this_axes.set_xticklabels(['before','after'])
        total_count = 0
        up_count = 0
        for parameter_index in range(results_before_change.shape[0]):
            total_count+=1
            value_before = results_before_change[parameter_index,4]
            value_after = results_after_change[parameter_index,4]
            if value_before<value_after:
                up_count+=1
                this_color = 'blue'
                this_alpha = 1.0
                this_z = 0
            else:
                this_color = 'green'
                this_alpha = 1.0
                this_z = 1
            if plot_option == 'lines':
                this_axes.plot([0,1],
                         [results_before_change[parameter_index,4],
                          results_after_change[parameter_index,4]],
                         color = this_color,
                         alpha = this_alpha)
        this_axes.set_title(r'$P_{up}$=' + '{:.2f}'.format(up_count/total_count))
        this_axes.set_ylabel('mRNA number')

        ## Absolute noise
        this_axes = axes[2,0]
        if plot_option == 'boxplot':
            this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,-1],
                                                            results_after_change[:,-1])),
                                            columns = ['before','after'])
            this_data_frame.boxplot(ax = this_axes)
        else: 
                this_axes.set_xticks([0,1])
                this_axes.set_xticklabels(['before','after'])
        total_count = 0
        up_count = 0
        for parameter_index in range(results_before_change.shape[0]):
            total_count+=1
            value_before = results_before_change[parameter_index,-1]
            value_after = results_after_change[parameter_index,-1]
            if value_before<value_after:
                up_count+=1
                this_color = 'blue'
                this_alpha = 1.0
                this_z = 0
            else:
                this_color = 'green'
                this_alpha = 1.0
                this_z = 1
            if plot_option == 'lines':
                this_axes.plot([0,1],
                         [results_before_change[parameter_index,-1],
                          results_after_change[parameter_index,-1]],
                         color = this_color,
                         alpha = this_alpha)
        this_axes.set_title(r'$P_{up}$=' + '{:.2f}'.format(up_count/total_count))
        this_axes.set_ylabel('Absolute noise')

        ## NOISE 
        this_axes = axes[2,1]
        if plot_option == 'boxplot':
            this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,-1]/np.power(results_before_change[:,1]*
                                                                                                 results_before_change[:,0],2),
                                                            results_after_change[:,-1]/np.power(results_after_change[:,1]*
                                                                                                results_after_change[:,0],2))),
                                            columns = ['before','after'])
            this_data_frame.boxplot(ax = this_axes)
        else: 
            this_axes.set_xticks([0,1])
            this_axes.set_xticklabels(['before','after'])
        total_count = 0
        up_count = 0
        for parameter_index in range(results_before_change.shape[0]):
            total_count+=1
            value_before = results_before_change[parameter_index,-1]/np.power(results_before_change[parameter_index,1]*
                                                                                             results_before_change[parameter_index,0],2)
            value_after = results_after_change[parameter_index,-1]/np.power(results_after_change[parameter_index,1]*
                                                                                            results_after_change[parameter_index,0],2)
            if value_before<value_after:
                up_count+=1
                this_color = 'blue'
                this_alpha = 1.0
                this_z = 0
            else:
                this_color = 'green'
                this_alpha = 1.0
                this_z = 1
            if plot_option == 'lines':
                this_axes.plot([0,1],
                         [results_before_change[parameter_index,-1]/np.power(results_before_change[parameter_index,1]*
                                                                                                 results_before_change[parameter_index,0],2),
                          results_after_change[parameter_index,-1]/np.power(results_after_change[parameter_index,1]*
                                                                                                results_after_change[parameter_index,0],2)],
                         color = this_color,
                         alpha = this_alpha)
        this_axes.set_title(r'$P_{up}$=' + '{:.2f}'.format(up_count/total_count))
        this_axes.set_ylabel('Noise proportion')
        
        ## LENGTHSCALES
        this_axes = axes[2,2]
        if plot_option == 'boxplot':
            this_data_frame = pd.DataFrame(np.column_stack((old_lengthscales,
                                                            new_lengthscales)),
                                            columns = ['before','after'])
            this_data_frame.boxplot(ax = this_axes)
        else: 
            this_axes.set_xticks([0,1])
            this_axes.set_xticklabels(['before','after'])
        total_count = 0
        up_count = 0
        for parameter_index in range(results_before_change.shape[0]):
            total_count+=1
            value_before = old_lengthscales[parameter_index]
            value_after = new_lengthscales[parameter_index] 
            if value_before<value_after:
                up_count+=1
                this_color = 'blue'
                this_alpha = 1.0
                this_z = 0
            else:
                this_color = 'green'
                this_alpha = 1.0
                this_z = 1
            if plot_option == 'lines':
                this_axes.plot([0,1],
                         [old_lengthscales[parameter_index],
                          new_lengthscales[parameter_index]],
                         color = this_color,
                         alpha = this_alpha)
        this_axes.set_title(r'$P_{up}$=' + '{:.2f}'.format(up_count/total_count))
        this_axes.set_ylabel('Fluctuation rates')
 
        plt.tight_layout()
#         plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation_repeated.pdf'))
#         plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation_' + plot_option +'.pdf'))
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_fit_' + change + '_translation_' + plot_option +'.pdf'))
#         plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation_' + plot_option +'.pdf'))
        
    def xest_investigate_mrna_and_expression_decrease(self):

        change = 'decreased'
        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradationtest')
        results_after_change = np.load(saving_path + '.npy')
        parameters_after_change = np.load(saving_path + '_parameters.npy')
        results_before_change = np.load(saving_path + '_old.npy')
        parameters_before_change = np.load(saving_path + '_parameters_old.npy')
        old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
        new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
#         weirdest_index = np.argmax(results_before_change[:,1]-results_after_change[:,1])

        weiredest_index = np.argmax(results_before_change[:,-1]/np.power(results_before_change[:,1]*
                                                                         results_before_change[:,0],2) -
                                    results_after_change[:,-1]/np.power(results_after_change[:,1]*
                                                                        results_after_change[:,0],2))
        weird_parameter_before = parameters_before_change[weirdest_index]
        weird_parameter_after = parameters_after_change[weirdest_index]
        
        trace_before = hes5.generate_langevin_trajectory(720.0,
                                                         weird_parameter_before[2], #repression_threshold, 
                                                         weird_parameter_before[4], #hill_coefficient,
                                                         weird_parameter_before[5], #mRNA_degradation_rate, 
                                                         weird_parameter_before[6], #protein_degradation_rate, 
                                                         weird_parameter_before[0], #basal_transcription_rate, 
                                                         weird_parameter_before[1], #translation_rate,
                                                         weird_parameter_before[3], #transcription_delay, 
                                                         10, #initial_mRNA, 
                                                         weird_parameter_before[2], #initial_protein,
                                                         2000)

        trace_after = hes5.generate_langevin_trajectory(720.0,
                                                         weird_parameter_after[2], #repression_threshold, 
                                                         weird_parameter_after[4], #hill_coefficient,
                                                         weird_parameter_after[5], #mRNA_degradation_rate, 
                                                         weird_parameter_after[6], #protein_degradation_rate, 
                                                         weird_parameter_after[0], #basal_transcription_rate, 
                                                         weird_parameter_after[1], #translation_rate,
                                                         weird_parameter_after[3], #transcription_delay, 
                                                         10, #initial_mRNA, 
                                                         weird_parameter_after[2], #initial_protein,
                                                         2000)

        

        plt.figure(figsize = (6.5, 4.5))
        plt.subplot(221)
        plt.plot(trace_before[:,0],
                 trace_before[:,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.ylim(3000,15000)
        plt.title('Before')
        plt.subplot(222)
        plt.plot(trace_after[:,0],
                 trace_after[:,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.ylim(3000,15000)
        plt.title('After')
        plt.subplot(223)
        plt.plot(trace_before[:,0],
                 trace_before[:,1])
        plt.ylabel('Hes mRNA')
        plt.xlabel('Time')
        plt.ylim(0,120)
        plt.subplot(224)
        plt.plot(trace_after[:,0],
                 trace_after[:,1])
        plt.ylabel('Hes mRNA')
        plt.xlabel('Time')
        plt.ylim(0,120)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_degradation_weird_examples.pdf'))

    def xest_plot_mRNA_change_examples(self):
#         change = 'decreased'
        change = 'increased'
#         saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation_repeated')
        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation_repeated')
        results_after_change = np.load(saving_path + '.npy')
        parameters_after_change = np.load(saving_path + '_parameters.npy')
        results_before_change = np.load(saving_path + '_old.npy')
        parameters_before_change = np.load(saving_path + '_parameters_old.npy')
        old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
        new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
        weirdest_index = np.argmax(results_after_change[:,-1]-results_before_change[:,-1])
#         weirdest_index = np.argmax(results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                         results_after_change[:,0],2) -
#                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                         results_before_change[:,0],2))
#         weirdest_index = np.argmax(new_lengthscales-old_lengthscales)
        example_parameter_before = parameters_before_change[weirdest_index]
        example_parameter_after = parameters_after_change[weirdest_index]
    
#         example_parameter_index = 0
#         example_parameter_before = parameters_before_change[example_parameter_index]
#         example_parameter_after = parameters_after_change[example_parameter_index]
        
        example_trace_before = hes5.generate_langevin_trajectory( 720, #duration 
                                                                  example_parameter_before[2], #repression_threshold, 
                                                                  example_parameter_before[4], #hill_coefficient,
                                                                  example_parameter_before[5], #mRNA_degradation_rate, 
                                                                  example_parameter_before[6], #protein_degradation_rate, 
                                                                  example_parameter_before[0], #basal_transcription_rate, 
                                                                  example_parameter_before[1], #translation_rate,
                                                                  example_parameter_before[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                  example_parameter_before[2], #initial_protein,
                                                                  2000)

        example_trace_after = hes5.generate_langevin_trajectory( 720, #duration 
                                                                  example_parameter_after[2], #repression_threshold, 
                                                                  example_parameter_after[4], #hill_coefficient,
                                                                  example_parameter_after[5], #mRNA_degradation_rate, 
                                                                  example_parameter_after[6], #protein_degradation_rate, 
                                                                  example_parameter_after[0], #basal_transcription_rate, 
                                                                  example_parameter_after[1], #translation_rate,
                                                                  example_parameter_after[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                  example_parameter_after[2], #initial_protein,
                                                                  2000)

        plt.figure(figsize = (6.5, 2.5))
        plt.subplot(121)
        plt.plot(example_trace_before[::6,0],
                 example_trace_before[::6,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.subplot(122)
        plt.plot(example_trace_after[::6,0],
                 example_trace_after[::6,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation_examples.pdf'))
#         plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation_examples.pdf'))

    def xest_plot_dual_sweep_change_examples(self):
        
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_large')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,2]<150)))))
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]
        
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_complete_matrix.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_all.npy'))
        dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_shifted_more.npy'))
        translation_changes = dual_sweep_results[0,0,:,1]
        degradation_changes = dual_sweep_results[0,:,0,0]
        fluctuation_rates_before = dual_sweep_results[:,9,9,-1]

        total_condition_mask = np.zeros(len(dual_sweep_results))
        list_of_indices = []
        corresponding_proportions = []
        periods_before = []
        periods_after = []
        for translation_index, translation_change in enumerate(translation_changes):
            for degradation_index, degradation_change in enumerate(degradation_changes):
                these_results_after = dual_sweep_results[:, 
                                                         degradation_index, 
                                                         translation_index, 
                                                         :]

                relative_noise_after = ( these_results_after[:,-1]/np.power(these_results_after[:,3]*
                                         these_results_after[:,2],2))
                relative_noise_before = ( my_posterior_results[:,-1]/np.power(my_posterior_results[:,1]*
                                          my_posterior_results[:,0],2))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                                 these_results_after[:,2]>my_posterior_results[:,0]*1.8)
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                                 these_results_after[:,5]<my_posterior_results[:,3]))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]>0.1,
#                                                 these_results_after[:,5]<my_posterior_results[:,3])))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2] >my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5] <my_posterior_results[:,3],
#                                                 these_results_after[:,4] <150)))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2] >my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5] <my_posterior_results[:,3],
#                                                 relative_noise_after>relative_noise_before)))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                 np.logical_and(relative_noise_after>relative_noise_before,
#                                                 these_results_after[:,4]<150))))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.5,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.5,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                                 these_fluctuation_rates_after[:,2]>fluctuation_rates_before)))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                                 these_results_after[:,-1]<fluctuation_rates_before)))
#                 condition_mask = np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                                 these_results_after[:,-1]>fluctuation_rates_before)
                condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
                                np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
                                np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
                                np.logical_and(these_results_after[:,4]<150,
                                                these_results_after[:,-1]>fluctuation_rates_before))))
#                                                 these_fluctuation_rates_after[:,2]>fluctuation_rates_before))))
#                 condition_mask = these_fluctuation_rates_after[:,2]>fluctuation_rates_before
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                 np.logical_and(these_results_after[:,4]<150,
#                                                 these_results_after[:,-1]>fluctuation_rates_before))))
                
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                  np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                  np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                  np.logical_and(relative_noise_after>relative_noise_before,
#                                                 these_results_after[:,4]<150))))

                these_indices = np.where(condition_mask)[0]
                if len(these_indices>0):
                    for item in these_indices:
                        list_of_indices.append(item)
                        corresponding_proportions.append((degradation_change, translation_change))
                        periods_before.append(my_posterior_results[item,2])
                        periods_after.append(these_results_after[item,4])
 
        print(list_of_indices)
        print(corresponding_proportions)
        reference_index = 5
        example_index = list_of_indices[reference_index]
        example_parameter_before = my_posterior_samples[example_index]
        example_parameter_after = np.copy(example_parameter_before)
        example_parameter_after[5]*=corresponding_proportions[reference_index][0]
        example_parameter_after[1]*=corresponding_proportions[reference_index][1]

        print(example_parameter_before)
        print(example_parameter_after)
        print(periods_before)
        print(periods_after)
        example_trace_before = hes5.generate_langevin_trajectory( 720, #duration 
                                                                  example_parameter_before[2], #repression_threshold, 
                                                                  example_parameter_before[4], #hill_coefficient,
                                                                  example_parameter_before[5], #mRNA_degradation_rate, 
                                                                  example_parameter_before[6], #protein_degradation_rate, 
                                                                  example_parameter_before[0], #basal_transcription_rate, 
                                                                  example_parameter_before[1], #translation_rate,
                                                                  example_parameter_before[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                  example_parameter_before[2], #initial_protein,
                                                                  2000)

        example_trace_after = hes5.generate_langevin_trajectory( 720, #duration 
                                                                  example_parameter_after[2], #repression_threshold, 
                                                                  example_parameter_after[4], #hill_coefficient,
                                                                  example_parameter_after[5], #mRNA_degradation_rate, 
                                                                  example_parameter_after[6], #protein_degradation_rate, 
                                                                  example_parameter_after[0], #basal_transcription_rate, 
                                                                  example_parameter_after[1], #translation_rate,
                                                                  example_parameter_after[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                  example_parameter_after[2], #initial_protein,
                                                                  2000)

        plt.figure(figsize = (6.5, 2.5))
        plt.subplot(121)
        plt.title('Control')
        plt.plot(example_trace_before[::6,0],
                 example_trace_before[::6,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.subplot(122)
        plt.title('MBS')
        plt.plot(example_trace_after[::6,0],
                 example_trace_after[::6,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','dual_change_example_extrinsic_noise.pdf'))
#         plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation_examples.pdf'))

    ### PERIOD PREDICTION
        period_pairs = np.zeros((len(list_of_indices),2))
        for reference_index, period_before in enumerate(periods_before):
            period_pairs[reference_index,0] = period_before
            period_pairs[reference_index,1] = periods_after[reference_index]

        plt.figure(figsize = (4.5, 2.5))
        for pair in period_pairs:
            plt.plot([0,1],pair, color = 'blue', marker = '.')
        plt.xlim(-0.5,1.5)
        plt.xticks([0,1], ['Control','MBS'])
        plt.ylabel('Period [min]')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','dual_change_period_prediction_extrinsic_noise.pdf'))

    def xest_plot_fluctuation_rate_change_examples(self):
#         change = 'decreased'
        change = 'increased'
#         saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation_repeated')
#         saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation_repeated')
        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
        results_after_change = np.load(saving_path + '.npy')
        parameters_after_change = np.load(saving_path + '_parameters.npy')
        results_before_change = np.load(saving_path + '_old.npy')
        parameters_before_change = np.load(saving_path + '_parameters_old.npy')
        old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
        new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
        weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#         weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                 results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                      results_before_change[:,0],2)<
#                                                 results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                     results_after_change[:,0],2)))
#  
#                                                 old_lengthscales<new_lengthscales))
        weird_parameters_before = parameters_before_change[weird_indices]
        weird_parameters_after = parameters_after_change[weird_indices]
        
        weird_old_lengthscales = old_lengthscales[weird_indices]
        weird_new_lengthscales = new_lengthscales[weird_indices]
        weird_results_before = results_before_change[weird_indices]
        weird_results_after = results_after_change[weird_indices]

#         weirdest_index = np.argmax(weird_results_after[:,-1]-weird_results_before[:,-1])
#         weirdest_index = np.argmax(results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                         results_after_change[:,0],2) -
#                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                         results_before_change[:,0],2))
        weirdest_index = np.argmax(weird_new_lengthscales-weird_old_lengthscales)
        example_parameter_before = weird_parameters_before[weirdest_index]
        example_parameter_after = weird_parameters_after[weirdest_index]
    
#         example_parameter_index = 0
#         example_parameter_before = parameters_before_change[example_parameter_index]
#         example_parameter_after = parameters_after_change[example_parameter_index]
        
        example_trace_before = hes5.generate_langevin_trajectory( 720, #duration 
                                                                  example_parameter_before[2], #repression_threshold, 
                                                                  example_parameter_before[4], #hill_coefficient,
                                                                  example_parameter_before[5], #mRNA_degradation_rate, 
                                                                  example_parameter_before[6], #protein_degradation_rate, 
                                                                  example_parameter_before[0], #basal_transcription_rate, 
                                                                  example_parameter_before[1], #translation_rate,
                                                                  example_parameter_before[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                  example_parameter_before[2], #initial_protein,
                                                                  2000)

        example_trace_after = hes5.generate_langevin_trajectory( 720, #duration 
                                                                  example_parameter_after[2], #repression_threshold, 
                                                                  example_parameter_after[4], #hill_coefficient,
                                                                  example_parameter_after[5], #mRNA_degradation_rate, 
                                                                  example_parameter_after[6], #protein_degradation_rate, 
                                                                  example_parameter_after[0], #basal_transcription_rate, 
                                                                  example_parameter_after[1], #translation_rate,
                                                                  example_parameter_after[3], #transcription_delay, 
                                                                  10, #initial_mRNA, 
                                                                  example_parameter_after[2], #initial_protein,
                                                                  2000)

        plt.figure(figsize = (6.5, 2.5))
        plt.subplot(121)
        plt.title('Wildtype')
        plt.plot(example_trace_before[::6,0],
                 example_trace_before[::6,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.subplot(122)
        plt.title('MBS (translation)')
        plt.plot(example_trace_after[::6,0],
                 example_trace_after[::6,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation_decreased_coherence_examples.pdf'))
#         plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation_examples.pdf'))

    def xest_plot_smfish_results(self):
        root = os.path.join('/home','y91198jk','hdd','smfishdata','231118deconvoluted','Experiment_2','repeat_1')
        items_in_root = os.listdir(root)
        view_folders = [ item for item in items_in_root if item.startswith('view')]
        data_files = dict()

        ## FIND RESULTS FILES
        for view in view_folders:
            this_results_folder = os.path.join(root,view,'fish_quant_results')
            these_result_items = os.listdir(this_results_folder)
            for item in these_result_items:
                if item.startswith('__FQ_batch_summary_MATURE_'):
                    this_results_file_name = item
                    break
            this_results_file = os.path.join(this_results_folder, this_results_file_name)
            data_files[view] = this_results_file
        print(data_files)
        
        ## READ DATA
        nuclear_dots = dict()
        total_dots = dict()
        cytoplasmic_dots = dict()
        for view in data_files:
            these_data = pd.read_csv(data_files[view], header = 4, sep = '\t') 
#                                         usecols = ['N_thresh_Total', 'N_thresh_Nuc'])
            total_number_of_dots = these_data['N_thres_Total']
            number_dots_in_nucleus = these_data['N_thres_Nuc']
            number_dots_in_cytoplasm = total_number_of_dots - number_dots_in_nucleus
            nuclear_dots[view] = number_dots_in_nucleus
            total_dots[view] = total_number_of_dots
            cytoplasmic_dots[view] = number_dots_in_cytoplasm
            
        ## Calculate totals
        total_number_of_dots = []
        total_cytoplasmic_number_of_dots = []
        total_nuclear_number_of_dots = []
        for view in data_files:
            total_number_of_dots += total_dots[view].tolist()
            total_cytoplasmic_number_of_dots += cytoplasmic_dots[view].tolist()
            total_nuclear_number_of_dots += nuclear_dots[view].tolist()
            
        nuclear_dots['total'] = total_nuclear_number_of_dots
        total_dots['total'] = total_number_of_dots
        cytoplasmic_dots['total'] = total_cytoplasmic_number_of_dots
        
        maximimal_number_of_dots = np.max(total_number_of_dots)

        ## PLOT DATA
        types_of_plot = {'total':total_dots,
                         'nuclear':nuclear_dots,
                         'cytoplasmic':cytoplasmic_dots}
        for view in total_dots:
            for type in types_of_plot:
                plt.figure(figsize = (4.5,2.5))
                these_dots = types_of_plot[type][view]
#                 these_dots = [number for number in these_dots if number != 0]
                plt.hist(these_dots,bins=100, range = [0,maximimal_number_of_dots])
                plt.axvline(np.median(these_dots), lw=1, color = 'black')
                plt.axvline(np.mean(these_dots), lw=1, color = 'black')
                plt.ylabel('occurrence')
                plt.xlabel('# ' + type + ' mRNA')
                plt.title('Mean: ' + '{:.2f}'.format(np.mean(these_dots)) + 
                          ', Median: ' + '{:.2f}'.format(np.median(these_dots)) +
                          ', #cells: ' + str(len(these_dots)))
                plt.tight_layout()
                plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                         'smfish_' + view + '_' + type + '.pdf'))

    def xest_play_with_Gaussian_Processes(self):
        
        np.random.seed(1)
        
        
        # the function to predict
        f = lambda x : x*np.sin(x)
        
        # ----------------------------------------------------------------------
        #  First the noiseless case
        X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
        print(X)
        
        # Observations
        y = f(X).ravel()
        print(y)
        
        # Mesh the input space for evaluations of the real function, the prediction and
        # its MSE
        x = np.atleast_2d(np.linspace(0, 10, 1000)).T
        
        # Instantiate a Gaussian Process model
        my_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        my_gp_regressor = GaussianProcessRegressor(kernel=my_kernel, n_restarts_optimizer=9)
        
        # Fit to data using Maximum Likelihood Estimation of the parameters
        my_gp_regressor.fit(X, y)
        
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred, sigma = my_gp_regressor.predict(x, return_std=True)
        
        # Plot the function, the prediction and the 95% confidence interval based on
        # the MSE
        plt.figure()
        plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
        plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
        plt.plot(x, y_pred, 'b-', label=u'Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma,
                                (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, 20)
        plt.legend(loc='upper left')
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'gp_example_1.pdf'))
        
        # ----------------------------------------------------------------------
        # now the noisy case
        X = np.linspace(0.1, 9.9, 20)
        X = np.atleast_2d(X).T
        
        # Observations and noise
        y = f(X).ravel()
        dy = 0.5 + 1.0 * np.random.random(y.shape)
        noise = np.random.normal(0, dy)
        y += noise
        
        # Instantiate a Gaussian Process model
        my_gp_regressor = GaussianProcessRegressor(kernel=my_kernel, alpha=dy ** 2,
                                      n_restarts_optimizer=10)
        
        # Fit to data using Maximum Likelihood Estimation of the parameters
        my_gp_regressor.fit(X, y)
        
        # Make the prediction on the meshed x-axis (ask for MSE as well)
        y_pred, sigma = my_gp_regressor.predict(x, return_std=True)
        
        # Plot the function, the prediction and the 95% confidence interval based on
        # the MSE
        plt.figure()
        plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
        plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
        plt.plot(x, y_pred, 'b-', label=u'Prediction')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y_pred - 1.9600 * sigma,
                                (y_pred + 1.9600 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, 20)
        plt.legend(loc='upper left')
        
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'gp_example_2.pdf'))
        
    def xest_try_OU_process_for_lengthscale(self):
        #generate a trace
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples = prior_samples[accepted_indices]

        example_parameter_index = 100
        example_parameter = my_posterior_samples[example_parameter_index]
        
        example_trace = hes5.generate_langevin_trajectory( 720, #duration 
                                                                  example_parameter[2], #repression_threshold, 
                                                                  example_parameter[4], #hill_coefficient,
                                                                  example_parameter[5], #mRNA_degradation_rate, 
                                                                  example_parameter[6], #protein_degradation_rate, 
                                                                  example_parameter[0], #basal_transcription_rate, 
                                                                  example_parameter[1], #translation_rate,
                                                                  example_parameter[3], #transcription_delay, 
                                                                  10, #initial_mRNA
                                                                  example_parameter[2], #initial_protein,
                                                                  2000)
        
        times = example_trace[:,0]
        times = times[:,np.newaxis]
        times = times/60.0
        protein_trace = example_trace[:,2] - np.mean(example_trace[:,2])
        
        ornstein_kernel = ConstantKernel(1.0, (1e-3, 1e3))*Matern(length_scale=0.1, length_scale_bounds=(1e-03, 100), nu=0.5)
        my_gp_regressor = GaussianProcessRegressor(kernel=ornstein_kernel, n_restarts_optimizer=10)
        my_fit = my_gp_regressor.fit(times, protein_trace)
        import pdb; pdb.set_trace()
        print(my_fit)
        print(my_fit.kernel_)
        my_parameters = my_gp_regressor.get_params()
        print(my_parameters)
        print(my_parameters['kernel__k2__length_scale'])
    
        protein_predicted, sigma_predicted = my_gp_regressor.predict(times, return_std=True)

        plt.figure(figsize = (6.5, 2.5))
        plt.plot(example_trace[:,0],
                 example_trace[:,2])
        plt.plot(example_trace[:,0],
                 protein_predicted,
                 )
        plt.fill(np.concatenate([times, times[::-1]])*60,
                 np.concatenate([protein_predicted - 1.9600 * sigma_predicted,
                                (protein_predicted + 1.9600 * sigma_predicted)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','gp_example.pdf'))

    def xest_do_gpy_example(self):
        import GPy
        X = np.random.uniform(-3.,3.,(20,1))
        Y = np.sin(X) + np.random.randn(20,1)*0.05

        kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        m = GPy.models.GPRegression(X,Y,kernel)
        print(m)
        fig = m.plot()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','gpy_example.pdf'))

        m.optimize(messages=True)
        m.optimize_restarts(num_restarts = 10)
        fig = m.plot()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','gpy_example2.pdf'))

    def xest_do_gpflow_example(self):
        # websites for this include: https://blog.dominodatalab.com/fitting-gaussian-process-models-python/
        # and https://gpflow.readthedocs.io/en/master/notebooks/regression.html
        N = 12
        X = np.random.rand(N,1)
        Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*0.1 + 3
        plt.figure()
        plt.plot(X, Y, 'kx', mew=2)
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','gpflow_example.pdf'))

        k = gpflow.kernels.Matern52(1, lengthscales=0.3)
        m = gpflow.models.GPR(X, Y, kern=k)
        m.likelihood.variance = 0.01

        xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)

        plt.figure()
        mean, var = m.predict_y(xx)
        plt.figure(figsize=(12, 6))
        plt.plot(X, Y, 'kx', mew=2)
        plt.plot(xx, mean, 'C0', lw=2)
        plt.fill_between(xx[:,0],
                     mean[:,0] - 2*np.sqrt(var[:,0]),
                     mean[:,0] + 2*np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)
        plt.xlim(-0.1, 1.1)
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','gpflow_example1.pdf'))

        k = gpflow.kernels.Matern52(1, lengthscales=0.3)
        meanf = gpflow.mean_functions.Linear(1.0, 0.0)
        m = gpflow.models.GPR(X, Y, k, meanf)
        m.likelihood.variance = 0.01
        print(m.as_pandas_table())
        gpflow.train.ScipyOptimizer().minimize(m)
        plt.figure()
        mean, var = m.predict_y(xx)
        plt.figure(figsize=(12, 6))
        plt.plot(X, Y, 'kx', mew=2)
        plt.plot(xx, mean, 'C0', lw=2)
        plt.fill_between(xx[:,0],
                     mean[:,0] - 2*np.sqrt(var[:,0]),
                     mean[:,0] + 2*np.sqrt(var[:,0]),
                     color='C0', alpha=0.2)
        plt.xlim(-0.1, 1.1)
        print(m.as_pandas_table())
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','gpflow_example2.pdf'))

    def xest_try_OU_process_for_lengthscale_with_gpflow(self):
        #generate a trace
        # trying to simulate what covOU does in here:
        # http://www.gaussianprocess.org/gpml/code/matlab/cov/
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples = prior_samples[accepted_indices]

        example_parameter_index = 100
        example_parameter = my_posterior_samples[example_parameter_index]
        
        example_trace = hes5.generate_langevin_trajectory( 720, #duration 
                                                                  example_parameter[2], #repression_threshold, 
                                                                  example_parameter[4], #hill_coefficient,
                                                                  example_parameter[5], #mRNA_degradation_rate, 
                                                                  example_parameter[6], #protein_degradation_rate, 
                                                                  example_parameter[0], #basal_transcription_rate, 
                                                                  example_parameter[1], #translation_rate,
                                                                  example_parameter[3], #transcription_delay, 
                                                                  10, #initial_mRNA
                                                                  example_parameter[2], #initial_protein,
                                                                  2000)
        
        times = example_trace[:,0]
        times = times[:,np.newaxis]
        times = times/60.0
        protein_trace = example_trace[:,2] - np.mean(example_trace[:,2])
        protein_trace = protein_trace[:,np.newaxis]
        
#         ornstein_kernel = ConstantKernel(1.0, (1e-3, 1e3))*Matern(length_scale=0.1, length_scale_bounds=(1e-03, 100), nu=0.5)
        ornstein_kernel = gpflow.kernels.Matern12(1, lengthscales=0.3)

        my_regression_model = gpflow.models.GPR(times, protein_trace, kern=ornstein_kernel)
        
        gpflow.train.ScipyOptimizer().minimize(my_regression_model)
        print(my_regression_model.as_pandas_table())
#         import pdb; pdb.set_trace()
        regression_values = my_regression_model.kern.read_values()
        this_lengthscale = regression_values['GPR/kern/lengthscales']
        print('this lengthscale is')
        print(this_lengthscale)
    
#         protein_predicted, sigma_predicted = my_gp_regressor.predict(times, return_std=True)
        predicted_mean, predicted_variance = my_regression_model.predict_y(times)
        plt.figure(figsize = (6.5, 2.5))
        plt.plot(example_trace[:,0],
                 example_trace[:,2])
        plt.plot(example_trace[:,0],
                 predicted_mean,
                 )
        plt.fill_between(times[:,0]*60,
                         predicted_mean[:,0] - 1.9600 * np.sqrt(predicted_variance)[:,0],
                         predicted_mean[:,0] + 1.9600 * np.sqrt(predicted_variance)[:,0],
                 alpha=.5, color='blue', label='95% confidence interval')
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','gp_example_gpflow.pdf'))
        
    def xest_get_OU_lengthscale_from_single_trace(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples = prior_samples[accepted_indices]

        example_parameter_index = 100
        example_parameter = my_posterior_samples[example_parameter_index]
        
        example_trace = hes5.generate_langevin_trajectory( 720, #duration 
                                                           example_parameter[2], #repression_threshold, 
                                                           example_parameter[4], #hill_coefficient,
                                                           example_parameter[5], #mRNA_degradation_rate, 
                                                           example_parameter[6], #protein_degradation_rate, 
                                                           example_parameter[0], #basal_transcription_rate, 
                                                           example_parameter[1], #translation_rate,
                                                           example_parameter[3], #transcription_delay, 
                                                           10, #initial_mRNA
                                                           example_parameter[2], #initial_protein,
                                                           2000)
 
        protein_trace = np.vstack((example_trace[:,0],example_trace[:,2])).transpose()
        
        this_fluctuation_rate = hes5.measure_fluctuation_rate_of_single_trace(protein_trace, method = 'gpflow')
         
        print('this gpflow fluctuation_rate is')
        print(this_fluctuation_rate)
        print(this_fluctuation_rate*60)
         
        this_fluctuation_rate = hes5.measure_fluctuation_rate_of_single_trace(protein_trace, method = 'sklearn')
         
        print('this sklearn fluctuation_rate is')
        print(this_fluctuation_rate)
        print(this_fluctuation_rate*60)
         
        this_fluctuation_rate = hes5.measure_fluctuation_rate_of_single_trace(protein_trace, method = 'gpy')
         
        print('this gpy fluctuation_rate is')
        print(this_fluctuation_rate)
        print(this_fluctuation_rate*60)
 
        this_fluctuation_rate = hes5.measure_fluctuation_rate_of_single_trace(protein_trace, method = 'george')
        
        print('this george fluctuation_rate is')
        print(this_fluctuation_rate)
        print(this_fluctuation_rate*60)
 
    def xest_get_fluctuation_rate_from_example_traces(self):
        periods = dict()
        times_values = dict()
        y_values = dict()
        periods['short'] = 0.2
        periods['medium'] = 2.0
        periods['long'] = 20.0
        periods['ten'] = 10.0
        periods['one'] = 1.0

        times_values['short'] = np.arange(0,10.0,0.01)
        times_values['medium'] = np.arange(0,20.0,0.1)
        times_values['long'] = np.arange(0,1000.0,1.0)
        times_values['ten'] = np.arange(0,100.0,0.1)
        times_values['one'] = np.arange(0,100.0,0.1)
        y_values = {key: 10.*np.sin(times_values[key]/periods[key]*2.0*np.pi) for key in periods}
        
        times_values['compound'] = np.arange(0,100.0,0.1)
        y_values['compound'] = (10.*np.sin(times_values['compound']/10*2.0*np.pi) + 
                                3.*np.sin(times_values['compound']*2.0*np.pi))
        
        
        times_values['noise'] = np.arange(0,100.0,0.1)
        y_values['noise'] = (10.*np.sin(times_values['noise']/10*2.0*np.pi) + 
                             np.random.randn(len(times_values['noise'])))
        periods['noise'] = 10.0
        
        for abbreviation in y_values:
            plt.figure(figsize = (4.5,2.5))
            these_y_values = y_values[abbreviation]
            compound_trajectory = np.vstack((times_values[abbreviation], these_y_values)).transpose()
            this_fluctuation_rate = hes5.measure_fluctuation_rate_of_single_trace(compound_trajectory)
            plt.plot(times_values[abbreviation],these_y_values)
            if abbreviation in periods:
                period_blurb ='Period: ' + '{:.2f}'.format(periods[abbreviation]) + ', ' 
            else:
                period_blurb = ''  
            plt.title(period_blurb + 
                      'fluctuation rate: '+ '{:.2f}'.format(this_fluctuation_rate))
            plt.xlabel('Time')
            plt.ylabel('normalised signal')
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                     'lengthscale_visualisation_' + abbreviation + '.pdf'))
            
    def xest_get_fluctuation_rates_from_multiple_traces(self):
#         saving_path = os.path.join(os.path.dirname(__file__), 'output',
#                                     'sampling_results_zebrafish')
#         model_results = np.load(saving_path + '.npy' )
#         prior_samples = np.load(saving_path + '_parameters.npy')
#         
#         accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
#                                     np.logical_and(model_results[:,0]<8000,
#                                     np.logical_and(model_results[:,2]<100,
#                                                    model_results[:,3]>0.3))))  
# 
#         my_posterior_samples = prior_samples[accepted_indices]
# 
#         example_parameter_index = 100
#         example_parameter = my_posterior_samples[example_parameter_index]
#  
#         mrna_traces, protein_traces = hes5.generate_multiple_langevin_trajectories(10,
#                                                                                     720, #duration 
#                                                                                     example_parameter[2], #repression_threshold, 
#                                                                                     example_parameter[4], #hill_coefficient,
#                                                                                     example_parameter[5], #mRNA_degradation_rate, 
#                                                                                     example_parameter[6], #protein_degradation_rate, 
#                                                                                     example_parameter[0], #basal_transcription_rate, 
#                                                                                     example_parameter[1], #translation_rate,
#                                                                                     example_parameter[3], #transcription_delay, 
#                                                                                     10, #initial_mRNA
#                                                                                     example_parameter[2], #initial_protein,
#                                                                                     2000)
# #  
#         these_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(protein_traces, method = 'gpflow')
 
#         np.save(os.path.join(os.path.dirname(__file__),'output',
#                 'fluctuation_rates.npy'), these_fluctuation_rates)

        these_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
                                        'fluctuation_rates.npy'))
        print(these_fluctuation_rates)
        plt.figure(figsize = (4.5,2.5))
        plt.hist(these_fluctuation_rates)
        plt.axvline(np.mean(these_fluctuation_rates), color = 'black')
        plt.xlabel('fluctuation rate')
        plt.ylabel('occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                     'fluctuation_rate_distribution_test.pdf'))
        
    def xest_get_fluctuation_rates_from_multiple_traces_double_duration(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples = prior_samples[accepted_indices]

        example_parameter_index = 100
        example_parameter = my_posterior_samples[example_parameter_index]
 
#         mrna_traces, protein_traces = hes5.generate_multiple_langevin_trajectories(100,
#                                                                                     720*2, #duration 
#                                                                                     example_parameter[2], #repression_threshold, 
#                                                                                     example_parameter[4], #hill_coefficient,
#                                                                                     example_parameter[5], #mRNA_degradation_rate, 
#                                                                                     example_parameter[6], #protein_degradation_rate, 
#                                                                                     example_parameter[0], #basal_transcription_rate, 
#                                                                                     example_parameter[1], #translation_rate,
#                                                                                     example_parameter[3], #transcription_delay, 
#                                                                                     10, #initial_mRNA
#                                                                                     example_parameter[2], #initial_protein,
#                                                                                     2000)
  
#         these_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(protein_traces)

#         np.save(os.path.join(os.path.dirname(__file__),'output',
#                 'fluctuation_rates_double.npy'), these_fluctuation_rates)

        these_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
                                        'fluctuation_rates_double.npy'))

        print(these_fluctuation_rates)
        plt.figure(figsize = (4.5,2.5))
        plt.hist(these_fluctuation_rates, range = (0,0.006))
        plt.axvline(np.mean(these_fluctuation_rates), color = 'black')
        plt.xlabel('fluctuation rate')
        plt.ylabel('occurrence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                     'fluctuation_rate_distribution_double.pdf'))    
        
    def xest_get_autocorrelation_function_from_power_spectrum(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.2))))  

        my_posterior_samples = prior_samples[accepted_indices]

        example_parameter_index = 10
        example_parameter = my_posterior_samples[example_parameter_index]
 
        number_of_traces = 10
        mrna_traces, protein_traces = hes5.generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
                                                                                           1500*5, #duration 
                                                                                           example_parameter[2], #repression_threshold, 
                                                                                           example_parameter[4], #hill_coefficient,
                                                                                           example_parameter[5], #mRNA_degradation_rate, 
                                                                                           example_parameter[6], #protein_degradation_rate, 
                                                                                           example_parameter[0], #basal_transcription_rate, 
                                                                                           example_parameter[1], #translation_rate,
                                                                                           example_parameter[3], #transcription_delay, 
                                                                                           10, #initial_mRNA, 
                                                                                           example_parameter[2], #initial_protein,
                                                                                           1000)
        
        power_spectrum, _, _ = hes5.calculate_power_spectrum_of_trajectories(protein_traces, normalize = False)
        
        auto_correlation_from_fourier = hes5.calculate_autocorrelation_from_power_spectrum(power_spectrum)
        print(auto_correlation_from_fourier)
#         all_auto_correlation = np.zeros(protein_traces.shape[0]/2)
        @jit(nopython = True)
        def calculate_autocorrelation(full_signal):
            signal = full_signal - np.mean(full_signal)
            auto_correlation = np.zeros_like(signal)
            auto_correlation[0] = np.sum(signal*signal)/len(signal)
            for lag in range(1,len(signal)):
                auto_correlation[lag] = np.sum(signal[:-lag]*signal[lag:])/len(signal[:-lag])
            return auto_correlation

        mean_auto_correlation = np.zeros_like(protein_traces[:,1])
#         import pdb; pdb.set_trace()
        plt.figure(figsize = (4.5,4.5))
        plt.subplot(211)
        plt.plot(power_spectrum[:,0], power_spectrum[:,1])
        plt.xlabel('Frequency')
        plt.xlim(0,0.02)
        plt.xlabel('Power')
        plt.subplot(212)
        for trace in protein_traces[:,1:].transpose():
            this_autocorrelation = calculate_autocorrelation(trace)
            mean_auto_correlation += this_autocorrelation
            plt.plot(protein_traces[:,0], this_autocorrelation, color = 'black', alpha = 0.1, lw = 0.5 )
        mean_auto_correlation/= number_of_traces
        plt.plot(protein_traces[:,0], mean_auto_correlation, color = 'black', lw = 0.5)
#         corrected_auto_correlation = auto_correlation_from_fourier[:,1]/(np.sqrt(len(trace)))
        corrected_auto_correlation = auto_correlation_from_fourier[:,1]
        plt.plot(auto_correlation_from_fourier[:,0],
                 corrected_auto_correlation, lw = 0.5, color = 'blue', ls = '--')
        plt.xlim(0,1000)
        plt.xlabel('Time')
        plt.ylabel('Autocorrelation')
#         plt.axhline(np.mean(auto_correlation_from_fourier)*2/(np.sqrt(len(trace))), color = 'purple', lw = 3)
#         plt.axhline(np.power(np.mean(protein_traces[:,1:]),2)+np.var(protein_traces[:,1:]), lw = 0.5)
#         plt.axhline(np.power(np.mean(protein_traces[:,1:]),2), lw = 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'autocorrelation_function_test.pdf'))    
        
    def xest_plot_gamma_dependence_of_ou_osc(self):

        gamma = np.linspace(0,2,100)
        power_spectrum_peak = np.sqrt(2*np.sqrt(np.power(gamma,4) + np.power(gamma,2)) -1 -np.power(gamma,2))
        power_spectrum_peak /= gamma
        plt.figure(figsize = (4.5,2.5))
#         plt.plot(gamma, gamma)
        plt.axhline(1.0)
        plt.plot(gamma, power_spectrum_peak)
        plt.xlabel(r'Oscillation quality $\beta$/$\alpha$')
        plt.ylabel(r'Period ratio $\omega$/$\beta$')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'gamma_dependency.pdf'))    
 
        ornstein_kernel = gpflow.kernels.Matern12(1, lengthscales=1.0)*gpflow.kernels.Cosine(1,lengthscales = 1.0/0.9)
        times = np.linspace(0,100,1000)
        times = times[:,np.newaxis]
        mean_function = np.zeros_like(times)
        my_regression_model = gpflow.models.GPR(times, mean_function, kern=ornstein_kernel)
        sample = my_regression_model.predict_f_samples(times, 1)
        plt.figure(figsize = (4.5,4.5))
        plt.plot(times, sample[0])
        plt.xlabel('Time')
        plt.ylabel('process value')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'ou_osc_prediction.pdf'))    
        print('expected_period is')
        gamma=0.9
        print(np.sqrt(2*np.sqrt(np.power(gamma,4) + np.power(gamma,2)) -1 -np.power(gamma,2)))
        data = np.column_stack((times,sample[0]))
        data_as_frame = pd.DataFrame(data)
        data_as_frame.to_excel(os.path.join(os.path.dirname(__file__),'output',
                               'ornstein_sample.xlsx'), index = False)

    def xest_compare_quality_and_coherence(self):
        gamma_values = np.linspace(1.0/np.sqrt(3),8,1000)
        theoretical_frequency_values = np.sqrt(2*np.sqrt(np.power(gamma_values,4) + np.power(gamma_values,2)) -1 -np.power(gamma_values,2))
        
        frequencies = np.linspace(0,80,10000)
        coherence_values = np.zeros_like(gamma_values)
        oscillation_frequency_values = np.zeros_like(gamma_values)
        for gamma_index, gamma_value in enumerate(gamma_values):
            power_spectrum = 1.0/(1+np.power(frequencies - gamma_value,2)) + 1.0/(1+np.power(frequencies + gamma_value,2))
            full_power_spectrum = np.vstack((frequencies/(2*np.pi), power_spectrum)).transpose()
            coherence, period = hes5.calculate_coherence_and_period_of_power_spectrum(full_power_spectrum)
            coherence_values[gamma_index] = coherence
            oscillation_frequency_values[gamma_index] = 2*np.pi/period

        this_gamma = 2.0
        power_spectrum = 1.0/(1+np.power(frequencies - this_gamma,2)) + 1.0/(1+np.power(frequencies + this_gamma,2))
        plt.figure(figsize = (4.5,2.5))
        plt.plot(frequencies, power_spectrum)
        plt.xlabel(r'Frequency $\omega$')
        plt.ylabel(r'Power')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'example_OUosc_power_spectrum.pdf'))    
 
        plt.figure(figsize = (4.5,2.5))
        plt.plot(gamma_values, oscillation_frequency_values/gamma_values)
        plt.plot(gamma_values, theoretical_frequency_values/gamma_values)
        plt.xlabel(r'Oscillation quality $\beta$/$\alpha$')
        plt.ylabel(r'Frequency ratio $\omega/\beta$')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'period_validation_for_coherence_calculation_OUosc.pdf'))    

        plt.figure(figsize = (4.5,2.5))
        plt.plot(gamma_values, coherence_values)
        plt.xlabel(r'Oscillation quality $\beta$/$\alpha$')
        plt.ylabel(r'Coherence')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'coherence_vs_quality_OUosc.pdf'))    
 
    def xest_get_lengthscale_from_simulated_power_spectrum(self):

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples = prior_samples[accepted_indices]

        example_parameter_index = 100
        example_parameter = my_posterior_samples[example_parameter_index]
 
        number_of_traces = 100
        mrna_traces, protein_traces = hes5.generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
                                                                                    1500*5, #duration 
                                                                                    example_parameter[2], #repression_threshold, 
                                                                                    example_parameter[4], #hill_coefficient,
                                                                                    example_parameter[5], #mRNA_degradation_rate, 
                                                                                    example_parameter[6], #protein_degradation_rate, 
                                                                                    example_parameter[0], #basal_transcription_rate, 
                                                                                    example_parameter[1], #translation_rate,
                                                                                    example_parameter[3], #transcription_delay, 
                                                                                    10, #initial_mRNA, 
                                                                                    example_parameter[2], #initial_protein,
                                                                                    1000)
        
        power_spectrum, _, _ = hes5.calculate_power_spectrum_of_trajectories(protein_traces, normalize = False)
        
        auto_correlation_from_fourier = hes5.calculate_autocorrelation_from_power_spectrum(power_spectrum)

#         this_fluctuation_rate, this_variance = hes5.estimate_fluctuation_rate_of_traces(protein_traces)
#         this_fluctuation_rate, this_variance = hes5.estimate_fluctuation_rate_of_traces_by_matrices(protein_traces)
        this_fluctuation_rate = 0.003580 
        this_variance = np.var(protein_traces[:,1:])
        
        print(this_fluctuation_rate)
        
        time_values = protein_traces[:,0]
        fitted_auto_correlation_values = this_variance*np.exp(-this_fluctuation_rate*time_values)
        fitted_power_spectrum_values = this_variance*2*this_fluctuation_rate/(this_fluctuation_rate*this_fluctuation_rate
                                                                              +4*np.pi*np.pi*power_spectrum[:,0]*power_spectrum[:,0])
        fitted_power_spectrum = np.vstack((power_spectrum[:,0], fitted_power_spectrum_values)).transpose()       
        tlest_correlation_function = hes5.calculate_autocorrelation_from_power_spectrum(fitted_power_spectrum)
        plt.figure(figsize = (4.5,4.5))
        plt.subplot(211)
        plt.plot(power_spectrum[:,0], power_spectrum[:,1])
        plt.plot(power_spectrum[:,0], fitted_power_spectrum_values)
        plt.xlabel('Frequency')
        plt.xlim(0,0.05)
        plt.ylabel('Power')
        plt.subplot(212)
        plt.plot(auto_correlation_from_fourier[:,0],
                auto_correlation_from_fourier[:,1], lw = 0.5, color = 'blue', ls = '-')
        plt.plot(time_values,
                fitted_auto_correlation_values, lw = 0.5, color = 'orange', ls = '--')
        plt.plot(tlest_correlation_function[:,0],
                tlest_correlation_function[:,1], lw = 0.5, color = 'green', ls = '--', alpha =0.5)
        plt.xlim(0,1000)
        plt.xlabel('Time')
        plt.ylabel('Autocorrelation')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'autocorrelation_function_fit.pdf'))    
 
    def xest_compare_direct_and_indirect_lengthscale_measurement(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples = prior_samples[accepted_indices]

#         example_parameter_index = 10
        example_parameter_index = 100
        example_parameter = my_posterior_samples[example_parameter_index]
 
        number_of_traces = 100
        mrna_traces, protein_traces = hes5.generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
                                                                                    1500*5, #duration 
                                                                                    example_parameter[2], #repression_threshold, 
                                                                                    example_parameter[4], #hill_coefficient,
                                                                                    example_parameter[5], #mRNA_degradation_rate, 
                                                                                    example_parameter[6], #protein_degradation_rate, 
                                                                                    example_parameter[0], #basal_transcription_rate, 
                                                                                    example_parameter[1], #translation_rate,
                                                                                    example_parameter[3], #transcription_delay, 
                                                                                    10, #initial_mRNA, 
                                                                                    example_parameter[2], #initial_protein,
                                                                                    1000)

        print('generated traces')
        this_fluctuation_rate, this_variance = hes5.estimate_fluctuation_rate_of_traces_by_matrices(protein_traces)
        print('estimated fluctuation rate')
        
        these_reduced_traces = protein_traces[:720]
        these_extended_reduced_traces = protein_traces[:720*2]
        
#         these_reduced_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_reduced_traces)
#         np.save(os.path.join(os.path.dirname(__file__),'output',
#                 'fluctuation_rates_for_convergence_2.npy'), these_reduced_fluctuation_rates)
        these_reduced_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
                                        'fluctuation_rates_for_convergence.npy'))
        print('measured fluctuation rate short')

#         these_extended_reduced_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_extended_reduced_traces)
#         np.save(os.path.join(os.path.dirname(__file__),'output',
#                 'fluctuation_rates_for_convergence_longer_2.npy'), these_extended_reduced_fluctuation_rates)
        these_extended_reduced_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
                                                            'fluctuation_rates_for_convergence_longer.npy'))
        print('measured fluctuation rate longer')
        print(np.mean(these_reduced_fluctuation_rates)/this_fluctuation_rate)
        print(np.mean(these_extended_reduced_fluctuation_rates)/this_fluctuation_rate)
        print(this_fluctuation_rate)
        
        correction_factor = 1.0
        plt.figure(figsize = (4.5,4.5))

        plt.subplot(211)
        plt.hist(these_reduced_fluctuation_rates, bins = 20, range = (0,0.015))
#         plt.hist(these_reduced_fluctuation_rates, bins = 20, range = (0,0.007))
#         plt.hist(these_reduced_fluctuation_rates, bins = 20)
        plt.axvline(np.mean(these_reduced_fluctuation_rates), color = 'blue')
        plt.axvline(np.median(these_reduced_fluctuation_rates), color = 'orange')
#         plt.axvline(this_fluctuation_rate/np.sqrt(2.0), color = 'green')
        plt.axvline(this_fluctuation_rate/correction_factor, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        plt.subplot(212)
#         plt.hist(these_extended_reduced_fluctuation_rates, bins = 20, range = (0,0.007))
        plt.hist(these_extended_reduced_fluctuation_rates, bins = 20, range = (0,0.015))
        plt.axvline(np.mean(these_extended_reduced_fluctuation_rates), color = 'blue')
        plt.axvline(np.median(these_extended_reduced_fluctuation_rates), color = 'orange')
        plt.axvline(this_fluctuation_rate/correction_factor, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'fluctuation_rate_convergence_2.pdf'))

    def xest_approximate_lengthscale_measurement(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples = prior_samples[accepted_indices]

#         example_parameter_index = 10
        example_parameter_index = 100
        example_parameter = my_posterior_samples[example_parameter_index]
 
        number_of_traces = 100
        mrna_traces, protein_traces = hes5.generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
                                                                                    1500*5, #duration 
                                                                                    example_parameter[2], #repression_threshold, 
                                                                                    example_parameter[4], #hill_coefficient,
                                                                                    example_parameter[5], #mRNA_degradation_rate, 
                                                                                    example_parameter[6], #protein_degradation_rate, 
                                                                                    example_parameter[0], #basal_transcription_rate, 
                                                                                    example_parameter[1], #translation_rate,
                                                                                    example_parameter[3], #transcription_delay, 
                                                                                    10, #initial_mRNA, 
                                                                                    example_parameter[2], #initial_protein,
                                                                                    1000)

        print('generated traces')
        this_fluctuation_rate = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces)
        
        these_reduced_traces = protein_traces[:720,:10]
        
#         these_reduced_traces = protein_traces[::10,:10]
        these_extended_reduced_traces = protein_traces[:720*2,:10]
        
        these_reduced_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_reduced_traces)
#         np.save(os.path.join(os.path.dirname(__file__),'output',
#                 'fluctuation_rates_for_convergence_2.npy'), these_reduced_fluctuation_rates)
#         these_reduced_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence.npy'))
        print('measured fluctuation rate short')
#         these_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_reduced_traces, method = 'sklearn')
#         this_fluctuation_rate = np.mean(these_fluctuation_rates)
        print('estimated fluctuation rate')

        these_extended_reduced_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_extended_reduced_traces)
#         np.save(os.path.join(os.path.dirname(__file__),'output',
#                 'fluctuation_rates_for_convergence_longer_2.npy'), these_extended_reduced_fluctuation_rates)
#         these_extended_reduced_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                                             'fluctuation_rates_for_convergence_longer.npy'))
        print('measured fluctuation rate longer')
        print(np.mean(these_reduced_fluctuation_rates)/this_fluctuation_rate)
        print(np.mean(these_extended_reduced_fluctuation_rates)/this_fluctuation_rate)
        print(this_fluctuation_rate)
        
        correction_factor = 1.0
        plt.figure(figsize = (4.5,4.5))

        plt.subplot(211)
        plt.hist(these_reduced_fluctuation_rates, bins = 20, range = (0,0.015))
#         plt.hist(these_reduced_fluctuation_rates, bins = 20, range = (0,0.007))
#         plt.hist(these_reduced_fluctuation_rates, bins = 20)
        plt.axvline(np.mean(these_reduced_fluctuation_rates), color = 'blue')
        plt.axvline(np.median(these_reduced_fluctuation_rates), color = 'orange')
#         plt.axvline(this_fluctuation_rate/np.sqrt(2.0), color = 'green')
        plt.axvline(this_fluctuation_rate/correction_factor, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        plt.subplot(212)
#         plt.hist(these_extended_reduced_fluctuation_rates, bins = 20, range = (0,0.007))
        plt.hist(these_extended_reduced_fluctuation_rates, bins = 20, range = (0,0.015))
        plt.axvline(np.mean(these_extended_reduced_fluctuation_rates), color = 'blue')
        plt.axvline(np.median(these_extended_reduced_fluctuation_rates), color = 'orange')
        plt.axvline(this_fluctuation_rate/correction_factor, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'fluctuation_rate_convergence_alternative.pdf'))

    def xest_illustrate_lengthscale_measurements_1(self):
        option = 'without_noise'
        # option = 'with_noise'
        number_traces_to_consider = 100
        noise_strength = 0.01
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples_1 = prior_samples[accepted_indices]
        example_parameter_index = 100
        example_parameter_1 = my_posterior_samples_1[example_parameter_index]

        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.2))))  
        my_posterior_samples_2 = prior_samples[accepted_indices]
#         example_parameter_index = 10
        example_parameter_index = 10
        example_parameter_2 = my_posterior_samples_2[example_parameter_index]
 
        number_of_traces = 100
        _, protein_traces_1 = hes5.generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
                                                                            1500*5, #duration 
                                                                            example_parameter_1[2], #repression_threshold, 
                                                                            example_parameter_1[4], #hill_coefficient,
                                                                            example_parameter_1[5], #mRNA_degradation_rate, 
                                                                            example_parameter_1[6], #protein_degradation_rate, 
                                                                            example_parameter_1[0], #basal_transcription_rate, 
                                                                            example_parameter_1[1], #translation_rate,
                                                                            example_parameter_1[3], #transcription_delay, 
                                                                            10, #initial_mRNA, 
                                                                            example_parameter_1[2], #initial_protein,
                                                                            1000)
        number_of_traces = 100
        _, protein_traces_2 = hes5.generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
                                                                            1500*5, #duration 
                                                                            example_parameter_2[2], #repression_threshold, 
                                                                            example_parameter_2[4], #hill_coefficient,
                                                                            example_parameter_2[5], #mRNA_degradation_rate, 
                                                                            example_parameter_2[6], #protein_degradation_rate, 
                                                                            example_parameter_2[0], #basal_transcription_rate, 
                                                                            example_parameter_2[1], #translation_rate,
                                                                            example_parameter_2[3], #transcription_delay, 
                                                                            10, #initial_mRNA,2
                                                                            example_parameter_2[2], #initial_protein,
                                                                            1000)
        
        if option == 'with_noise':
            protein_traces_1[:,1:] += np.random.randn(1500*5,100)*noise_strength*np.mean(protein_traces_1[:,1:])
            protein_traces_2[:,1:] += np.random.randn(1500*5,100)*noise_strength*np.mean(protein_traces_2[:,1:])

        plt.figure(figsize = (6.5,10.5))
        ## Row 1 - traces
        plt.subplot(421)
        this_trace = protein_traces_1[:,(0,1)]
        plt.plot(this_trace[:,0], this_trace[:,1])
        plt.xlim(0,1000)
        plt.xlabel('Time [min]')
        plt.ylabel('# Her6')
        plt.subplot(422)
        this_trace = protein_traces_2[:,(0,1)]
        plt.plot(this_trace[:,0], this_trace[:,1])
        plt.xlim(0,1000)
        plt.xlabel('Time [min]')
        plt.ylabel('# Her6')

        ## Row 2 - histogram from 12 hours
        plt.subplot(423)
        these_shortened_traces_1 = protein_traces_1[:720,:number_traces_to_consider+1]
        these_measured_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_shortened_traces_1)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_shortened_1_' + option + '_' + 
                str(number_traces_to_consider) + '_' + str(noise_strength) + '.npy'), 
                these_measured_fluctuation_rates)
#         these_measured_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence_shortened_1.npy'))
        this_fluctuation_rate_1 = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces_1, sampling_interval = 1)
#         plt.hist(these_measured_fluctuation_rates, bins = 20, range = (0,0.008))
        plt.hist(these_measured_fluctuation_rates, bins = 20)
        plt.axvline(np.mean(these_measured_fluctuation_rates), color = 'blue', label = 'Mean')
        plt.axvline(this_fluctuation_rate_1, color = 'green', label = 'Theory')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')
        plt.legend(ncol=1, loc = 'upper left', bbox_to_anchor = (-0.1,1.2), framealpha = 1.0)

        plt.subplot(424)
        these_shortened_traces_2 = protein_traces_2[:720,:number_traces_to_consider+1]
        these_measured_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_shortened_traces_2)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_shortened_2' + option + '_' + 
                str(number_traces_to_consider) + '_' + str(noise_strength) + '.npy'), these_measured_fluctuation_rates)
#         these_measured_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence_2.npy'))
        this_fluctuation_rate_2 = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces_2, sampling_interval = 1)
#         plt.hist(these_measured_fluctuation_rates, bins = 20, range = (0,0.015))
        plt.hist(these_measured_fluctuation_rates, bins = 20)
        plt.axvline(np.mean(these_measured_fluctuation_rates), color = 'blue')
        plt.axvline(this_fluctuation_rate_2, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        ## Row 3 - histogram from 24 hours
        plt.subplot(425)
        these_shortened_traces_1 = protein_traces_1[:720*2,:number_traces_to_consider+1]
        these_measured_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_shortened_traces_1)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_less_shortened_1' + option + '_' + 
                str(number_traces_to_consider) + '_' + str(noise_strength) + '.npy'), these_measured_fluctuation_rates)
#         these_measured_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence_less_shortened_1.npy'))
#         plt.hist(these_measured_fluctuation_rates, bins = 20, range = (0,0.008))
        plt.hist(these_measured_fluctuation_rates, bins = 20)
        plt.axvline(np.mean(these_measured_fluctuation_rates), color = 'blue')
        plt.axvline(this_fluctuation_rate_1, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        plt.subplot(426)
        these_shortened_traces_2 = protein_traces_2[:720*2,:number_traces_to_consider+1]
        these_measured_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_shortened_traces_2)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_less_shortened_2' + option + '_' + 
                str(number_traces_to_consider) + '_' + str(noise_strength) + '.npy'), these_measured_fluctuation_rates)
#         these_measured_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence_less_shortened_2.npy'))
#         plt.hist(these_measured_fluctuation_rates, bins = 20, range = (0,0.015))
        plt.hist(these_measured_fluctuation_rates, bins = 20)
        plt.axvline(np.mean(these_measured_fluctuation_rates), color = 'blue')
        plt.axvline(this_fluctuation_rate_2, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        ## Row 4 - histogram from 12 hours, lower sampling rate
        plt.subplot(427)
        these_short_downsampled_protein_traces_1 = protein_traces_1[:720:10,:number_traces_to_consider+1]
        these_measured_fluctuation_rates_1 = hes5.measure_fluctuation_rates_of_traces(these_short_downsampled_protein_traces_1)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_downsampled_1' + option + '_' + 
                str(number_traces_to_consider) + '_' + str(noise_strength) + '.npy'), these_measured_fluctuation_rates_1)
#         these_measured_fluctuation_rates_1 = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence_downsampled_1.npy'))
        this_estimated_fluctuation_rate_1 = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces_1,
                                                                                                      sampling_interval = 10)
        plt.hist(these_measured_fluctuation_rates_1, bins = 20)
        plt.axvline(np.mean(these_measured_fluctuation_rates_1), color = 'blue')
        plt.axvline(this_estimated_fluctuation_rate_1, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        plt.subplot(428)
        these_short_downsampled_protein_traces_2 = protein_traces_2[:720:10,:number_traces_to_consider+1]
        these_measured_fluctuation_rates_2 = hes5.measure_fluctuation_rates_of_traces(these_short_downsampled_protein_traces_2)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_downsampled_2' + option + '_' + 
                str(number_traces_to_consider) + '_' + str(noise_strength) + '.npy'), these_measured_fluctuation_rates_2)
#         these_measured_fluctuation_rates_2 = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence_downsampled_2.npy'))
        this_estimated_fluctuation_rate_2 = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces_2,
                                                                                                      sampling_interval = 10)
        plt.hist(these_measured_fluctuation_rates_2, bins = 20)
        plt.axvline(np.mean(these_measured_fluctuation_rates_2), color = 'blue')
        plt.axvline(this_estimated_fluctuation_rate_2, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'fluctuation_rate_illustration_panels_' + option + '_' + 
                                  str(number_traces_to_consider) + '_' + str(noise_strength) + 
                                 '.pdf'))
        
        ## make the second plot
        this_halfway_sampling_rate_1 = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces_1,
                                                                                                 sampling_interval = 6)

        this_halfway_sampling_rate_2 = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces_2,
                                                                                                 sampling_interval = 6)
        
        plt.figure(figsize = (6.5,4.5))
        plt.subplot(221)
        this_trace = protein_traces_1[:,(0,1)]
        plt.plot(this_trace[:,0], this_trace[:,1], color = 'blue')
        plt.xlabel('Time [min]')
        plt.ylabel('# Her6')
        plt.xlim(0,1000)

        plt.subplot(222)
        this_trace = protein_traces_2[:,(0,1)]
        plt.plot(this_trace[:,0], this_trace[:,1], color = 'green')
        plt.xlabel('Time [min]')
        plt.ylabel('# Her6')
        plt.xlim(0,1000)
        plt.subplot(223)
        plt.plot([1,6,10], 
                 [this_fluctuation_rate_1, this_halfway_sampling_rate_1, this_estimated_fluctuation_rate_1],
                 marker = 'o',
                 color = 'blue')
        plt.plot([1,6,10], 
                 [this_fluctuation_rate_2, this_halfway_sampling_rate_2, this_estimated_fluctuation_rate_2],
                 marker = 'o',
                 color = 'green')
        plt.xlabel('Sampling interval [min]')
        plt.ylabel('Fluctuation rate [1/min]')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'fluctuation_rate_illustration_short_' + option + '_' + 
                                  str(number_traces_to_consider) + '_' + str(noise_strength) + 
                                  '.pdf'))

    def xest_illustrate_lengthscale_measurements_with_noise(self):
        option = 'without_noise'
        # option = 'with_noise'
        number_traces_to_consider = 100
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples_1 = prior_samples[accepted_indices]
        example_parameter_index = 100
        example_parameter_1 = my_posterior_samples_1[example_parameter_index]

        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.2))))  
        my_posterior_samples_2 = prior_samples[accepted_indices]
#         example_parameter_index = 10
        example_parameter_index = 10
        example_parameter_2 = my_posterior_samples_2[example_parameter_index]
 
        number_of_traces = 100
        _, protein_traces_1 = hes5.generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
                                                                            1500*5, #duration 
                                                                            example_parameter_1[2], #repression_threshold, 
                                                                            example_parameter_1[4], #hill_coefficient,
                                                                            example_parameter_1[5], #mRNA_degradation_rate, 
                                                                            example_parameter_1[6], #protein_degradation_rate, 
                                                                            example_parameter_1[0], #basal_transcription_rate, 
                                                                            example_parameter_1[1], #translation_rate,
                                                                            example_parameter_1[3], #transcription_delay, 
                                                                            10, #initial_mRNA, 
                                                                            example_parameter_1[2], #initial_protein,
                                                                            1000)
        number_of_traces = 100
        _, protein_traces_2 = hes5.generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
                                                                            1500*5, #duration 
                                                                            example_parameter_2[2], #repression_threshold, 
                                                                            example_parameter_2[4], #hill_coefficient,
                                                                            example_parameter_2[5], #mRNA_degradation_rate, 
                                                                            example_parameter_2[6], #protein_degradation_rate, 
                                                                            example_parameter_2[0], #basal_transcription_rate, 
                                                                            example_parameter_2[1], #translation_rate,
                                                                            example_parameter_2[3], #transcription_delay, 
                                                                            10, #initial_mRNA,2
                                                                            example_parameter_2[2], #initial_protein,
                                                                            1000)
        
        noises_to_investigate = [0.0,0.01,0.05,0.1]
        fluctuation_rates_for_noise = np.zeros((len(noises_to_investigate),3))
        fluctuation_rates_for_noise[:,0] = noises_to_investigate
        
        for noise_index, noise_strength in enumerate(fluctuation_rates_for_noise[:,0]):
            this_signal_1 = protein_traces_1[:,1:] + np.random.randn(1500*5,100)*noise_strength*np.mean(protein_traces_1[:,1:])
            this_signal_2 = protein_traces_2[:,1:] + np.random.randn(1500*5,100)*noise_strength*np.mean(protein_traces_2[:,1:])
            this_fluctuation_rate_1 = hes5.approximate_fluctuation_rate_of_traces_theoretically(this_signal_1,
                                                                                                 sampling_interval = 6)
            this_fluctuation_rate_2 = hes5.approximate_fluctuation_rate_of_traces_theoretically(this_signal_2,
                                                                                                 sampling_interval = 6)
            fluctuation_rates_for_noise[noise_index,1] = this_fluctuation_rate_1
            fluctuation_rates_for_noise[noise_index,2] = this_fluctuation_rate_2
        
        plt.figure(figsize = (6.5,4.5))
        plt.subplot(221)
        this_trace = protein_traces_1[:,(0,1)]
        plt.plot(this_trace[:,0], this_trace[:,1], color = 'blue')
        plt.xlabel('Time [min]')
        plt.ylabel('# Her6')
        plt.xlim(0,1000)

        plt.subplot(222)
        this_trace = protein_traces_2[:,(0,1)]
        plt.plot(this_trace[:,0], this_trace[:,1], color = 'green')
        plt.xlabel('Time [min]')
        plt.ylabel('# Her6')
        plt.xlim(0,1000)
        plt.subplot(223)
        plt.plot(fluctuation_rates_for_noise[:,0], 
                 fluctuation_rates_for_noise[:,1],
                 marker = 'o',
                 color = 'blue')
        plt.plot(fluctuation_rates_for_noise[:,0], 
                 fluctuation_rates_for_noise[:,2],
                 marker = 'o',
                 color = 'green')
        plt.xlabel('noise std/signal mean')
        plt.ylabel('Fluctuation rate [1/min]')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'fluctuation_rate_illustration_noise_dependence.pdf'))

    def xest_get_get_correlation_matrices(self):
        times = np.linspace(0,10,100)
        distance_vector = scipy.spatial.distance.pdist(times[:,np.newaxis]) 
        distance_matrix = scipy.spatial.distance.squareform(distance_vector) 
        covariance_matrix = np.exp(-distance_matrix)
        print(distance_matrix)
        print(covariance_matrix)
        
        # alternatively we can do
        new_times = np.linspace(0,10,100)
        correlation_function = np.exp(-new_times)
        indices = np.arange(0,times.shape[0],1)
        new_distance_vector = scipy.spatial.distance.pdist(indices[:,np.newaxis]).astype(np.int)
        print(new_distance_vector)
        new_distance_matrix = scipy.spatial.distance.squareform(new_distance_vector) 
        print(new_distance_matrix)
        new_covariance_matrix = correlation_function[new_distance_matrix]
        print(new_covariance_matrix)
        
    def xest_calculate_high_frequency_weight_of_power_spectrum(self):
        
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples_1 = prior_samples[accepted_indices]
        example_parameter_index = 100
        example_parameter_1 = my_posterior_samples_1[example_parameter_index]

        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.2))))  
        my_posterior_samples_2 = prior_samples[accepted_indices]
#         example_parameter_index = 10
        example_parameter_index = 10
        example_parameter_2 = my_posterior_samples_2[example_parameter_index]
 
        number_of_traces = 100
        _, protein_traces_1 = hes5.generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
                                                                            1500*5, #duration 
                                                                            example_parameter_1[2], #repression_threshold, 
                                                                            example_parameter_1[4], #hill_coefficient,
                                                                            example_parameter_1[5], #mRNA_degradation_rate, 
                                                                            example_parameter_1[6], #protein_degradation_rate, 
                                                                            example_parameter_1[0], #basal_transcription_rate, 
                                                                            example_parameter_1[1], #translation_rate,
                                                                            example_parameter_1[3], #transcription_delay, 
                                                                            10, #initial_mRNA, 
                                                                            example_parameter_1[2], #initial_protein,
                                                                            1000)
        number_of_traces = 100
        _, protein_traces_2 = hes5.generate_multiple_langevin_trajectories( number_of_traces, # number_of_trajectories 
                                                                            1500*5, #duration 
                                                                            example_parameter_2[2], #repression_threshold, 
                                                                            example_parameter_2[4], #hill_coefficient,
                                                                            example_parameter_2[5], #mRNA_degradation_rate, 
                                                                            example_parameter_2[6], #protein_degradation_rate, 
                                                                            example_parameter_2[0], #basal_transcription_rate, 
                                                                            example_parameter_2[1], #translation_rate,
                                                                            example_parameter_2[3], #transcription_delay, 
                                                                            10, #initial_mRNA,2
                                                                            example_parameter_2[2], #initial_protein,
                                                                            1000)

        power_spectrum_1,_,_ = hes5.calculate_power_spectrum_of_trajectories(protein_traces_1, normalize = False)
        power_spectrum_2,_,_ = hes5.calculate_power_spectrum_of_trajectories(protein_traces_2, normalize = False)
        
        frequency_cutoff = 1.0/30
        for power_spectrum_index, power_spectrum in enumerate([power_spectrum_1, power_spectrum_2]):
            first_left_index = np.min(np.where(power_spectrum[:,0]>frequency_cutoff))
            integration_axis = np.hstack(([frequency_cutoff], power_spectrum[first_left_index:,0]))
            power_spectrum_interpolation = scipy.interpolate.interp1d(power_spectrum[:,0], power_spectrum[:,1])
            interpolation_values = power_spectrum_interpolation(integration_axis)
            noise_area = np.trapz(interpolation_values, integration_axis)
            if power_spectrum_index == 0:
                noise_weight_1 = noise_area
            else:
                noise_weight_2 = noise_area
        noise_weight_1_by_function = hes5.calculate_noise_weight_from_power_spectrum(power_spectrum_1)
        self.assertAlmostEqual(noise_weight_1, noise_weight_1_by_function)
        noise_weight_2_by_function = hes5.calculate_noise_weight_from_power_spectrum(power_spectrum_2)
        self.assertAlmostEqual(noise_weight_2, noise_weight_2_by_function)
        plt.figure(figsize = (6.5,4.5))
        plt.subplot(221)
        this_trace = protein_traces_1[:,(0,1)]
        plt.plot(this_trace[:,0], this_trace[:,1], color = 'blue')
        plt.xlabel('Time [min]')
        plt.ylabel('# Her6')
        plt.xlim(0,1000)

        plt.subplot(222)
        this_trace = protein_traces_2[:,(0,1)]
        plt.plot(this_trace[:,0], this_trace[:,1], color = 'blue')
        plt.xlabel('Time [min]')
        plt.ylabel('# Her6')
        plt.xlim(0,1000)

        plt.subplot(223)
        plt.plot(power_spectrum_1[:,0], power_spectrum_1[:,1]/1e7, lw = 1)
        first_left_index = np.min(np.where(power_spectrum_1[:,0]>frequency_cutoff))
        plt.fill_between(power_spectrum_1[first_left_index:,0], 
                         power_spectrum_1[first_left_index:,1]/1e7,
                         0, color = 'green')
        plt.xlabel('Frequency [1/min]')
        plt.xlim(0,0.1)
        plt.ylim(0,2)
        plt.axvline(frequency_cutoff,ymin = 0)
        plt.title('Noise weight is ' + '{:.2f}'.format(noise_weight_1))
        plt.ylabel('Power [1e7min]')
        plt.title
        
        plt.subplot(224)
        theoretical_power_spectrum_1 = hes5.calculate_theoretical_power_spectrum_at_parameter_point(
                                                                basal_transcription_rate = example_parameter_2[0],
                                                                translation_rate = example_parameter_2[1],
                                                                repression_threshold = example_parameter_2[2],
                                                                transcription_delay = example_parameter_2[3],
                                                                mRNA_degradation_rate = example_parameter_2[5],
                                                                protein_degradation_rate = example_parameter_2[6],
                                                                hill_coefficient = example_parameter_2[4],
                                                                normalise = False,
                                                                limits = [0,0.1])
        plt.plot(power_spectrum_2[:,0], power_spectrum_2[:,1]/1e7, lw = 1)
#         plt.plot(theoretical_power_spectrum_1[:,0], theoretical_power_spectrum_1[:,1]/1e7)
        first_left_index = np.min(np.where(power_spectrum_2[:,0]>frequency_cutoff))
        plt.fill_between(power_spectrum_2[first_left_index:,0], 
                         power_spectrum_2[first_left_index:,1]/1e7,
                         0, color = 'green')
        first_left_index_1 = np.min(np.where(power_spectrum_2[:,0]>0.01))
        first_left_index_2 = np.min(np.where(theoretical_power_spectrum_1[:,0]>0.01))
#         print(theoretical_power_spectrum_1[first_left_index_1,1]/power_spectrum_2[first_left_index_2,1])
        print(power_spectrum_2.shape)
        power_area_1 = np.trapz(power_spectrum_2[:,1], power_spectrum_2[:,0])
        power_area = np.trapz(theoretical_power_spectrum_1[:,1], theoretical_power_spectrum_1[:,0])
        ratio = power_area/power_area_1
        print(power_area/power_area_1)
        print(protein_traces_1.shape)
        print(protein_traces_1.shape[0]*np.sqrt(protein_traces_1.shape[0])/ratio)
        print(power_spectrum_2.shape[0]*np.sqrt(power_spectrum_2.shape[0])/ratio)
        print(power_spectrum_2.shape[0]*np.sqrt(protein_traces_1.shape[0])/ratio)
        print(protein_traces_1.shape[0]*np.sqrt(protein_traces_1.shape[0])/np.sqrt(2*np.pi)/ratio)
        plt.axvline(frequency_cutoff,ymin = 0)
        plt.xlim(0,0.1)
#         plt.ylim(0,100)
        plt.xlabel('Frequency [1/min]')
        plt.title('Noise weight is ' + '{:.2f}'.format(noise_weight_2))
        plt.ylabel('Power [1e7min]')
        plt.ylim(0,2)

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'noise_weight_illustration.pdf'))

    def xest_noise_impact_on_mean_expression(self):

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,2]<150)))))
#                                     np.logical_and(prior_samples[:,2]<model_results[:,0],
#                                                    model_results[:,2]<150))))))

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]
        
        ## calculate second derivatives:
        mean_proteins = my_model_results[:,0]
        repression_thresholds = my_posterior_samples[:,2]
        hill_coefficients = my_posterior_samples[:,4]
        repression_factors = mean_proteins/repression_thresholds
        second_derivatives = ( hill_coefficients*np.power(repression_factors, hill_coefficients-2)/
                               repression_thresholds**2*(1+np.power(repression_factors,hill_coefficients))**3*
                               (np.power(repression_factors,hill_coefficients)*(hill_coefficients + 1) - hill_coefficients - 1)) 
        expected_change = ( my_posterior_samples[:,0]*my_posterior_samples[:,1]/
                            (my_posterior_samples[:,-1]*my_posterior_samples[:,-2])*mean_proteins*
                            second_derivatives)
#         best_index = np.argmax(my_model_results[:,0] - my_posterior_samples[:,2])
#         best_index = np.argmax((my_model_results[:,0] - my_posterior_samples[:,2])/my_model_results[:,0])
        best_index = np.argmax(expected_change)
#         best_index = np.argmax(second_derivatives)
#         example_parameter_index = 1
        example_parameter = my_posterior_samples[best_index]
        example_results = my_model_results[best_index]
        print(example_results)
        print(example_parameter)
        mRNA_noise = example_parameter[5]*example_results[4]*2
        example_normal_trace = hes5.generate_langevin_trajectory(720.0,
                                                         example_parameter[2], #repression_threshold, 
                                                         example_parameter[4], #hill_coefficient,
                                                         example_parameter[5], #mRNA_degradation_rate, 
                                                         example_parameter[6], #protein_degradation_rate, 
                                                         example_parameter[0], #basal_transcription_rate, 
                                                         example_parameter[1], #translation_rate,
                                                         example_parameter[3], #transcription_delay, 
                                                         10, #initial_mRNA, 
                                                         example_parameter[2], #initial_protein,
                                                         2000)

        example_noise_trace = hes5.generate_agnostic_noise_trajectory(720.0,
                                                         example_parameter[2], #repression_threshold, 
                                                         example_parameter[4], #hill_coefficient,
                                                         example_parameter[5], #mRNA_degradation_rate, 
                                                         example_parameter[6], #protein_degradation_rate, 
                                                         example_parameter[0], #basal_transcription_rate, 
                                                         example_parameter[1], #translation_rate,
                                                         example_parameter[3], #transcription_delay, 
                                                         mRNA_noise, #mrna_noise, 
                                                         0, #protein_noise, 
                                                         10, #initial_mRNA, 
                                                         example_parameter[2], #initial_protein,
                                                         2000)

        example_noisier_trace = hes5.generate_agnostic_noise_trajectory(720.0,
                                                         example_parameter[2], #repression_threshold, 
                                                         example_parameter[4], #hill_coefficient,
                                                         example_parameter[5], #mRNA_degradation_rate, 
                                                         example_parameter[6], #protein_degradation_rate, 
                                                         example_parameter[0], #basal_transcription_rate, 
                                                         example_parameter[1], #translation_rate,
                                                         example_parameter[3], #transcription_delay, 
                                                         mRNA_noise, #mrna_noise, 
#                                                         250000, #protein_noise, 
#                                                         400000, #protein_noise, 
#                                                         200000, #protein_noise, 
                                                        150000, #protein_noise, 
#                                                         1000000,
                                                         10, #initial_mRNA, 
                                                         example_parameter[2], #initial_protein,
                                                         2000)
        
#         noise_strengths = np.array([0,1,10,20,50,100,500,1000,10000,
#                                     100000,400000])
        noise_strengths = np.linspace(0,400000,5)

        new_parameters = np.zeros((len(noise_strengths), len(example_parameter)+2))

        for noise_index, protein_noise_strength in enumerate(noise_strengths):
            new_parameters[noise_index,:len(example_parameter)] = example_parameter
            new_parameters[noise_index,-2] = mRNA_noise
            new_parameters[noise_index,-1] = protein_noise_strength
            
        new_summary_stats = hes5.calculate_summary_statistics_at_parameters(new_parameters, model = 'agnostic')
        
        plt.figure(figsize = (4.5, 6.5))
        plt.subplot(511)
        plt.plot(noise_strengths,new_summary_stats[:,0])
        plt.xlabel('noise rate [1/min]')
        plt.ylabel('mean expression')
#         plt.ylim(2000,10000)
        plt.subplot(512)
        plt.plot(noise_strengths,new_summary_stats[:,1])
        plt.xlabel('noise rate [1/min]')
        plt.ylabel('relative std')
        plt.axhline(my_model_results[best_index,1])
        plt.subplot(513)
        plt.plot(example_normal_trace[:,0],example_normal_trace[:,2])
        plt.xlabel('time')
        plt.ylabel('expression')
        plt.ylim(1000,11000)
        plt.subplot(514)
        plt.plot(example_noise_trace[:,0],example_noise_trace[:,2])
        plt.xlabel('time')
        plt.ylabel('expression')
        plt.ylim(1000,11000)
        plt.subplot(515)
        plt.plot(example_noisier_trace[:,0],example_noisier_trace[:,2])
        plt.xlabel('time')
        plt.ylabel('expression')
        plt.ylim(1000,11000)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'noise_vs_mean_different.pdf'))
        
    def xest_flucutation_rate_dependant_activation(self):
        times = np.linspace(0,15,100000)
#         input_signal_before = np.ones_like(times)*11
#         input_signal_before = np.zeros_like(times)
        input_signal_before = 3*np.sin(2*np.pi*times/2) + 3
        input_signal_after = 3*np.sin(2*np.pi*times/0.5) + 3
        
        delta_t = times[1] - times[0]
        outputs = []
        for signal_index, signal in enumerate([input_signal_before, input_signal_after]):
            index = 0
            output = np.zeros_like(input_signal_before)
            for time in times[:-1]:
                x = output[index]
                this_signal = signal[index]
                dx = 0.5/(1+np.power(this_signal/3,4)) + 10/(1+np.power(x/0.4,-4)) - 3*x
                index+=1
                output[index] = x+dx*delta_t
            outputs.append(output)
        
        plt.figure(figsize = (4.5,4.5))
        plt.subplot(221)
        plt.title('Slow input')
        plt.plot(times,input_signal_before)
        plt.ylabel('Input Signal (Her6)')
        plt.xlabel('Time')
        plt.subplot(222)
        plt.plot(times,outputs[0])
        plt.ylabel('Downstream Response')
        plt.xlabel('Time')
        plt.ylim(0,4)
        plt.subplot(223)
        plt.plot(times,input_signal_after)
        plt.title('Fast input')
        plt.ylabel('Input Signal (Her6)')
        plt.xlabel('Time')
        plt.subplot(224)
        plt.plot(times,outputs[1])
        plt.xlabel('Time')
        plt.ylabel('Downstream Response')
        plt.ylim(0,4)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'fluctuation_rate_dependent_activation.pdf'))

    def xest_stochastic_flucutation_rate_dependant_activation(self):
        
#         fluctuation_rates = [0.25,0.5,0.7,1.0,1.5,2.0,10]
#         fluctuation_rates = np.linspace(0.5,1.5,21)
        fluctuation_rates = [0.05,2.0]
        for fluctuation_rate in fluctuation_rates:
            times, y, x = hes5.simulate_downstream_response_at_fluctuation_rate(fluctuation_rate)
            plt.figure(figsize = (4.5,2.25))
            plt.subplot(121)
            plt.plot(times,y)
            plt.ylim(0,10)
            plt.ylabel('Input Signal (Her6)')
            plt.xlabel('Time')
            plt.subplot(122)
            plt.plot(times,x)
            plt.ylabel('Downstream Response')
            plt.xlabel('Time')
            plt.ylim(0,4)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                     'stochastic_fluctuation_rate_dependent_activation_' + 
                                     '{:.2f}'.format(fluctuation_rate) + '.pdf'))

    def xest_stochastic_flucutation_rate_dependant_activation_figure_draft(self):
        
        number_of_traces = 4
#         fluctuation_rates = [0.25,0.5,0.7,1.0,1.5,2.0,10]
#         fluctuation_rates = np.linspace(0.5,1.5,21)
        color_list = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        fluctuation_rates = [2,15,100]
        for fluctuation_rate in fluctuation_rates:
            times, y, x = hes5.simulate_downstream_response_at_fluctuation_rate(fluctuation_rate, number_of_traces)
            this_figure = plt.figure(figsize = (4.5,2.25))
            outer_grid = matplotlib.gridspec.GridSpec(1, 2 )
            this_left_grid = matplotlib.gridspec.GridSpecFromSubplotSpec(number_of_traces, 1,
                    subplot_spec=outer_grid[0], hspace=0.0)
            for subplot_index in range(number_of_traces):
                this_axis = plt.Subplot(this_figure, this_left_grid[subplot_index])
                this_figure.add_subplot(this_axis)
                plt.plot(times,y.transpose()[subplot_index], lw = 0.5, color = color_list[subplot_index] )
                plt.yticks([2,7], fontsize = 8)
                plt.ylim(0,10)
            plt.ylabel('Input Signal Y')
            plt.xlabel('Time')
            this_axis.yaxis.set_label_coords(-0.15,2.0)
            plt.subplot(122)
            this_axis = plt.Subplot(this_figure, outer_grid[0])
            for trace_index, x_trace in enumerate(x.transpose()):
                plt.plot(times, x_trace, color = color_list[trace_index] )
            plt.ylabel('Downstream Response X')
            plt.xlabel('Time')
            plt.ylim(0,4)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                     'stochastic_multiple_fluctuation_rate_dependent_activation_' + 
                                     '{:.2f}'.format(fluctuation_rate) + '.pdf'))

    def xest_multiple_stochastic_flucutation_rate_dependant_activation(self):
        
        number_of_traces = 4
#         fluctuation_rates = [0.25,0.5,0.7,1.0,1.5,2.0,10]
#         fluctuation_rates = np.linspace(0.5,1.5,21)
        fluctuation_rates = [2,25,100]
        for fluctuation_rate in fluctuation_rates:
            times, y, x = hes5.simulate_downstream_response_at_fluctuation_rate(fluctuation_rate, number_of_traces)
            plt.figure(figsize = (4.5,2.25))
            plt.subplot(121)
            for y_trace in y.transpose():
                plt.plot(times,y_trace, lw = 0.5)
            plt.ylim(0,10)
            plt.ylabel('Input Signal (Her6)')
            plt.xlabel('Time')
            plt.subplot(122)
            for x_trace in x.transpose():
                plt.plot(times,x_trace)
            plt.ylabel('Downstream Response')
            plt.xlabel('Time')
            plt.ylim(0,4)
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                     'stochastic_multiple_fluctuation_rate_dependent_activation_' + 
                                     '{:.2f}'.format(fluctuation_rate) + '.pdf'))

    def xest_switching_vs_fluctuation_rate_paper_figure_draft(self):
        
#         fluctuation_rates = [0.25,0.5,0.7,1.0,1.5,2.0,10]
#         fluctuation_rates = np.linspace(0.5,1.5,21)
#         fluctuation_rates = np.array([0.05,2.0])
#         fluctuation_rates = np.logspace(0,3,10)
        fluctuation_rates = np.linspace(2,100,20)
        number_of_traces = 1000
#         number_of_traces = 10
        percentages = np.zeros_like(fluctuation_rates)
        activation_times = np.zeros_like(fluctuation_rates)
        activation_time_deviations = np.zeros_like(fluctuation_rates) 

        for fluctuation_index, fluctuation_rate in enumerate(fluctuation_rates):
            times, y, x = hes5.simulate_downstream_response_at_fluctuation_rate(fluctuation_rate, number_of_traces)
            turned_on_targets = x[-1,:]>2
            percentages[fluctuation_index] = np.sum(turned_on_targets)/number_of_traces
            active_level_bools = x>2
            these_activation_times = np.zeros(number_of_traces)
            for column_index, column in enumerate(active_level_bools.transpose()):
                entries = np.nonzero(column)
                if len(entries[0]) > 0:
                    minimum_entry = np.min(entries)
                    time = times[minimum_entry]
                    these_activation_times[column_index] = time
                else:
                    these_activation_times[column_index] = times[-1]
            activation_times[fluctuation_index] = np.mean(these_activation_times)
            activation_time_deviations[fluctuation_index] = np.std(these_activation_times)

        plt.figure(figsize = (2.25,2.25))
        plt.plot(fluctuation_rates, percentages)
        plt.xlabel('Y aperiodic lengthscale')
        plt.ylabel('Switching probability')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'stochastic_fluctuation_rate_probability_draft_figure.pdf'))

    def xest_switching_vs_fluctuation_rate(self):
        
#         fluctuation_rates = [0.25,0.5,0.7,1.0,1.5,2.0,10]
#         fluctuation_rates = np.linspace(0.5,1.5,21)
#         fluctuation_rates = np.array([0.05,2.0])
#         fluctuation_rates = np.logspace(0,3,10)
        fluctuation_rates = np.linspace(2,100,20)
        number_of_traces = 400
        percentages = np.zeros_like(fluctuation_rates)
        activation_times = np.zeros_like(fluctuation_rates)
        activation_time_deviations = np.zeros_like(fluctuation_rates) 

        for fluctuation_index, fluctuation_rate in enumerate(fluctuation_rates):
            times, y, x = hes5.simulate_downstream_response_at_fluctuation_rate(fluctuation_rate, number_of_traces)
            turned_on_targets = x[-1,:]>2
            percentages[fluctuation_index] = np.sum(turned_on_targets)/number_of_traces
            active_level_bools = x>2
            these_activation_times = np.zeros(number_of_traces)
            for column_index, column in enumerate(active_level_bools.transpose()):
                entries = np.nonzero(column)
                if len(entries[0]) > 0:
                    minimum_entry = np.min(entries)
                    time = times[minimum_entry]
                    these_activation_times[column_index] = time
                else:
                    these_activation_times[column_index] = times[-1]
            activation_times[fluctuation_index] = np.mean(these_activation_times)
            activation_time_deviations[fluctuation_index] = np.std(these_activation_times)

        plt.figure(figsize = (4.5,2.25))
        plt.subplot(121)
#         plt.gca().set_xscale("log", nonposx='clip')
        plt.errorbar(fluctuation_rates, activation_times, yerr = activation_time_deviations)
        plt.ylabel('Time to activation')
        plt.xlabel('Her6 aperiodic lengthscale')
        plt.subplot(122)
#         plt.gca().set_xscale("log", nonposx='clip')
        plt.plot(fluctuation_rates, percentages)
        plt.xlabel('Her6 aperiodic lengthscale')
        plt.ylabel('Switching probability')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'stochastic_fluctuation_rate_probability_analysis.pdf'))

    def xest_e_make_relative_parameter_variation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
#         number_of_parameter_points = 2
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
#         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,2]<150)))))
       
        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories,
                                                                                     relative = True,
                                                                                     relative_range = (0.1,2.0))
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_relative_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_d_make_dual_parameter_variation(self, 
                                             quadrant_index = 'all',
                                             model = 'standard'):
        number_of_trajectories = 200

        degradation_ranges = dict()
        degradation_ranges[1] = (0.6, 1.0)
        degradation_ranges[2] = (0.6, 1.0)
        degradation_ranges[3] = (0.1, 0.5)
        degradation_ranges[4] = (0.1, 0.5)
        degradation_ranges[5] = (0.6, 1.0)
        degradation_ranges[6] = (0.1, 0.5)
        degradation_ranges[7] = (0.1, 0.5)
        degradation_ranges[8] = (0.6, 1.0)
        degradation_ranges[9] = (1.1, 1.5)
        degradation_ranges[10] = (1.1, 1.5)
        degradation_ranges[11] = (1.1, 1.5)
        degradation_ranges[12] = (1.1, 1.5)
        degradation_ranges[13] = (1.6, 2.0)
        degradation_ranges[14] = (1.6, 2.0)
        degradation_ranges[15] = (1.6, 2.0)
        degradation_ranges[16] = (1.6, 2.0)
        degradation_ranges['all'] = (0.1, 2.0)
        degradation_ranges['shifted'] = (0.1, 2.0)
        degradation_ranges['shifted_more'] = (0.1, 2.0)
        degradation_ranges['shifted_final'] = (0.3, 1.0)

        degradation_interval_numbers = { i: 5 for i in range(1,17)}
        degradation_interval_numbers['all'] = 20
        degradation_interval_numbers['shifted'] = 20
        degradation_interval_numbers['shifted_more'] = 20
        degradation_interval_numbers['shifted_final'] = 8
        
        translation_ranges = dict()
        translation_ranges[1] = (1.0, 1.5)
        translation_ranges[2] = (1.6, 2.0)
        translation_ranges[3] = (1.0, 1.5)
        translation_ranges[4] = (1.6, 2.0)
        translation_ranges[5] = (0.5, 0.9)
        translation_ranges[6] = (0.5, 0.9)
        translation_ranges[7] = (0.1, 0.4)
        translation_ranges[8] = (0.1, 0.4)
        translation_ranges[9] = (1.0, 1.5)
        translation_ranges[10] = (1.6, 2.0)
        translation_ranges[11] = (0.5, 0.9)
        translation_ranges[12] = (0.1, 0.4)
        translation_ranges[13] = (1.0, 1.5)
        translation_ranges[14] = (0.5, 0.9)
        translation_ranges[15] = (1.6, 2.0)
        translation_ranges[16] = (0.1, 0.4)
        translation_ranges['all'] = (0.1, 2.0)
        translation_ranges['shifted'] = (0.9, 3.1)
        translation_ranges['shifted_more'] = (3.2, 4.1)
        translation_ranges['shifted_final'] = (2.5, 4.5)

        translation_interval_numbers = dict()
        translation_interval_numbers[1] = 6
        translation_interval_numbers[2] = 5
        translation_interval_numbers[3] = 6
        translation_interval_numbers[4] = 5
        translation_interval_numbers[5] = 5
        translation_interval_numbers[6] = 5
        translation_interval_numbers[7] = 4
        translation_interval_numbers[8] = 4
        translation_interval_numbers[9] = 6
        translation_interval_numbers[10] = 5
        translation_interval_numbers[11] = 5
        translation_interval_numbers[12] = 4
        translation_interval_numbers[13] = 6
        translation_interval_numbers[14] = 5
        translation_interval_numbers[15] = 5
        translation_interval_numbers[16] = 4
        translation_interval_numbers['all'] = 20
        translation_interval_numbers['shifted'] = 23
        translation_interval_numbers['shifted_more'] = 10
        translation_interval_numbers['shifted_final'] = 21

#         number_of_parameter_points = 2
#         number_of_trajectories = 2

        if model == 'standard':
            saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_delay')
#             saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish')
        if model == 'standard_large':
            saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_large')
        elif model == 'extrinsic_noise':
            saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise_delay')
#             saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise')
        elif model == 'extrinsic_noise_large':
            saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise_massive')
            
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
#         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,2]<150)))))
       
        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_dual_parameter_sweep_at_parameters(my_posterior_samples,
                                                                                     degradation_range = degradation_ranges[quadrant_index],
                                                                                     translation_range = translation_ranges[quadrant_index],
                                                                                     degradation_interval_number = degradation_interval_numbers[quadrant_index],
                                                                                     translation_interval_number = translation_interval_numbers[quadrant_index],
                                                                                     number_of_traces_per_parameter = number_of_trajectories)
        
#         self.assertEqual(my_parameter_sweep_results.shape, (len(my_posterior_samples),
#                                                             number_of_parameter_points,
#                                                             number_of_parameter_points,
#                                                             13))
        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_' + model 
                             + '_' + str(quadrant_index) +'.npy'),
                    my_parameter_sweep_results)

    def xest_reconstruct_dual_parameter_variation_matrix(self): 
        saving_path_root = os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_')
        all_sub_matrices = []
        for quadrant_index in range(1,13):
            this_saving_path = saving_path_root + str(quadrant_index) + '.npy'
            all_sub_matrices.append(np.load(this_saving_path))
            
        this_full_matrix = np.zeros((len(all_sub_matrices[0]),15,20,14))
        for parameter_index in range(len(all_sub_matrices[0])):
            this_12_matrix = np.hstack((all_sub_matrices[7][parameter_index],
                                        all_sub_matrices[4][parameter_index],
                                        all_sub_matrices[0][parameter_index],
                                        all_sub_matrices[1][parameter_index]))
            this_34_matrix = np.hstack((all_sub_matrices[6][parameter_index],
                                        all_sub_matrices[5][parameter_index],
                                        all_sub_matrices[2][parameter_index],
                                        all_sub_matrices[3][parameter_index]))
            this_9_10_matrix = np.hstack((all_sub_matrices[11][parameter_index],
                                          all_sub_matrices[10][parameter_index],
                                          all_sub_matrices[8][parameter_index],
                                          all_sub_matrices[9][parameter_index]))
            this_full_matrix[parameter_index] = np.vstack((this_34_matrix,
                                                           this_12_matrix, 
                                                           this_9_10_matrix))
            
        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_complete_matrix.npy'),
                    this_full_matrix)

    def xest_e_make_dual_parameter_variation_low_res(self):
        number_of_parameter_points = 10
        number_of_trajectories = 200
#         number_of_parameter_points = 2
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
#         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,2]<150)))))
        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_dual_parameter_sweep_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories,
                                                                                     relative_range = (0.2,1.8))
        
#         self.assertEqual(my_parameter_sweep_results.shape, (len(my_posterior_samples),
#                                                             number_of_parameter_points,
#                                                             number_of_parameter_points,
#                                                             13))
        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_low_res.npy'),
                    my_parameter_sweep_results)


    def xest_make_dual_parameter_variation_lengthscale_calculation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
        relative_range = (0.1,2.0)
#         number_of_parameter_points = 2
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
#         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,2]<150)))))

        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]

        dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps.npy'))

        # first: make a table of 7d parameters
        total_number_of_parameters_required = my_posterior_samples.shape[0]*(number_of_parameter_points**2)
        kept_parameter_values = np.zeros((total_number_of_parameters_required, 7)) 
        kept_sample_index = 0
        parameters_counted = 0
        list_of_reference_indices = []
        for sample_index, sample in enumerate(my_posterior_samples):
            degradation_proportion_index = 0
            for degradation_proportion in np.linspace(relative_range[0],relative_range[1],number_of_parameter_points):
                translation_proportion_index = 0
                for translation_proportion in np.linspace(relative_range[0],relative_range[1],number_of_parameter_points):
                    this_result_after = dual_sweep_results[sample_index,
                                                           degradation_proportion_index,
                                                           translation_proportion_index]
                    condition = ( this_result_after[2]< my_posterior_results[sample_index,0]*2.2 and
                                  this_result_after[2]> my_posterior_results[sample_index,0]*1.8 and
                                  this_result_after[5]< my_posterior_results[sample_index,3])
                    if condition:
                        kept_parameter_values[kept_sample_index] = sample
                        # now replace the parameter of interest with the actual parameter value
                        # degradation rate
                        kept_parameter_values[kept_sample_index, 5] *= degradation_proportion
                        # translation rate
                        kept_parameter_values[kept_sample_index, 1] *= translation_proportion
                        kept_sample_index += 1
                        list_of_reference_indices.append(sample_index)
                    translation_proportion_index += 1
                degradation_proportion_index += 1
        
        list_of_reference_indices_as_np = np.array(list_of_reference_indices)
        unique_list_of_reference_indices = np.unique(list_of_reference_indices_as_np)
        for additional_sample_index in unique_list_of_reference_indices:
            kept_parameter_values[kept_sample_index] = my_posterior_samples[additional_sample_index]
            kept_sample_index += 1
            
        kept_parameter_values = kept_parameter_values[:kept_sample_index]
        print(kept_parameter_values)
        print(len(kept_parameter_values))
    
        # pass these parameters to the calculate_summary_statistics_at_parameter_points
        all_fluctuation_rates = hes5.calculate_fluctuation_rates_at_parameters(parameter_values = kept_parameter_values, 
                                                                            number_of_traces_per_sample = number_of_trajectories,
                                                                            number_of_cpus = number_of_available_cores,
                                                                            sampling_duration = 12*60)
        
        # unpack and wrap the results in the output format
        fluctuation_rate_results = np.zeros((my_posterior_samples.shape[0], 
                                             number_of_parameter_points,
                                             number_of_parameter_points, 
                                             3))
        kept_sample_index = 0
        for sample_index, sample in enumerate(my_posterior_samples):
            degradation_proportion_index = 0
            for degradation_proportion in np.linspace(relative_range[0],relative_range[1],number_of_parameter_points):
                translation_proportion_index = 0
                for translation_proportion in np.linspace(relative_range[0],relative_range[1],number_of_parameter_points):
                    this_result_after = dual_sweep_results[sample_index,
                                                           degradation_proportion_index,
                                                           translation_proportion_index]
                    condition = ( this_result_after[2]< my_posterior_results[sample_index,0]*2.2 and
                                  this_result_after[2]> my_posterior_results[sample_index,0]*1.8 and
                                  this_result_after[5]< my_posterior_results[sample_index,3])
                    fluctuation_rate_results[sample_index,degradation_proportion_index,
                                             translation_proportion_index,:2] = [degradation_proportion, translation_proportion]
                    if condition:
                        this_fluctuation_rate = all_fluctuation_rates[kept_sample_index]
                        # the first entry gets the degradation rate
                        # the remaining entries get the summary statistics. We discard the last summary statistic, 
                        # which is the mean mRNA
                        fluctuation_rate_results[sample_index,degradation_proportion_index,
                                      translation_proportion_index,2] = this_fluctuation_rate
                        kept_sample_index+= 1
                    else:
                        fluctuation_rate_results[sample_index,degradation_proportion_index,
                                      translation_proportion_index,2] = np.nan
                    translation_proportion_index+=1
                degradation_proportion_index+=1
 
        for additional_sample_index in unique_list_of_reference_indices:
            this_fluctuation_rate = all_fluctuation_rates[kept_sample_index]
            fluctuation_rate_results[additional_sample_index,9,9,2] = this_fluctuation_rate
            kept_sample_index += 1

        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_fluctuation_rates.npy'),
                    fluctuation_rate_results)

    def xest_make_dual_parameter_variation_full_lengthscale_calculation(self):
        number_of_parameter_points = 20
        number_of_trajectories = 200
        relative_range = (0.1,2.0)
#         number_of_parameter_points = 2
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
#         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,2]<150)))))

        my_posterior_samples = prior_samples[accepted_indices][:2]
        my_posterior_results = model_results[accepted_indices][:2]

        dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps.npy'))[:2]

        # first: make a table of 7d parameters
        total_number_of_parameters_required = my_posterior_samples.shape[0]*(number_of_parameter_points**2)
        kept_parameter_values = np.zeros((total_number_of_parameters_required, 7)) 
        kept_sample_index = 0
        parameters_counted = 0
        list_of_reference_indices = []
        for sample_index, sample in enumerate(my_posterior_samples):
            degradation_proportion_index = 0
            for degradation_proportion in np.linspace(relative_range[0],relative_range[1],number_of_parameter_points):
                translation_proportion_index = 0
                for translation_proportion in np.linspace(relative_range[0],relative_range[1],number_of_parameter_points):
                    this_result_after = dual_sweep_results[sample_index,
                                                           degradation_proportion_index,
                                                           translation_proportion_index]
                    kept_parameter_values[kept_sample_index] = sample
                    # now replace the parameter of interest with the actual parameter value
                    # degradation rate
                    kept_parameter_values[kept_sample_index, 5] *= degradation_proportion
                    # translation rate
                    kept_parameter_values[kept_sample_index, 1] *= translation_proportion
                    kept_sample_index += 1
                    translation_proportion_index += 1
                degradation_proportion_index += 1
        
        print(kept_parameter_values)
        print(len(kept_parameter_values))
    
        # pass these parameters to the calculate_summary_statistics_at_parameter_points
        all_fluctuation_rates = hes5.calculate_fluctuation_rates_at_parameters(parameter_values = kept_parameter_values, 
                                                                            number_of_traces_per_sample = number_of_trajectories,
                                                                            number_of_cpus = number_of_available_cores,
                                                                            sampling_duration = 12*60)
        
        # unpack and wrap the results in the output format
        fluctuation_rate_results = np.zeros((my_posterior_samples.shape[0], 
                                             number_of_parameter_points,
                                             number_of_parameter_points, 
                                             3))
        kept_sample_index = 0
        for sample_index, sample in enumerate(my_posterior_samples):
            degradation_proportion_index = 0
            for degradation_proportion in np.linspace(relative_range[0],relative_range[1],number_of_parameter_points):
                translation_proportion_index = 0
                for translation_proportion in np.linspace(relative_range[0],relative_range[1],number_of_parameter_points):
                    this_result_after = dual_sweep_results[sample_index,
                                                           degradation_proportion_index,
                                                           translation_proportion_index]
                    fluctuation_rate_results[sample_index,degradation_proportion_index,
                                             translation_proportion_index,:2] = [degradation_proportion, translation_proportion]
                    this_fluctuation_rate = all_fluctuation_rates[kept_sample_index]
                    # the first entry gets the degradation rate
                    # the remaining entries get the summary statistics. We discard the last summary statistic, 
                    # which is the mean mRNA
                    fluctuation_rate_results[sample_index,degradation_proportion_index,
                                  translation_proportion_index,2] = this_fluctuation_rate
                    kept_sample_index+= 1
                    translation_proportion_index+=1
                degradation_proportion_index+=1
 
        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_fluctuation_rates_full.npy'),
                    fluctuation_rate_results)

    def xest_plot_dual_parameter_change(self):
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish')
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_delay')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise_delay')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,2]<150)))))
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]
        
        print('number of accepted samples')
        print(len(my_posterior_results))
        dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_shifted_final.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_standard_shifted_final.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_shifted_more.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_standard_shifted_more.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_shifted.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_standard_shifted.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_complete_matrix.npy'))
#         dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_extrinsic_noise_all.npy'))
#         fluctuation_rate_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_fluctuation_rates.npy'))
#         fluctuation_rate_results = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                                         'zebrafish_dual_sweeps_fluctuation_rates_full.npy'))

        translation_changes = dual_sweep_results[0,0,:,1]
        degradation_changes = dual_sweep_results[0,:,0,0]
        X, Y = np.meshgrid(translation_changes, degradation_changes)

        # need to replace this with something
        likelihoods = np.zeros((degradation_changes.shape[0],
                                translation_changes.shape[0]))
        
#         fluctuation_rates_before = fluctuation_rate_results[:,9,9,2]
#         fluctuation_rates_before = dual_sweep_results[:,9,9,-1]
        fluctuation_rates_before = my_posterior_results[:,-1]
#         print(fluctuation_rates_before)
        total_condition_mask = np.zeros(len(dual_sweep_results))
        for translation_index, translation_change in enumerate(translation_changes):
            for degradation_index, degradation_change in enumerate(degradation_changes):
                these_results_after = dual_sweep_results[:, 
                                                         degradation_index, 
                                                         translation_index, 
                                                         :]
#                 these_fluctuation_rates_after = fluctuation_rate_results[:, 
#                                                          degradation_index, 
#                                                          translation_index, 
#                                                          :]

                relative_noise_after = ( these_results_after[:,-1]/np.power(these_results_after[:,3]*
                                         these_results_after[:,2],2))
                relative_noise_before = ( my_posterior_results[:,-1]/np.power(my_posterior_results[:,1]*
                                          my_posterior_results[:,0],2))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                                 these_results_after[:,2]>my_posterior_results[:,0]*1.8)
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                                 these_results_after[:,5]<my_posterior_results[:,3]))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]>0.1,
#                                                 these_results_after[:,5]<my_posterior_results[:,3])))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2] >my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5] <my_posterior_results[:,3],
#                                                 these_results_after[:,4] <150)))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2] >my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5] <my_posterior_results[:,3],
#                                                 relative_noise_after>relative_noise_before)))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                 np.logical_and(relative_noise_after>relative_noise_before,
#                                                 these_results_after[:,4]<150))))
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.5,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.5,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                                 these_fluctuation_rates_after[:,2]>fluctuation_rates_before)))
                condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
                                np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
                                np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
                                                these_results_after[:,-1]>fluctuation_rates_before)))
#                 condition_mask = np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                                 these_results_after[:,-1]>fluctuation_rates_before)
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                 np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                 np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
#                                 np.logical_and(these_results_after[:,4]<150,
#                                                 these_results_after[:,-1]>fluctuation_rates_before))))
#                                                 these_fluctuation_rates_after[:,2]>fluctuation_rates_before))))
#                 condition_mask = these_fluctuation_rates_after[:,2]>fluctuation_rates_before
#                 condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                                 these_results_after[:,2]>my_posterior_results[:,0]*1.8)
#                 condition_mask = np.logical_and(these_results_after[:,2]<5000,
#                                 np.logical_and(these_results_after[:,2]>2000,
#                                                 these_results_after[:,5]<my_posterior_results[:,3]))
                
                total_condition_mask += condition_mask
                likelihoods[degradation_index, translation_index] = np.sum(condition_mask)
                
        print(likelihoods)
        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_likelihoods.npy'),
                total_condition_mask)
        
        print(translation_changes)
        translation_step = translation_changes[1] - translation_changes[0]
        left_translation_boundary = translation_changes[0] - 0.5*translation_step
        right_translation_boundary = translation_changes[-1] + 0.5*translation_step
        translation_bin_edges = np.linspace(left_translation_boundary,right_translation_boundary, len(translation_changes) +1)
        print(translation_bin_edges)

        degradation_step = degradation_changes[1] - degradation_changes[0]
        left_degradation_boundary = degradation_changes[0] - 0.5*degradation_step
        right_degradation_boundary = degradation_changes[-1] + 0.5*degradation_step
        degradation_bin_edges = np.linspace(left_degradation_boundary,right_degradation_boundary, len(degradation_changes) +1)
        print(degradation_bin_edges)
        
#         print('likelihood of dec. coh. and incr. fluct. at 1.5 transl.')
#         print(likelihoods[9,14])

        this_figure = plt.figure(figsize = (2.5,1.9))
        colormesh = plt.pcolormesh(translation_bin_edges,degradation_bin_edges,likelihoods, rasterized = True)
#         colormesh = plt.pcolormesh(translation_bin_edges,degradation_bin_edges,likelihoods, rasterized = True,
#                                    vmin = 0, vmax = 10)
#         colormesh = plt.pcolormesh(likelihoods, rasterized = True)
#         colormesh = plt.pcolormesh(degradation_bin_edges,translation_bin_edges,likelihoods, rasterized = True,
#                                     vmin = 0, vmax = 100)
#         plt.pcolor(X,Y,expected_coherence)
#         plt.scatter(np.log(2)/90, np.log(2)/30)
        plt.xlabel("Translation proportion", labelpad = 1.3)
#         plt.xlim(0.95,2.05)
#         plt.ylim(0.25,1.05)
#         plt.ylim(0.05,1.05)
        plt.ylabel("Degradation\nproportion", y=0.4)
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.new_vertical(size=0.07, pad=0.5, pack_start=True)
        this_figure.add_axes(cax)

        tick_locator = mpl.ticker.MaxNLocator(nbins=5)
        this_colorbar = this_figure.colorbar(colormesh, cax = cax, orientation = 'horizontal')
        this_colorbar.locator = tick_locator
        this_colorbar.update_ticks()
#         for ticklabel in this_colorbar.ax.get_xticklabels():
#             ticklabel.set_horizontalalignment('left') 
        this_colorbar.ax.set_ylabel('Likelihood\nof change', rotation = 0, verticalalignment = 'top', labelpad = 30)
        plt.tight_layout(pad = 0.05)
#         plt.tight_layout()

        file_name = os.path.join(os.path.dirname(__file__),
#                                  'output','zebrafish_likelihood_plot_extrinsic_noise')
                                 'output','zebrafish_likelihood_plot_shifted_final')
 
        plt.savefig(file_name + '.pdf', dpi = 600)
        plt.savefig(file_name + '.eps', dpi = 600)
        plt.savefig(file_name + '.png', dpi = 600)

    def xest_fit_model_by_optimization(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_large')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                                   model_results[:,2]<150)))))
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]
        
        dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_complete_matrix.npy'))
        translation_changes = dual_sweep_results[0,0,:,1]
        degradation_changes = dual_sweep_results[0,:,0,0]
        fluctuation_rates_before = dual_sweep_results[:,9,9,-1]

        total_condition_mask = np.zeros(len(dual_sweep_results))
        list_of_indices = []
        corresponding_proportions = []
        for translation_index, translation_change in enumerate(translation_changes):
            if translation_change>0.9:
                for degradation_index, degradation_change in enumerate(degradation_changes):
                    these_results_after = dual_sweep_results[:, 
                                                             degradation_index, 
                                                             translation_index, 
                                                             :]

                    relative_noise_after = ( these_results_after[:,-1]/np.power(these_results_after[:,3]*
                                             these_results_after[:,2],2))
                    relative_noise_before = ( my_posterior_results[:,-1]/np.power(my_posterior_results[:,1]*
                                              my_posterior_results[:,0],2))
#                     condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
#                                     np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
#                                     np.logical_and(these_results_after[:,5]<0.9*my_posterior_results[:,3],
#                                                     these_results_after[:,4]<140)))
                    condition_mask = np.logical_and(these_results_after[:,2]<my_posterior_results[:,0]*2.2,
                                    np.logical_and(these_results_after[:,2]>my_posterior_results[:,0]*1.8,
                                    np.logical_and(these_results_after[:,5]<my_posterior_results[:,3],
                                                    these_results_after[:,-1]>fluctuation_rates_before)))
#                                                     these_results_after[:,4]<140)))
#                                     np.logical_and(these_results_after[:,4]<150,
#                                                     these_results_after[:,-1]>fluctuation_rates_before))))
#                     

                    these_indices = np.where(condition_mask)[0]
                    if len(these_indices>0):
                        for item in these_indices:
                            list_of_indices.append(item)
                            corresponding_proportions.append((degradation_change, translation_change))
 
        reference_index = 0
        example_index = list_of_indices[reference_index]
        example_parameter_before = my_posterior_samples[example_index]
        example_parameter_after = np.copy(example_parameter_before)
        degradation_proportion_after=corresponding_proportions[reference_index][0]
        translation_proportion_after=corresponding_proportions[reference_index][1]
        
        full_initial_parameter = np.zeros(9)
        full_initial_parameter[:7] = example_parameter_before
        full_initial_parameter[7:] = [degradation_proportion_after, translation_proportion_after]

        mean_expression_before = lambda parameter : hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter[:-2])[0]
        mean_expression_before_constraint = scipy.optimize.NonlinearConstraint(mean_expression_before,
                                                                               1000,2500)
        def relative_expression_after(parameter):
            this_mean_expression_before = mean_expression_before(parameter)
            parameter_after = parameter[:-2].copy()
            parameter_after[5]*=parameter[-2]
            parameter_after[1]*=parameter[-1]
            this_mean_expression_after = hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter_after[:-2])[0]
            relative_expression_after = this_mean_expression_after/this_mean_expression_before
            print('relative_expression_after is')
            print(parameter)
            print(relative_expression_after)
            return relative_expression_after
        mean_expression_after_constraint = scipy.optimize.NonlinearConstraint(relative_expression_after,
                                                                              1.8,2.2)
        
        period_before = lambda parameter : hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter[:-2])[2]
        period_before_constraint = scipy.optimize.NonlinearConstraint(period_before,
                                                                      0,150)
        def period_after(parameter):
            parameter_after = parameter[:-2].copy()
            parameter_after[5]*=parameter[-2]
            parameter_after[1]*=parameter[-1]
            period_after = hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter[:-2])[2]
            print('period after is')
            print(parameter)
            print(period_after)
            return period_after
        period_after_constraint = scipy.optimize.NonlinearConstraint(period_after,
                                                                     0,150)
        
        def period_difference_after(parameter):
            this_period_before = period_before(parameter)
            this_period_after = period_after(parameter)
            print('period_difference after is')
            print(parameter)
            print(this_period_after - this_period_before)
            return this_period_after - this_period_before

        coherence_before = lambda parameter : hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter[:-2])[3]
        def coherence_difference_after(parameter):
            this_coherence_before = coherence_before(parameter)
            parameter_after = parameter[:-2].copy()
            parameter_after[5]*=parameter[-2]
            parameter_after[1]*=parameter[-1]
            this_coherence_after = hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter[:-2])[3]
            print('coherence_difference_after is')
            print(parameter)
            print(this_coherence_after - this_coherence_before)
            return this_coherence_after - this_coherence_before
        coherence_after_constraint = scipy.optimize.NonlinearConstraint(coherence_difference_after,
                                                                        -100,0)
        
        fluctuation_rate_before = lambda parameter : hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter[:-2])[-1]
        def fluctuation_rate_difference(parameter):
            this_fluctuation_rate_before = fluctuation_rate_before(parameter)
            parameter_after = parameter[:-2].copy()
            parameter_after[5]*=parameter[-2]
            parameter_after[1]*=parameter[-1]
            this_fluctuation_rate_after = hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter[:-2])[-1]
            print('fluctuation_rate_difference is')
            print(parameter)
            print(- (this_fluctuation_rate_after - this_fluctuation_rate_before))
            return - (this_fluctuation_rate_after - this_fluctuation_rate_before)
        flucutation_rate_difference_constraint = scipy.optimize.NonlinearConstraint(fluctuation_rate_difference,
                                                                                    -10000,0)
 
        def fluctuation_rate_difference_minimisation_function(parameter):
            parameter_after = parameter[:-2].copy()
            parameter_after[5]*=parameter[-2]
            parameter_after[1]*=parameter[-1]
            summary_statistics_before = hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter[:-2])
            summary_statistics_after = hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter_after)
            print('next parameter and summary stats are')
            print(parameter)
            print(summary_statistics_before)
            print(summary_statistics_after)
            this_fluctuation_rate_difference = - (summary_statistics_after[-1] - summary_statistics_before[-1])
            this_mean_expression_before = summary_statistics_before[0]
            this_mean_expression_after = summary_statistics_after[0]
            this_relative_mean_expression_after = this_mean_expression_after/this_mean_expression_before
            this_period_before = summary_statistics_before[2]
            this_period_after = summary_statistics_after[2]
            this_period_difference_after = this_period_after - this_period_before
            this_coherence_before = summary_statistics_before[3]
            this_coherence_after = summary_statistics_after[3]
            this_coherence_difference_after = this_coherence_after - this_coherence_before
            if not (this_mean_expression_before<2500 and
                    this_mean_expression_before>1000 and
                    this_period_before<150 and
                    this_period_after<150 and
                    this_coherence_difference_after <0 and
                    parameter[0]>0.001 and
                    parameter[0]<60 and
                    parameter[1]>0.001 and
                    parameter[1]<40 and
                    parameter[2]>0 and
                    parameter[2]<6000 and
                    parameter[3]>5 and
                    parameter[3]<40 and
                    parameter[4]>2 and
                    parameter[4]<6 and
                    parameter[5]>np.log(2)/11 and
                    parameter[5]<np.log(2)/1 and
                    parameter[6]>np.log(2)/11.1 and
                    parameter[6]<np.log(2)/10.9 and
                    parameter[7]>0.0 and
                    parameter[7]<1.0 and
                    parameter[8]>0.99 and
                    parameter[8]<10.0
                    ):
                this_fluctuation_rate_difference = 1000
            print(this_fluctuation_rate_difference)
            return this_fluctuation_rate_difference
                
        def period_difference_minimisation_function(parameter):
            parameter_after = parameter[:-2].copy()
            parameter_after[5]*=parameter[-2]
            parameter_after[1]*=parameter[-1]
            summary_statistics_before = hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter[:-2])
            summary_statistics_after = hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter_after)
            print('next parameter and summary stats are')
            print(parameter)
            print(summary_statistics_before)
            print(summary_statistics_after)
            this_fluctuation_rate_difference = summary_statistics_after[-1] - summary_statistics_before[-1]
            this_mean_expression_before = summary_statistics_before[0]
            this_mean_expression_after = summary_statistics_after[0]
            this_relative_mean_expression_after = this_mean_expression_after/this_mean_expression_before
            this_period_before = summary_statistics_before[2]
            this_period_after = summary_statistics_after[2]
            this_period_difference_after = this_period_after - this_period_before
            this_coherence_before = summary_statistics_before[3]
            this_coherence_after = summary_statistics_after[3]
            this_coherence_difference_after = this_coherence_after - this_coherence_before
            if not (this_mean_expression_before<2500 and
                    this_mean_expression_before>1000 and
                    this_period_before<150 and
                    this_coherence_difference_after <0 and
                    this_fluctuation_rate_difference > 0 and
                    parameter[0]>0.001 and
                    parameter[0]<60 and
                    parameter[1]>0.001 and
                    parameter[1]<40 and
                    parameter[2]>0 and
                    parameter[2]<6000 and
                    parameter[3]>5 and
                    parameter[3]<40 and
                    parameter[4]>2 and
                    parameter[4]<6 and
                    parameter[5]>np.log(2)/11 and
                    parameter[5]<np.log(2)/1 and
                    parameter[6]>np.log(2)/11.1 and
                    parameter[6]<np.log(2)/10.9 and
                    parameter[7]>0.0 and
                    parameter[7]<1.0 and
                    parameter[8]>0.99 and
                    parameter[8]<10.0
                    ):
                this_period_difference_after = 100000
            print(this_period_difference_after)
            return this_period_difference_after
 
#         result = scipy.optimize.minimize(fluctuation_rate_difference,full_initial_parameter, 
#                                          constraints = (mean_expression_before_constraint,
#                                                         mean_expression_after_constraint,
#                                                         period_before_constraint,
#                                                         period_after_constraint,
#                                                         coherence_after_constraint),
#                                          bounds = [(0.001,60),(0.001,40),(0,6000),(5,40),(2,6),(np.log(2)/11,np.log(2)/1),
#                                                    (np.log(2)/11,np.log(2)/11),(0.0,1.0),(1.0,10.0)], 
#                                          options = {'disp':True},
#                                          tol = 0.01)

#         result = scipy.optimize.minimize(period_difference_after,full_initial_parameter, 
#                                          constraints = (mean_expression_before_constraint,
#                                                         mean_expression_after_constraint,
#                                                         period_before_constraint,
#                                                         period_after_constraint,
#                                                         flucutation_rate_difference_constraint,
#                                                         coherence_after_constraint),
#                                          bounds = [(0.001,60),(0.001,40),(0,6000),(5,40),(2,6),(np.log(2)/11,np.log(2)/1),
#                                                    (np.log(2)/11,np.log(2)/11),(0.0,1.0),(1.0,10.0)], 
#                                          options = {'disp':True},
#                                          tol = 10.0)
#         

#         result = scipy.optimize.minimize(fluctuation_rate_difference_minimisation_function,
#                                         full_initial_parameter, 
#                                         method = 'Nelder-Mead',
#                                         method = 'Powell',
#                                         options = {'disp':True},
#                                         tol = 0.001)

        result = scipy.optimize.minimize(period_difference_minimisation_function,
                                        full_initial_parameter, 
                                        method = 'Nelder-Mead',
                                        options = {'disp':True},
                                        tol = 1)

#         result = scipy.optimize.minimize(fluctuation_rate_difference_minimisation_function,
#                                         full_initial_parameter, 
#                                         options = {'disp':True},
#                                         tol = 0.1)

        print(result.x)
        this_parameter = np.array(result.x)
        this_parameter_after = this_parameter[:-2].copy()
        this_parameter_after[5]*=this_parameter[-2]
        this_parameter_after[1]*=this_parameter[-1]
        results_before = hes5.calculate_langevin_summary_statistics_at_parameter_point(this_parameter[:-2])
        results_after = hes5.calculate_langevin_summary_statistics_at_parameter_point(this_parameter_after)
        print(results_before)
        print(results_after)
        
    def xest_extrinsic_noise_trajectory_and_summary_stats(self):
        my_trajectory = hes5.generate_langevin_trajectory( duration = 1500,
                                                           repression_threshold = 2000,
                                                           mRNA_degradation_rate = np.log(2)/11,
                                                           protein_degradation_rate = np.log(2)/11,
                                                           translation_rate = 26,
                                                           basal_transcription_rate = 9,
                                                           transcription_delay = 18,
                                                           hill_coefficient = 3,
                                                           initial_mRNA = 3,
                                                           initial_protein = 2000,
                                                           extrinsic_noise_rate = 100 )

        
        self.assertGreaterEqual(np.min(my_trajectory),0.0)
        figuresize = (4,2.5)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1]*100, label = 'mRNA*100', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.text(0.95, 0.4, 'Mean protein number: ' + str(np.mean(my_trajectory[:,2])),
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=plt.gca().transAxes)
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','her6_intrinsic_noise_langevin_trajectory.pdf'))
        
        print('attempting to calculate summary statistics from model')
        this_parameter = np.array([9,26,2000,18,3,np.log(2)/8,np.log(2)/11,10])
        these_summary_statistics = hes5.calculate_langevin_summary_statistics_at_parameter_point(this_parameter)
        print(these_summary_statistics)
    
    def test_perform_abc_with_extrinsic_noise(self):
        print('starting zebrafish abc')
        ## generate posterior samples
        total_number_of_samples = 200000
#         total_number_of_samples = 5
#         total_number_of_samples = 100
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.6,60),
                        'translation_rate' : (0.04,40),
                        'repression_threshold' : (0,5000),
                        'time_delay' : (1,30),
                        'hill_coefficient' : (2,6),
                        'protein_degradation_rate' : ( np.log(2)/11.0, np.log(2)/11.0 ),
                        'mRNA_half_life' : ( 1, 11),
                        'extrinsic_noise_rate' : (0.1,1000) }

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_zebrafish_extrinsic_noise_delay_large',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'extrinsic_noise',
                                                                logarithmic = True )
 

    def xest_plot_zebrafish_inference_extrinsic_noise(self):
#         option = 'prior'
#         option = 'mean_period_and_coherence'
#         option = 'mean_longer_periods_and_coherence'
#         option = 'mean_and_std'
#         option = 'mean_std_period'
#         option = 'mean_std_period_coherence'
#         option = 'mean_std_period_coherence_noise'
#         option = 'coherence_decrease_translation'
#         option = 'coherence_decrease_degradation'
        option = 'dual_coherence_decrease'
#         option = 'mean'
#         option = 'dual_coherence_and_lengthscale_decrease'
#         option = 'mean_std_period_fewer_samples'
#         option = 'mean_std_period_coherence'
#         option = 'weird_decrease'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish_extrinsic_noise')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                       model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                                       model_results[:,0]<1500))  #standard deviation
#                                                        model_results[:,1]>0.05)))  #standard deviation
        elif option == 'prior':
            accepted_indices = range(len(prior_samples))
        elif option == 'coherence':
            accepted_indices = np.where( model_results[:,3]>0.3 )  #standard deviation
        elif option == 'period':
            accepted_indices = np.where( model_results[:,2]<100 )  #standard deviation
        elif option == 'period_and_coherence':
            accepted_indices = np.where( np.logical_and( model_results[:,2]<100,
                                                         model_results[:,3]>0.3 ))  
        elif option == 'mean_period_and_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<8000,
                                        np.logical_and(model_results[:,2]<100,
                                                       model_results[:,3]>0.3))))  
        elif option == 'mean_longer_periods_and_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<8000,
                                        np.logical_and(model_results[:,2]<150,
                                        np.logical_and(model_results[:,3]>0.25,
                                                       model_results[:,3]<0.4)))))
        elif option == 'mean_and_std':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                                       model_results[:,1]>0.05))))
        elif option == 'mean_std_period_fewer_samples':
            accepted_indices = np.where(np.logical_and(model_results[:4000,0]>1000, #protein number
                                        np.logical_and(model_results[:4000,0]<2500,
                                        np.logical_and(model_results[:4000,1]<0.15,
                                        np.logical_and(model_results[:4000,1]>0.05,
                                                       model_results[:4000,2]<150)))))
        elif option == 'mean_std_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
        elif option == 'mean_std_period_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                        np.logical_and(model_results[:,3]>0.8,
                                                       model_results[:,2]<150))))))
        elif option == 'mean_std_period_coherence_noise':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                        np.logical_and(model_results[:,3]>0.4,
                                        np.logical_and(prior_samples[:,-1]>10, #noise
                                                       model_results[:,2]<150)))))))
        elif option == 'amplitude_and_coherence':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                        np.logical_and(model_results[:,0]<4000, #protein_number
#                                         np.logical_and(model_results[:,4]>40,
#                                         np.logical_and(model_results[:,4]>60, #mrna number
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,3]>0.15)))) #standard deviation
        elif option == 'deterministic': 
             accepted_indices = np.where(np.logical_and(model_results[:,5]>2000, #protein number
                                         np.logical_and(model_results[:,5]<4000, #protein_number
                                         np.logical_and(model_results[:,9]>40,
                                         np.logical_and(model_results[:,9]<60, #mrna number
                                                        model_results[:,6]>0.05)))))  #standard deviation
        elif option == 'weird_decrease':
            change = 'decreased'
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradationtest')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
#             weird_indices = np.where(results_before_change[:,0]>results_after_change[:,0])
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
        elif option == 'coherence_decrease_degradation':
            change = 'decreased'
#             change = 'increased'
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
        elif option == 'coherence_decrease_translation':
#             change = 'decreased'
            change = 'increased'
#             saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_translation')
            results_after_change = np.load(saving_path + '.npy')
            parameters_after_change = np.load(saving_path + '_parameters.npy')
            results_before_change = np.load(saving_path + '_old.npy')
            parameters_before_change = np.load(saving_path + '_parameters_old.npy')
            old_lengthscales = np.load(saving_path + '_old_lengthscales.npy')
            new_lengthscales = np.load(saving_path + '_new_lengthscales.npy')
        
            weird_indices = np.where(results_before_change[:,3]>results_after_change[:,3])
#             weird_indices = np.where(np.logical_and(results_before_change[:,3]>results_after_change[:,3],
#                                                     results_before_change[:,-1]/np.power(results_before_change[:,1]*
#                                                                                          results_before_change[:,0],2)<
#                                                     results_after_change[:,-1]/np.power(results_after_change[:,1]*
#                                                                                         results_after_change[:,0],2)))
#  
#                                                     old_lengthscales<new_lengthscales))
            weird_parameters_before = parameters_before_change[weird_indices]
            weird_parameters_after = parameters_after_change[weird_indices]
        elif option == 'dual_coherence_decrease':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_likelihoods.npy')
            conditions = np.load(saving_path)
            positive_indices = np.where(conditions>0)
            accepted_indices = (accepted_indices[0][positive_indices],)
        elif option == 'dual_coherence_and_lengthscale_decrease':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                        np.logical_and(model_results[:,0]<2500,
                                        np.logical_and(model_results[:,1]<0.15,
                                        np.logical_and(model_results[:,1]>0.05,
                                                       model_results[:,2]<150)))))
            saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_dual_sweeps_likelihoods.npy')
            conditions = np.load(saving_path)
            positive_indices = np.where(conditions>0)
            accepted_indices = (accepted_indices[0][positive_indices],)
        else:
            ValueError('could not identify posterior option')
#       
        if option not in ['weird_decrease', 'coherence_decrease_degradation',
                          'coherence_decrease_translation']:
            my_posterior_samples = prior_samples[accepted_indices]
        else:
            my_posterior_samples = weird_parameters_before

        print('Number of accepted samples is ')
        print(len(my_posterior_samples))
        print('minimal transcription is')
        print(np.min(my_posterior_samples[:,0]))
        print('and in log space')
        print(np.min(np.log10(my_posterior_samples[:,0])))
        print('minimal translation is')
        print(np.min(my_posterior_samples[:,1]))
        print('and in log space')
        print(np.min(np.log10(my_posterior_samples[:,1])))

        my_posterior_samples[:,2]/=1000

        print(my_posterior_samples.shape)
#         my_pairplot = hes5.plot_posterior_distributions(my_posterior_samples)

        data_frame = pd.DataFrame( data = my_posterior_samples[:,(0,1,2,3,4,5,7)],
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e3', 
                                             'Transcription delay',
                                             'Hill coefficient',
                                             'mRNA degradation',
                                             'Extrinsic noise rate'])

        ### PAIRGRID
#         my_adjusted_posterior_samples = np.copy(my_posterior_samples)
#         my_adjusted_posterior_samples[:,5] = np.log(2)/my_adjusted_posterior_samples[:,5]
#         my_adjusted_posterior_samples[:,0] = np.log10(my_adjusted_posterior_samples[:,0])
#         my_adjusted_posterior_samples[:,1] = np.log10(my_adjusted_posterior_samples[:,1])
#         new_data_frame = pd.DataFrame( data = my_adjusted_posterior_samples[:,:6],
#                                    columns= ['log10(Transcription rate)', 
#                                              'log10(Translation rate)', 
#                                              'Repression threshold/1e3', 
#                                              'Transcription delay',
#                                              'Hill coefficient',
#                                              'mRNA half life'])
#         my_pairplot = sns.PairGrid(new_data_frame)
# #         my_pairplot = sns.pairplot(new_data_frame)
#         my_pairplot.map_upper(plt.scatter, alpha = 0.02, color = 'black', rasterized = True)
# #         my_pairplot.map_upper(sns.kdeplot,rasterized = True)
#         my_pairplot.map_diag(plt.hist)
#         my_pairplot.map_lower(sns.kdeplot, cmap = 'Reds', rasterized = True)
# #         my_pairplot.axes[-1,0].set_xscale("log")
# #         my_pairplot.axes[-1,1].set_xscale("log")
#         my_pairplot.savefig(os.path.join(os.path.dirname(__file__),
#                                          'output',
#                                          'pairplot_zebrafish_abc_' +  option + '.pdf'))
#         ### END PAIRGRID

        sns.set(font_scale = 1.1, rc = {'ytick.labelsize': 6})
#         font = {'size'   : 28}
#         plt.rc('font', **font)
        my_figure = plt.figure(figsize= (11,3))

        my_figure.add_subplot(171)
#         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(np.log10(0.6),np.log10(60.0),20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Transcription rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                    bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
        plt.gca().set_xlim(-0.5,np.log10(60.0))
        plt.ylabel("Probability", labelpad = 20)
        plt.xlabel("Transcription rate \n [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,1)
#         plt.gca().set_ylim(0,1)
#         plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
        plt.xticks([0,1], [r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(172)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(np.log10(0.04),np.log10(40),20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black',
                                 'alpha' : None},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(-2,1)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([-1,0], [r'$10^{-1}$',r'$10^0$'])
        plt.xlabel("Translation rate \n [1/min]")
        plt.gca().set_ylim(0,1)
#         plt.gca().set_ylim(0,1.0)
#         plt.yticks([])
 
        my_figure.add_subplot(173)
        sns.distplot(data_frame['Repression threshold/1e3'],
                     kde = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e3]")
        plt.gca().set_ylim(0,0.5)
        plt.gca().set_xlim(0,5)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(174))
        time_delay_bins = np.linspace(5,40,10)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                     bins = time_delay_bins)
        plt.gca().set_xlim(5,40)
        plt.gca().set_ylim(0,0.07)
#         plt.gca().set_ylim(0,0.04)
        plt.gca().locator_params(axis='x', tight = True, nbins=5)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xlabel(" Transcription delay \n [min]")
#         plt.yticks([])
 
        plots_to_shift.append(my_figure.add_subplot(175))
        sns.distplot(data_frame['Hill coefficient'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.gca().set_ylim(0,0.4)
        plt.gca().set_xlim(2,6)
        plt.gca().locator_params(axis='x', tight = True, nbins=3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        my_figure.add_subplot(176)
#         translation_rate_bins = np.logspace(0,2.3,20)
#         degradation_rate_bins = np.linspace(np.log(2.0)/15.0,np.log(2)/1.0,20)
#         histogram, bin_edges = np.histogram(data_frame['mRNA degradation'], degradation_rate_bins, 
#                                             density = True)
#         plt.hist(histogram[::-1], np.log(2)/bin_edges[::-1] )

        half_lifes = np.log(2)/data_frame['mRNA degradation']
        print(half_lifes)
        half_life_bins = np.linspace(1,11,20)
#         half_life_histogram, _ = np.histogram(half_lifes, half_life_bins, density = True)
#         print(half_life_histogram)
#         prior_histogram, _ = np.histogram( np.log(2)/prior_samples[:,5], half_life_bins, density = True )
#         corrected_histogram = half_life_histogram/prior_histogram
#         corrected_histogram = half_life_histogram
#         print(corrected_histogram)
#         bin_centres = (half_life_bins[:-1] + half_life_bins[1:])/2
#         width = 0.7*(half_life_bins[1] - half_life_bins[0])
         
#         plt.bar(bin_centres, corrected_histogram, align = 'center' , width = width )
        sns.distplot(half_lifes,
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                    bins = half_life_bins)
#
#         sns.distplot(data_frame['mRNA degradation'],
#                      kde = False,
#                      rug = False,
#                      norm_hist = True,
#                      hist_kws = {'edgecolor' : 'black'},
#                      bins = degradation_rate_bins)
# #         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
#         plt.gca().set_xlim(-2,0)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.xticks([-1,0], [r'$10^{-1}$',r'$10^0$'])
        plt.xlabel("mRNA half-life \n [min]")
#         plt.gca().set_ylim(0,4.0)
#         plt.gca().set_ylim(0,1.0)
#         plt.yticks([])

        ## EXTRINSIC NOISE
        my_figure.add_subplot(177)
#         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(np.log10(0.1),np.log10(1000),20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Extrinsic noise rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black',
                                'alpha' : None},
                    bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
#         plt.gca().set_xlim(-0.5,np.log10(60.0))
        plt.xlabel("Extrinsic noise\nrate [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,1)
#         plt.gca().set_ylim(0,1)
#         plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.xticks([0,1], [r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        plt.tight_layout(w_pad = 0.0001)
#         plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inference_for_zebrafish_extrinsic_noise_' + option + '.pdf'))


