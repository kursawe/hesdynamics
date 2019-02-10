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
import scipy.optimize
import pandas as pd
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
import gpflow
from numba import jit, autojit

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
        
    def test_make_abc_samples(self):
        ## generate posterior samples
        total_number_of_samples = 200000
#         total_number_of_samples = 100
        acceptance_ratio = 0.02

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.01,60),
                        'translation_rate' : (0.01,40),
                        'repression_threshold' : (0,16000),
                        'time_delay' : (5,40),
                        'hill_coefficient' : (2,6),
                        'protein_degradation_rate' : ( np.log(2)/15.0, np.log(2)/15.0 ),
                        'mRNA_half_life' : ( 1, 15) }

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_zebrafish',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'full',
                                                                logarithmic = True )
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 7))
        
    def test_plot_zebrafish_inference(self):
        option = 'prior'
#         option = 'mean_period_and_coherence'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
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
            accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                                       model_results[:,0]<8000))  #standard deviation
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
        else:
            ValueError('could not identify posterior option')
#       
        my_posterior_samples = prior_samples[accepted_indices]
        print('Number of accepted samples is ')
        print(len(my_posterior_samples))

        my_posterior_samples[:,2]/=1000

        data_frame = pd.DataFrame( data = my_posterior_samples[:,:6],
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e3', 
                                             'Transcription delay',
                                             'Hill coefficient',
                                             'mRNA degradation'])

        sns.set(font_scale = 1.1, rc = {'ytick.labelsize': 6})
#         font = {'size'   : 28}
#         plt.rc('font', **font)
        my_figure = plt.figure(figsize= (11,3))

        my_figure.add_subplot(161)
#         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
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
        plt.gca().set_xlim(-1,np.log10(60.0))
        plt.ylabel("Probability", labelpad = 20)
        plt.xlabel("Transcription rate \n [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,1)
#         plt.gca().set_ylim(0,1)
        plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(162)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(-2,0,20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black',
                                 'alpha' : None},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(-2,0)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([-1,0], [r'$10^{-1}$',r'$10^0$'])
        plt.xlabel("Translation rate \n [1/min]")
        plt.gca().set_ylim(0,3.8)
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
        plt.gca().set_ylim(0,0.27)
        plt.gca().set_xlim(0,12)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(164))
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
        half_life_bins = np.linspace(1,15,20)
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
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where( model_results[:,3]>0.2)  #standard deviation

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
                     kde = False,
                     rug = False,
                     norm_hist = True,
                     hist_kws = {'edgecolor' : 'black'},
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled period [h]")
        plt.xlim(0,300)
#         plt.ylim(0,0.2)
#         plt.ylim(0,0.0003)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','zebrafish_period_distribution.pdf'))
 
    def xest_plot_zebrafish_coherence_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
#                                    'sampling_results_MCF7')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        my_posterior_samples = prior_samples
        my_model_results = model_results

        sns.set()
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
                     hist_kws = {'edgecolor' : 'black'},
                     bins = 100)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
        plt.xlabel("Modelled coherence")
        plt.xlim(0.2,)
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
                                 'output','zebrafish_coherence_distribution.pdf'))
        
    def test_increase_mRNA_degradation(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples = prior_samples[accepted_indices]
        old_model_results = model_results[accepted_indices]
        my_posterior_samples_increased_degradation = np.copy(my_posterior_samples)
        my_posterior_samples_increased_degradation[:,5]*=1.5
        new_model_results = hes5.calculate_summary_statistics_at_parameters( my_posterior_samples_increased_degradation, 
                                                                        number_of_traces_per_sample=200 )

        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_increased_degradation')

        np.save(saving_path + '.npy', new_model_results)
        np.save(saving_path + '_parameters.npy', my_posterior_samples_increased_degradation )
        np.save(saving_path + '_old.npy', old_model_results)
        np.save(saving_path + '_parameters_old.npy', my_posterior_samples )
        
    def test_decrease_mRNA_degradation(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                    'sampling_results_zebrafish')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>2000, #protein number
                                    np.logical_and(model_results[:,0]<8000,
                                    np.logical_and(model_results[:,2]<100,
                                                   model_results[:,3]>0.3))))  

        my_posterior_samples = prior_samples[accepted_indices]
        old_model_results = model_results[accepted_indices]
        my_posterior_samples_increased_degradation = np.copy(my_posterior_samples)
        my_posterior_samples_increased_degradation[:,5]*=2.0/3.0
        new_model_results = hes5.calculate_summary_statistics_at_parameters( my_posterior_samples_increased_degradation, 
                                                                        number_of_traces_per_sample=200 )

        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_decreased_degradation')

        np.save(saving_path + '.npy', new_model_results)
        np.save(saving_path + '_parameters.npy', my_posterior_samples_increased_degradation )
        np.save(saving_path + '_old.npy', old_model_results)
        np.save(saving_path + '_parameters_old.npy', my_posterior_samples )
 
    def test_plot_mrna_change_results(self):
        
        change = 'decreased'
#         change = 'increased'
        
        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
        results_after_change = np.load(saving_path + '.npy')
        parameters_after_change = np.load(saving_path + '_parameters.npy')
        results_before_change = np.load(saving_path + '_old.npy')
        parameters_before_change = np.load(saving_path + '_parameters_old.npy')
    
        this_figure, axes = plt.subplots(2,3,figsize = (6.5,4.5))

        ## DEGRADATION
        this_data_frame = pd.DataFrame(np.column_stack((parameters_before_change[:,5],
                                                       parameters_after_change[:,5])),
                                        columns = ['before','after'])
        this_axes = axes[0,0]
        this_data_frame.boxplot(ax = axes[0,0])
        this_axes.set_ylabel('mRNA degradation')

        ## EXPRESSION
        this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,0],
                                                        results_after_change[:,0])),
                                        columns = ['before','after'])
        this_axes = axes[0,1]
        this_data_frame.boxplot(ax = axes[0,1])
        this_axes.set_ylabel('Hes expression')

        ## STANDARD DEVIATION
        this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,1],
                                                        results_after_change[:,1])),
                                        columns = ['before','after'])
        this_axes = axes[0,2]
        this_data_frame.boxplot(ax = this_axes)
        this_axes.set_ylabel('Hes std')

        ## PERIOD
        this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,2],
                                                        results_after_change[:,2])),
                                        columns = ['before','after'])
        this_axes = axes[1,0]
        this_data_frame.boxplot(ax = this_axes)
        this_axes.set_ylabel('Period')

        ## COHERENCE
        this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,3],
                                                        results_after_change[:,3])),
                                        columns = ['before','after'])
        this_axes = axes[1,1]
        this_data_frame.boxplot(ax = this_axes)
        this_axes.set_ylabel('Coherence')

        ## MRNA
        this_data_frame = pd.DataFrame(np.column_stack((results_before_change[:,4],
                                                        results_after_change[:,4])),
                                        columns = ['before','after'])
        this_axes = axes[1,2]
        this_data_frame.boxplot(ax = this_axes)
        this_axes.set_ylabel('mRNA number')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation.pdf'))

    def xest_plot_mRNA_change_examples(self):
        change = 'decreased'
        saving_path = os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation')
        results_after_change = np.load(saving_path + '.npy')
        parameters_after_change = np.load(saving_path + '_parameters.npy')
        results_before_change = np.load(saving_path + '_old.npy')
        parameters_before_change = np.load(saving_path + '_parameters_old.npy')
    
        example_parameter_index = 100
        example_parameter_before = parameters_before_change[example_parameter_index]
        example_parameter_after = parameters_after_change[example_parameter_index]
        
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
        plt.plot(example_trace_before[:,0],
                 example_trace_before[:,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.subplot(122)
        plt.plot(example_trace_after[:,0],
                 example_trace_after[:,2])
        plt.ylabel('Hes expression')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output','zebrafish_' + change + '_degradation_examples.pdf'))

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
        gamma_values = np.linspace(0.5,8,100)
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
        plt.ylabel(r'Frequency')
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
        test_correlation_function = hes5.calculate_autocorrelation_from_power_spectrum(fitted_power_spectrum)
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
        plt.plot(test_correlation_function[:,0],
                test_correlation_function[:,1], lw = 0.5, color = 'green', ls = '--', alpha =0.5)
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

    def xest_illustrate_lengthscale_measurements(self):
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
        these_shortened_traces_1 = protein_traces_1[:720]
        these_measured_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_shortened_traces_1)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_shortened_1.npy'), these_measured_fluctuation_rates)
#         these_measured_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence.npy'))
        this_fluctuation_rate_1 = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces_1)
        plt.hist(these_measured_fluctuation_rates, bins = 20, range = (0,0.008))
        plt.axvline(np.mean(these_measured_fluctuation_rates), color = 'blue', label = 'Mean')
        plt.axvline(this_fluctuation_rate_1, color = 'green', label = 'Theory')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')
        plt.legend(ncol=1, loc = 'upper left', bbox_to_anchor = (-0.1,1.2), framealpha = 1.0)

        plt.subplot(424)
        these_shortened_traces_2 = protein_traces_2[:720]
        these_measured_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_shortened_traces_2)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_shortened_2.npy'), these_measured_fluctuation_rates)
#         these_measured_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence_2.npy'))
        this_fluctuation_rate_2 = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces_2)
        plt.hist(these_measured_fluctuation_rates, bins = 20, range = (0,0.015))
        plt.axvline(np.mean(these_measured_fluctuation_rates), color = 'blue')
        plt.axvline(this_fluctuation_rate_2, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        ## Row 3 - histogram from 24 hours
        plt.subplot(425)
        these_shortened_traces_1 = protein_traces_1[:720*2]
        these_measured_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_shortened_traces_1)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_less_shortened_1.npy'), these_measured_fluctuation_rates)
#         these_measured_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence_longer.npy'))
        plt.hist(these_measured_fluctuation_rates, bins = 20, range = (0,0.008))
        plt.axvline(np.mean(these_measured_fluctuation_rates), color = 'blue')
        plt.axvline(this_fluctuation_rate_1, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        plt.subplot(426)
        these_shortened_traces_2 = protein_traces_2[:720*2]
        these_measured_fluctuation_rates = hes5.measure_fluctuation_rates_of_traces(these_shortened_traces_2)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_less_shortened_2.npy'), these_measured_fluctuation_rates)
#         these_measured_fluctuation_rates = np.load(os.path.join(os.path.dirname(__file__),'output',
#                                         'fluctuation_rates_for_convergence_longer_2.npy'))
        plt.hist(these_measured_fluctuation_rates, bins = 20, range = (0,0.015))
        plt.axvline(np.mean(these_measured_fluctuation_rates), color = 'blue')
        plt.axvline(this_fluctuation_rate_2, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        ## Row 4 - histogram from 12 hours, lower sampling rate
        plt.subplot(427)
        these_short_downsampled_protein_traces_1 = protein_traces_1[:720:10]
        these_measured_fluctuation_rates_1 = hes5.measure_fluctuation_rates_of_traces(these_short_downsampled_protein_traces_1)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_downsampled_1.npy'), these_measured_fluctuation_rates_1)
        this_estimated_fluctuation_rate_1 = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces_1,
                                                                                                      sampling_interval = 10)
        plt.hist(these_measured_fluctuation_rates_1, bins = 20)
        plt.axvline(np.mean(these_measured_fluctuation_rates_1), color = 'blue')
        plt.axvline(this_estimated_fluctuation_rate_1, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        plt.subplot(428)
        these_short_downsampled_protein_traces_2 = protein_traces_2[:720:10]
        these_measured_fluctuation_rates_2 = hes5.measure_fluctuation_rates_of_traces(these_short_downsampled_protein_traces_2)
        np.save(os.path.join(os.path.dirname(__file__),'output',
                'fluctuation_rates_for_convergence_downsampled_2.npy'), these_measured_fluctuation_rates_2)
        this_estimated_fluctuation_rate_2 = hes5.approximate_fluctuation_rate_of_traces_theoretically(protein_traces_2,
                                                                                                      sampling_interval = 10)
        plt.hist(these_measured_fluctuation_rates_2, bins = 20)
        plt.axvline(np.mean(these_measured_fluctuation_rates_2), color = 'blue')
        plt.axvline(this_estimated_fluctuation_rate_2, color = 'green')
        plt.xlabel('Fluctuation rate')
        plt.ylabel('Occurrence')

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'fluctuation_rate_illustration_panels.pdf'))
        
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
                                 'fluctuation_rate_illustration_short.pdf'))

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
        plt.title('Noise weight is ' + '{:.2f}'.format(noise_weight_1) + '/min')
        plt.ylabel('Power/1e7')
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
        plt.title('Noise weight is ' + '{:.2f}'.format(noise_weight_2) + '/min')
        plt.ylabel('Power/1e7')
        plt.ylim(0,2)

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),'output',
                                 'noise_weight_illustration.pdf'))

