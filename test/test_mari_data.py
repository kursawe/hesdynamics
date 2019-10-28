import unittest
import os
os.environ["OMP_NUM_THREADS"] = "1"
import os.path
import sys
import matplotlib as mpl
import matplotlib.gridspec 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np
import scipy.signal
import pandas as pd
import seaborn as sns

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestNSQuiescence(unittest.TestCase):
                                 
    def xest_a_make_abc_samples(self):
        print('making abc samples')
        ## total number of prior samples:
        total_number_of_samples = 200000
        
        ## this needs to go - it has no impact on the prior

#         total_number_of_samples = 10

        prior_bounds = {'basal_transcription_rate' : (0.1,60),
                        'translation_rate' : (0.01,40),
                        'repression_threshold' : (0,20000),
                        'time_delay' : (5,40),
                        'hill_coefficient' : (2,6),
                        # assume 20 minute half life for  protein and mrna
                        'protein_degradation_rate' : (np.log(2)/20,np.log(2)/20),
                        'mRNA_degradation_rate' : (np.log(2)/20,np.log(2)/20)}

        # generate prior samples should be the name of this function
        my_prior_samples, my_prior_results = hes5.generate_lookup_tables_for_abc( total_number_of_samples,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'sampling_results_quiescense',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'full',
                                                                logarithmic = True,
                                                                simulation_duration = 1500*5 )
        
        self.assertEquals(my_prior_samples.shape, 
                          (total_number_of_samples, 5))

    def xest_plot_posterior_distributions(self):
        
        # this will extract posterior distributions according to some experimentally informed summary statistics
        
        option = 'mean_and_mrna'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        # results table columns are
        # mean protein, cov of protein, period of protein, coherence of protein, mean mrna, cov mrna, [all the same from the deterministic model,
        # noise weight in the power spectrum, aperiodic lengthscale
        
        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                                       model_results[:,0]<12000)) #protein_number
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,3]>20))))) #time_delay
        elif option == 'mean_and_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000, #protein_number
                                                       model_results[:,2]<300)))  #standard deviation
        elif option == 'mean_and_mrna':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000,
                                                       model_results[:,4]<50))) #protein_number
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                                       model_results[:,0]<12000)) #protein_number
        elif option == 'oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'not_oscillating': 
             accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]<0.1)))))  #standard deviation
        elif option == 'deterministic': 
             accepted_indices = np.where(np.logical_and(model_results[:,5]>55000, #protein number
                                         np.logical_and(model_results[:,5]<65000, #protein_number
#                                          np.logical_and(model_results[:,6]<0.15,  #standard deviation
                                                        model_results[:,6]>0.05)))
        else:
            ValueError('could not identify posterior option')
#       
        my_posterior_samples = prior_samples[accepted_indices]
        
#         pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
#         pairplot.savefig(os.path.join(os.path.dirname(__file__),
#                                       'output','pairplot_extended_abc_' + option + '.pdf'))

        print('Number of accepted samples is ')
        print(len(my_posterior_samples))

#         my_posterior_samples[:,2]/=1000

        data_frame = pd.DataFrame( data = my_posterior_samples,
                                   columns= ['Transcription rate', 
                                             'Translation rate', 
                                             'Repression threshold/1e4', 
                                             'Transcription delay',
                                             'Hill coefficient',
                                             'mRNA degradation rate',
                                             'protein degradation rate'])

        sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
        font = {'size'   : 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize= (11,3))

        my_figure.add_subplot(151)
#         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
#         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'], 
#                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Transcription rate']),
                    kde = False,
                    rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                    bins = transcription_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(0.1,100)
        plt.gca().set_xlim(-1,np.log10(60.0))
        plt.ylabel("Probability", labelpad = 20)
        plt.xlabel("Transcription rate \n [1/min]")
        plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
        plt.gca().set_ylim(0,1.0)
        plt.xticks([-1,0,1], [r'$10^{-1}$',r'$10^0$',r'$10^1$'])
#         plt.yticks([])
 
        my_figure.add_subplot(152)
#         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(-1,np.log10(40),20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde = False,
                     rug = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = translation_rate_bins)
#         plt.gca().set_xscale("log")
#         plt.gca().set_xlim(1,200)
        plt.gca().set_xlim(-1,np.log10(40))
        plt.gca().set_ylim(0,1.3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xticks([0,1], [r'$10^0$',r'$10^1$'])
        plt.xlabel("Translation rate \n [1/min]")
        plt.gca().set_ylim(0,2.0)
#         plt.yticks([])
 
        my_figure.add_subplot(153)
        sns.distplot(data_frame['Repression threshold/1e4'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e4]")
#         plt.gca().set_ylim(0,0.22)
#         plt.gca().set_xlim(0,2)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(154))
        time_delay_bins = np.linspace(5,40,10)
        sns.distplot(data_frame['Transcription delay'],
                     kde = False,
                     rug = False,
                    norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     bins = time_delay_bins)
        plt.gca().set_xlim(5,40)
#         plt.gca().set_ylim(0,0.035)
        plt.gca().set_ylim(0,0.04)
        plt.gca().locator_params(axis='x', tight = True, nbins=5)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
        plt.xlabel(" Transcription delay \n [min]")
#         plt.yticks([])
 
        plots_to_shift.append(my_figure.add_subplot(155))
        sns.distplot(data_frame['Hill coefficient'],
                     kde = False,
                     norm_hist = True,
                    hist_kws = {'edgecolor' : 'black'},
                     rug = False,
                     bins = 20)
#         plt.gca().set_xlim(1,200)
        plt.gca().set_ylim(0,0.35)
        plt.gca().set_xlim(2,6)
        plt.gca().locator_params(axis='x', tight = True, nbins=3)
        plt.gca().locator_params(axis='y', tight = True, nbins=2)
#         plt.yticks([])

        plt.tight_layout(w_pad = 0.0001)
#         plt.tight_layout()
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','inference_for_mari_' + option + '.pdf'))

    def xest_plot_period_distribution_for_paper(self):
        # in this function we make a Bayesian posterior prediction for the periods,
        # now that we know the posterior distributions from the previous plot
        option = 'mean_and_mrna'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
 
        if option == 'prior':
            accepted_indices = (range(len(prior_samples)),)
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                                       model_results[:,0]<12000)) #protein_number
        elif option == 'mean_and_mrna':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000,
                                                       model_results[:,4]<50))) #protein_number
        elif option == 'degradation_rate_decrease':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000,
                                                       model_results[:,4]<50))) #protein_number

            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'output',
                                                          'maris_relative_sweeps_' + 
                                                          'time_delay' + '.npy'))

            decrease_indices = np.where(np.logical_and( my_parameter_sweep_results[:,9,3] < 300,
                                                       my_parameter_sweep_results[:,4,5] < 
                                                       my_parameter_sweep_results[:,9,5]*0.5))

            accepted_indices = (accepted_indices[0][decrease_indices],)
        elif option == 'mean_and_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000, #protein_number
                                                       model_results[:,2]<300)))  #standard deviation
        elif option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,-1]<20))))) #noise strength
        elif option == 'oscillating': 
            accepted_indices = np.where(model_results[:,3]>0.3)  #standard deviation
        elif option == 'mean_and_oscillating': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                                       model_results[:,3]>0.3)))  #standard deviation
        elif option == 'mean_and_period': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                        np.logical_and(model_results[:,2]>240,
                                                       model_results[:,2]<300))))  #standard deviation
        elif option == 'mean_and_period_and_coherence': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                        np.logical_and(model_results[:,2]>240,
                                        np.logical_and(model_results[:,2]<300,
                                                       model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'lower_amplitude':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                                       model_results[:,1]>0.05)))  #standard deviation
        elif option == 'agnostic_prior':
            saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                       'sampling_results_agnostic')
            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = (range(len(prior_samples)),)
        elif option == 'agnostic_mean':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')
            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                                       model_results[:,0]<65000)) #protein_number
        elif option == 'agnostic_mean_and_coherence':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')

            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                                       model_results[:,3]>0.3)))  #standard deviation
        elif option == 'not_oscillating': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]<0.1)))))  #standard deviation
            my_posterior_samples = prior_samples[accepted_indices]
        else:
            ValueError('could not identify posterior option')

        my_posterior_samples = prior_samples[accepted_indices]
        print('so many posterior samples')
        print(len(my_posterior_samples))
        my_model_results = model_results[accepted_indices]
#         my_model_results = my_parameter_sweep_results[:,4,1:]

        my_posterior_samples[:,2]/=10000

        sns.set()
#         sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
#         font = {'size'   : 28}
#         plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_periods = my_model_results[:,2]
        print('maximal standard_deviation is')
        print(np.max(all_periods))
        print('number of samples above 0.15')
#         print(np.sum(all_standard_deviations>0.15))
        sns.distplot(all_periods,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                    bins = np.linspace(0,400,20),
                     hist_kws = {'edgecolor' : 'black'},
                     )
#                      bins = 20)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
#         plt.xlabel("Standard deviation/mean HES5")
        plt.xlabel('Period [min]')
        plt.xlim(0,600)
#         plt.xlim(0,0.25)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','period_distribution_for_mari_' + option + '.pdf'))
 
    def xest_plot_amplitude_distribution_for_paper(self):
        option = 'degradation_rate_decrease'

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
 
        if option == 'prior':
            accepted_indices = (range(len(prior_samples)),)
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                                       model_results[:,0]<12000)) #protein_number
        elif option == 'mean_and_mrna':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000,
                                                       model_results[:,4]<50))) #protein_number
        elif option == 'degradation_rate_decrease':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000,
                                                       model_results[:,4]<50))) #protein_number

            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'output',
                                                          'maris_relative_sweeps_' + 
                                                          'time_delay' + '.npy'))

            decrease_indices = np.where(np.logical_and( my_parameter_sweep_results[:,9,3] < 300,
                                                       my_parameter_sweep_results[:,4,5] < 
                                                       my_parameter_sweep_results[:,9,5]*0.5))

            accepted_indices = (accepted_indices[0][decrease_indices],)
        elif option == 'mean_and_period':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000, #protein_number
                                                       model_results[:,2]<300)))  #standard deviation
        elif option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                    model_results[:,1]>0.05))))  #standard deviation
#                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
#                                                     prior_samples[:,-1]<20))))) #noise strength
        elif option == 'oscillating': 
            accepted_indices = np.where(model_results[:,3]>0.3)  #standard deviation
        elif option == 'mean_and_oscillating': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                                       model_results[:,3]>0.3)))  #standard deviation
        elif option == 'mean_and_period': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                        np.logical_and(model_results[:,2]>240,
                                                       model_results[:,2]<300))))  #standard deviation
        elif option == 'mean_and_period_and_coherence': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                        np.logical_and(model_results[:,2]>240,
                                        np.logical_and(model_results[:,2]<300,
                                                       model_results[:,3]>0.3)))))  #standard deviation
        elif option == 'lower_amplitude':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                                       model_results[:,1]>0.05)))  #standard deviation
        elif option == 'agnostic_prior':
            saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                       'sampling_results_agnostic')
            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = (range(len(prior_samples)),)
        elif option == 'agnostic_mean':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')
            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                                       model_results[:,0]<65000)) #protein_number
        elif option == 'agnostic_mean_and_coherence':
            saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                       'sampling_results_agnostic')

            model_results = np.load(saving_path + '.npy' )
            prior_samples = np.load(saving_path + '_parameters.npy')

            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                        np.logical_and(model_results[:,0]<65000,
                                                       model_results[:,3]>0.3)))  #standard deviation
        elif option == 'not_oscillating': 
            accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                         np.logical_and(model_results[:,0]<65000, #protein_number
                                         np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                         np.logical_and(model_results[:,1]>0.05,
                                                        model_results[:,3]<0.1)))))  #standard deviation
            my_posterior_samples = prior_samples[accepted_indices]
        else:
            ValueError('could not identify posterior option')

        my_posterior_samples = prior_samples[accepted_indices]
        print('so many posterior samples')
        print(len(my_posterior_samples))
        my_model_results = model_results[accepted_indices]
        my_model_results = my_parameter_sweep_results[:,4,1:]

        my_posterior_samples[:,2]/=10000

        sns.set()
#         sns.set(font_scale = 1.5)
#         sns.set(font_scale = 1.3, rc = {'ytick.labelsize': 6})
#         font = {'size'   : 28}
#         plt.rc('font', **font)
        my_figure = plt.figure(figsize= (4.5,2.5))

# #         dataframe = pd.DataFrame({'Model': all_periods, 
#                                     'Data' : np.array(real_data)*60})
        all_standard_deviations = my_model_results[:,2]
        print('maximal standard_deviation is')
        print(np.max(all_standard_deviations))
        print('number of samples above 0.15')
#         print(np.sum(all_standard_deviations>0.15))
        sns.distplot(all_standard_deviations,
                     kde = False,
                     rug = False,
                     norm_hist = True,
                    bins = np.linspace(0,400,20),
                     hist_kws = {'edgecolor' : 'black'},
                     )
#                      bins = 20)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood", labelpad = 20)
#         plt.xlabel("Standard deviation/mean HES5")
        plt.xlabel('periods [min]')
        plt.xlim(0,600)
#         plt.xlim(0,0.25)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','mari_standard_deviation_' + option + '.pdf'))
 
    def xest_plot_mrna_distribution_for_mari(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        option = 'mean_and_mrna'

        if option == 'mean_and_mrna':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000,
                                                       model_results[:,4]<50))) #protein_number
            number_of_bins = 40
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                                       model_results[:,0]<12000))
            number_of_bins = 400

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

#         my_posterior_samples[:,2] /= 10000

        my_figure = plt.figure(figsize= (4.5,2.5))

        all_mrna = my_model_results[:,4]
        print('minimum and maximum are')
        print(np.min(all_mrna))
        print(np.max(all_mrna))
        print('so many samples above 100')
        print(np.sum(all_mrna>100))
        mrna_histogram, bins = np.histogram(all_mrna, bins = 400) 
        maximum_index = np.argmax(mrna_histogram)
        print('max bin is')
        print(bins[maximum_index])
        print(bins[maximum_index+1])
        print(bins[maximum_index+2])
        print(bins[maximum_index-1])

#         sns.distplot(all_mrna[all_mrna<80],
        sns.distplot(all_mrna,
                     kde = False,
                     rug = False,
                     hist_kws = {'edgecolor' : 'black'},
                     norm_hist = True,
#                      norm_hist = True,
                     bins = number_of_bins)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood" )
        plt.xlabel("mean mRNA number")
        plt.xlim(0,100)
#         plt.ylim(0,0.06)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','mrna_distribution_for_mari_' + option + '.pdf'))

    def xest_make_relative_parameter_variation(self):
        # This function makes parameter sweeps
#         number_of_parameter_points = 20
#         number_of_trajectories = 200
        number_of_parameter_points = 3
        number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
#         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                    np.logical_and(model_results[:,0]<12000,
                                                   model_results[:,4]<50))) #protein_number

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories,
                                                                                     relative = True,
                                                                                     simulation_duration = 1500*5)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','maris_relative_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def xest_plot_bayes_factors_for_models(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        sns.set()

        accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                    np.logical_and(model_results[:,0]<12000,
                                                   model_results[:,4]<50))) #protein_number

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]

        number_of_absolute_samples = len(accepted_indices[0])
        print('base model accepted that many indices')
        print(number_of_absolute_samples)
        parameter_names = ['basal_transcription_rate',
                            'translation_rate',
                            'repression_threshold',
                            'time_delay',
                            'mRNA_degradation_rate',
                            'protein_degradation_rate',
                            'hill_coefficient']

        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'Transcription rate'
        x_labels['translation_rate'] = 'Translation rate'
        x_labels['repression_threshold'] = 'Repression threshold' 
        x_labels['time_delay'] = 'Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'mRNA degradation'
        x_labels['protein_degradation_rate'] = 'Protein degradation'
        x_labels['hill_coefficient'] = 'Hill coefficient'

        decrease_ratios = dict()
        increase_ratios = dict()
        bardata = []
        for parameter_name in parameter_names:
            print('investigating ' + parameter_name)
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'output',
                                                          'maris_relative_sweeps_' + 
                                                          parameter_name + '.npy'))
 
            print('these accepted base samples are')
            number_of_absolute_samples = len(np.where(my_parameter_sweep_results[:,9,3] < 300)[0])
            print(number_of_absolute_samples)
            
            decrease_indices = np.where(np.logical_and( my_parameter_sweep_results[:,9,3] < 300,
#                                                        my_parameter_sweep_results[:,4,5] < 
#                                                        my_parameter_sweep_results[:,9,5]*0.5))
                                        np.logical_and(my_parameter_sweep_results[:,4,3] > 
                                                        my_parameter_sweep_results[:,9,3],
                                                        my_parameter_sweep_results[:,4,5] < 
                                                        my_parameter_sweep_results[:,9,5])))

            decrease_ratios[parameter_name] = len(decrease_indices[0])/float(number_of_absolute_samples)
            print('these decrease samples are')
            number_of_decrease_samples = len(decrease_indices[0])
            print(number_of_decrease_samples)

            increase_indices = np.where(np.logical_and( my_parameter_sweep_results[:,9,3] < 300,
#                                                        my_parameter_sweep_results[:,14,5] < 
#                                                        my_parameter_sweep_results[:,9,5]*0.5))
                                        np.logical_and(my_parameter_sweep_results[:,14,3] > 
                                                        my_parameter_sweep_results[:,9,3],
                                                        my_parameter_sweep_results[:,14,5] < 
                                                        my_parameter_sweep_results[:,9,5])))

            increase_ratios[parameter_name] = len(increase_indices[0])/float(number_of_absolute_samples)
            print('these increase samples are')
            number_of_increase_samples = len(increase_indices[0])
            print(number_of_increase_samples)
                
        increase_bars = [increase_ratios[parameter_name] for parameter_name
                         in parameter_names]

        decrease_bars = [decrease_ratios[parameter_name] for parameter_name
                         in parameter_names]

        increase_positions = np.arange(len(increase_bars))
        decrease_positions = np.arange(len(decrease_bars)) + len(increase_bars)
        all_positions = np.hstack((increase_positions, decrease_positions))
        
        all_bars = np.array( increase_bars + decrease_bars)

        labels_up = [x_labels[parameter_name] + ' up' for parameter_name in parameter_names]
        labels_down = [x_labels[parameter_name] + ' down' for parameter_name in parameter_names]
        
        all_labels = labels_up + labels_down
        sorting_indices = np.argsort(all_bars)
        sorted_labels = [all_labels[sorting_index] for
                         sorting_index in sorting_indices]
        sorted_bars = np.sort(all_bars)
        sorted_bars/= np.sum(sorted_bars)

        my_figure = plt.figure( figsize = (4.5, 1.5) )
        plt.bar(all_positions, sorted_bars[::-1])
        sorted_labels.reverse()
#         plt.xticks( all_positions + 0.4 , 
        plt.xticks( all_positions, 
                    sorted_labels,
                    rotation = 30,
                    fontsize = 5,
                    horizontalalignment = 'right')
        plt.xlim(all_positions[0] - 0.5, all_positions[-1] + 0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=4)
        plt.ylim(0,sorted_bars[-1]*1.2)
        plt.ylabel('Likelihood')

        my_figure.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output',
                                     'period_likelihood_plot_for_mari.pdf'))

    def xest_plot_bayes_2D(self):
        # This is the purple heat map thing
        saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        sns.set()

        accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                    np.logical_and(model_results[:,0]<12000,
                                                   model_results[:,4]<50))) #protein_number

        my_posterior_samples = prior_samples[accepted_indices]
        
        accepted_model_results = model_results[accepted_indices]

        number_of_absolute_samples = len(accepted_indices[0])
        print('base model accepted that many indices')
        print(number_of_absolute_samples)
        parameter_names = ['basal_transcription_rate',
                            'translation_rate',
                            'repression_threshold',
                            'time_delay',
                            'mRNA_degradation_rate',
                            'protein_degradation_rate',
                            'hill_coefficient']

        x_labels = dict()
        x_labels['basal_transcription_rate'] = 'Transcription rate'
        x_labels['translation_rate'] = 'Translation rate'
        x_labels['repression_threshold'] = 'Repression threshold' 
        x_labels['time_delay'] = 'Transcription delay'
        x_labels['mRNA_degradation_rate'] = 'mRNA degradation'
        x_labels['protein_degradation_rate'] = 'Protein degradation'
        x_labels['hill_coefficient'] = 'Hill coefficient'
        
        all_labels_in_order = [ x_labels[parameter_name] for parameter_name in parameter_names ]
        
        number_parameters = len(x_labels)
        
        likelihood_table = np.zeros((20,number_parameters))
        relative_change_values = np.linspace(0.1,2.0,20)

        decrease_ratios = dict()
        increase_ratios = dict()
        bardata = []
        for parameter_index, parameter_name in enumerate(parameter_names):
            print('investigating ' + parameter_name)
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                          'output',
                                                          'maris_relative_sweeps_' + 
                                                          parameter_name + '.npy'))
            these_results_before = my_parameter_sweep_results[:,9,:]
            for change_index, relative_change in enumerate(relative_change_values):
                these_results_after = my_parameter_sweep_results[:,change_index,:]
                # This is to extract information from the parameter sweep that 
                # fulfill the experimental observations 
                # the conditions on the right define the experimental observations
                this_condition_mask = np.logical_and(these_results_after[:,3] > 
                                                     1.3*these_results_before[:,3],
                                      np.logical_and(these_results_after[:,3] < 
                                                     2.5*these_results_before[:,3],
                                      np.logical_and(these_results_after[:,5] < 0.6*these_results_before[:,5],
                                                     these_results_after[:,5] > 0.0*these_results_before[:,5])))
                this_likelihood = np.sum(this_condition_mask)
                likelihood_table[change_index, parameter_index] = this_likelihood
         
        x_bin_edges = np.linspace(0.5,7.5,8)
        y_bin_edges = np.linspace(0.05,2.05,21)

        x_bin_centers = np.linspace(1.0,7.0,7)
        this_figure = plt.figure(figsize = (4.5,4.5))
        colormesh = plt.pcolormesh(x_bin_edges,y_bin_edges,likelihood_table, rasterized = True)
        plt.xticks( x_bin_centers, 
                    all_labels_in_order,
                    rotation = 30,
                    fontsize = 10,
                    horizontalalignment = 'right')
#         plt.pcolor(X,Y,expected_coherence)
#         plt.scatter(np.log(2)/90, np.log(2)/30)
        plt.ylabel("Relative change", y=0.4)
        
        divider = make_axes_locatable(plt.gca())
        cax = divider.new_vertical(size=0.07, pad=1.0, pack_start=True)
        this_figure.add_axes(cax)

        tick_locator = mpl.ticker.MaxNLocator(nbins=5)
        this_colorbar = this_figure.colorbar(colormesh, cax = cax, orientation = 'horizontal')
        this_colorbar.locator = tick_locator
        this_colorbar.update_ticks()
#         for ticklabel in this_colorbar.ax.get_xticklabels():
#             ticklabel.set_horizontalalignment('left') 
        this_colorbar.ax.set_ylabel('Likelihood', rotation = 0, verticalalignment = 'top', labelpad = 30)
        plt.tight_layout(pad = 0.05)
#         plt.tight_layout()

        file_name = os.path.join(os.path.dirname(__file__),
                                 'output','mari_2D_likelihood_plot')
 
        plt.savefig(file_name + '.pdf', dpi = 600)
        plt.savefig(file_name + '.eps', dpi = 600)
        plt.savefig(file_name + '.png', dpi = 600)

        my_figure = plt.figure( figsize = (4.5, 1.5) )
        plt.bar(x_bin_centers, np.sum(likelihood_table, axis = 0))
#         plt.xticks( all_positions + 0.4 , 
        plt.xticks( x_bin_centers, 
                    all_labels_in_order,
                    rotation = 30,
                    fontsize = 5,
                    horizontalalignment = 'right')
        plt.gca().locator_params(axis='y', tight = True, nbins=4)
        plt.ylabel('Likelihood')

        my_figure.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                     'output',
                                     'alternative_period_likelihood_plot_for_mari.pdf'))

    def xest_plot_mean_level_distribution_for_protein_degradation_change(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        option = 'mean_and_mrna'

        if option == 'mean_and_mrna':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000,
                                                       model_results[:,4]<50))) #protein_number
            number_of_bins = 40
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                                       model_results[:,0]<12000))
            number_of_bins = 400

        parameter_name = 'protein_degradation_rate'
        print('investigating ' + parameter_name)
        these_mean_expression_levels = []
        relative_change_values = np.linspace(0.1,2.0,20)
        my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 
                                                      'output',
                                                      'maris_relative_sweeps_' + 
                                                      parameter_name + '.npy'))
        these_results_before = my_parameter_sweep_results[:,9,:]
        for change_index, relative_change in enumerate(relative_change_values):
            these_results_after = my_parameter_sweep_results[:,change_index,:]
            this_condition_mask = np.logical_and(these_results_after[:,3] > 
                                                 1.3*these_results_before[:,3],
                                  np.logical_and(these_results_after[:,3] < 
                                                 2.5*these_results_before[:,3],
                                  np.logical_and(these_results_after[:,5] < 0.6*these_results_before[:,5],
                                                 these_results_after[:,5] > 0.0*these_results_before[:,5])))
            these_mean_expression_levels += these_results_after[this_condition_mask,1].tolist()
 
#         my_posterior_samples[:,2] /= 10000

        my_figure = plt.figure(figsize= (4.5,2.5))

        number_of_bis = 20
        sns.distplot(these_mean_expression_levels,
                     kde = False,
                     rug = False,
                     hist_kws = {'edgecolor' : 'black'},
                     norm_hist = True,
#                      norm_hist = True,
                     bins = number_of_bins)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood" )
        plt.xlabel("Mean Hes1 expression")
#         plt.xlim(0,100)
#         plt.ylim(0,0.06)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','mari_bayesian_posterior_prediction_level_changes.pdf'))

    def test_plot_mean_level_distribution_for_protein_degradation_change(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   'sampling_results_quiescense')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        option = 'mean_and_mrna'

        if option == 'mean_and_mrna':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                        np.logical_and(model_results[:,0]<12000,
                                                       model_results[:,4]<50))) #protein_number
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:,0]>8000, #protein number
                                                       model_results[:,0]<12000))

        my_posterior_results = model_results[accepted_indices]
        these_mean_expression_levels = my_posterior_results[:,0]
        my_figure = plt.figure(figsize= (4.5,2.5))

        number_of_bins = 20
        sns.distplot(these_mean_expression_levels,
                     kde = False,
                     rug = False,
                     hist_kws = {'edgecolor' : 'black'},
                     norm_hist = True,
#                      norm_hist = True,
                     bins = number_of_bins)
#         plt.gca().set_xlim(-1,2)
        plt.ylabel("Likelihood" )
        plt.xlabel("Mean Hes1 expression")
#         plt.xlim(0,100)
#         plt.ylim(0,0.06)
#         plt.ylim(0,0.5)
        plt.gca().locator_params(axis='y', tight = True, nbins=3)
#         plt.gca().locator_params(axis='y', tight = True, nbins=2, labelsize = 'small')
#         plt.gca().set_ylim(0,1.0)
#         plt.xticks([-1,0,1,2], [r'$10^{-1}$',r'$10^0$',r'$10^1$',r'$10^2$'])
#         plt.yticks([])
 
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','mari_bayesian_posterior_prediction_level_changes_before.pdf'))
