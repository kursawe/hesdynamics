import unittest
import os
import os.path

import sys
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10,
        'sans-serif' : 'Arial'}
plt.rc('font', **font)

import numpy as np
import scipy.signal
import pandas as pd
import seaborn as sns


# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestInfrastructure(unittest.TestCase):

    def xest_make_relative_parameter_variation(self):
        number_of_parameter_points = 2
        number_of_trajectories = 200
        # this is a test comment to see whether git push still works
        #         number_of_parameter_points = 2
        #         number_of_trajectories = 2

        #         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        #         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'output', 'sampling_results_mive')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:, 0] > 5000,  # protein number
                                                   np.logical_and(model_results[:, 0] < 65000,  # protein_number
                                                                  #                                     np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                                  model_results[:, 1] > 0.05)))  # standard deviation

        my_posterior_samples = prior_samples[accepted_indices][:10]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories,
                                                                                     relative=True,
                                                                                     relative_range=(0.5, 1.5))

        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output',
                                 'repeated_relative_sweeps_mive_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])

    def test_a_make_abc_samples(self):
        print('making abc samples')
        ## generate posterior samples
        total_number_of_samples = 100000

        #         total_number_of_samples = 10

        prior_bounds = {'basal_transcription_rate': (0,120),
                        'translation_rate': (10,60),
                        'repression_threshold': (0, 40000),
                        'time_delay': (5, 40),
                        'hill_coefficient': (2, 6)}

        my_prior_samples, my_results = hes5.generate_lookup_tables_for_abc(total_number_of_samples,
                                                                           number_of_traces_per_sample=200,
                                                                           saving_name='sampling_results_MiVe',
                                                                           prior_bounds=prior_bounds,
                                                                           prior_dimension='hill',
                                                                           logarithmic=True,
                                                                           simulation_timestep=1.0,
                                                                           simulation_duration=1500 * 5)

        self.assertEquals(my_prior_samples.shape,
                          (total_number_of_samples, 5))


####################################################
    def xest_plot_posterior_distributions(self):

        option = 'deterministic'
        protein_low = 5000
        protein_high = 65000

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #         saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #                                     'sampling_results_repeated')
                                   #         saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #                                     'sampling_results_massive')
                                   'sampling_results_mive')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')

        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:, 0] > protein_low,  # protein number
                                                       np.logical_and(model_results[:, 0] < protein_high,  # protein_number
                                                                      np.logical_and(model_results[:, 1] < 0.15,
                                                                                     # standard deviation
                                                                                     model_results[:,
                                                                                     1] > 0.05))))  # standard deviation
        #                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
        #                                                     prior_samples[:,3]>20))))) #time_delay
        elif option == 'amplitude':
            accepted_indices = np.where(np.logical_and(model_results[:, 0] > protein_low,  # protein number
                                                       np.logical_and(model_results[:, 0] < protein_high,  # protein_number
                                                                      model_results[:,
                                                                      1] > 0.05)))  # standard deviation
        elif option == 'prior':
            accepted_indices = np.where(model_results[:, 0] > 0)
        elif option == 'increase_is_possible':
            accepted_indices = np.where(np.logical_and(model_results[:, 0] > 3000,  # protein number
                                                       np.logical_and(model_results[:, 0] < protein_high,  # protein_number
                                                                      model_results[:,
                                                                      1] > 0.05)))  # standard deviation
            parameter_name = 'hill_coefficient'
            my_parameter_sweep_results = np.load(os.path.join(os.path.dirname(__file__),
                                                              #                                                           'data',
                                                              'output',
                                                              #                                                           'narrowed_relative_sweeps_' +
                                                              'repeated_relative_sweeps_mive_' +
                                                              #                                                           'extended_relative_sweeps_' +
                                                              parameter_name + '.npy'))

            print('these accepted base samples are')
            possible_samples = np.where(np.logical_or(my_parameter_sweep_results[:, 9, 3] > 600,
                                                      my_parameter_sweep_results[:, 9, 4] < 0.1))
            number_of_absolute_samples = len(np.where(np.logical_or(my_parameter_sweep_results[:, 9, 3] > 600,
                                                                    my_parameter_sweep_results[:, 9, 4] < 0.1))[0])
            print(number_of_absolute_samples)

            decrease_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:, 9, 4] < 0.1,
                                                                     my_parameter_sweep_results[:, 9, 3] > 600),
                                                       np.logical_and(my_parameter_sweep_results[:, 4, 3] < 300,
                                                                      my_parameter_sweep_results[:, 4, 4] > 0.1)))

            print('these decrease samples are')
            number_of_decrease_samples = len(decrease_indices[0])
            print(number_of_decrease_samples)

            increase_indices = np.where(np.logical_and(np.logical_or(my_parameter_sweep_results[:, 9, 4] < 0.1,
                                                                     my_parameter_sweep_results[:, 9, 3] > 600),
                                                       np.logical_and(my_parameter_sweep_results[:, 14, 3] < 300,
                                                                      my_parameter_sweep_results[:, 14, 4] > 0.1)))

            print('these increase samples are')
            number_of_increase_samples = len(increase_indices[0])
            print(number_of_increase_samples)
            #             accepted_indices = (accepted_indices[0][decrease_indices],)
            #             accepted_indices = (accepted_indices[0][increase_indices],)
            accepted_indices = (accepted_indices[0][possible_samples],)
        elif option == 'mean':
            accepted_indices = np.where(np.logical_and(model_results[:, 0] > protein_low,  # protein number
                                                       model_results[:, 0] < protein_high))  # protein_number
        elif option == 'oscillating':
            accepted_indices = np.where(np.logical_and(model_results[:, 0] > protein_low,  # protein number
                                                       np.logical_and(model_results[:, 0] < protein_high,  # protein_number
                                                                      np.logical_and(model_results[:, 1] < 0.15,
                                                                                     # standard deviation
                                                                                     np.logical_and(
                                                                                         model_results[:, 1] > 0.05,
                                                                                         model_results[:,
                                                                                         3] > 0.3)))))  # standard deviation
        elif option == 'not_oscillating':
            accepted_indices = np.where(np.logical_and(model_results[:, 0] > protein_low,  # protein number
                                                       np.logical_and(model_results[:, 0] < protein_high,  # protein_number
                                                                      np.logical_and(model_results[:, 1] < 0.15,
                                                                                     # standard deviation
                                                                                     np.logical_and(
                                                                                         model_results[:, 1] > 0.05,
                                                                                         model_results[:,
                                                                                         3] < 0.1)))))  # standard deviation
        elif option == 'deterministic':
            accepted_indices = np.where(np.logical_and(model_results[:, 0] > 3000,  # protein number
                                                       np.logical_and(model_results[:, 0] < protein_high,  # protein_number
                                                                      #                                          np.logical_and(model_results[:,6]<0.15,  #standard deviation
                                                                      model_results[:, 1] > 0.05)))
        elif option == 'coherence_goes_down':
            accepted_indices = np.where(np.logical_and(model_results[:, 0] > protein_low,  # protein number
                                                       np.logical_and(model_results[:, 0] < protein_high,  # protein_number
                                                                      model_results[:,
                                                                      1] > 0.05)))  # standard deviation
            my_degradation_sweep_results = np.load(os.path.join(os.path.dirname(__file__), 'output',
                                                                #                                                           'extended_degradation_sweep.npy'))
                                                                'repeated_degradation_sweep.npy'))
            my_filtered_indices = np.where(np.logical_and(my_degradation_sweep_results[:, 9, 4] -
                                                          my_degradation_sweep_results[:, 3, 4] >
                                                          my_degradation_sweep_results[:, 3, 4] * 1.0,
                                                          my_degradation_sweep_results[:, 3, 4] > 0.1))

            accepted_indices = (accepted_indices[0][my_filtered_indices],)
        else:
            ValueError('could not identify posterior option')
        #
        my_posterior_samples = prior_samples[accepted_indices]

        #pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        #pairplot.savefig(os.path.join(os.path.dirname(__file__),
        #                              'output','pairplot_extended_abc_mive' + option + '.pdf'))

        print('Number of accepted samples is ')
        print(len(my_posterior_samples))

        my_posterior_samples[:, 2] /= 10000

        data_frame = pd.DataFrame(data=my_posterior_samples,
                                  columns=['Transcription rate',
                                           'Translation rate',
                                           'Repression threshold/1e4',
                                           'Transcription delay',
                                           'Hill coefficient'])

        sns.set(font_scale=1.3, rc={'ytick.labelsize': 6})
        font = {'size': 28}
        plt.rc('font', **font)
        my_figure = plt.figure(figsize=(11, 3))

        my_figure.add_subplot(151)
        #         transcription_rate_bins = np.logspace(-1,2,20)
        transcription_rate_bins = np.linspace(-1, np.log10(60.0), 20)
        #         transcription_rate_histogram,_ = np.histogram( data_frame['Transcription delay'],
        #                                                        bins = time_delay_bins )
        sns.distplot(np.log10(data_frame['Transcription rate']),
                     kde=False,
                     rug=False,
                     norm_hist=True,
                     hist_kws={'edgecolor': 'black'})
                     #bins=transcription_rate_bins)
        #         plt.gca().set_xscale("log")
        #         plt.gca().set_xlim(0.1,100)
        #plt.gca().set_xlim(-1, np.log10(60.0))
        plt.ylabel("Probability", labelpad=20)
        plt.xlabel("Transcription rate \n [min$^{-1}$]")
        plt.gca().locator_params(axis='y', tight=True, nbins=2, labelsize='small')
        if option != 'deterministic':
            plt.gca().set_ylim(0, 1.0)
        plt.xticks([-1, 0, 1], [r'10$^{-1}$', r'10$^0$', r'10$^1$'])
        #         plt.yticks([])

        my_figure.add_subplot(152)
        #         translation_rate_bins = np.logspace(0,2.3,20)
        translation_rate_bins = np.linspace(0, np.log10(40), 20)
        sns.distplot(np.log10(data_frame['Translation rate']),
                     kde=False,
                     rug=False,
                     norm_hist=True,
                     hist_kws={'edgecolor': 'black'},
                     bins=translation_rate_bins)
        #         plt.gca().set_xscale("log")
        #         plt.gca().set_xlim(1,200)
        #plt.gca().set_xlim(0, np.log10(40))
        if option != 'deterministic':
            plt.gca().set_ylim(0, 2.0)
        plt.gca().locator_params(axis='y', tight=True, nbins=2)
        plt.xticks([0, 1], [r'10$^0$', r'10$^1$'])
        plt.xlabel("Translation rate \n [min$^{-1}$]")
        #         plt.yticks([])

        my_figure.add_subplot(153)
        sns.distplot(data_frame['Repression threshold/1e4'],
                     kde=False,
                     norm_hist=True,
                     hist_kws={'edgecolor': 'black',
                               'range': (0, 12)},
                     rug=False,
                     bins=20)
        #         plt.gca().set_xlim(1,200)
        plt.xlabel("Repression threshold \n [1e4]")
        if option != 'deterministic':
            plt.gca().set_ylim(0, 0.22)
        #plt.gca().set_xlim(0, 12)
        plt.gca().locator_params(axis='x', tight=True, nbins=4)
        plt.gca().locator_params(axis='y', tight=True, nbins=2)
        #         plt.yticks([])

        plots_to_shift = []
        plots_to_shift.append(my_figure.add_subplot(154))
        if option == 'deterministic':
            time_delay_bins = np.linspace(5, 40, 20)
        else:
            time_delay_bins = np.linspace(5, 40, 10)
        sns.distplot(data_frame['Transcription delay'],
                     kde=False,
                     rug=False,
                     norm_hist=True,
                     hist_kws={'edgecolor': 'black'})
                    # bins=time_delay_bins)
        #plt.gca().set_xlim(5, 40)
        #         plt.gca().set_ylim(0,0.035)
        if option != 'deterministic':
            plt.gca().set_ylim(0, 0.04)
        plt.gca().locator_params(axis='x', tight=True, nbins=5)
        plt.gca().locator_params(axis='y', tight=True, nbins=2)
        plt.xlabel(" Transcription delay \n [min]")
        #         plt.yticks([])

        plots_to_shift.append(my_figure.add_subplot(155))
        sns.distplot(data_frame['Hill coefficient'],
                     kde=False,
                     norm_hist=True,
                     hist_kws={'edgecolor': 'black',
                               'range': (2, 6)},
                     rug=False,
                     bins=20)
        #         plt.gca().set_xlim(1,200)
        if option != 'deterministic':
            plt.gca().set_ylim(0, 0.35)
        plt.gca().set_xlim(2, 6)
        plt.gca().locator_params(axis='x', tight=True, nbins=3)
        plt.gca().locator_params(axis='y', tight=True, nbins=2)
        #         plt.yticks([])

        plt.tight_layout(w_pad=0.0001)
        #         plt.tight_layout()

        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output', 'inference_for_mive_' + option + '.pdf'))

