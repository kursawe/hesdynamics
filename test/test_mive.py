import unittest
import os
import os.path

import sys

os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rc

font = {'size': 10,
        'sans-serif': 'Arial'}
plt.rc('font', **font)

import numpy as np
import scipy.signal
import pandas as pd
import seaborn as sns

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

# make sure we find the right python module

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
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

    def xest_a_make_abc_samples(self):
        print('making abc samples')
        ## generate posterior samples
        total_number_of_samples = 100000
        # total_number_of_samples = 10

        #         total_number_of_samples = 10

        prior_bounds = {'basal_transcription_rate': (0.01, 120),
                        'translation_rate': (0.01, 60),
                        'repression_threshold': (0.01, 40000),
                        'time_delay': (5, 40),
                        'hill_coefficient': (2, 6)}

        my_prior_samples, my_results = hes5.generate_lookup_tables_for_abc(total_number_of_samples,
                                                                           number_of_traces_per_sample=200,
                                                                           saving_name='sampling_results_MiVe_expanded',
                                                                           prior_bounds=prior_bounds,
                                                                           prior_dimension='hill',
                                                                           logarithmic=True,
                                                                           simulation_timestep=1.0,
                                                                           simulation_duration=1500 * 5)

        self.assertEquals(my_prior_samples.shape,
                          (total_number_of_samples, 5))

    def test_plot_posterior_distributions(self):
        #Load data
        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #         saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #                                     'sampling_results_repeated')
                                   #         saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #                                     'sampling_results_massive')
                                   'sampling_results_MiVe_expanded')
        prior_performances = np.load(saving_path + '.npy')
        prior_parameters = np.load(saving_path + '_parameters.npy')

        protein_low = 11000
        protein_high = 29000


        option = 'priors'
        if option == 'full':
            accepted_indices = np.where(np.logical_and(prior_performances[:, 0] > protein_low,  # protein number
                                                       np.logical_and(prior_performances[:, 0] < protein_high,
                                                                      # protein_number
                                                                      np.logical_and(prior_performances[:, 1] < 0.15,
                                                                                     # standard deviation
                                                                                     prior_performances[:,
                                                                                     1] > 0.05))))  # standard deviation
        #                                         np.logical_and(prior_performances[:,1]>0.05,  #standard deviation
        #                                                     prior_parameters[:,3]>20))))) #time_delay

        elif option == 'posteriors':
            accepted_indices = np.where(np.logical_and(prior_performances[:, 0] > protein_low,  # protein number
                                                       prior_performances[:, 0] < protein_high))
            # protein_number
            #                                          np.logical_and(prior_performances[:,6]<0.15,  #standard deviation
            # prior_performances[:, 1] > 0.05)))
        elif option == 'priors':
            accepted_indices = np.where(prior_performances[:, 0] > 0)  # protein number

        else:
            ValueError('could not identify posterior option')
        #
        my_posterior_parameters = prior_parameters[accepted_indices]
        performances = prior_performances[accepted_indices]

        # Organise data
        my_posterior_parameters[:, 2] /= 10000 #Repression threshold 1e-4
        parameter_frame = pd.DataFrame(data=my_posterior_parameters,
                                       columns=['Transcription rate',
                                                'Translation rate',
                                                'Repression threshold 1e-4',
                                                'Transcription delay',
                                                'Hill coefficient'])
        performance_frame = pd.DataFrame(data=performances[:, range(0, 5)],
                                         columns=['Mean protein stochastic',
                                                  'STD stochastic',
                                                  'Period stochastic',
                                                  'Coherence stochastic',
                                                  'Mean mRNA stochastic'])
        parameter_bounds = pd.DataFrame({'Transcription rate': (0.01, 120),
                                         'Translation rate': (0.01, 60),
                                         'Repression threshold 1e-4': (0.000001, 4),
                                         'Transcription delay': (5, 40),
                                         'Hill coefficient': (2, 6)})

        performance_bounds = pd.DataFrame({'Mean protein stochastic': (11000, 29000),
                                           'STD stochastic':(performances[:, 1].min(), performances[:, 1].max()),
                                           'Period stochastic':(performances[:, 2].min(), performances[:, 2].max()),
                                           'Coherence stochastic':(performances[:, 3].min(), performances[:, 3].max()),
                                           'Mean mRNA stochastic':(performances[:, 4].min(), performances[:, 4].max())})

        parameter_pairplot = hes5.plot_posterior_distributions_MiVe(parameter_frame, parameter_bounds)
        parameter_pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output', 'Parameter_pairplot_extended_abc_MiVe_' + option + '.png'))
        performance_pairplot = hes5.plot_posterior_distributions_MiVe(performance_frame,performance_bounds,logarithmic = False)
        performance_pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                                'output', 'Performances_pairplot_extended_abc_MiVe_' + option + '.png'))

        print('Number of accepted samples is ')
        print(len(my_posterior_parameters))

         # performance_frame = pd.DataFrame(data=performances,
        #                                  columns=['Mean protein stochastic',
        #                                           'STD stochastic',
        #                                           'Period stochastic',
        #                                           'Coherence stochastic',
        #                                           'Mean mRNA stochastic',
        #                                           'Mean protein deterministic',
        #                                           'STD deterministic',
        #                                           'Period deterministic',
        #                                           'Coherence deterministic',
        #                                           'Mean mRNA deterministic',
        #                                           'High frequency weight',
        #                                           'Fluctuation weight'])



    def xest_plot_model_traces(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output', 'sweeping_results_MiVe_')
        # Plot design
        font = {'family': 'serif',
                'weight': 'light',
                'size': 8}
        mpl.rc('font', **font)  # pass in the font dict as kwargs
        # Generate langevin traces sweeping parameters
        parameter_bounds = {'basal_transcription_rate': (0.5, 3),
                            'translation_rate': (0.5, 10),
                            'repression_threshold': (0.01, 40000),
                            'transcription_delay': (5, 40),
                            'hill_coefficient': (2, 6)}
        parameter_bounds = pd.DataFrame(parameter_bounds)
        parameter_defaults = {'basal_transcription_rate': [1],
                              'translation_rate': [1],
                              'repression_threshold': [10000],
                              'transcription_delay': [29],
                              'hill_coefficient': [5]}
        parameter_defaults = pd.DataFrame(parameter_defaults)
        noPartitions = 3
        # Transcription rate
        l = []
        for parameter in parameter_bounds.columns:
            grid = np.linspace(parameter_bounds.loc[0, parameter], parameter_bounds.loc[1, parameter], noPartitions)
            parameters = parameter_defaults
            plt.figure(parameter)
            for i in range(0, noPartitions):
                parameterValue = grid[i]
                parameters.loc[0, parameter] = parameterValue
                trace = hes5.generate_langevin_trajectory(duration=2000,
                                                          repression_threshold=parameters.loc[
                                                              0, 'repression_threshold'],
                                                          hill_coefficient=parameters.loc[0, 'hill_coefficient'],
                                                          mRNA_degradation_rate=np.log(2) / 30,
                                                          protein_degradation_rate=np.log(2) / 90,
                                                          basal_transcription_rate=parameters.loc[
                                                              0, 'basal_transcription_rate'],
                                                          translation_rate=parameters.loc[0, 'translation_rate'],
                                                          transcription_delay=parameters.loc[0, 'transcription_delay'],
                                                          initial_mRNA=0,
                                                          initial_protein=0,
                                                          equilibration_time=0.0,
                                                          extrinsic_noise_rate=0.0,
                                                          transcription_noise_amplification=1.0,
                                                          timestep=0.5
                                                          )
                trace = pd.DataFrame({'time': trace[:, 0],
                                      'mRNA': trace[:, 1],
                                      'protein': trace[:, 2]})
                plt.subplot(2, 1, 1)
                line, = plt.plot(trace.loc[:, 'time'], trace.loc[:, 'mRNA'])
                l.append(line)
                # plt.title(str(round(parameterValue,2)))
                # plt.xlabel('time')
                if i == 0:
                    plt.ylabel('mRNA')
                plt.subplot(2, 1, 2)
                plt.plot(trace.loc[:, 'time'], trace.loc[:, 'protein'])
                # plt.title(parameter + ' = ' + str(parameterValue))
                plt.xlabel('time')
                if i == 0:
                    plt.ylabel('protein')
            plt.suptitle((parameter.replace('_', ' ')).upper())
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.8, top=0.8, wspace=0.6, hspace=0.4)
            plt.figlegend((l[0], l[1], l[2]), (str(grid[0]), str(grid[1]), str(grid[2])), loc='upper right')
            plt.savefig(saving_path + parameter)

        # Repression threshold
        # Transcription delay
        # Hill coefficient
