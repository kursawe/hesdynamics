import unittest
import os
import os.path

import sys

os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt

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

        prior_bounds = {'basal_transcription_rate': (0.1, 120),
                        'translation_rate': (0.1, 60),
                        'repression_threshold': (0.1, 40000),
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
    def test_plot_posterior_distributions(self):

        option = 'first_filter'
        protein_low = 11000
        protein_high = 30000

        saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #         saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #                                     'sampling_results_repeated')
                                   #         saving_path = os.path.join(os.path.dirname(__file__), 'output',
                                   #                                     'sampling_results_massive')
                                   'sampling_results_MiVe')
        model_results = np.load(saving_path + '.npy')
        prior_samples = np.load(saving_path + '_parameters.npy')

        if option == 'full':
            accepted_indices = np.where(np.logical_and(model_results[:, 0] > protein_low,  # protein number
                                                       np.logical_and(model_results[:, 0] < protein_high,
                                                                      # protein_number
                                                                      np.logical_and(model_results[:, 1] < 0.15,
                                                                                     # standard deviation
                                                                                     model_results[:,
                                                                                     1] > 0.05))))  # standard deviation
        #                                         np.logical_and(model_results[:,1]>0.05,  #standard deviation
        #                                                     prior_samples[:,3]>20))))) #time_delay

        elif option == 'first_filter':
            accepted_indices = np.where(np.logical_and(model_results[:, 0] > protein_low,  # protein number
                                                       np.logical_and(model_results[:, 0] < protein_high,
                                                                      # protein_number
                                                                      #                                          np.logical_and(model_results[:,6]<0.15,  #standard deviation
                                                                      model_results[:, 1] > 0.05)))
        elif option == 'no_filter':
            accepted_indices = np.where(model_results[:, 0] > 0)  # protein number

        else:
            ValueError('could not identify posterior option')
        #
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_performances = model_results[accepted_indices]

        # pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        # pairplot.savefig(os.path.join(os.path.dirname(__file__),
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

        parameter_frame = pd.DataFrame(data=my_posterior_samples,
                                       columns=['Transcription rate',
                                                'Translation rate',
                                                'Repression threshold/1e4',
                                                'Transcription delay',
                                                'Hill coefficient'])

        performance_frame = pd.DataFrame(data=my_posterior_performances[:,range(0,5)],
                                         columns=['Mean protein stochastic',
                                                  'STD stochastic',
                                                  'Period stochastic',
                                                  'Coherence stochastic',
                                                  'Mean mRNA stochastic'])
        # performance_frame = pd.DataFrame(data=my_posterior_performances,
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

        sns.set(font_scale=1.3, rc={'ytick.labelsize': 6})
        font = {'size': 28}
        plt.rc('font', **font)



        my_parameter_pairplot = sns.pairplot(parameter_frame)
        my_parameter_pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                                   'output', 'parameterPairplot_MiVe_' + option)) #+ '.pdf'))
        my_performance_pairplot = sns.pairplot(performance_frame)
        my_performance_pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                                     'output', 'performancePairplot_MiVe_' + option)) # + '.pdf'))
