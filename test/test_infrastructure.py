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

class TestInfrastructure(unittest.TestCase):

    def test_generate_single_oscillatory_trajectory(self):
        #First: run the model for 100 minutes
        my_trajectory = hes5.generate_deterministic_trajectory( duration = 720,
                                                         repression_threshold = 100,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = 18.5,
                                                         initial_mRNA = 3,
                                                         initial_protein = 100 )

        #Second, plot the model

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

    def test_stochastic_trajectory(self):
        my_trajectory = hes5.generate_stochastic_trajectory( duration = 720,
                                                             repression_threshold = 100,
                                                             mRNA_degradation_rate = 0.03,
                                                             protein_degradation_rate = 0.03,
                                                             transcription_delay = 18.5,
                                                             initial_mRNA = 3,
                                                             initial_protein = 100 )

        #Second, plot the model

        figuresize = (4,2.75)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1], label = 'mRNA', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2]*0.03, label = 'Hes protein (scaled)', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','stochastic_trajectory.pdf'))

    def test_equlibrate_stochastic_trajectory(self):
        #for profiling
        np.random.seed(0)
        my_trajectory = hes5.generate_stochastic_trajectory( duration = 1500,
                                                         repression_threshold = 31400,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         translation_rate = 29,
                                                         basal_transcription_rate = 11,
                                                         transcription_delay = 29,
                                                         initial_mRNA = 3,
                                                         initial_protein = 31400,
                                                         equilibration_time = 1000)

        
        figuresize = (4,2.5)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1]*1000, label = 'mRNA*1000', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.text(0.95, 0.4, 'Mean protein number: ' + str(np.mean(my_trajectory[:,2])),
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=plt.gca().transAxes)
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','hes5_stochastic_trajectory_equilibrated.pdf'))
        
    def test_calculate_power_spectrum_of_specific_trace(self):
        interval_length = 100
        x_values = np.linspace(1,interval_length,1000)
        function_values = 3*np.sin(2*np.pi*0.5*x_values) + 2*np.sin(2*np.pi*0.2*x_values) + 10.0
        number_of_data_points = len(x_values)
        trajectory = np.vstack((x_values, function_values)).transpose()
#         fourier_transform = np.fft.fft(function_values)/number_of_data_points
#         fourier_frequencies = np.arange(0,number_of_data_points/(2.0*interval_length), 1.0/(interval_length) )
        power_spectrum,_,_ = hes5.calculate_power_spectrum_of_trajectory(trajectory)

        my_figure = plt.figure()
        my_figure.add_subplot(211)
        plt.plot(x_values, 
                 function_values, label = r'$3sin(2\pi 0.5x) + 2sin(2\pi 0.2x)$', color = 'black')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()

        my_figure.add_subplot(212)
        plt.plot(power_spectrum[:,0], 
                 power_spectrum[:,1], color = 'black')
        plt.xlim(0,1)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','power_spectrum_test.pdf'))

    def test_stochastic_hes_trajectory_with_langevin(self):
        # same plot as before for different transcription ("more_mrna") - not yet
        # our preferred hes5 values
        my_trajectory = hes5.generate_langevin_trajectory( duration = 1500,
                                                         repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         translation_rate = 26,
                                                         basal_transcription_rate = 9,
                                                         transcription_delay = 29,
                                                         initial_mRNA = 3,
                                                         initial_protein = 23000)

        
        self.assertGreaterEqual(np.min(my_trajectory),0.0)
        figuresize = (4,2.5)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1]*10000, label = 'mRNA*1000', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.text(0.95, 0.4, 'Mean protein number: ' + str(np.mean(my_trajectory[:,2])),
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=plt.gca().transAxes)
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','hes5_langevin_trajectory.pdf'))

    def test_equlibrate_langevin_trajectory(self):
        import time
        np.random.seed(0)

        start = time.clock()
        my_trajectory = hes5.generate_langevin_trajectory( duration = 1500,
                                                         repression_threshold = 31400,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         translation_rate = 29,
                                                         basal_transcription_rate = 11,
                                                         transcription_delay = 29,
                                                         initial_mRNA = 3,
                                                         initial_protein = 31400,
                                                         equilibration_time = 1000)
        end = time.clock()
        
        print 'needed ' + str(end-start) + ' seconds'

        
        figuresize = (4,2.5)
        my_figure = plt.figure()
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1]*1000, label = 'mRNA*1000', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.text(0.95, 0.4, 'Mean protein number: ' + str(np.mean(my_trajectory[:,2])),
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=plt.gca().transAxes)
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','hes5_langevin_trajectory_equilibrated.pdf'))

    def test_multiple_equlibrated_langevin_trajectories(self):
        mRNA_trajectories, protein_trajectories = hes5.generate_multiple_langevin_trajectories( number_of_trajectories = 100,
                                                                                        duration = 1500,
                                                         repression_threshold = 31400,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         translation_rate = 29,
                                                         basal_transcription_rate = 11,
                                                         transcription_delay = 29,
                                                         initial_mRNA = 3,
                                                         initial_protein = 31400,
                                                         equilibration_time = 1000)

        np.save(os.path.join(os.path.dirname(__file__),
                                       'output','protein_traces.npy'), protein_trajectories)
        np.save(os.path.join(os.path.dirname(__file__),
                                       'output','rna_traces.npy'), mRNA_trajectories)

        mean_protein_trajectory = np.mean(protein_trajectories[:,1:], axis = 1)
        protein_deviation = np.std(mRNA_trajectories[:,1:])
        mean_mRNA_trajectory = np.mean(mRNA_trajectories[:,1:], axis = 1)
        mRNA_deviation = np.std(mRNA_trajectories[:,1:])
        
        figuresize = (4,2.75)
        my_figure = plt.figure()
        # want to plot: protein and mRNA for stochastic and deterministic system,
        # example stochastic system
        plt.plot( mRNA_trajectories[:,0],
                  mRNA_trajectories[:,1]*1000., label = 'mRNA example', color = 'black' )
        plt.plot( protein_trajectories[:,0],
                  protein_trajectories[:,1], label = 'Protein example', color = 'black', ls = '--' )
        plt.plot( mRNA_trajectories[:,0],
                  mean_mRNA_trajectory*1000, label = 'Mean mRNA*1000', color = 'blue' )
        plt.plot( protein_trajectories[:,0],
                  mean_protein_trajectory, label = 'Mean protein*1000', color = 'blue', ls = '--' )
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','average_hes5_langevin_behaviour.pdf'))

    def test_make_abc_example(self):
        ## generate posterior samples
        total_number_of_samples = 200
        acceptance_ratio = 0.1

#         total_number_of_samples = 10
#         acceptance_ratio = 0.5

        prior_bounds = {'basal_transcription_rate' : (0.1,100),
                        'translation_rate' : (1,200),
                        'repression_threshold' : (0,100000),
                        'time_delay' : (5,40),
                        'hill_coefficient' : (2,6)}
#                         'mRNA_degradation_rate': (np.log(2)/500, np.log(2)/5),
#                         'protein_degradation_rate': (np.log(2)/500, np.log(2)/5)}
#                         'mRNA_degradation_rate': (0.001, 0.04),
#                         'protein_degradation_rate': (0.001, 0.04),

        my_posterior_samples = hes5.generate_posterior_samples( total_number_of_samples,
                                                                acceptance_ratio,
                                                                number_of_traces_per_sample = 200,
                                                                saving_name = 'test_sampling_results',
                                                                prior_bounds = prior_bounds,
                                                                prior_dimension = 'hill',
                                                                logarithmic = True)
        
        self.assertEquals(my_posterior_samples.shape, 
                          (int(round(total_number_of_samples*acceptance_ratio)), 5))

        # plot distribution of accepted parameter samples
        pairplot = hes5.plot_posterior_distributions( my_posterior_samples )
        pairplot.savefig(os.path.join(os.path.dirname(__file__),
                                      'output','pairplot_hill_abc_logarithmic_prior_' +  str(total_number_of_samples) + '_'
                                      + str(acceptance_ratio) + '.pdf'))
 
    def test_make_logarithmic_degradation_rate_sweep(self):
        number_of_parameter_points = 5
        number_of_trajectories = 10
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','test_sampling_results')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_sweep_results = hes5.conduct_parameter_sweep_at_parameters('protein_degradation_rate',
                                          my_posterior_samples,
                                          number_of_sweep_values = number_of_parameter_points,
                                          number_of_traces_per_parameter = number_of_trajectories,
                                          relative = False)

        np.save(os.path.join(os.path.dirname(__file__), 'output','test_degradation_sweep.npy'),
                    my_sweep_results)

    def test_make_logarithmic_relative_parameter_variation(self):
        number_of_parameter_points = 5
        number_of_trajectories = 10
#         number_of_parameter_points = 3
#         number_of_trajectories = 2

#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','test_sampling_results')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #cell number
                                    np.logical_and(model_results[:,0]<65000, #cell_number
                                    np.logical_and(model_results[:,1]<0.15, #standard deviation
                                                   model_results[:,1]>0.05)))) #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_all_parameter_sweeps_at_parameters(my_posterior_samples,
                                                                                     number_of_parameter_points,
                                                                                     number_of_trajectories,
                                                                                     relative = True)
        
        for parameter_name in my_parameter_sweep_results:
            np.save(os.path.join(os.path.dirname(__file__), 'output','test_relative_sweeps_' + parameter_name + '.npy'),
                    my_parameter_sweep_results[parameter_name])
