import unittest
import os.path
import sys
import matplotlib as mpl
from scipy.signal.spectral import coherence
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5

class TestSimpleHes5Model(unittest.TestCase):
                                 
    def xest_generate_single_oscillatory_trajectory(self):
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

    def xest_generate_hes5_predicted_trajectory(self):
        #First: run the model for 100 minutes
        my_trajectory = hes5.generate_deterministic_trajectory( duration = 720,
                                                         repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         translation_rate = 230,
                                                         transcription_delay = 29,
                                                         initial_mRNA = 3,
                                                         initial_protein = 23000,
                                                         for_negative_times = 'initial' )

        
        number_of_data_points = len(my_trajectory[:,2])
        fourier_transform = np.fft.fft(my_trajectory[:,2])/number_of_data_points
        interval_length = 720.0
        fourier_frequencies = np.arange(0,number_of_data_points/(2.0*interval_length), 1/(interval_length) )
        print fourier_transform.shape
        #Second, plot the model

        figuresize = (4,4.5)
        my_figure = plt.figure()
        my_figure.add_subplot(211)
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1], label = 'mRNA', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.text(0.95, 0.4, 'Mean protein number: ' + str(np.mean(my_trajectory[:,2])),
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=plt.gca().transAxes)
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.add_subplot(212)
        plt.plot(fourier_frequencies, 
                 np.abs(fourier_transform[:(number_of_data_points/2)]), color = 'black')
        plt.xlim(0,0.02)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','hes5_trajectory.pdf'))

    def xest_that_I_apply_DFT_correctly(self):
        interval_length = 100
        x_values = np.linspace(1,interval_length,1000)
        function_values = 3*np.sin(2*np.pi*0.5*x_values) + 2*np.sin(2*np.pi*0.2*x_values) + 10.0
        number_of_data_points = len(x_values)
        fourier_transform = np.fft.fft(function_values)/number_of_data_points
        fourier_frequencies = np.arange(0,number_of_data_points/(2.0*interval_length), 1.0/(interval_length) )

        my_figure = plt.figure()
        my_figure.add_subplot(211)
        plt.plot(x_values, 
                 function_values, label = r'$3sin(2\pi 0.5x) + 2sin(2\pi 0.2x)$', color = 'black')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        my_figure.add_subplot(212)
        plt.plot(fourier_frequencies, 
                 np.abs(fourier_transform[:(number_of_data_points/2)]), color = 'black')
        plt.xlim(0,1)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','fourier_test.pdf'))


    def xest_generate_non_dimensionalised_trajectory(self):
        #First: run the model for 100 minutes
#         my_trajectory = hes5.generate_deterministic_trajectory( duration = 720/29.0,
        my_trajectory = hes5.generate_deterministic_trajectory( duration = 60,
                                                         repression_threshold = 100.0/np.power(29.0,2),
                                                         mRNA_degradation_rate = np.log(2)/30*29.0,
                                                         protein_degradation_rate = np.log(2)/90*29.0,
                                                         transcription_delay = 29.0/29.0,
                                                         initial_mRNA = 3.0/(29),
                                                         initial_protein = 100.0/np.power(29.0,2) )

        #Second, plot the model

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
                                       'output','hes5_rescaled_trajectory.pdf'))
    
    def xest_extract_period_from_signal(self):

        # decaying but obvious oscillation
        interval_length = 10
        x_values = np.linspace(1,interval_length,1000)
        period = 0.5
        function_values = 10 + 3*np.sin(2*np.pi/period*x_values)*np.exp(-0.5*x_values)

        this_period, this_relative_amplitude, this_relative_amplitude_variance = hes5.measure_period_and_amplitude_of_signal(x_values, function_values)

        figuresize = (4,4.5)
        my_figure = plt.figure(figsize = figuresize)
        my_figure.add_subplot(211)
        plt.plot(x_values, 
                 function_values, label = r'$3sin(2\pi x/0.5)exp(-0.5x)$', color = 'black')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Measured period: ' + str(this_period) + '\n' +
                  'Relative amplitude: ' + str(this_relative_amplitude) + '\n' +
                  'Relative amplitude variation: ' + str(this_relative_amplitude_variance) + '\n')
#         plt.gca().text(label_x, label_y, 'A', transform=plt.gca().transAxes)
        self.assertAlmostEqual(0.5, this_period, places=2)
        plt.legend()
        
        # flat signal
        function_values = np.ones_like(function_values)*10
        this_period, this_relative_amplitude, this_relative_amplitude_variance = hes5.measure_period_and_amplitude_of_signal(x_values, function_values)
        my_figure.add_subplot(212)
        plt.plot(x_values, 
                 function_values, label = r'10', color = 'black')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Measured period: ' + str(this_period) + '\n' +
                  'Relative amplitude: ' + str(this_relative_amplitude) + '\n' +
                  'Relative amplitude variation: ' + str(this_relative_amplitude_variance) + '\n')
#         plt.gca().text(label_x, label_y, 'A', transform=plt.gca().transAxes)
        self.assertAlmostEqual(0.0, this_period)
        plt.legend()
        plt.tight_layout()
        
        
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','extract_frequency.pdf'))

    def xest_stochastic_trajectory(self):
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
                 my_trajectory[:,2], label = 'Hes protein', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','stochastic_trajectory.pdf'))

    def xtest_validate_stochastic_implementation(self):
        mRNA_trajectories, protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = 10,
                                                                     duration = 720,
                                                                     repression_threshold = 100000,
                                                                     mRNA_degradation_rate = 0.03,
                                                                     protein_degradation_rate = 0.03,
                                                                     transcription_delay = 18.5,
                                                                     basal_transcription_rate = 1000.0,
                                                                     translation_rate = 1.0,
                                                                     initial_mRNA = 3000,
                                                                     initial_protein = 100000 )
        
        mean_protein_trajectory = np.mean(protein_trajectories[:,1:], axis = 1)
        protein_deviation = np.std(mRNA_trajectories[:,1:])
        mean_mRNA_trajectory = np.mean(mRNA_trajectories[:,1:], axis = 1)
        mRNA_deviation = np.std(mRNA_trajectories[:,1:])
        
        deterministic_trajectory = hes5.generate_deterministic_trajectory( duration = 720,
                                                                           repression_threshold = 100000,
                                                                           mRNA_degradation_rate = 0.03,
                                                                           protein_degradation_rate = 0.03,
                                                                           transcription_delay = 18.5,
                                                                           basal_transcription_rate = 1000.0,
                                                                           translation_rate = 1.0,
                                                                           initial_mRNA = 3000,
                                                                           initial_protein = 100000,
                                                                           for_negative_times = 'no_negative' )

        figuresize = (4,2.75)
        my_figure = plt.figure()
        # want to plot: protein and mRNA for stochastic and deterministic system,
        # example stochastic system
        plt.plot( mRNA_trajectories[:,0],
                  mRNA_trajectories[:,1]/1000., label = 'mRNA example', color = 'black' )
        plt.plot( protein_trajectories[:,0],
                  protein_trajectories[:,1]/10000., label = 'Protein example', color = 'black', ls = '--' )
        plt.plot( mRNA_trajectories[:,0],
                  mean_mRNA_trajectory/1000., label = 'Mean mRNA', color = 'blue' )
        plt.plot( protein_trajectories[:,0],
                  mean_protein_trajectory/10000., label = 'Mean protein', color = 'blue', ls = '--' )
        plt.plot( deterministic_trajectory[:,0],
                  deterministic_trajectory[:,1]/1000., label = 'Deterministic mRNA', color = 'green' )
        plt.plot( deterministic_trajectory[:,0],
                  deterministic_trajectory[:,2]/10000., label = 'Deterministic Protein', color = 'green', ls = '--' )
        plt.xlabel('Time')
        plt.ylabel('Scaled expression')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','stochastic_model_validation.pdf'))

    def xest_stochastic_hes_trajectory(self):
        #First: run the model for 100 minutes
        my_trajectory = hes5.generate_stochastic_trajectory( duration = 720,
                                                         repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         translation_rate = 250,
                                                         transcription_delay = 29,
                                                         initial_mRNA = 3,
                                                         initial_protein = 23000)

        
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
                                       'output','hes5_stochastic_trajectory.pdf'))

    def xest_stochastic_hes_trajectory_different_transcription(self):
        my_trajectory = hes5.generate_stochastic_trajectory( duration = 1500,
                                                         repression_threshold = 23000,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         translation_rate = 26,
                                                         basal_transcription_rate = 9,
                                                         transcription_delay = 29,
                                                         initial_mRNA = 3,
                                                         initial_protein = 23000)

        
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
                                       'output','hes5_stochastic_trajectory_more_rna.pdf'))
    
    def xest_equlibrate_stochastic_trajectory(self):
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

    def xest_multiple_equlibrated_trajectories(self):
        mRNA_trajectories, protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = 100,
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
                                       'output','average_hes5_behaviour.pdf'))

    def xest_average_trajectories_in_oscillating_regime(self):
        mRNA_trajectories, protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = 100,
                                                                                        duration = 1500,
                                                         repression_threshold = 100,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = 18.5,
                                                         initial_mRNA = 3,
                                                         initial_protein = 100,
                                                         equilibration_time = 1000 )

        mean_protein_trajectory = np.mean(protein_trajectories[:,1:], axis = 1)
        protein_deviation = np.std(mRNA_trajectories[:,1:])
        mean_mRNA_trajectory = np.mean(mRNA_trajectories[:,1:], axis = 1)
        mRNA_deviation = np.std(mRNA_trajectories[:,1:])
        
        figuresize = (4,2.75)
        my_figure = plt.figure()
        # want to plot: protein and mRNA for stochastic and deterministic system,
        # example stochastic system
        plt.plot( mRNA_trajectories[:,0],
                  mRNA_trajectories[:,1]*10., label = 'mRNA example', color = 'black' )
        plt.plot( protein_trajectories[:,0],
                  protein_trajectories[:,1], label = 'Protein example', color = 'black', ls = '--' )
        plt.plot( mRNA_trajectories[:,0],
                  mean_mRNA_trajectory*10, label = 'Mean mRNA*10', color = 'blue' )
        plt.plot( protein_trajectories[:,0],
                  mean_protein_trajectory, label = 'Mean protein*10', color = 'blue', ls = '--' )
        plt.ylabel('Copy number')
        plt.legend()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','average_oscillating_behaviour.pdf'))

    def xest_power_spectra_of_mean_behaviours(self):
        ## oscillating power spectrum?
        mRNA_trajectories, protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = 100,
                                                                                        duration = 1500,
                                                         repression_threshold = 100,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = 18.5,
                                                         initial_mRNA = 3,
                                                         initial_protein = 100,
                                                         equilibration_time = 1000 )

        mean_protein_trajectory = np.mean(protein_trajectories[:,1:], axis = 1)
        protein_deviation = np.std(mRNA_trajectories[:,1:])
        mean_mRNA_trajectory = np.mean(mRNA_trajectories[:,1:], axis = 1)
        mRNA_deviation = np.std(mRNA_trajectories[:,1:])
 
        number_of_data_points = len(mean_mRNA_trajectory)
        interval_length = protein_trajectories[-1,0]
        oscillating_fourier_transform = np.fft.fft(mean_protein_trajectory)/number_of_data_points
        oscillating_fourier_frequencies = np.arange( 0,number_of_data_points/(2*interval_length), 
                                                     1.0/(interval_length) )
        oscillating_power_spectrum = np.power(np.abs(oscillating_fourier_transform),2)[1:]

        ## Calculate coherence:
        max_index = np.argmax(oscillating_power_spectrum)
        coherence_boundary_left = int(np.round(max_index - max_index*0.1))
        coherence_boundary_right = int(np.round(max_index + max_index*0.1))
        coherence_area = np.trapz(oscillating_power_spectrum[coherence_boundary_left:(coherence_boundary_right+1)])
        full_area = np.trapz(oscillating_power_spectrum)
        oscillating_coherence = coherence_area/full_area
        
        import pdb; pdb.set_trace()
        my_figure = plt.figure()
        figuresize = (4,3.5)
        my_figure.add_subplot(211)
        plt.plot(oscillating_fourier_frequencies[1:], 
                 np.power(np.abs(oscillating_fourier_transform[1:(number_of_data_points/2)]), 2), color = 'black')
        plt.xlim(0,0.01)
#         plt.ylim(0,100)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
#         import pdb; pdb.set_trace()
        plt.text(0.95, 0.4, 'Coherence: ' + str(oscillating_coherence),
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=plt.gca().transAxes)

        my_figure.add_subplot(212)
        mRNA_trajectories = np.load( os.path.join(os.path.dirname(__file__),
                                    'output','rna_traces.npy') )
        protein_trajectories = np.load( os.path.join(os.path.dirname(__file__),
                                      'output','protein_traces.npy') )
 
        mean_protein_trajectory = np.mean(protein_trajectories[:,1:], axis = 1)
        protein_deviation = np.std(mRNA_trajectories[:,1:])
        mean_mRNA_trajectory = np.mean(mRNA_trajectories[:,1:], axis = 1)
        mRNA_deviation = np.std(mRNA_trajectories[:,1:])
#  
        number_of_data_points = len(mean_mRNA_trajectory)
        interval_length = protein_trajectories[-1,0]
        hes5_fourier_transform = np.fft.fft(mean_protein_trajectory)/number_of_data_points
        hes5_fourier_frequencies = np.arange(0,number_of_data_points/(2*interval_length), 
                                                    1.0/(interval_length) )
        hes5_power_spectrum = np.power(np.abs(hes5_fourier_transform),2)[1:]
         ## Calculate coherence:
        max_index = np.argmax(hes5_power_spectrum)
        coherence_boundary_left = int(np.round(max_index - max_index*0.1))
        coherence_boundary_right = int(np.round(max_index + max_index*0.1))
        coherence_area = np.trapz(hes5_power_spectrum[coherence_boundary_left:(coherence_boundary_right+1)])
        full_area = np.trapz(hes5_power_spectrum)
        hes5_coherence = coherence_area/full_area
        
        plt.plot(hes5_fourier_frequencies[1:], 
                 np.power(np.abs(hes5_fourier_transform[1:(number_of_data_points/2)]), 2), color = 'black')
#         plt.xlim(0,1)
        plt.xlim(0,0.01)
        plt.text(0.95, 0.4, 'Coherence: ' + str(hes5_coherence),
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=plt.gca().transAxes)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','fourier_spectra.pdf'))

    def test_power_spectra_of_mean_behaviours_fewer_trajectories(self):
        ##
        # Hes1 samples
        ##
        oscillating_mRNA_trajectories, oscillating_protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = 100,
                                                                                        duration = 1500,
                                                         repression_threshold = 100,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = 18.5,
                                                         initial_mRNA = 3,
                                                         initial_protein = 100,
                                                         equilibration_time = 1000 )

        oscillating_power_spectrum, oscillating_coherence, oscillating_period = hes5.calculate_power_spectrum_of_trajectories(oscillating_protein_trajectories)
        
        mean_oscillating_protein_trajectory = np.mean(oscillating_protein_trajectories[:,1:], axis = 1)
        mean_oscillating_mRNA_trajectory = np.mean(oscillating_mRNA_trajectories[:,1:], axis = 1)
        
        figuresize = (6,6)
        my_figure = plt.figure(figsize = figuresize)
        my_figure.add_subplot(321)
        plt.plot( oscillating_mRNA_trajectories[:,0],
                  oscillating_mRNA_trajectories[:,1]*10., label = 'mRNA example*10', color = 'black',
                  lw = 0.5 )
        plt.plot( oscillating_protein_trajectories[:,0],
                  oscillating_protein_trajectories[:,1], label = 'Protein example', color = 'black', ls = '--',
                  lw = 0.5, dashes = [1,1] )
        plt.plot( oscillating_mRNA_trajectories[:,0],
                  mean_oscillating_mRNA_trajectory*10, label = 'Mean mRNA*10', color = 'blue',
                  lw = 0.5 )
        plt.plot( oscillating_protein_trajectories[:,0],
                  mean_oscillating_protein_trajectory, label = 'Mean protein', color = 'blue', ls = '--',
                  lw = 0.5, dashes = [1,1] )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.title('Monk (2003)')
#         plt.legend()
       
        my_figure.add_subplot(322)
        plt.plot(oscillating_power_spectrum[:,0],
                 oscillating_power_spectrum[:,1], color = 'black')
        plt.xlim(0,0.01)
#         plt.ylim(0,100)
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
#         import pdb; pdb.set_trace()
        plt.text(0.05, 0.95, 'Coherence:\n' + "{:.2f}".format(oscillating_coherence) + 
                 '\nPeriod:\n' +  "{:.2f}".format(oscillating_period) ,
                 verticalalignment='top', horizontalalignment='left',
                 transform=plt.gca().transAxes)

        ##
        # Hes5 samples
        ##
        hes5_mRNA_trajectories, hes5_protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = 100,
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

        hes5_power_spectrum, hes5_coherence, hes5_period = hes5.calculate_power_spectrum_of_trajectories(hes5_protein_trajectories)
        
        mean_hes5_protein_trajectory = np.mean(hes5_protein_trajectories[:,1:], axis = 1)
        mean_hes5_rna_trajectory = np.mean(hes5_mRNA_trajectories[:,1:], axis = 1)

        my_figure.add_subplot(323)
        mrna_example, = plt.plot( hes5_mRNA_trajectories[:,0],
                  hes5_mRNA_trajectories[:,1]*1000., label = 'mRNA example*1000', color = 'black',
                  lw = 0.5 )
        protein_example, = plt.plot( hes5_protein_trajectories[:,0],
                  hes5_protein_trajectories[:,1], label = 'Protein example', color = 'black', ls = '--',
                  lw = 0.5, dashes = [1,1] )
        mean_rna, = plt.plot( hes5_mRNA_trajectories[:,0],
                  mean_hes5_rna_trajectory*1000., label = 'Mean mRNA*10', color = 'blue',
                  lw = 0.5 )
        mean_protein, = plt.plot( hes5_protein_trajectories[:,0],
                  mean_hes5_protein_trajectory, label = 'Mean protein', color = 'blue', ls = '--',
                  lw = 0.5, dashes = [1,1] )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.title('Hes5')
#         plt.legend()

        my_figure.add_subplot(324)
        plt.plot(hes5_power_spectrum[:,0],
                 hes5_power_spectrum[:,1], color = 'black')
        plt.xlim(0,0.01)
#         plt.ylim(0,100)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
#         import pdb; pdb.set_trace()
        plt.text(0.95, 0.95, 'Coherence:\n' + "{:.2f}".format(hes5_coherence) + 
                 '\nPeriod:\n' +  "{:.2f}".format(hes5_period) ,
                 verticalalignment='top', horizontalalignment='right',
                 transform=plt.gca().transAxes)
        
        ##
        # Random samples
        ##
        # generate the random samples:
        random_trajectories = np.zeros((100,101))
        times_of_trajectories = np.linspace(0,1500,100)
        random_trajectories[:,0] = times_of_trajectories
        for trajectory_index in range(100):
            for time_index in range(1,100):
                random_trajectories[time_index, trajectory_index+1] = random_trajectories[time_index-1, trajectory_index+1]\
                                                                    + np.random.randn()*1.0

        random_trajectories[:,1:] += 100
        # generate power spectrum, measure period etc
        random_power_spectrum, random_coherence, random_period = \
            hes5.calculate_power_spectrum_of_trajectories(random_trajectories)
        
        mean_random_trajectory = np.mean(random_trajectories[:,1:], axis = 1)

        my_figure.add_subplot(325)
        plt.plot( random_trajectories[:,0],
                  random_trajectories[:,1], label = 'Protein example', color = 'black', ls = '--',
                  lw = 0.5, dashes = [1,1] )
        plt.plot( random_trajectories[:,0],
                  mean_random_trajectory, label = 'Mean protein', color = 'blue', ls = '--',
                  lw = 0.5, dashes = [1,1] )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)
        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.title('Random traces')
#         plt.legend()

        my_figure.add_subplot(326)
        plt.plot(random_power_spectrum[:,0],
                 random_power_spectrum[:,1], color = 'black')
        plt.xlim(0,0.01)
#         plt.ylim(0,100)
        plt.xlabel('Frequency')
        plt.ylabel('Occurence')
#         import pdb; pdb.set_trace()
        plt.text(0.95, 0.95, 'Coherence: ' + "{:.2f}".format(random_coherence) + 
                 '\nPeriod: ' +  "{:.2f}".format(random_period) ,
                 verticalalignment='top', horizontalalignment='right',
                 transform=plt.gca().transAxes)

        plt.tight_layout()
        my_figure.legend((mrna_example, protein_example, mean_rna, mean_protein), 
                       ('mRNA example (scaled)', 'Protein example',
                        'mean mRNA (scaled)', 'Mean protein'), 
                       loc = 'upper right', ncol = 2 )
        plt.gca().locator_params(axis='x', tight = True, nbins=4)

        plt.subplots_adjust(top = 0.85, hspace = 0.7)
       
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','fourier_spectra_illustration_10.pdf'))

    def xest_plot_100_hes_trajectories(self):
        hes5_mRNA_trajectories, hes5_protein_trajectories = hes5.generate_multiple_trajectories( number_of_trajectories = 100,
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

        mean_hes5_protein_trajectory = np.mean(hes5_protein_trajectories[:,1:], axis = 1)
        mean_hes5_rna_trajectory = np.mean(hes5_mRNA_trajectories[:,1:], axis = 1)
        figuresize = (4,2.5)
        my_figure = plt.figure(figsize = figuresize)
        for trajectory_index in range(1,101):
            plt.plot( hes5_mRNA_trajectories[:,0],
                      hes5_mRNA_trajectories[:,trajectory_index]*1000., color = 'black',
                      lw = 0.5, alpha = 0.1 )
            plt.plot( hes5_protein_trajectories[:,0],
                      hes5_protein_trajectories[:,trajectory_index], color = 'black', ls = '--',
                      lw = 0.5, dashes = [1,1], alpha = 0.1 )
        plt.plot( hes5_mRNA_trajectories[:,0],
                  mean_hes5_rna_trajectory*1000., label = 'mRNA*10', color = 'blue',
                  lw = 0.5 )
        plt.plot( hes5_protein_trajectories[:,0],
                  mean_hes5_protein_trajectory, label = 'Protein', color = 'blue', ls = '--',
                  lw = 0.5, dashes = [1,1] )

        plt.xlabel('Time')
        plt.ylabel('Copy number')
        plt.legend()
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','100_trajectories.pdf'))