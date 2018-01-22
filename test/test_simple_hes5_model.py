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
        x_values = np.linspace(1,interval_length,100000)
        function_values = 3*np.sin(2*np.pi*0.5*x_values) + 2*np.sin(2*np.pi*0.2*x_values)
        number_of_data_points = len(x_values)
        fourier_transform = np.fft.fft(function_values)/number_of_data_points
        fourier_frequencies = np.arange(0,number_of_data_points/(2*interval_length), 1.0/(interval_length) )

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
    
    def test_equlibrate_stochastic_trajectories(self):
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
 