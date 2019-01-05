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

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import switching
import hes5

class TestSwitching(unittest.TestCase):

    def xest_generate_single_gillespie_trajectory(self):
        my_trajectory = switching.generate_switching_trajectory( duration = 1500,
                                                         repression_threshold = 10000,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = 19,
                                                         initial_mRNA = 3,
                                                         initial_protein = 100,
                                                         basal_transcription_rate = 1.0,
                                                         translation_rate = 4.0,
                                                         equilibration_time = 1000,
                                                         switching_rate = 0.1)

        #Second, plot the model
        figuresize = (4,4)
        my_figure = plt.figure(figsize=figuresize)
        plt.subplot(211)
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1], label = 'mRNA', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2]*0.03, label = 'Hes protein', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Scaled expression')
        plt.ylim(0,600)
        plt.legend(ncol=2)
        plt.subplot(212)
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,3], label = 'environment', color = 'black')
        plt.xlabel('Time')
        plt.ylabel('environmental state')
        plt.ylim(-0.2, 1.2)
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','switching_trajectory.pdf'))

    def xest_generate_single_langevin_trajectory(self):
        my_trajectory = switching.generate_switching_langevin_trajectory( duration = 1500,
                                                         repression_threshold = 10000,
                                                         mRNA_degradation_rate = np.log(2)/30,
                                                         protein_degradation_rate = np.log(2)/90,
                                                         transcription_delay = 19,
                                                         initial_mRNA = 3,
                                                         initial_protein = 100,
                                                         basal_transcription_rate = 1.0,
                                                         translation_rate = 4.0,
                                                         equilibration_time = 1000,
                                                         switching_rate = 0.1)

        #Second, plot the model
        figuresize = (4,2.75)
        my_figure = plt.figure(figsize=figuresize)
        plt.plot(my_trajectory[:,0], 
                 my_trajectory[:,1], label = 'mRNA', color = 'black')
        plt.plot(my_trajectory[:,0],
                 my_trajectory[:,2]*0.03, label = 'Hes protein', color = 'black', ls = '--')
        plt.xlabel('Time')
        plt.ylabel('Scaled expression')
        plt.ylim(0,600)
        plt.legend(ncol=2)
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','switching_langevin_trajectory.pdf'))

    def xest_validate_stochastic_langevin_implementation(self):
        mRNA_trajectories, protein_trajectories = switching.generate_multiple_switching_langevin_trajectories( number_of_trajectories = 10,
                                                                     duration = 720,
                                                                     repression_threshold = 100000,
                                                                     mRNA_degradation_rate = 0.03,
                                                                     protein_degradation_rate = 0.03,
                                                                     transcription_delay = 19,
                                                                     basal_transcription_rate = 1000.0,
                                                                     translation_rate = 1.0,
                                                                     initial_mRNA = 3000,
                                                                     initial_protein = 100000,
                                                                     switching_rate = 1000.0 )
        
        mean_protein_trajectory = np.mean(protein_trajectories[:,1:], axis = 1)
        protein_deviation = np.std(mRNA_trajectories[:,1:])
        mean_mRNA_trajectory = np.mean(mRNA_trajectories[:,1:], axis = 1)
        mRNA_deviation = np.std(mRNA_trajectories[:,1:])
        
        deterministic_trajectory = hes5.generate_deterministic_trajectory( duration = 720,
                                                                           repression_threshold = 100000,
                                                                           mRNA_degradation_rate = 0.03,
                                                                           protein_degradation_rate = 0.03,
                                                                           transcription_delay = 19,
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
                                       'output','switching_langevin_model_validation.pdf'))


    def xest_validate_stochastic_implementation(self):
        mRNA_trajectories, protein_trajectories = switching.generate_multiple_switching_trajectories( number_of_trajectories = 10,
                                                                     duration = 720,
                                                                     repression_threshold = 100000,
                                                                     mRNA_degradation_rate = 0.03,
                                                                     protein_degradation_rate = 0.03,
                                                                     transcription_delay = 19,
                                                                     basal_transcription_rate = 1000.0,
                                                                     translation_rate = 1.0,
                                                                     initial_mRNA = 3000,
                                                                     initial_protein = 100000,
                                                                     switching_rate = 1000.0 )
        
        mean_protein_trajectory = np.mean(protein_trajectories[:,1:], axis = 1)
        protein_deviation = np.std(mRNA_trajectories[:,1:])
        mean_mRNA_trajectory = np.mean(mRNA_trajectories[:,1:], axis = 1)
        mRNA_deviation = np.std(mRNA_trajectories[:,1:])
        
        deterministic_trajectory = hes5.generate_deterministic_trajectory( duration = 720,
                                                                           repression_threshold = 100000,
                                                                           mRNA_degradation_rate = 0.03,
                                                                           protein_degradation_rate = 0.03,
                                                                           transcription_delay = 19,
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
                                       'output','stochastic_switching_model_validation.pdf'))

    def xest_reproduce_tobias_example_gillespie(self):
        system_size = 1000
        deterministic_trajectory = hes5.generate_deterministic_trajectory( duration = 100*60,
                                                         repression_threshold = 10*system_size,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = 18.7,
                                                         initial_mRNA = 1,
                                                         initial_protein = 10,
                                                         basal_transcription_rate = 1.0*system_size,
                                                         translation_rate = 1.0,
                                                         hill_coefficient = 4.1)


        my_slow_trajectory = switching.generate_switching_trajectory( duration = 100*60,
                                                        repression_threshold = 10*system_size,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        transcription_delay = 18.7,
                                                        initial_mRNA = 1,
                                                        initial_protein = 10,
                                                        basal_transcription_rate = 1.0*system_size,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1,
                                                        equilibration_time = 10*60,
                                                        switching_rate = 2)
# 
        my_fast_trajectory = switching.generate_switching_trajectory( duration = 100*60,
                                                        repression_threshold = 10*system_size,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        transcription_delay = 18.7,
                                                        initial_mRNA = 1,
                                                        initial_protein = 10,
                                                        basal_transcription_rate = 1.0*system_size,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1,
                                                        equilibration_time = 100*60,
                                                        switching_rate = 10)
#
#         my_fast_trajectory = switching.generate_switching_trajectory( duration = 100*60,
#                                                          repression_threshold = 10*system_size,
#                                                          mRNA_degradation_rate = 0.03,
#                                                          protein_degradation_rate = 0.03,
#                                                          transcription_delay = 18.7,
#                                                          initial_mRNA = 1,
#                                                          initial_protein = 10,
#                                                          basal_transcription_rate = 1.0*system_size,
#                                                          hill_coefficient = 4.1,
#                                                          translation_rate = 1.0,
#                                                          equilibration_time = 1000,
#                                                          switching_rate = 500.0)
# 
# 
        steady_state = hes5.calculate_steady_state_of_ode(repression_threshold = 10*system_size,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        basal_transcription_rate = 1.0*system_size,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1)
        #Second, plot the model
        figuresize = (4,4)
        my_figure = plt.figure(figsize=figuresize)
        plt.subplot(211)
#         plt.plot(deterministic_trajectory[:,0]/60, 
#                  deterministic_trajectory[:,1])
        plt.plot(my_slow_trajectory[:,0]/60, 
                my_slow_trajectory[:,1])
        plt.plot(my_fast_trajectory[:,0]/60,
                my_fast_trajectory[:,1])
        plt.axhline(steady_state[0])
#         plt.ylim(0,2)
        plt.xlabel('Time')
        plt.ylabel('mRNA expression')

        plt.subplot(212)
#         plt.plot(deterministic_trajectory[:,0]/60, 
#                 deterministic_trajectory[:,2])
        plt.plot(my_slow_trajectory[:,0]/60, 
                my_slow_trajectory[:,2])
        plt.plot(my_fast_trajectory[:,0]/60,
                my_fast_trajectory[:,2])
#         plt.ylim(0,5000)
        plt.axhline(steady_state[1])
        plt.xlabel('Time')
        plt.ylabel('Protein expression')
#         plt.ylim(20,35)
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','switching_lambda_dependence.pdf'))


    def xest_reproduce_tobias_example_langevin(self):
        system_size = 1000000
        deterministic_trajectory = hes5.generate_deterministic_trajectory( duration = 100*60,
                                                         repression_threshold = 10*system_size,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = 18.7,
                                                         initial_mRNA = 1,
                                                         initial_protein = 10,
                                                         basal_transcription_rate = 1.0*system_size,
                                                         translation_rate = 1.0,
                                                         hill_coefficient = 4.1)


        my_slow_trajectory = switching.generate_switching_langevin_trajectory( duration = 100*60,
                                                        repression_threshold = 10*system_size,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        transcription_delay = 18.7,
                                                        initial_mRNA = 1,
                                                        initial_protein = 10,
                                                        basal_transcription_rate = 1.0*system_size,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1,
                                                        equilibration_time = 100*60,
                                                        switching_rate = 2)
# 
        my_fast_trajectory = switching.generate_switching_langevin_trajectory( duration = 100*60,
                                                        repression_threshold = 10*system_size,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        transcription_delay = 18.7,
                                                        initial_mRNA = 1,
                                                        initial_protein = 10,
                                                        basal_transcription_rate = 1.0*system_size,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1,
                                                        equilibration_time = 100*60,
                                                        switching_rate = 10)
#
#         my_fast_trajectory = switching.generate_switching_trajectory( duration = 100*60,
#                                                          repression_threshold = 10*system_size,
#                                                          mRNA_degradation_rate = 0.03,
#                                                          protein_degradation_rate = 0.03,
#                                                          transcription_delay = 18.7,
#                                                          initial_mRNA = 1,
#                                                          initial_protein = 10,
#                                                          basal_transcription_rate = 1.0*system_size,
#                                                          hill_coefficient = 4.1,
#                                                          translation_rate = 1.0,
#                                                          equilibration_time = 1000,
#                                                          switching_rate = 500.0)
# 
# 
        steady_state = hes5.calculate_steady_state_of_ode(repression_threshold = 10*system_size,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        basal_transcription_rate = 1.0*system_size,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1)
        #Second, plot the model
        figuresize = (4,4)
        my_figure = plt.figure(figsize=figuresize)
        plt.subplot(211)
#         plt.plot(deterministic_trajectory[:,0]/60, 
#                  deterministic_trajectory[:,1])
        plt.plot(my_slow_trajectory[:,0]/60, 
                my_slow_trajectory[:,1])
        plt.plot(my_fast_trajectory[:,0]/60,
                my_fast_trajectory[:,1])
        plt.axhline(steady_state[0])
#         plt.ylim(0,2)
        plt.xlabel('Time')
        plt.ylabel('mRNA expression')

        plt.subplot(212)
#         plt.plot(deterministic_trajectory[:,0]/60, 
#                 deterministic_trajectory[:,2])
        plt.plot(my_slow_trajectory[:,0]/60, 
                my_slow_trajectory[:,2])
        plt.plot(my_fast_trajectory[:,0]/60,
                my_fast_trajectory[:,2])
#         plt.ylim(0,5000)
        plt.axhline(steady_state[1])
        plt.xlabel('Time')
        plt.ylabel('Protein expression')
#         plt.ylim(20,35)
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','switching_lambda_dependence_langevin.pdf'))

    def test_difference_langevin_gillespie(self):
        system_size = 1000

        slow_gillespie_mrna, slow_gillespie_protein,_ = switching.generate_multiple_switching_trajectories( 
                                                        number_of_trajectories = 100,
                                                        duration = 1000*60,
                                                        repression_threshold = 10*system_size,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        transcription_delay = 18.7,
                                                        initial_mRNA = 1,
                                                        initial_protein = 10,
                                                        basal_transcription_rate = 1.0*system_size,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1,
                                                        equilibration_time = 10*60,
                                                        switching_rate = 2)

        slow_langevin_mrna, slow_langevin_protein = switching.generate_multiple_switching_langevin_trajectories( 
                                                        number_of_trajectories = 100,
                                                        duration = 1000*60,
                                                        repression_threshold = 10*system_size,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        transcription_delay = 18.7,
                                                        initial_mRNA = 1,
                                                        initial_protein = 10,
                                                        basal_transcription_rate = 1.0*system_size,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1,
                                                        equilibration_time = 10*60,
                                                        switching_rate = 2)
# 
        mean_langevin_mRNA = np.mean(slow_langevin_mrna[:,1])
        mean_langevin_protein = np.mean(slow_langevin_protein[:,1])
        mean_gillespie_mRNA = np.mean(slow_gillespie_mrna[:,1])
        mean_gillespie_protein = np.mean(slow_gillespie_protein[:,1])

        std_langevin_mRNA = np.std(slow_langevin_mrna[:,1])
        std_langevin_protein = np.std(slow_langevin_protein[:,1])
        std_gillespie_mRNA = np.std(slow_gillespie_mrna[:,1])
        std_gillespie_protein = np.std(slow_gillespie_protein[:,1])
        
        print 'langevin mean'
        print(mean_langevin_protein)
        print 'langevin std'
        print(std_langevin_protein)
        print 'gilespie mean'
        print(mean_gillespie_protein)
        print 'gilespie std'
        print(std_gillespie_protein)
        my_fast_trajectory = switching.generate_switching_trajectory( duration = 100*60,
                                                        repression_threshold = 10*system_size,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        transcription_delay = 18.7,
                                                        initial_mRNA = 1,
                                                        initial_protein = 10,
                                                        basal_transcription_rate = 1.0*system_size,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1,
                                                        equilibration_time = 100*60,
                                                        switching_rate = 10)
#
#         my_fast_trajectory = switching.generate_switching_trajectory( duration = 100*60,
#                                                          repression_threshold = 10*system_size,
#                                                          mRNA_degradation_rate = 0.03,
#                                                          protein_degradation_rate = 0.03,
#                                                          transcription_delay = 18.7,
#                                                          initial_mRNA = 1,
#                                                          initial_protein = 10,
#                                                          basal_transcription_rate = 1.0*system_size,
#                                                          hill_coefficient = 4.1,
#                                                          translation_rate = 1.0,
#                                                          equilibration_time = 1000,
#                                                          switching_rate = 500.0)
# 
# 
        steady_state = hes5.calculate_steady_state_of_ode(repression_threshold = 10*system_size,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        basal_transcription_rate = 1.0*system_size,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1)
        #Second, plot the model
        figuresize = (4,4)
        my_figure = plt.figure(figsize=figuresize)
        plt.subplot(211)
#         plt.plot(deterministic_trajectory[:,0]/60, 
#                  deterministic_trajectory[:,1])
        plt.plot(my_slow_trajectory[:,0]/60, 
                my_slow_trajectory[:,1])
        plt.plot(my_fast_trajectory[:,0]/60,
                my_fast_trajectory[:,1])
        plt.axhline(steady_state[0])
#         plt.ylim(0,2)
        plt.xlabel('Time')
        plt.ylabel('mRNA expression')

        plt.subplot(212)
#         plt.plot(deterministic_trajectory[:,0]/60, 
#                 deterministic_trajectory[:,2])
        plt.plot(my_slow_trajectory[:,0]/60, 
                my_slow_trajectory[:,2])
        plt.plot(my_fast_trajectory[:,0]/60,
                my_fast_trajectory[:,2])
#         plt.ylim(0,5000)
        plt.axhline(steady_state[1])
        plt.xlabel('Time')
        plt.ylabel('Protein expression')
#         plt.ylim(20,35)
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','switching_lambda_dependence.pdf'))

