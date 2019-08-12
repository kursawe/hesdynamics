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
import functools

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import switching
import hes5
import pickle

parameter_set_dict = {}
parameter_set_dict['standard'] = dict()
parameter_set_dict['standard']['repression_threshold'] = 10
parameter_set_dict['standard']['mRNA_degradation_rate'] = 0.03
parameter_set_dict['standard']['protein_degradation_rate'] = 0.03
parameter_set_dict['standard']['transcription_delay'] = 18.7
parameter_set_dict['standard']['initial_mRNA'] = 1
parameter_set_dict['standard']['initial_protein'] = 10
parameter_set_dict['standard']['basal_transcription_rate'] = 1.0
parameter_set_dict['standard']['translation_rate'] = 1.0
parameter_set_dict['standard']['hill_coefficient'] = 4.1
parameter_set_dict['standard']['equilibration_time'] = 5000

parameter_set_dict['hes5'] = dict()
parameter_set_dict['hes5']['repression_threshold'] = 50000
parameter_set_dict['hes5']['mRNA_degradation_rate'] = np.log(2)/30
parameter_set_dict['hes5']['protein_degradation_rate'] = np.log(2)/90
parameter_set_dict['hes5']['transcription_delay'] = 30
parameter_set_dict['hes5']['initial_mRNA'] = 1
parameter_set_dict['hes5']['initial_protein'] = 10
parameter_set_dict['hes5']['basal_transcription_rate'] = 1.0
parameter_set_dict['hes5']['translation_rate'] = 15.0
parameter_set_dict['hes5']['hill_coefficient'] = 4
parameter_set_dict['hes5']['equilibration_time'] = 5000

parameter_set_dict['her6'] = dict()
parameter_set_dict['her6']['repression_threshold'] = 2000
parameter_set_dict['her6']['mRNA_degradation_rate'] = np.log(2)/7
parameter_set_dict['her6']['protein_degradation_rate'] = np.log(2)/11
parameter_set_dict['her6']['transcription_delay'] = 5
parameter_set_dict['her6']['initial_mRNA'] = 1
parameter_set_dict['her6']['initial_protein'] = 10.0
parameter_set_dict['her6']['basal_transcription_rate'] = 20.0
parameter_set_dict['her6']['translation_rate'] = 1.0
parameter_set_dict['her6']['hill_coefficient'] = 4
parameter_set_dict['her6']['equilibration_time'] = 5000

parameter_points_dict = {}
for parameter_name, parameter_values in parameter_set_dict.items():
    parameter_points_dict[parameter_name] = np.array([parameter_values['basal_transcription_rate'],
                                                      parameter_values['translation_rate'],
                                                      parameter_values['repression_threshold'],
                                                      parameter_values['transcription_delay'],
                                                      parameter_values['hill_coefficient'],
                                                      parameter_values['mRNA_degradation_rate'],
                                                      parameter_values['protein_degradation_rate']])


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

    def xest_generate_single_ode_switching_trajectory(self):
        my_trajectory = switching.generate_ode_switching_trajectory( duration = 1500,
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
                                       'output','ode_switching_trajectory.pdf'))

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

    def xest_validate_stochastic_ode_implementation(self):
        mRNA_trajectories, protein_trajectories = switching.generate_multiple_switching_ode_trajectories( number_of_trajectories = 10,
                                                                     duration = 720,
                                                                     repression_threshold = 100000.0,
                                                                     mRNA_degradation_rate = 0.03,
                                                                     protein_degradation_rate = 0.03,
                                                                     transcription_delay = 19,
                                                                     basal_transcription_rate = 1000.0,
                                                                     translation_rate = 1.0,
                                                                     initial_mRNA = 3000,
                                                                     initial_protein = 100000,
                                                                     switching_rate = 100.0,
                                                                     discretisation_time_step = 0.0001,
                                                                     initial_environment_on = False)
        
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
                                       'output','switching_ode_model_validation.pdf'))

    def xest_difference_langevin_gillespie(self):
        system_size = 1000

        slow_gillespie_mrna, slow_gillespie_protein,_ = switching.generate_multiple_switching_trajectories( 
                                                        number_of_trajectories = 1000,
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
                                                        switching_rate = 20)

        slow_langevin_mrna, slow_langevin_protein = switching.generate_multiple_switching_langevin_trajectories( 
                                                        number_of_trajectories = 1000,
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
                                                        switching_rate = 20)
# 
        mean_langevin_mRNA = np.mean(slow_langevin_mrna[:,1])
        mean_langevin_protein = np.mean(slow_langevin_protein[:,1])
        mean_gillespie_mRNA = np.mean(slow_gillespie_mrna[:,1])
        mean_gillespie_protein = np.mean(slow_gillespie_protein[:,1])

        std_langevin_mRNA = np.std(slow_langevin_mrna[:,1])
        std_langevin_protein = np.std(slow_langevin_protein[:,1])
        std_gillespie_mRNA = np.std(slow_gillespie_mrna[:,1])
        std_gillespie_protein = np.std(slow_gillespie_protein[:,1])
        
        print('langevin mean')
        print(mean_langevin_protein)
        print('langevin std')
        print(std_langevin_protein)
        print('gilespie mean')
        print(mean_gillespie_protein)
        print('gilespie std')
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
                                                        switching_rate = 20)
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
#         steady_state = hes5.calculate_steady_state_of_ode(repression_threshold = 10*system_size,
#                                                         mRNA_degradation_rate = 0.03,
#                                                         protein_degradation_rate = 0.03,
#                                                         basal_transcription_rate = 1.0*system_size,
#                                                         translation_rate = 1.0,
#                                                         hill_coefficient = 4.1)
#         #Second, plot the model
#         figuresize = (4,4)
#         my_figure = plt.figure(figsize=figuresize)
#         plt.subplot(211)
# #         plt.plot(deterministic_trajectory[:,0]/60, 
# #                  deterministic_trajectory[:,1])
#         plt.plot(my_slow_trajectory[:,0]/60, 
#                 my_slow_trajectory[:,1])
#         plt.plot(my_fast_trajectory[:,0]/60,
#                 my_fast_trajectory[:,1])
#         plt.axhline(steady_state[0])
# #         plt.ylim(0,2)
#         plt.xlabel('Time')
#         plt.ylabel('mRNA expression')
# 
#         plt.subplot(212)
# #         plt.plot(deterministic_trajectory[:,0]/60, 
# #                 deterministic_trajectory[:,2])
#         plt.plot(my_slow_trajectory[:,0]/60, 
#                 my_slow_trajectory[:,2])
#         plt.plot(my_fast_trajectory[:,0]/60,
#                 my_fast_trajectory[:,2])
# #         plt.ylim(0,5000)
#         plt.axhline(steady_state[1])
#         plt.xlabel('Time')
#         plt.ylabel('Protein expression')
# #         plt.ylim(20,35)
#         plt.tight_layout()
#         my_figure.savefig(os.path.join(os.path.dirname(__file__),
#                                        'output','switching_lambda_dependence.pdf'))
# 

    def xest_measure_time_average(self):
        
        # define a set of timesteps 
        timestep = 0.1
        durations = np.arange(0.1,5.0,timestep)
        time_averages = np.zeros((durations.shape[0],2))
        time_averages[:,0] = durations
        number_of_trajectories = 10000
        for duration_index, duration in enumerate(durations):
            print(duration)
            _,_,my_full_trajectories = switching.generate_multiple_switching_trajectories( 
                                                        number_of_trajectories = number_of_trajectories,
                                                        duration = duration,
                                                        repression_threshold = 1.0,
                                                        mRNA_degradation_rate = 0.0,
                                                        protein_degradation_rate = 0.0,
                                                        transcription_delay = 0.0,
                                                        initial_mRNA = 0,
                                                        initial_protein = 2.0,
                                                        basal_transcription_rate = 0.0,
                                                        translation_rate = 0.0,
                                                        hill_coefficient = 1,
                                                        equilibration_time = 100,
                                                        switching_rate = 1.0,
                                                        sampling_timestep = 0.01)
            for trajectory in my_full_trajectories[:,1:].transpose():
                average_a = np.trapz(trajectory,
                                     my_full_trajectories[:,0])
                average_a/=duration
                time_averages[duration_index,1] += average_a
            time_averages[duration_index,1] /= number_of_trajectories

        plt.figure()
        plt.plot(time_averages[:,0], time_averages[:,1])
        plt.ylim(0,1)
        plt.xlabel('time window')
        plt.ylabel('time average')
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                        'output','time_average_distribution.pdf'))
        
    
    def xest_generate_power_spectrum(self):
#         number_of_traces = 2000
        number_of_traces = 5
        switching_rate = 100.0
        _, these_real_traces = switching.generate_multiple_switching_ode_trajectories(number_of_trajectories = number_of_traces,
                                                         duration = 1500*20,
                                                         repression_threshold = 10,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = 18.7,
                                                         initial_mRNA = 1,
                                                         initial_protein = 10,
                                                         basal_transcription_rate = 1.0,
                                                         translation_rate = 1.0,
                                                         hill_coefficient = 4.1,
                                                         equilibration_time = 5000,
                                                         switching_rate = switching_rate)
 
        _, these_langevin_traces = switching.generate_multiple_switching_langevin_trajectories(number_of_trajectories = number_of_traces,
                                                         duration = 1500*20,
                                                         repression_threshold = 10,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = 18.7,
                                                         initial_mRNA = 1,
                                                         initial_protein = 10,
                                                         basal_transcription_rate = 1.0,
                                                         translation_rate = 1.0,
                                                         hill_coefficient = 4.1,
                                                         equilibration_time = 5000,
                                                         switching_rate = switching_rate,
                                                         model = 'switching_only')
 
        _, these_lna_traces = switching.generate_multiple_switching_langevin_trajectories(number_of_trajectories = number_of_traces,
                                                         duration = 1500*20,
                                                         repression_threshold = 10,
                                                         mRNA_degradation_rate = 0.03,
                                                         protein_degradation_rate = 0.03,
                                                         transcription_delay = 18.7,
                                                         initial_mRNA = 1,
                                                         initial_protein = 10,
                                                         basal_transcription_rate = 1.0,
                                                         translation_rate = 1.0,
                                                         hill_coefficient = 4.1,
                                                         equilibration_time = 5000,
                                                         switching_rate = switching_rate,
                                                         model = 'switching_only_lna')
 
 
        np.save(os.path.join(os.path.dirname(__file__), 'output','real_switching_trajectories_faster_lambda.npy'),
                    these_real_traces)
        
        np.save(os.path.join(os.path.dirname(__file__), 'output','langevin_switching_trajectories_faster_lambda.npy'),
                    these_langevin_traces)
 
        np.save(os.path.join(os.path.dirname(__file__), 'output','lna_switching_trajectories_faster_lambda.npy'),
                    these_lna_traces)

#         these_real_traces = np.load(os.path.join(os.path.dirname(__file__), 'output','real_switching_trajectories.npy'))
#          
#         these_langevin_traces = np.load(os.path.join(os.path.dirname(__file__), 'output','langevin_switching_trajectories.npy'))
#  
#         these_lna_traces = np.load(os.path.join(os.path.dirname(__file__), 'output','lna_switching_trajectories.npy'))


        print('Real timestep')
        print(these_real_traces[1,0] - these_real_traces[0,0])
        real_standard_deviation = np.var(these_real_traces[:,1:])
        langevin_standard_deviation = np.var(these_langevin_traces[:,1:])
        lna_standard_deviation = np.var(these_lna_traces[:,1:])
        print('Real variance')
        print(real_standard_deviation)
        print('Langevin variance')
        print(langevin_standard_deviation)
        print('lna variance')
        print(lna_standard_deviation)

        real_mean = np.mean(these_real_traces[:,1:])
        langevin_mean = np.mean(these_langevin_traces[:,1:])
        lna_mean = np.mean(these_lna_traces[:,1:])
        print('real mean')
        print(real_mean)
        print('langevin mean')
        print(langevin_mean)
        print('lna mean')
        print(lna_mean)

        this_real_power_spectrum, _, _ = hes5.calculate_power_spectrum_of_trajectories(these_real_traces, normalize = False)
        this_langevin_power_spectrum, _, _ = hes5.calculate_power_spectrum_of_trajectories(these_langevin_traces, normalize = False)
        this_lna_power_spectrum, _, _ = hes5.calculate_power_spectrum_of_trajectories(these_lna_traces, normalize = False)
        theoretical_power_spectrum = switching.calculate_theoretical_power_spectrum_at_parameter_point(
                                                        repression_threshold = 10,
                                                        mRNA_degradation_rate = 0.03,
                                                        protein_degradation_rate = 0.03,
                                                        transcription_delay = 18.7,
                                                        basal_transcription_rate = 1.0,
                                                        translation_rate = 1.0,
                                                        hill_coefficient = 4.1,
                                                        switching_rate = switching_rate)
#         theoretical_power_spectrum[:,1]*=np.sqrt(np.pi)
        
        power_area_theoretical = np.trapz(theoretical_power_spectrum[:,1], theoretical_power_spectrum[:,0])
        power_area_real = np.trapz(this_real_power_spectrum[:,1], this_real_power_spectrum[:,0])
        power_area_langevin = np.trapz(this_langevin_power_spectrum[:,1], this_langevin_power_spectrum[:,0])
        power_area_lna = np.trapz(this_lna_power_spectrum[:,1], this_lna_power_spectrum[:,0])
        ratio = power_area_theoretical/power_area_lna
        print('theoretical area')
        print(power_area_theoretical)
        print('real area')
        print(power_area_real)
        print('langevin area')
        print(power_area_langevin)
        print('lna area')
        print(power_area_lna)
        print('real traces length')
        print(len(these_real_traces))
        print('langevin traces length')
        print(len(these_langevin_traces))
        print('lna traces length')
        print(len(these_lna_traces))
        print('theoretical area over real area')
        print(ratio)

        plt.figure()
        plt.plot(this_real_power_spectrum[:,0], this_real_power_spectrum[:,1], lw = 1, label = 'full')
#         plt.plot(this_langevin_power_spectrum[:,0], this_langevin_power_spectrum[:,1], lw = 1, label = 'langevin')
#         plt.plot(this_lna_power_spectrum[:,0], this_lna_power_spectrum[:,1], lw = 1, label = 'lna')
        plt.plot(theoretical_power_spectrum[:,0], theoretical_power_spectrum[:,1], lw =1, label = 'theory' )
        # times = these_real_traces[:,0]
        # all_power_spectra = np.zeros((this_real_power_spectrum.shape[0], these_real_traces.shape[1] - 1))
        # frequency_values = this_real_power_spectrum[:,0]
        # trajectory_index = 0
        # for trajectory in these_real_traces[:,1:].transpose():
        #     this_compound_trajectory = np.vstack((times, trajectory)).transpose()
        #     this_power_spectrum,_,_ = hes5.calculate_power_spectrum_of_trajectory(this_compound_trajectory,
        #                                                                           normalize = False)
        #     all_power_spectra[:,trajectory_index] = this_power_spectrum[:,1]
        #     # plt.plot(this_power_spectrum[:,0], this_power_spectrum[:,1], lw = 1, color = 'grey', alpha = 0.1)
        #     trajectory_index+=1
        # mean_power_spectrum_without_frequencies = np.mean(all_power_spectra, axis = 1)
        # mean_power_spectrum = np.vstack((frequency_values, mean_power_spectrum_without_frequencies)).transpose()
        # power_integral = np.trapz(mean_power_spectrum[:,1], mean_power_spectrum[:,0])
        # mean_power_spectrum[:,1]/=power_integral
        # plt.plot(mean_power_spectrum[:,0], mean_power_spectrum[:,1], lw = 1)
        plt.xlim(0.004,0.01)
        plt.legend()
        plt.xlabel('Frequency [1/min]')
        plt.ylabel('Power [min]')
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                        'output','power_spectrum_validation_faster_lambda.pdf'))


    def xest_plot_example_parameter_sets_wo_switching(self):
        for parameter_set_name in parameter_set_dict:
            this_trace = hes5.generate_langevin_trajectory(**parameter_set_dict[parameter_set_name])
            these_summary_statistics = hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter_points_dict[parameter_set_name])
            plt.figure(figsize = (4.5,2.5))
            plt.plot(this_trace[:,0], this_trace[:,1])
            plt.plot(this_trace[:,0], this_trace[:,2])
            plt.xlabel('Time')
            plt.ylabel('Copy number')
            plt.title('Coherence: ' + "{:.2f}".format(these_summary_statistics[3]) + 
                       ', Mean: ' + "{:.2f}".format(these_summary_statistics[0]) +
                       ', Period: ' + "{:.2f}".format(these_summary_statistics[2]))
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__),
                        'output','switching_example_max_lik_parameters_'+parameter_set_name +'.pdf'))

    def test_lambda_dependance(self):
        number_of_traces = 1
        repetition_factor = 20
        
        switching_rates = [0.01,0.1,1,10,50]
        simulation_methods = dict()
        simulation_methods['full'] = switching.generate_multiple_switching_ode_trajectories
        simulation_methods['langevin'] = functools.partial(
            switching.generate_multiple_switching_langevin_trajectories,
            model = 'switching_only')
        simulation_methods['lna'] = functools.partial(
            switching.generate_multiple_switching_langevin_trajectories,
            model = 'switching_only_lna')


        for parameter_set_name in parameter_set_dict.keys():
            simulation_results = dict()
            for simulation_method in simulation_methods:
                simulation_results[simulation_method] = np.zeros(len(switching_rates))
 
            for rate_index, switching_rate in enumerate(switching_rates):
                for simulation_method, simulation_function in simulation_methods.items():
                    _, these_simulated_traces = simulation_function(number_of_trajectories = number_of_traces,
                                                                 duration = 1500*repetition_factor,
                                                                 **parameter_set_dict[parameter_set_name])
                    this_mean = np.mean(these_simulated_traces[:,1:])
                    this_std = np.std(these_simulated_traces[:,1:])
                    simulation_results[simulation_method][rate_index] = this_std
      
#             with open(os.path.join(os.path.dirname(__file__),
#                                             'output','lambda_depdendance.pickle'), 'wb') as f:
#                 Pickle the 'data' dictionary using the highest protocol available.
#                 pickle.dump(simulation_results, f, pickle.HIGHEST_PROTOCOL)

#             simulation_results = pickle.load(open(os.path.join(os.path.dirname(__file__),'output','lambda_depdendance.pickle'), 'rb'))

            plt.figure()
            for simulation_method, simulation_result in simulation_results.items():
                plt.semilogx(switching_rates,simulation_result,label = simulation_method)
            plt.xlabel('switching rate [1/min]')
            plt.ylabel('Protein std')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__),
                                            'output','lambda_depdendance_new_'+
                                            parameter_set_name + '.pdf'))
