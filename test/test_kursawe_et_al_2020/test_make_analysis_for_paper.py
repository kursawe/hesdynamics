import os.path
import os
os.environ["OMP_NUM_THREADS"] = "1"
import unittest
import sys
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
import matplotlib.gridspec 
from mpl_toolkits.axes_grid1 import make_axes_locatable
font = {'size'   : 10}
plt.rc('font', **font)
import numpy as np
import scipy.optimize
import pandas as pd
import seaborn as sns
import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)
try:
    import gpflow
except ImportError:
    print('Could not import gpflow. This may affect GP regression tests.')
from numba import jit, autojit

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..','src'))
import hes5

import socket
import multiprocessing as mp
domain_name = socket.getfqdn()
if domain_name == 'jochen-ThinkPad-S1-Yoga-12':
    number_of_available_cores = 2
else:
#     number_of_available_cores = 1
    number_of_available_cores = mp.cpu_count()

class TestMakeAnalysisForPaper(unittest.TestCase):

    def xest_make_abc_samples(self):
        print('starting zebrafish abc')
        ## generate posterior samples
        total_number_of_samples = 2000000
#         total_number_of_samples = 5
#         total_number_of_samples = 100

#         total_number_of_samples = 10

        prior_bounds = {'basal_transcription_rate' : (1.0,120),
                        'translation_rate' : (0.1,40),
                        'repression_threshold' : (0,4000),
                        'time_delay' : (1,12),
                        'hill_coefficient' : (2,6),
                        'protein_degradation_rate' : ( np.log(2)/11.0, np.log(2)/11.0 ),
                        'mRNA_half_life' : ( 1, 11) }

        my_prior_samples, my_prior_results = hes5.generate_lookup_tables_for_abc( total_number_of_samples,
                                                                                  number_of_traces_per_sample = 2000,
                                                                                  saving_name = 'sampling_results_zebrafish_delay_large_extra',
                                                                                  prior_bounds = prior_bounds,
                                                                                  prior_dimension = 'full',
                                                                                  logarithmic = True,
                                                                                  power_spectrum_smoothing_window = 0.02 )
        
        self.assertEquals(my_prior_samples.shape, 
                          (total_number_of_samples, 7))

    def xest_perform_abc_with_extrinsic_noise(self):
        print('starting zebrafish abc')
        ## generate posterior samples
        total_number_of_samples = 2000000
#         total_number_of_samples = 5
#         total_number_of_samples = 100

#         total_number_of_samples = 10

        prior_bounds = {'basal_transcription_rate' : (1.0,120),
                        'translation_rate' : (0.1,40),
                        'repression_threshold' : (0,4000),
                        'time_delay' : (1,12),
                        'hill_coefficient' : (2,6),
                        'protein_degradation_rate' : ( np.log(2)/11.0, np.log(2)/11.0 ),
                        'mRNA_half_life' : ( 1, 11),
                        'extrinsic_noise_rate' : (0.1,7000) }

        my_prior_samples, my_prior_results = hes5.generate_lookup_tables_for_abc( total_number_of_samples,
                                                                    number_of_traces_per_sample = 2000,
                                                                    saving_name = 'sampling_results_zebrafish_extrinsic_noise_delay_large_extra',
                                                                    prior_bounds = prior_bounds,
                                                                    prior_dimension = 'extrinsic_noise',
                                                                    logarithmic = True,
                                                                    power_spectrum_smoothing_window = 0.02 )
 
    def xest_d_make_dual_parameter_variation(self, 
                                             quadrant_index = 'all',
                                             model = 'standard'):

        #This test is splitting up the calculations for the dual parameter variation into chunks that can be sent to individual nodes on a 
        # HPC cluster. There are a couple of options for parameter ranges that are not used in the final version of the paper, the important
        # ranges that we used to make the paper figures are calculated by evaluating this function on all quadrant indices between 100 and 139.
        # The relevant model options are 'standard_extra' for the model without extrinsic noise, and 'extrinsic_noise_extra' for the model with extrinsic noise.
        number_of_trajectories = 2000

        degradation_ranges = dict()
        degradation_ranges[1] = (0.6, 1.0)
        degradation_ranges[2] = (0.6, 1.0)
        degradation_ranges[3] = (0.1, 0.5)
        degradation_ranges[4] = (0.1, 0.5)
        degradation_ranges[5] = (0.6, 1.0)
        degradation_ranges[6] = (0.1, 0.5)
        degradation_ranges[7] = (0.1, 0.5)
        degradation_ranges[8] = (0.6, 1.0)
        degradation_ranges[9] = (1.1, 1.5)
        degradation_ranges[10] = (1.1, 1.5)
        degradation_ranges[11] = (1.1, 1.5)
        degradation_ranges[12] = (1.1, 1.5)
        degradation_ranges[13] = (1.6, 2.0)
        degradation_ranges[14] = (1.6, 2.0)
        degradation_ranges[15] = (1.6, 2.0)
        degradation_ranges[16] = (1.6, 2.0)
        degradation_ranges['all'] = (0.1, 2.0)
        degradation_ranges['shifted'] = (0.1, 2.0)
        degradation_ranges['shifted_more'] = (0.1, 2.0)
        degradation_ranges['shifted_final'] = (0.3, 1.0)

        degradation_interval_numbers = { i: 5 for i in range(1,17)}
        degradation_interval_numbers['all'] = 20
        degradation_interval_numbers['shifted'] = 20
        degradation_interval_numbers['shifted_more'] = 20
        degradation_interval_numbers['shifted_final'] = 8
        
        translation_ranges = dict()
        translation_ranges[1] = (1.0, 1.5)
        translation_ranges[2] = (1.6, 2.0)
        translation_ranges[3] = (1.0, 1.5)
        translation_ranges[4] = (1.6, 2.0)
        translation_ranges[5] = (0.5, 0.9)
        translation_ranges[6] = (0.5, 0.9)
        translation_ranges[7] = (0.1, 0.4)
        translation_ranges[8] = (0.1, 0.4)
        translation_ranges[9] = (1.0, 1.5)
        translation_ranges[10] = (1.6, 2.0)
        translation_ranges[11] = (0.5, 0.9)
        translation_ranges[12] = (0.1, 0.4)
        translation_ranges[13] = (1.0, 1.5)
        translation_ranges[14] = (0.5, 0.9)
        translation_ranges[15] = (1.6, 2.0)
        translation_ranges[16] = (0.1, 0.4)
        translation_ranges['all'] = (0.1, 2.0)
        translation_ranges['shifted'] = (0.9, 3.1)
        translation_ranges['shifted_more'] = (3.2, 4.1)
        translation_ranges['shifted_final'] = (2.5, 4.5)

        translation_interval_numbers = dict()
        translation_interval_numbers[1] = 6
        translation_interval_numbers[2] = 5
        translation_interval_numbers[3] = 6
        translation_interval_numbers[4] = 5
        translation_interval_numbers[5] = 5
        translation_interval_numbers[6] = 5
        translation_interval_numbers[7] = 4
        translation_interval_numbers[8] = 4
        translation_interval_numbers[9] = 6
        translation_interval_numbers[10] = 5
        translation_interval_numbers[11] = 5
        translation_interval_numbers[12] = 4
        translation_interval_numbers[13] = 6
        translation_interval_numbers[14] = 5
        translation_interval_numbers[15] = 5
        translation_interval_numbers[16] = 4
        translation_interval_numbers['all'] = 20
        translation_interval_numbers['shifted'] = 23
        translation_interval_numbers['shifted_more'] = 10
        translation_interval_numbers['shifted_final'] = 21

#         additional_index = 17
        additional_index = 7
        for degradation_change_start in [0.7,0.3]:
#             for translation_change_start in np.linspace(7.5,3.0,16):
            for translation_change_start in np.linspace(10.5,3.0,26):
                degradation_ranges[additional_index] = (degradation_change_start, 
                                                        degradation_change_start + 0.3)
                translation_ranges[additional_index] = (translation_change_start, 
                                                        translation_change_start + 0.2)
                degradation_interval_numbers[additional_index] = 4
                translation_interval_numbers[additional_index] = 3
                additional_index += 1
            
        additional_index = 100
        for translation_change_start in np.linspace(2.0,13.4,39):
            degradation_ranges[additional_index] = (0.75,1.0)
            translation_ranges[additional_index] = (translation_change_start, 
                                                    translation_change_start)
            degradation_interval_numbers[additional_index] = 6
            translation_interval_numbers[additional_index] = 1
            additional_index += 1
 
        additional_index = 200
        for translation_change_start in np.linspace(2.0,4.1,8):
            degradation_ranges[additional_index] = (0.5,0.7)
            translation_ranges[additional_index] = (translation_change_start, 
                                                    translation_change_start)
            degradation_interval_numbers[additional_index] = 5
            translation_interval_numbers[additional_index] = 1
            additional_index += 1
 
        print(additional_index)
        print(translation_ranges)
        print(degradation_ranges)
#         number_of_parameter_points = 2
#         number_of_trajectories = 2

        if model == 'standard':
            saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_delay')
#             saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish')
        if model == 'standard_large':
            saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_delay_large')
        if model == 'standard_extra':
            saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_zebrafish_delay_large_extra')
        elif model == 'extrinsic_noise':
            saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise_delay')
#             saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise')
        elif model == 'extrinsic_noise_large':
            saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_extrinsic_noise_delay_large')
        elif model == 'extrinsic_noise_extra':
            saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_zebrafish_extrinsic_noise_delay_large_extra')
        elif model == 'transcription_amplification':
            saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_zebrafish_transcription_amplification')
            
#         saving_path = os.path.join(os.path.dirname(__file__), 'output','sampling_results_all_parameters')
#         saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                    np.logical_and(model_results[:,3]>0.1,
                                                   model_results[:,2]<150))))))
       
        my_posterior_samples = prior_samples[accepted_indices]
        print('number of accepted samples is')
        print(len(my_posterior_samples))

        my_parameter_sweep_results = hes5.conduct_dual_parameter_sweep_at_parameters(my_posterior_samples,
                                                                                     degradation_range = degradation_ranges[quadrant_index],
                                                                                     translation_range = translation_ranges[quadrant_index],
                                                                                     degradation_interval_number = degradation_interval_numbers[quadrant_index],
                                                                                     translation_interval_number = translation_interval_numbers[quadrant_index],
                                                                                     number_of_traces_per_parameter = number_of_trajectories)
        
#         self.assertEqual(my_parameter_sweep_results.shape, (len(my_posterior_samples),
#                                                             number_of_parameter_points,
#                                                             number_of_parameter_points,
#                                                             13))
        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_' + model 
                             + '_' + str(quadrant_index) +'.npy'),
                    my_parameter_sweep_results)

    def xest_reconstruct_further_dual_parameter_variation_matrix(self): 
        #This test will stick all the calculated chunks together in the right order
#         model = 'standard'
        model = 'extrinsic_noise_extra'
#         model = 'standard_extra'
#         model = 'transcription_amplification'
        saving_path_root = os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_' + model + '_')
        all_sub_matrices = []
        for quadrant_index in range(100,137):
            try:
                this_saving_path = saving_path_root + str(quadrant_index) + '.npy'
                all_sub_matrices.append(np.load(this_saving_path))
            except FileNotFoundError:
                all_sub_matrices.append(np.zeros_like(all_sub_matrices[0]))
                
            
        this_full_matrix = np.zeros((len(all_sub_matrices[0]),6,25,14))
        for parameter_index in range(len(all_sub_matrices[0])):
            this_upper_matrix = all_sub_matrices[0][parameter_index]
            for submatrix_index in range(1,25):
                this_upper_matrix = np.hstack((this_upper_matrix,all_sub_matrices[submatrix_index][parameter_index]))
            this_full_matrix[parameter_index] = this_upper_matrix
            
        np.save(os.path.join(os.path.dirname(__file__), 'output','zebrafish_dual_sweeps_' + model + '_complete_matrix.npy'),
                    this_full_matrix)

    def xest_generate_results_without_noise(self):
        model = 'extrinsic_noise_extra'
        saving_path = os.path.join(os.path.dirname(__file__), 'data','sampling_results_zebrafish_extrinsic_noise_delay_large_extra')
        dual_sweep_results = np.load(os.path.join(os.path.dirname(__file__),'data','zebrafish_dual_sweeps_extrinsic_noise_extra_complete_matrix.npy'))
        relevant_indices = np.load(os.path.join(os.path.dirname(__file__), 'data','zebrafish_dual_sweeps_indices_' + model + '.npy'))
        corresponding_proportions = np.load(os.path.join(os.path.dirname(__file__), 'data','zebrafish_dual_sweeps_change_proportions_' + model + '.npy'))

        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>1000, #protein number
                                    np.logical_and(model_results[:,0]<2500,
                                    np.logical_and(model_results[:,1]<0.15,
                                    np.logical_and(model_results[:,1]>0.05,
                                    np.logical_and(model_results[:,3]>0.1,
                                                   model_results[:,2]<150))))))
#                                     np.logical_and(model_results[:,3]>0.1,
#                                                     model_results[:,2]<150))))))

        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_results = model_results[accepted_indices]
        
#         relevant_indices = relevant_indices[:100]
#         corresponding_proportions = corresponding_proportions[:100]
        print('number of accepted samples is')
        print(len(my_posterior_samples))
        
        my_selected_posterior_samples = my_posterior_samples[relevant_indices]
        my_selected_posterior_samples_after = np.copy(my_selected_posterior_samples)
        my_selected_posterior_samples_after[:,1]*=corresponding_proportions[:,1]
        my_selected_posterior_samples_after[:,5]*=corresponding_proportions[:,0]
        
        np.save(os.path.join(os.path.dirname(__file__),'output','zebrafish_noise_comparison_real_parameters_before.npy'),
                my_selected_posterior_samples)

        np.save(os.path.join(os.path.dirname(__file__),'output','zebrafish_noise_comparison_real_parameters_after.npy'),
                my_selected_posterior_samples_after)

        print('number of selected samples is')
        print(len(my_selected_posterior_samples))

        my_no_noise_samples_before = np.copy(my_selected_posterior_samples)
        my_no_noise_samples_after = np.copy(my_selected_posterior_samples_after)
        my_no_noise_samples_before[:,7] = 0.0
        my_no_noise_samples_after[:,7] = 0.0

        my_selected_results_before = my_posterior_results[relevant_indices]

        my_selected_results_after = hes5.calculate_summary_statistics_at_parameters(my_selected_posterior_samples_after,
                                                                                    number_of_traces_per_sample = 2000,
                                                                                    power_spectrum_smoothing_window = 0.02)
        my_no_noise_results_before = hes5.calculate_summary_statistics_at_parameters(my_no_noise_samples_before,
                                                                                     number_of_traces_per_sample = 2000,
                                                                                     power_spectrum_smoothing_window = 0.02)
        my_no_noise_results_after = hes5.calculate_summary_statistics_at_parameters(my_no_noise_samples_after,
                                                                                    number_of_traces_per_sample = 2000,
                                                                                    power_spectrum_smoothing_window = 0.02)
        
        np.save(os.path.join(os.path.dirname(__file__),'output','zebrafish_noise_comparison_no_noise_before.npy'),
                my_no_noise_results_before)

        np.save(os.path.join(os.path.dirname(__file__),'output','zebrafish_noise_comparison_no_noise_after.npy'),
                my_no_noise_results_after)
        
        np.save(os.path.join(os.path.dirname(__file__),'output','zebrafish_noise_comparison_actual_before.npy'),
                my_selected_results_before)
        
        np.save(os.path.join(os.path.dirname(__file__),'output','zebrafish_noise_comparison_actual_after.npy'),
                my_selected_results_after)
 