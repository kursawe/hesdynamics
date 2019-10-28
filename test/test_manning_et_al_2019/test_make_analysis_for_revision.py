import unittest
import os
os.environ["OMP_NUM_THREADS"] = "1"
import os.path
import sys
import matplotlib as mpl
import matplotlib.gridspec 
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
sys.path.append(os.path.join(os.path.dirname(__file__),'..','..','src'))
import hes5

class TestMakeAnalysisForRevision(unittest.TestCase):
                                 
    def xest_make_time_dependent_repression_threshold(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
 
        # pick out a medium coherence value
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    np.logical_and(model_results[:,3]>0.2, #coherence
                                    np.logical_and(model_results[:,3]<0.3, #coherence
                                                   model_results[:,1]>0.05)))))  #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]
        
        example_parameter_index = 4
        example_parameter = my_posterior_samples[example_parameter_index]
        
        this_transcription_rate = example_parameter[0]
        this_translation_rate = example_parameter[1]
        this_repression_threshold = example_parameter[2]
        this_time_delay = example_parameter[3]
        this_hill_coefficient = example_parameter[4]
        
        this_duration = 1250.0
        these_times = np.arange(0,1250.0,1.0)
        these_repression_thresholds = np.zeros_like(these_times)
        for time_index, time in enumerate(these_times):
            if time<600:
                these_repression_thresholds[time_index] = this_repression_threshold
            else:
                these_repression_thresholds[time_index] = this_repression_threshold*(1-(time - 600)/1200.0)
        
        these_transcription_rates = np.array([this_transcription_rate]*len(these_times))
        these_translation_rates = np.array([this_translation_rate]*len(these_times))
        these_time_delays = np.array([this_time_delay]*len(these_times))
        these_hill_coefficients = np.array([this_hill_coefficient]*len(these_times))
        these_protein_degradations = np.array([np.log(2)/90]*len(these_times))
        these_mRNA_degradations = np.array([np.log(2)/30]*len(these_times))
        
        this_trajectory = hes5.generate_time_dependent_langevin_trajectory( duration = this_duration, 
                                  repression_threshold = these_repression_thresholds,
                                  hill_coefficient = these_hill_coefficients,
                                  mRNA_degradation_rate = these_mRNA_degradations,
                                  protein_degradation_rate = these_protein_degradations, 
                                  basal_transcription_rate = these_transcription_rates,
                                  translation_rate = these_translation_rates,
                                  transcription_delay = these_time_delays,
                                  initial_mRNA = 0,
                                  initial_protein = 0,
                                  equilibration_time = 2000.0)

        my_figure = plt.figure(figsize= (4.5,2.9))
        plt.plot(this_trajectory[:,0], this_trajectory[:,2])
        plt.xlabel('Time')
        plt.ylabel('Protein expression')
        plt.tight_layout()
        file_name = os.path.join(os.path.dirname(__file__), 'output',
                                   'trace_with_decreasing_repression_threshold.pdf')
        plt.savefig(file_name)

    def test_make_time_dependent_transcription_rate(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')
 
        # pick out a medium coherence value
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    np.logical_and(model_results[:,3]>0.2, #coherence
                                    np.logical_and(model_results[:,3]<0.3, #coherence
                                                   model_results[:,1]>0.05)))))  #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]
        
        example_parameter_index = 4
        example_parameter = my_posterior_samples[example_parameter_index]
        
        this_transcription_rate = example_parameter[0]
        this_translation_rate = example_parameter[1]
        this_repression_threshold = example_parameter[2]
        this_time_delay = example_parameter[3]
        this_hill_coefficient = example_parameter[4]
        
        this_duration = 1250.0
        these_times = np.arange(0,1250.0,1.0)
        these_transcription_rates = np.zeros_like(these_times)
        for time_index, time in enumerate(these_times):
            if time<300:
                these_transcription_rates[time_index] = this_transcription_rate
            else:
                these_transcription_rates[time_index] = this_transcription_rate*(1-(time - 300)/800.0)
        
        these_repression_thresholds = np.array([this_repression_threshold]*len(these_times))
        these_translation_rates = np.array([this_translation_rate]*len(these_times))
        these_time_delays = np.array([this_time_delay]*len(these_times))
        these_hill_coefficients = np.array([this_hill_coefficient]*len(these_times))
        these_protein_degradations = np.array([np.log(2)/90]*len(these_times))
        these_mRNA_degradations = np.array([np.log(2)/30]*len(these_times))
        
        this_trajectory = hes5.generate_time_dependent_langevin_trajectory( duration = this_duration, 
                                  repression_threshold = these_repression_thresholds,
                                  hill_coefficient = these_hill_coefficients,
                                  mRNA_degradation_rate = these_mRNA_degradations,
                                  protein_degradation_rate = these_protein_degradations, 
                                  basal_transcription_rate = these_transcription_rates,
                                  translation_rate = these_translation_rates,
                                  transcription_delay = these_time_delays,
                                  initial_mRNA = 0,
                                  initial_protein = 0,
                                  equilibration_time = 2000.0)

        my_figure = plt.figure(figsize= (4.5,2.9))
        plt.plot(this_trajectory[:,0], this_trajectory[:,2])
        plt.xlabel('Time')
        plt.ylabel('Protein expression')
        plt.tight_layout()
        file_name = os.path.join(os.path.dirname(__file__), 'output',
                                   'trace_with_decreasing_transcription_rate.pdf')
        plt.savefig(file_name)

        