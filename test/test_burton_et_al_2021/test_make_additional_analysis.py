import unittest
import os.path
import sys
import argparse
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import statsmodels.api as sm
# mpl.use('Agg')
mpl.rcParams['mathtext.default'] = 'regular'
# mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
font = {'size'   : 20}
plt.rc('font', **font)
import numpy as np
import multiprocessing as mp
import multiprocessing.pool as mp_pool
from jitcdde import jitcdde,y,t
import time
from scipy.spatial.distance import euclidean
from scipy import stats
import pymc3 as pm
import arviz as az
# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5
import hes_inference

number_of_cpus = mp.cpu_count()
font_size = 25
cm_to_inches = 0.3937008
class TestInference(unittest.TestCase):

    def xest_mala_analysis(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output/figure_5','')
        chain_path_strings = [i for i in os.listdir(loading_path) if 'ps6_1_cells_8_minutes_3' in i
                                                                  if '.npy' in i]
        for chain_path_string in chain_path_strings:
            mala = np.load(loading_path + chain_path_string)
            mala = mala[[0,1,2,4,5,6,7],:,:]
            # mala[:,:,[2,3]] = np.exp(mala[:,:,[2,3]])
            chains = az.convert_to_dataset(mala)
            print('\n' + chain_path_string + '\n')
            print('\nrhat:\n',az.rhat(chains))
            print('\ness:\n',az.ess(chains))
            az.plot_trace(chains); plt.savefig(loading_path + 'traceplot_' + chain_path_string[:-4] + '.png'); plt.close()
            az.plot_posterior(chains); plt.savefig(loading_path + 'posterior_' + chain_path_string[:-4] + '.png'); plt.close()
            az.plot_pair(chains,kind='kde'); plt.savefig(loading_path + 'pairplot_' + chain_path_string[:-4] + '.png'); plt.close()
            # np.save(loading_path + chain_path_string,mala)

    def xest_relationship_between_steady_state_mean_and_variance(self):
        model_parameters = [10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0]
        mean = hes5.calculate_steady_state_of_ode(repression_threshold=model_parameters[0],
                                                  hill_coefficient=model_parameters[1],
                                                  mRNA_degradation_rate=model_parameters[2],
                                                  protein_degradation_rate=model_parameters[3],
                                                  basal_transcription_rate=model_parameters[4],
                                                  translation_rate=model_parameters[5])

        LNA_mRNA_variance = np.power(hes5.calculate_approximate_mRNA_standard_deviation_at_parameter_point(repression_threshold=model_parameters[0],
                                                                                                           hill_coefficient=model_parameters[1],
                                                                                                           mRNA_degradation_rate=model_parameters[2],
                                                                                                           protein_degradation_rate=model_parameters[3],
                                                                                                           basal_transcription_rate=model_parameters[4],
                                                                                                           translation_rate=model_parameters[5],
                                                                                                           transcription_delay=model_parameters[6]),2)

        LNA_protein_variance = np.power(hes5.calculate_approximate_protein_standard_deviation_at_parameter_point(repression_threshold=model_parameters[0],
                                                                                                                 hill_coefficient=model_parameters[1],
                                                                                                                 mRNA_degradation_rate=model_parameters[2],
                                                                                                                 protein_degradation_rate=model_parameters[3],
                                                                                                                 basal_transcription_rate=model_parameters[4],
                                                                                                                 translation_rate=model_parameters[5],
                                                                                                                 transcription_delay=model_parameters[6]),2)

        print('mean =',mean)
        print('mRNA_variance/mRNA_mean =',LNA_mRNA_variance/mean[0])
        print('protein_variance/protein_mean =',LNA_protein_variance/mean[1])

    def xest_mala_multiple_experimental_traces(self,experiment_date='280317p1',cluster='1'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/selected_data_for_mala/')
        data_filename = experiment_date + '_cluster_' + cluster + '.npy'
        protein_at_observations = np.array([np.load(os.path.join(saving_path,i)) for i in os.listdir(saving_path) if 'detrended' in i
                                                                               if 'cluster_'+cluster in i
                                                                               if experiment_date in i])

        measurement_variance = np.power(np.round(np.load(saving_path + experiment_date + "_measurement_variance_detrended.npy"),0),2)

        # define known parameters
        all_parameters = {'repression_threshold' : [0,None],
                          'hill_coefficient' : [1,None],
                          'mRNA_degradation_rate' : [2,np.log(np.log(2)/30)],
                          'protein_degradation_rate' : [3,np.log(np.log(2)/90)],
                          'basal_transcription_rate' : [4,None],
                          'translation_rate' : [5,None],
                          'transcription_delay' : [6,None]}

        known_parameters = {k:all_parameters[k] for k in ('mRNA_degradation_rate',
                                                          'protein_degradation_rate') if k in all_parameters}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(all_parameters)) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        number_of_samples = 80000
        number_of_chains = 8
        step_size = 0.001

        run_mala_for_dataset(data_filename,
                             protein_at_observations,
                             measurement_variance,
                             number_of_parameters,
                             known_parameters,
                             step_size,
                             number_of_chains,
                             number_of_samples)
