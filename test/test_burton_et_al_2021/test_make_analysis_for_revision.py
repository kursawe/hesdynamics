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

    def xest_runtime_mh_vs_mala(self):
        saving_path                          = os.path.join(os.path.dirname(__file__), 'data','kalman_test_trace')
        fixed_langevin_trace                 = np.load(saving_path + '_true_data.npy')
        fixed_protein_observations           = np.load(saving_path + '_observations.npy')
        # true_kalman_prediction_mean          = np.load(saving_path + '_prediction_mean.npy')
        # true_kalman_prediction_variance      = np.load(saving_path + '_prediction_variance.npy')
        # true_kalman_prediction_distributions = np.load(saving_path + '_prediction_distributions.npy')
        # true_kalman_negative_log_likelihood_derivative = np.load(saving_path + '_negative_log_likelihood_derivative.npy')

        # run the current kalman filter using the same parameters and observations, then compare
        parameters = np.array([10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0])

        # first run for JIT compilation
        _ = hes_inference.calculate_log_likelihood_at_parameter_point(parameters,
                                                                      np.array([fixed_protein_observations]),
                                                                      measurement_variance=10000)
        from time import time
        start_time_mh = time()
        for _ in range(5000):
            _ = hes_inference.calculate_log_likelihood_at_parameter_point(parameters,
                                                                          np.array([fixed_protein_observations]),
                                                                          measurement_variance=10000)
        mh_time = time()-start_time_mh

        # first run for JIT compilation
        _, _ = hes_inference.calculate_log_likelihood_and_derivative_at_parameter_point(fixed_protein_observations,
                                                                                        parameters,
                                                                                        5000,
                                                                                        measurement_variance=10000)

        # MALA
        start_time_mala = time()
        for _ in range(5000):
            _, _ = hes_inference.calculate_log_likelihood_and_derivative_at_parameter_point(fixed_protein_observations,
                                                                                            parameters,
                                                                                            5000,
                                                                                            measurement_variance=10000)
        mala_time = time()-start_time_mala
        print("MH time: ",mh_time," seconds")
        print("MALA time: ",mala_time," seconds")
        print("Factor: ",mala_time/mh_time)


    def qest_coherence_vs_sampling_frequency(self):
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        sampling_frequencies = [1,5,8,12,15,30,50]
        parameter_set_strings = ['ps6','ps7','ps8','ps9','ps10','ps11','ps12']
        parameter_sets = np.zeros((len(parameter_set_strings),5))
        for index, parameter_set_string in enumerate(parameter_set_strings):
            parameter_sets[index,:] = np.load(saving_path + parameter_set_string + '_parameter_values.npy')[[4,5,0,6,1]]

        coherence_values = np.zeros((len(parameter_set_strings),len(sampling_frequencies)))

        for parameter_set_index, parameter_set in enumerate(parameter_sets):
            for sampling_index, frequency in enumerate(sampling_frequencies):
                summary_statistics = hes5.calculate_langevin_summary_statistics_at_parameter_point(parameter_set,
                                                                                                   number_of_traces = 200,
                                                                                                   simulation_duration = 5*1500,
                                                                                                   power_spectrum_smoothing_window = 0.001,
                                                                                                   sampling_frequency = frequency)
                coherence_values[parameter_set_index,sampling_index] = summary_statistics[3]

        fig, ax = plt.subplots(figsize=(10,5))
        # ax.set_ylim(0,.05)
        for i in range(len(parameter_set_strings)):
            ax.plot(sampling_frequencies,coherence_values[i],color = '#20948B')
        ax.set_xlabel('Sampling interval (mins)')
        ax.set_ylabel('Coherence')

        plt.tight_layout()
        plt.savefig(loading_path + 'sampling_frequency_vs_coherence.pdf')

    def qest_state_space_mean_mRNA_uncertainty_plots(self):
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','')

        protein_observations = np.load(os.path.join(saving_path,'figure_5/protein_observations_ps9_1_cells_12_minutes_2.npy'))
        true_parameters = np.load(os.path.join(saving_path,'ps9_parameter_values.npy'))[:7]
        chain = np.load(os.path.join(loading_path,'final_parallel_mala_output_protein_observations_ps9_1_cells_12_minutes_2.npy'))
        mRNA_chain = np.load(os.path.join(loading_path,'final_parallel_mala_output_protein_observations_with_mRNA_ps9_1_cells_12_minutes_2.npy'))

        # reshape
        chain = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])
        mRNA_chain = mRNA_chain.reshape(mRNA_chain.shape[0]*mRNA_chain.shape[1],mRNA_chain.shape[2])
        # chain[:,[2,3]] = np.exp(chain[:,[2,3]])

        posterior_mean = np.zeros(7)
        posterior_mean[[2,3]] = [np.log(2)/30,np.log(2)/90]
        posterior_mean[[0,1,4,5,6]] = np.mean(chain,axis=0)
        posterior_mean[[4,5]] = np.exp(posterior_mean[[4,5]])

        posterior_mode = np.zeros(7)
        posterior_mode[[2,3]] = [np.log(2)/30,np.log(2)/90]
        counter = 0
        for index,value in enumerate([0,1,4,5,6]):
            heights, bins, _ = plt.hist(chain[:,index],bins=45)
            posterior_mode[value] = bins[heights.argmax()]
        posterior_mode[[4,5]] = np.exp(posterior_mode[[4,5]])

        # mRNA_mode = np.zeros(7)
        # mRNA_mode[[2,3]] = [np.log(2)/30,np.log(2)/90]
        # counter = 0
        # for index,value in enumerate([0,1,4,5,6]):
        #     heights, bins, _ = plt.hist(mRNA_chain[:,index],bins=45)
        #     mRNA_mode[value] = bins[heights.argmax()]
        # mRNA_mode[[4,5]] = np.exp(mRNA_mode[[4,5]])

        true_state_space_mean, true_state_space_variance, _,_,_,_,_ = hes_inference.kalman_filter(protein_observations,
                                                                                                  true_parameters,
                                                                                                  measurement_variance=1000000)

        mode_state_space_mean, mode_state_space_variance, _,_,_,_,_ = hes_inference.kalman_filter(protein_observations,
                                                                                                  posterior_mode,
                                                                                                  measurement_variance=1000000)

        # mRNA_state_space_mean, mRNA_state_space_variance, _,_,_,_,_ = hes_inference.kalman_filter(protein_observations,
        #                                                                                           mRNA_mode,
        #                                                                                           measurement_variance=1000000)

        # ground truth error
        number_of_states = true_state_space_mean.shape[0]
        true_mRNA_covariance_matrix = true_state_space_variance[:number_of_states,:number_of_states]
        true_mRNA_variance = np.diagonal(true_mRNA_covariance_matrix)
        true_mRNA_error = np.sqrt(true_mRNA_variance)*2
        # remove negatives from error bar calc
        true_mRNA_error = np.zeros((2,len(true_mRNA_variance)))
        true_mRNA_error[0,:] = np.minimum(true_state_space_mean[:,1],np.sqrt(true_mRNA_variance)*2)
        true_mRNA_error[1,:] = np.sqrt(true_mRNA_variance)*2

        # without mRNA error
        number_of_states = mode_state_space_mean.shape[0]
        mode_mRNA_covariance_matrix = mode_state_space_variance[:number_of_states,:number_of_states]
        mode_mRNA_variance = np.diagonal(mode_mRNA_covariance_matrix)
        # remove negatives from error bar calc
        mode_mRNA_error = np.zeros((2,len(mode_mRNA_variance)))
        mode_mRNA_error[0,:] = np.minimum(mode_state_space_mean[:,1],np.sqrt(mode_mRNA_variance)*2)
        mode_mRNA_error[1,:] = np.sqrt(mode_mRNA_variance)*2

        # # with mRNA error
        # number_of_states = mRNA_state_space_mean.shape[0]
        # mRNA_mRNA_covariance_matrix = mRNA_state_space_variance[:number_of_states,:number_of_states]
        # mRNA_mRNA_variance = np.diagonal(mRNA_mRNA_covariance_matrix)
        # # remove negatives from error bar calc
        # mRNA_mRNA_error = np.zeros((2,len(mRNA_mRNA_variance)))
        # mRNA_mRNA_error[0,:] = np.minimum(mRNA_state_space_mean[:,1],np.sqrt(mRNA_mRNA_variance)*2)
        # mRNA_mRNA_error[1,:] = np.sqrt(mRNA_mRNA_variance)*2

        fig, ax = plt.subplots(figsize=(10,5))
        # plt.subplot(2,1,1)
        # ground truth
        ax.scatter(true_state_space_mean[np.int(true_parameters[-1])::12,0],true_state_space_mean[np.int(true_parameters[-1])::12,1],s=6,label='inferred mRNA copy numbers at ground truth',color='#20948B',zorder=2)
        ax.errorbar(true_state_space_mean[:,0],true_state_space_mean[:,1],yerr=true_mRNA_error,ecolor='#98DBC6',alpha=0.25,zorder=1)
        # without mRNA posterior mode
        ax.scatter(mode_state_space_mean[np.int(posterior_mode[-1])::12,0],mode_state_space_mean[np.int(posterior_mode[-1])::12,1],s=6,label='inferred mRNA copy numbers (without mRNA data)',color='#F18D9E',zorder=2)
        ax.errorbar(mode_state_space_mean[:,0],mode_state_space_mean[:,1],yerr=mode_mRNA_error,ecolor='#f8c6ce',alpha=0.25,zorder=1)
        # with mRNA posterior mode
        # ax.scatter(mRNA_state_space_mean[np.int(mRNA_mode[-1])::12,0],mRNA_state_space_mean[np.int(mRNA_mode[-1])::12,1],s=6,label='posterior mode (with mRNA)',color='#F69454',zorder=2)
        # ax.errorbar(mRNA_state_space_mean[:,0],mRNA_state_space_mean[:,1],yerr=mRNA_mRNA_error,ecolor='#F9be98',alpha=0.25,zorder=1)
        ax.set_xlabel('Time')
        ax.set_ylabel('mRNA Copy Numbers')
        plt.legend(fontsize='x-small')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','state_space_uncertainty_posterior_mode.pdf'))

    def qest_state_space_mRNA_uncertainty(self):
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','figure_5/')

        protein_observations = np.load(os.path.join(saving_path,'figure_5/protein_observations_ps9_1_cells_12_minutes_2.npy'))
        true_parameters = np.load(os.path.join(saving_path,'ps9_parameter_values.npy'))[:7]
        chain = np.load(os.path.join(loading_path,'final_parallel_mala_output_protein_observations_ps9_1_cells_12_minutes_2.npy'))

        # reshape and sample
        chain = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])

        number_of_samples = 1000
        samples = np.zeros((number_of_samples,7))
        samples[:,[2,3]] = [np.log(2)/30,np.log(2)/90]
        import random
        samples[:,[0,1,4,5,6]] = chain[random.sample(range(chain.shape[0]),number_of_samples)]
        samples[:,[4,5]] = np.exp(samples[:,[4,5]])


        true_state_space_mean, true_state_space_variance, _,_,_,_,_ = hes_inference.kalman_filter(protein_observations,
                                                                                                  true_parameters,
                                                                                                  measurement_variance=1000000)
        # remove negative times
        zero_index = np.where(true_state_space_mean[:,0]==0)[0][0]
        # ground truth error
        true_state_space_mean = true_state_space_mean[zero_index:]
        number_of_states = true_state_space_mean.shape[0]
        true_mRNA_covariance_matrix = true_state_space_variance[zero_index:number_of_states+zero_index,
                                                                zero_index:number_of_states+zero_index]
        true_mRNA_variance = np.diagonal(true_mRNA_covariance_matrix)
        # remove negatives from error bar calc
        true_mRNA_error = np.zeros((2,len(true_mRNA_variance)))
        true_mRNA_error[0,:] = np.minimum(true_state_space_mean[:,1],np.sqrt(true_mRNA_variance)*2)
        true_mRNA_error[1,:] = np.sqrt(true_mRNA_variance)*2

        total_means = np.zeros((number_of_samples,733))
        total_variances = np.zeros((number_of_samples,733))
        for index, sample in enumerate(samples):
            # mean
            state_space_mean, state_space_variance, _,_,_,_,_ = hes_inference.kalman_filter(protein_observations,
                                                                                            sample,
                                                                                            measurement_variance=1000000)

            # remove negative times
            zero_index = np.where(state_space_mean[:,0]==0)[0][0]
            #mean
            total_means[index] = state_space_mean[zero_index:,1]
            # variance
            number_of_states = state_space_mean[zero_index:,1].shape[0]
            mRNA_covariance_matrix = state_space_variance[zero_index:number_of_states+zero_index,
                                                          zero_index:number_of_states+zero_index]
            total_variances[index] = np.diagonal(mRNA_covariance_matrix)
            # mRNA_std = np.sqrt(mRNA_variance) # *2 for 2 standard deviations?

        mean_of_total_means = np.mean(total_means,axis=0)
        variance_of_total_variances = np.mean(total_variances,axis=0) + np.var(total_means,axis=0)
        # remove negatives from error bar calc
        total_mRNA_error = np.zeros((2,len(mean_of_total_means)))
        total_mRNA_error[0,:] = np.minimum(mean_of_total_means,np.sqrt(variance_of_total_variances)*2)
        total_mRNA_error[1,:] = np.sqrt(variance_of_total_variances)*2

        fig, ax = plt.subplots(figsize=(10,5))
        # plt.subplot(2,1,1)
        # ground truth
        ax.scatter(np.arange(0,733,12),true_state_space_mean[::12,1],s=6,label='inferred mRNA copy numbers at ground truth',color='#20948B',zorder=2)
        ax.errorbar(np.arange(0,733,1),true_state_space_mean[:,1],yerr=true_mRNA_error,ecolor='#98DBC6',alpha=0.25,zorder=1)
        # without mRNA posterior mode



        ax.scatter(np.arange(0,733,12),mean_of_total_means[::12],s=6,label='inferred mRNA copy numbers (without mRNA data)',color='#F18D9E',zorder=2)
        ax.errorbar(np.arange(0,733,1),mean_of_total_means,yerr=total_mRNA_error,ecolor='#f8c6ce',alpha=0.25,zorder=1)
        # with mRNA posterior mode
        # ax.scatter(mRNA_state_space_mean[np.int(mRNA_mode[-1])::12,0],mRNA_state_space_mean[np.int(mRNA_mode[-1])::12,1],s=6,label='posterior mode (with mRNA)',color='#F69454',zorder=2)
        # ax.errorbar(mRNA_state_space_mean[:,0],mRNA_state_space_mean[:,1],yerr=mRNA_mRNA_error,ecolor='#F9be98',alpha=0.25,zorder=1)
        ax.set_xlabel('Time (mins)')
        ax.set_ylabel('mRNA Copy Numbers')
        plt.legend(fontsize='x-small')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','state_space_uncertainty_posterior_samples.pdf'))

    def xest_instantaneous_transcription_rate_with_and_without_mRNA(self):
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','figure_5/')
        mRNA_loading_path = os.path.join(os.path.dirname(__file__),'output','mRNA_output/')

        chain_path_strings = [i for i in os.listdir(loading_path) if '.npy' in i if 'ps6_1_cells_5_minutes_4' in i]
        # mRNA_chain_path_strings = [i for i in os.listdir(mRNA_loading_path) if '.npy' in i]

        number_of_samples = 25000
        import random
        def hill_function(protein,repression_threshold,hill_coefficient):
            return 1 / (1 + np.power(protein/repression_threshold,hill_coefficient))

        for string in chain_path_strings:
            ps_string = string[string.find('ions')+5:]
            protein_observations = np.load(os.path.join(saving_path,'figure_5/protein_observations_' + ps_string))
            mean_protein = np.mean(protein_observations[:,1])
            true_parameters = np.load(os.path.join(saving_path,ps_string[:ps_string.find('_')] + '_parameter_values.npy'))[:7]
            if os.path.exists(os.path.join(mRNA_loading_path,
                                           'final_parallel_mala_output_protein_observations_with_mRNA_' + ps_string)):
                mRNA_chain = np.load(os.path.join(mRNA_loading_path,'final_parallel_mala_output_protein_observations_with_mRNA_' + ps_string))
                mRNA_chain = mRNA_chain.reshape(mRNA_chain.shape[0]*mRNA_chain.shape[1],mRNA_chain.shape[2])
            else:
                continue
            chain = np.load(os.path.join(loading_path,'final_parallel_mala_output_protein_observations_' + ps_string))
            chain = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])

            samples = np.zeros((number_of_samples,7))
            samples[:,[2,3]] = [np.log(2)/30,np.log(2)/90]
            samples[:,[0,1,4,5,6]] = chain[random.sample(range(chain.shape[0]),number_of_samples)]
            samples[:,[4,5]] = np.exp(samples[:,[4,5]])

            mRNA_samples = np.zeros((number_of_samples,7))
            mRNA_samples[:,[2,3]] = [np.log(2)/30,np.log(2)/90]
            mRNA_samples[:,[0,1,4,5,6]] = mRNA_chain[random.sample(range(chain.shape[0]),number_of_samples)]
            mRNA_samples[:,[4,5]] = np.exp(mRNA_samples[:,[4,5]])

            instant_transcription_rate = [i[4]*hill_function(mean_protein,i[0],i[1]) for i in samples]
            mRNA_instant_transcription_rate = [i[4]*hill_function(mean_protein,i[0],i[1]) for i in mRNA_samples]

            fig, ax = plt.subplots(figsize=(0.7*7,0.7*5))
            heights, bins, _ = ax.hist(instant_transcription_rate,bins=60,density=True,label='Without mRNA data')
            # heights, bins, _ = ax.hist(mRNA_instant_transcription_rate,bins=60,density=True,alpha=0.6,label='With mRNA data')
            ax.vlines(true_parameters[4]*hill_function(mean_protein,true_parameters[0],true_parameters[1]),
                      0,1.1*max(heights),color='k',lw=2,label='True value')
            mean_transcription = np.mean(instant_transcription_rate)#bins[np.argmax(heights)]
            mean_transcription_mRNA = np.mean(mRNA_instant_transcription_rate)#bins_mRNA[np.argmax(heights_mRNA)]
            ground_truth = true_parameters[4]*hill_function(mean_protein,true_parameters[0],true_parameters[1])
            print("relative error w/o mRNA",np.abs(mean_transcription-ground_truth)/ground_truth)
            print("relative error w mRNA",np.abs(mean_transcription_mRNA-ground_truth)/ground_truth)
            print(az.hdi(np.array(mRNA_instant_transcription_rate),0.9),ground_truth)
            import pdb; pdb.set_trace()
            # plt.subplot(2,1,1)
            # ground truth
            ax.set_xlabel('$\\alpha_T$ [1/min]')
            ax.set_ylabel('Probability')
            ax.set_xlim(0,1)
            plt.legend(fontsize='x-small')
            plt.tight_layout()
            # plt.savefig(mRNA_loading_path + 'alpha_t_histograms_without_mRNA_' + ps_string[:-4] + '.png')

    def xest_instantaneous_transcription_rate_with_and_without_mRNA_CoV(self):
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','figure_5/')
        mRNA_loading_path = os.path.join(os.path.dirname(__file__),'output','mRNA_output/')

        chain_path_strings = [i for i in os.listdir(loading_path) if '.npy' in i
                                                                  if '1_cells_15_minutes' in i
                                                                  if 'ps9' in i
                                                                  if 'png' not in i]
        mRNA_chain_path_strings = [i for i in os.listdir(mRNA_loading_path) if '.npy' in i
                                                                            if '1_cells_15_minutes' in i
                                                                            if 'ps9' in i
                                                                            if 'png' not in i]

        number_of_samples = 25000
        import random
        def hill_function(protein,repression_threshold,hill_coefficient):
            return 1 / (1 + np.power(protein/repression_threshold,hill_coefficient))

        cov_values = np.zeros(len(mRNA_chain_path_strings))
        mRNA_cov_values = np.zeros(len(mRNA_chain_path_strings))

        for cov_index, string in enumerate(chain_path_strings):
            ps_string = string[string.find('ions')+5:]
            protein_observations = np.load(os.path.join(saving_path,'figure_5/protein_observations_' + ps_string))
            mean_protein = np.mean(protein_observations[:,1])
            true_parameters = np.load(os.path.join(saving_path,ps_string[:ps_string.find('_')] + '_parameter_values.npy'))[:7]
            if os.path.exists(os.path.join(mRNA_loading_path,
                                           'final_parallel_mala_output_protein_observations_with_mRNA_' + ps_string)):
                mRNA_chain = np.load(os.path.join(mRNA_loading_path,'final_parallel_mala_output_protein_observations_with_mRNA_' + ps_string))
                mRNA_chain = mRNA_chain.reshape(mRNA_chain.shape[0]*mRNA_chain.shape[1],mRNA_chain.shape[2])
            else:
                continue
            chain = np.load(os.path.join(loading_path,'final_parallel_mala_output_protein_observations_' + ps_string))
            chain = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])

            samples = np.zeros((number_of_samples,7))
            samples[:,[2,3]] = [np.log(2)/30,np.log(2)/90]
            samples[:,[0,1,4,5,6]] = chain[random.sample(range(chain.shape[0]),number_of_samples)]
            samples[:,[4,5]] = np.exp(samples[:,[4,5]])

            mRNA_samples = np.zeros((number_of_samples,7))
            mRNA_samples[:,[2,3]] = [np.log(2)/30,np.log(2)/90]
            mRNA_samples[:,[0,1,4,5,6]] = mRNA_chain[random.sample(range(chain.shape[0]),number_of_samples)]
            mRNA_samples[:,[4,5]] = np.exp(mRNA_samples[:,[4,5]])

            instant_transcription_rate = [i[4]*hill_function(mean_protein,i[0],i[1]) for i in samples]
            mRNA_instant_transcription_rate = [i[4]*hill_function(mean_protein,i[0],i[1]) for i in mRNA_samples]

            cov_values[cov_index] = np.std(instant_transcription_rate)/np.mean(instant_transcription_rate)
            mRNA_cov_values[cov_index] = np.std(mRNA_instant_transcription_rate)/np.mean(mRNA_instant_transcription_rate)

        import pdb; pdb.set_trace()
        fig, ax = plt.subplots(figsize=(8.63,6.95))
        ax.plot([0.5,2.5],[cov_values,mRNA_cov_values],'o-',color='#b5aeb0')
        ax.set_xticks([0.5,2.5])
        ax.set_ylim(0,0.7)
        ax.set_xticklabels(['Without mRNA','With mRNA'])
        ax.tick_params(axis='x',rotation=30)

        ax.plot([0.5,2.5],[np.mean(cov_values),np.mean(mRNA_cov_values)],color='#F18D9E',alpha=0.5)
        ax.fill_between([0.5,2.5],[np.mean(cov_values)-np.std(cov_values),
                                   np.mean(mRNA_cov_values)-np.std(mRNA_cov_values)],
                                  [np.mean(cov_values)+np.std(cov_values),
                                   np.mean(mRNA_cov_values)+np.std(mRNA_cov_values)],alpha=0.2,color='#F18D9E')

        ax.set_ylabel('Uncertainty, $\\alpha_T$')
        # plt.legend(fontsize='x-small')
        plt.tight_layout()
        plt.savefig(mRNA_loading_path + 'ps6_instant_transcription_rate_cov.png')
        # import pdb; pdb.set_trace()

    def xest_pair_plots_with_and_without_mRNA_CoV(self):
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','figure_5/')
        mRNA_loading_path = os.path.join(os.path.dirname(__file__),'output','mRNA_output/')

        chain_path_strings = [i for i in os.listdir(loading_path) if '.npy' in i
                                                                  if 'ps6_1_cells_8_minutes_4' in i
                                                                  if 'png' not in i]
        mRNA_chain_path_strings = [i for i in os.listdir(mRNA_loading_path) if '.npy' in i
                                                                            if 'ps6_1_cells_8_minutes_4' in i
                                                                            if 'png' not in i]

        number_of_samples = 25000
        parameter_names = np.array(["$P_0$",
                                    "$\\log(\\alpha_m$)"])
        import random
        from scipy.stats import pearsonr
        def corrfunc(x, y, ax=None, **kws):
            """Plot the correlation coefficient in the top left hand corner of a plot."""
            # import pdb; pdb.set_trace()
            r, _ = pearsonr(x, y)
            ax = ax or plt.gca()
            ax.annotate(f'$\\nu$ = {r:.2f}', xy=(.1, .5), xycoords=ax.transAxes)
            # ax.set_axis_off()

        from matplotlib.colors import LinearSegmentedColormap
        colors = ['#000000','#20948B','#FFFFFF']  # Black -> color -> White
        n_bins = 200  # Discretizes the interpolation into bins
        cmap_name = 'my_list'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        for cov_index, string in enumerate(chain_path_strings):
            ps_string = string[string.find('ions')+5:]
            protein_observations = np.load(os.path.join(saving_path,'figure_5/protein_observations_' + ps_string))
            mean_protein = np.mean(protein_observations[:,1])
            true_parameters = np.load(os.path.join(saving_path,ps_string[:ps_string.find('_')] + '_parameter_values.npy'))[:7]
            if os.path.exists(os.path.join(mRNA_loading_path,
                                           'final_parallel_mala_output_protein_observations_with_mRNA_' + ps_string)):
                mRNA_chain = np.load(os.path.join(mRNA_loading_path,'final_parallel_mala_output_protein_observations_with_mRNA_' + ps_string))
                mRNA_chain = mRNA_chain.reshape(mRNA_chain.shape[0]*mRNA_chain.shape[1],mRNA_chain.shape[2])
                mRNA_chain = mRNA_chain[:,[0,2]]
            else:
                continue
            chain = np.load(os.path.join(loading_path,'final_parallel_mala_output_protein_observations_' + ps_string))
            chain = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])
            chain = chain[:,[0,2]]

            samples = chain[random.sample(range(chain.shape[0]),number_of_samples)]
            mRNA_samples = mRNA_chain[random.sample(range(mRNA_chain.shape[0]),number_of_samples)]

            # output[:,[2,3]] = np.exp(output[:,[2,3]])
            df = pd.DataFrame(samples,columns=parameter_names)
            mRNA_df = pd.DataFrame(mRNA_samples,columns=parameter_names)

            # without mRNA
            fig, ax = plt.subplots(figsize=(7.92*6.85,5.94*6.85))
            # Create a pair grid instance
            grid = sns.PairGrid(data= df[parameter_names])
            # grid = sns.pairplot(data= df[parameter_names])
            # Map the plots to the locations
            grid = grid.map_upper(corrfunc)
            grid = grid.map_lower(sns.scatterplot, alpha=0.002,color='#20948B')
            grid = grid.map_lower(sns.kdeplot,color='k')
            grid = grid.map_diag(sns.histplot, bins = 20,color='#20948B');
            grid.axes[1,0].set_xticks([50000,100000])
            grid.axes[1,0].set_xticklabels(["5","10"])
            grid.axes[1,0].set_xlabel("$P_0$ [10e4]")
            plt.savefig(mRNA_loading_path + 'pairplot_' + ps_string[:-4] + '.png'); plt.close()

            # with mRNA
            fig, ax = plt.subplots(figsize=(7.92*10.85,5.94*10.85))
            # Create a pair grid instance
            grid = sns.PairGrid(data= mRNA_df[parameter_names])
            # grid = sns.pairplot(data = mRNA_df[parameter_names])
            # Map the plots to the locations
            grid = grid.map_upper(corrfunc)
            grid = grid.map_lower(sns.scatterplot, alpha=0.002,color='#20948B')
            grid = grid.map_lower(sns.kdeplot,color='k')
            grid = grid.map_diag(sns.histplot, bins = 20,color='#20948B');
            grid.axes[1,0].set_xticks([50000,100000])
            grid.axes[1,0].set_xticklabels(["5","10"])
            grid.axes[1,0].set_xlabel("$P_0$ [10e4]")
            plt.savefig(mRNA_loading_path + 'pairplot_' + ps_string[:-4] + '_with_mRNA.png'); plt.close()

    def qest_mRNA_distribution(self):
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','mRNA_output')

        protein_observations = np.load(os.path.join(saving_path,'figure_5/protein_observations_ps9_1_cells_12_minutes_2.npy'))
        chain = np.load(os.path.join(loading_path,'final_parallel_mala_output_protein_observations_with_mRNA_ps9_1_cells_12_minutes_2.npy'))

        # reshape and sample
        chain = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])

        number_of_samples = 2500
        samples = np.zeros((number_of_samples,7))
        samples[:,[2,3]] = [np.log(2)/30,np.log(2)/90]
        import random
        samples[:,[0,1,4,5,6]] = chain[random.sample(range(chain.shape[0]),number_of_samples)]
        samples[:,[4,5]] = np.exp(samples[:,[4,5]])

        mRNA_samples = np.zeros(number_of_samples)
        from scipy.stats import norm
        for index, sample in enumerate(samples):
            # mean
            state_space_mean, state_space_variance, _,_,_,_,_ = hes_inference.kalman_filter(protein_observations,
                                                                                            sample,
                                                                                            measurement_variance=1000000)
            # remove negative times
            zero_index = np.where(state_space_mean[:,0]==0)[0][0]
            #mean
            mRNA_mean = state_space_mean[zero_index:,1]
            # variance
            number_of_states = state_space_mean[zero_index:,1].shape[0]
            mRNA_covariance_matrix = state_space_variance[zero_index:number_of_states+zero_index,
                                                          zero_index:number_of_states+zero_index]
            mRNA_std = np.sqrt(np.diagonal(mRNA_covariance_matrix))
            random_index = random.sample(range(number_of_states),1)[0]
            mRNA_samples[index] = np.maximum(norm(mRNA_mean[random_index],mRNA_std[random_index]).rvs(1),0)[0]

        fig, ax = plt.subplots(figsize=(10,5))
        ax.hist(mRNA_samples,color='#20948B',density=True,bins=30)
        ax.set_xlabel('mRNA copy numbers')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','mRNA_distribution__with_mRNA_test.pdf'))

    def qest_get_mRNA_information_from_parameter_set(self,ps_strings=['ps6','ps7','ps8','ps9','ps10','ps11','ps12']):
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','')

        for ps_string in ps_strings:
            parameters = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))
            mRNA, _ = hes5.generate_multiple_langevin_trajectories( number_of_trajectories = 5000,
                                                                    duration = 5000,
                                                                    repression_threshold = parameters[0],
                                                                    hill_coefficient = parameters[1],
                                                                    mRNA_degradation_rate = parameters[2],
                                                                    protein_degradation_rate = parameters[3],
                                                                    basal_transcription_rate = parameters[4],
                                                                    translation_rate = parameters[5],
                                                                    transcription_delay = parameters[6],
                                                                    equilibration_time = 1000.0)
            all_mRNA_counts = mRNA[:,1:].reshape(5000*5000)
            mean_value = np.mean(all_mRNA_counts)
            std_value = np.std(all_mRNA_counts)
            np.save(saving_path + ps_string + "_mRNA_distribution.npy",np.array([mean_value,std_value]))

    def qest_mala_with_mRNA_information(self,data_filename = 'protein_observations_ps10_1_cells_12_minutes_2.npy'):
        data_filename = data_filename[:data_filename.find('ps')] + 'with_mRNA' + data_filename[data_filename.find('ps')-1:]
        saving_path = os.path.join(os.path.dirname(__file__),'data','figure_5')
        loading_path = os.path.join(os.path.dirname(__file__),'data','')
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('_cells')-2
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]

        protein_at_observations = np.array([np.load(os.path.join(saving_path,"protein_observations_" + data_filename[ps_string_index_start:]))])
        mRNA = np.load(os.path.join(loading_path,ps_string + "_mRNA_distribution.npy"))
        true_parameter_values = np.load(os.path.join(loading_path,ps_string + '_parameter_values.npy'))
        measurement_variance = np.power(true_parameter_values[-1],2)

        # define known parameters
        all_parameters = {'repression_threshold' : [0,true_parameter_values[0]],
                          'hill_coefficient' : [1,true_parameter_values[1]],
                          'mRNA_degradation_rate' : [2,np.log(true_parameter_values[2])],
                          'protein_degradation_rate' : [3,np.log(true_parameter_values[3])],
                          'basal_transcription_rate' : [4,np.log(true_parameter_values[4])],
                          'translation_rate' : [5,np.log(true_parameter_values[5])],
                          'transcription_delay' : [6,true_parameter_values[6]]}

        known_parameters = {k:all_parameters[k] for k in ('mRNA_degradation_rate',
                                                          'protein_degradation_rate') if k in all_parameters}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(all_parameters)) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        number_of_samples = 80000
        number_of_chains = 8
        step_size = 0.001

        run_mala_for_dataset_with_mRNA(data_filename,
                                       protein_at_observations,
                                       mRNA,
                                       measurement_variance,
                                       number_of_parameters,
                                       known_parameters,
                                       step_size,
                                       number_of_chains,
                                       number_of_samples)

    def xest_different_accuracy_metrics(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','output_jan_17/')
        chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps')]

        total_mean_error_values = np.zeros((5,len([i for i in chain_path_strings])))
        total_median_error_values = np.zeros((5,len([i for i in chain_path_strings])))
        total_cov_error_values = np.zeros((5,len([i for i in chain_path_strings])))
        total_prior_error_values = np.zeros((5,len([i for i in chain_path_strings])))

        for parameter_index in range(5):
            mean_error_values = np.zeros(len([i for i in chain_path_strings]))
            median_error_values = np.zeros(len([i for i in chain_path_strings]))
            cov_error_values = np.zeros(len([i for i in chain_path_strings]))
            prior_error_values = np.zeros(len([i for i in chain_path_strings]))

            for chain_index, chain_path_string in enumerate(chain_path_strings):
                mean_protein = np.mean(np.load(loading_path + '../../data/figure_5_coherence/protein_observations_' +
                                               chain_path_string[chain_path_string.find('ps'):])[:,1])
                mala = np.load(loading_path + chain_path_string)
                # import pdb; pdb.set_trace()
                samples = mala.reshape(mala.shape[0]*mala.shape[1],mala.shape[2])
                samples[:,[2,3]] = np.exp(samples[:,[2,3]])
                parameter_set_string = chain_path_string[chain_path_string.find('ps'):chain_path_string.find('_fig5')]
                true_values = np.load(loading_path + '../../data/' + parameter_set_string + '_parameter_values.npy')[[0,1,4,5,6]]
                # true_values[[2,3]] = np.log(true_values[[2,3]])
                sample_mean = np.mean(samples,axis=0)
                sample_std = np.std(samples,axis=0)
                sample_median = np.quantile(samples,0.5,axis=0)
                prior_widths = [2*mean_protein-50,4,120,40,40]
                # prior_widths = [2*mean_protein-50,4,np.log(120)-np.log(0.01),np.log(40)-np.log(0.01),40]

                mean_error = np.abs(true_values-sample_mean)/prior_widths
                mean_error_values[chain_index] = mean_error[parameter_index]/np.sum(mean_error)

                median_error = np.abs(true_values-sample_median)/true_values
                median_error_values[chain_index] = median_error[parameter_index]/np.sum(median_error)

                cov_error = sample_std/np.abs(true_values)
                cov_error_values[chain_index] = cov_error[parameter_index]/np.sum(cov_error)

                prior_error = sample_std/prior_widths
                prior_error_values[chain_index] = prior_error[parameter_index]/np.sum(prior_error)

            total_mean_error_values[parameter_index,:] = mean_error_values
            total_median_error_values[parameter_index,:] = median_error_values
            total_cov_error_values[parameter_index,:] = cov_error_values
            total_prior_error_values[parameter_index,:] = prior_error_values

        figure_strings = ["Mean Error","Median Error","Coefficient of Variation","Relative Uncertainty"]
        for string_index, error_values in enumerate([total_mean_error_values,total_median_error_values,total_cov_error_values,total_prior_error_values]):
            fig, ax = plt.subplots(figsize=(8,6))
            labels = ['$P_0$', '$h$', '$\\alpha_m$', '$\\alpha_p$', '$\\tau$']
            means = np.mean(error_values,axis=1)
            std = np.std(error_values,axis=1)
            width = 0.35
            ax.bar(labels, means, width, yerr=std)
            ax.set_ylabel('Proportion')
            ax.set_title(figure_strings[string_index])
            plt.tight_layout()
            plt.savefig(loading_path + figure_strings[string_index].replace(" ", "_").lower() + ".png")

    def xest_accuracy_with_or_without_mrna(self):
        fig5_loading_path = os.path.join(os.path.dirname(__file__),'output','figure_5/')
        mRNA_loading_path = os.path.join(os.path.dirname(__file__),'output','mRNA_output/')
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        ps6_true_parameter_values = np.load(os.path.join(saving_path,'ps6_parameter_values.npy'))[[0,1,4,5,6]]
        ps6_measurement_variance = np.power(ps6_true_parameter_values[-1],2)
        ps9_true_parameter_values = np.load(os.path.join(saving_path,'ps9_parameter_values.npy'))[[0,1,4,5,6]]
        ps9_measurement_variance = np.power(ps9_true_parameter_values[-1],2)


        ps6_mRNA_chain_path_strings = [i for i in os.listdir(mRNA_loading_path) if i.startswith('final_parallel_mala_output_protein_observations_with_mRNA_ps6_1_cells_15_minutes')]
        ps9_mRNA_chain_path_strings = [i for i in os.listdir(mRNA_loading_path) if i.startswith('final_parallel_mala_output_protein_observations_with_mRNA_ps9_1_cells_15_minutes')]
        ps6_fig5_chain_path_strings = [i for i in os.listdir(fig5_loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps6_1_cells_15_minutes')]
        ps9_fig5_chain_path_strings = [i for i in os.listdir(fig5_loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps9_1_cells_15_minutes')]
        ps6_mRNA_datasets = {}
        ps9_mRNA_datasets = {}
        ps6_fig5_datasets = {}
        ps9_fig5_datasets = {}

        for string in ps6_mRNA_chain_path_strings:
            ps6_mRNA_datasets[string] = np.load(mRNA_loading_path + string)
        for string in ps6_fig5_chain_path_strings:
            ps6_fig5_datasets[string] = np.load(fig5_loading_path + string)

        for string in ps9_mRNA_chain_path_strings:
            ps9_mRNA_datasets[string] = np.load(mRNA_loading_path + string)
        for string in ps9_fig5_chain_path_strings:
            ps9_fig5_datasets[string] = np.load(fig5_loading_path + string)

        ps6_mRNA_mean_error_dict = {}
        ps6_mRNA_cov_dict = {}
        ps9_mRNA_mean_error_dict = {}
        ps9_mRNA_cov_dict = {}

        ps6_fig5_mean_error_dict = {}
        ps6_fig5_cov_dict = {}
        ps9_fig5_mean_error_dict = {}
        ps9_fig5_cov_dict = {}

        for key in ps6_mRNA_datasets.keys():
            mean_protein = np.mean(np.load(mRNA_loading_path + '../../data/figure_5/protein_observations_' +
                                           key[key.find('ps'):])[:,1])
            prior_widths = [2*mean_protein-50,4,120,40,40]
            ps6_mRNA_mean_error_dict[key] = 0
            ps6_mRNA_cov_dict[key] = 0
            short_chains = ps6_mRNA_datasets[key].reshape(ps6_mRNA_datasets[key].shape[0]*ps6_mRNA_datasets[key].shape[1],5)
            short_chains[:,[2,3]] = np.exp(short_chains[:,[2,3]])
            short_chains_mean = np.mean(short_chains,axis=0)
            short_chains_std = np.std(short_chains,axis=0)
            # coefficient of variation
            # short_chain_cov = np.sum(short_chains_std/ps6_true_parameter_values)
            short_chain_cov = np.sum(short_chains_std/prior_widths)
            # print(short_chains_std/ps6_true_parameter_values)
            # relative mean
            relative_mean = np.sum(np.abs(ps6_true_parameter_values - short_chains_mean)/prior_widths)
            ps6_mRNA_mean_error_dict[key] = relative_mean
            ps6_mRNA_cov_dict[key] = short_chain_cov

        for key in ps9_mRNA_datasets.keys():
            mean_protein = np.mean(np.load(mRNA_loading_path + '../../data/figure_5/protein_observations_' +
                                           key[key.find('ps'):])[:,1])
            prior_widths = [2*mean_protein-50,4,120,40,40]
            ps9_mRNA_mean_error_dict[key] = 0
            ps9_mRNA_cov_dict[key] = 0
            short_chains = ps9_mRNA_datasets[key].reshape(ps9_mRNA_datasets[key].shape[0]*ps9_mRNA_datasets[key].shape[1],5)
            short_chains[:,[2,3]] = np.exp(short_chains[:,[2,3]])
            short_chains_mean = np.mean(short_chains,axis=0)
            short_chains_std = np.std(short_chains,axis=0)
            # coefficient of variation
            # short_chain_cov = np.sum(short_chains_std/ps9_true_parameter_values)
            short_chain_cov = np.sum(short_chains_std/prior_widths)
            # relative mean
            relative_mean = np.sum(np.abs(ps9_true_parameter_values - short_chains_mean)/prior_widths)
            ps9_mRNA_mean_error_dict[key] = relative_mean
            ps9_mRNA_cov_dict[key] = short_chain_cov

        for key in ps6_fig5_datasets.keys():
            mean_protein = np.mean(np.load(fig5_loading_path + '../../data/figure_5/protein_observations_' +
                                           key[key.find('ps'):])[:,1])
            prior_widths = [2*mean_protein-50,4,120,40,40]
            ps6_fig5_mean_error_dict[key] = 0
            ps6_fig5_cov_dict[key] = 0
            short_chains = ps6_fig5_datasets[key].reshape(ps6_fig5_datasets[key].shape[0]*ps6_fig5_datasets[key].shape[1],5)
            short_chains[:,[2,3]] = np.exp(short_chains[:,[2,3]])
            short_chains_mean = np.mean(short_chains,axis=0)
            short_chains_std = np.std(short_chains,axis=0)
            # coefficient of variation
            # short_chain_cov = np.sum(short_chains_std/ps6_true_parameter_values)
            short_chain_cov = np.sum(short_chains_std/prior_widths)
            # relative mean
            relative_mean = np.sum(np.abs(ps6_true_parameter_values - short_chains_mean)/prior_widths)
            ps6_fig5_mean_error_dict[key] = relative_mean
            ps6_fig5_cov_dict[key] = short_chain_cov

        for key in ps9_fig5_datasets.keys():
            mean_protein = np.mean(np.load(fig5_loading_path + '../../data/figure_5/protein_observations_' +
                                           key[key.find('ps'):])[:,1])
            prior_widths = [2*mean_protein-50,4,120,40,40]
            ps9_fig5_mean_error_dict[key] = 0
            ps9_fig5_cov_dict[key] = 0
            short_chains = ps9_fig5_datasets[key].reshape(ps9_fig5_datasets[key].shape[0]*ps9_fig5_datasets[key].shape[1],5)
            short_chains[:,[2,3]] = np.exp(short_chains[:,[2,3]])
            short_chains_mean = np.mean(short_chains,axis=0)
            short_chains_std = np.std(short_chains,axis=0)
            # coefficient of variation
            # short_chain_cov = np.sum(short_chains_std/ps9_true_parameter_values)
            short_chain_cov = np.sum(short_chains_std/prior_widths)
            # relative mean
            relative_mean = np.sum(np.abs(ps9_true_parameter_values - short_chains_mean)/prior_widths)
            ps9_fig5_mean_error_dict[key] = relative_mean
            ps9_fig5_cov_dict[key] = short_chain_cov

        plotting_strings = ['1_cells_15_minutes']
        dataset_string = ['1.npy',
                          '2.npy',
                          '3.npy',
                          '4.npy',
                          '5.npy',]

        fig, ax = plt.subplots(1,2,figsize=(8.63*2,6.95))
        mRNA_mean_and_sd_covs = np.zeros(2)
        mRNA_mean_and_sd_means = np.zeros(2)
        fig5_mean_and_sd_covs = np.zeros(2)
        fig5_mean_and_sd_means = np.zeros(2)
        for string in plotting_strings:
            mRNA_mean_and_sd_covs[0] = np.mean([y for x, y in ps6_mRNA_cov_dict.items()])
            mRNA_mean_and_sd_covs[1] = np.std([y for x, y in ps6_mRNA_cov_dict.items()])
            mRNA_mean_and_sd_means[0] = np.mean([y for x, y in ps6_mRNA_mean_error_dict.items()])
            mRNA_mean_and_sd_means[1] = np.std([y for x, y in ps6_mRNA_mean_error_dict.items()])
            fig5_mean_and_sd_covs[0] = np.mean([y for x, y in ps6_fig5_cov_dict.items()])
            fig5_mean_and_sd_covs[1] = np.std([y for x, y in ps6_fig5_cov_dict.items()])
            fig5_mean_and_sd_means[0] = np.mean([y for x, y in ps6_fig5_mean_error_dict.items()])
            fig5_mean_and_sd_means[1] = np.std([y for x, y in ps6_fig5_mean_error_dict.items()])

            for index, substring in enumerate(dataset_string):
                # 1/cov
                mRNA_cov = [y for x, y in ps6_mRNA_cov_dict.items() if substring in x][0]
                fig5_cov = [y for x, y in ps6_fig5_cov_dict.items() if substring in x][0]
                # xcoords = [np.int(string[string.find('lls_')+4:string.find('_min')])]*len(covs)
                ax[0].plot([0.5,2.5],[fig5_cov,mRNA_cov],'o-',label='Without mRNA', color='#b5aeb0')
                # ax[0].scatter(1,mRNA_cov,label='With mRNA', color='#b5aeb0')
                # ax[0].set_xlabel("Sampling interval (mins)",fontsize=font_size)
                ax[0].set_xlim(0,3)
                ax[0].set_ylim(.5,1.2)
                ax[0].set_xticks([0.5,2.5])
                ax[0].set_xticklabels(['Without mRNA','With mRNA'])
                ax[0].set_ylabel("Uncertainty (All Parameters)",fontsize=font_size)
                ax[0].tick_params(axis='x',rotation=30)
                # mean error
                mRNA_mean_errors = [y for x, y in ps6_mRNA_mean_error_dict.items() if substring in x][0]
                fig5_mean_errors = [y for x, y in ps6_fig5_mean_error_dict.items() if substring in x][0]
                ax[1].plot([0.5,2.5],[fig5_mean_errors,mRNA_mean_errors],'o-',label='Without mRNA', color='#b5aeb0')
                # ax[1].scatter(0,fig5_mean_errors,label=string, color='#b5aeb0')
                # ax[1].scatter(1,mRNA_mean_errors,label=string, color='#b5aeb0')
                # ax[1].set_xlim(15.5,4.5) # backwards for comparison to length
                # ax[1].set_xlabel("Sampling interval (mins)",fontsize=font_size)
                ax[1].set_xlim(0,3)
                ax[1].set_xticks([0.5,2.5])
                ax[1].set_xticklabels(['Without mRNA','With mRNA'])
                ax[1].set_ylabel("Relative mean error",fontsize=font_size)
                ax[1].tick_params(axis='x',rotation=30)
                # plt.legend()


            ax[0].plot([0.5,2.5],[fig5_mean_and_sd_covs[0],mRNA_mean_and_sd_covs[0]],color='#F18D9E',alpha=0.5)
            ax[0].fill_between([0.5,2.5], [fig5_mean_and_sd_covs[0]-fig5_mean_and_sd_covs[1],
                                                              mRNA_mean_and_sd_covs[0]-mRNA_mean_and_sd_covs[1]],
                                                             [fig5_mean_and_sd_covs[0]+fig5_mean_and_sd_covs[1],
                                                              mRNA_mean_and_sd_covs[0]+mRNA_mean_and_sd_covs[1]],alpha=0.2,color='#F18D9E')
            ax[1].plot([0.5,2.5],[fig5_mean_and_sd_means[0],mRNA_mean_and_sd_means[0]],color='#F18D9E',alpha=0.5)
            ax[1].fill_between([0.5,2.5], [fig5_mean_and_sd_means[0]-fig5_mean_and_sd_means[1],
                                                              mRNA_mean_and_sd_means[0]-mRNA_mean_and_sd_means[1]],
                                                             [fig5_mean_and_sd_means[0]+fig5_mean_and_sd_means[1],
                                                              mRNA_mean_and_sd_means[0]+mRNA_mean_and_sd_means[1]],alpha=0.2,color='#F18D9E')
        plt.tight_layout()
        plt.savefig(mRNA_loading_path + 'ps6_with_and_without_mRNA_error_values_frequency_prior.png')
        import pdb; pdb.set_trace()

        fig, ax = plt.subplots(1,2,figsize=(8.63*2,6.95))
        mRNA_mean_and_sd_covs = np.zeros(2)
        mRNA_mean_and_sd_means = np.zeros(2)
        fig5_mean_and_sd_covs = np.zeros(2)
        fig5_mean_and_sd_means = np.zeros(2)

        for string in plotting_strings:
            mRNA_mean_and_sd_covs[0] = np.mean([y for x, y in ps9_mRNA_cov_dict.items()])
            mRNA_mean_and_sd_covs[1] = np.std([y for x, y in ps9_mRNA_cov_dict.items()])
            mRNA_mean_and_sd_means[0] = np.mean([y for x, y in ps9_mRNA_mean_error_dict.items()])
            mRNA_mean_and_sd_means[1] = np.std([y for x, y in ps9_mRNA_mean_error_dict.items()])
            fig5_mean_and_sd_covs[0] = np.mean([y for x, y in ps9_fig5_cov_dict.items()])
            fig5_mean_and_sd_covs[1] = np.std([y for x, y in ps9_fig5_cov_dict.items()])
            fig5_mean_and_sd_means[0] = np.mean([y for x, y in ps9_fig5_mean_error_dict.items()])
            fig5_mean_and_sd_means[1] = np.std([y for x, y in ps9_fig5_mean_error_dict.items()])

            for index, substring in enumerate(dataset_string):
                # 1/cov
                mRNA_cov = [y for x, y in ps9_mRNA_cov_dict.items() if substring in x][0]
                fig5_cov = [y for x, y in ps9_fig5_cov_dict.items() if substring in x][0]
                # xcoords = [np.int(string[string.find('lls_')+4:string.find('_min')])]*len(covs)
                ax[0].plot([0.5,2.5],[fig5_cov,mRNA_cov],'o-',label='Without mRNA', color='#b5aeb0')
                # ax[0].scatter(1,mRNA_cov,label='With mRNA', color='#b5aeb0')
                # ax[0].set_xlim(15.5,4.5) # backwards for comparison to length
                # ax[0].set_xlabel("Sampling interval (mins)",fontsize=font_size)
                ax[0].set_ylim(.5,1.2)
                ax[0].set_xlim(0,3)
                ax[0].set_xticks([0.5,2.5])
                ax[0].set_xticklabels(['Without mRNA','With mRNA'])
                ax[0].set_ylabel("Uncertainty (All Parameters)",fontsize=font_size)
                ax[0].tick_params(axis='x',rotation=30)
                # mean error
                mRNA_mean_errors = [y for x, y in ps9_mRNA_mean_error_dict.items() if substring in x][0]
                fig5_mean_errors = [y for x, y in ps9_fig5_mean_error_dict.items() if substring in x][0]
                ax[1].plot([0.5,2.5],[fig5_mean_errors,mRNA_mean_errors],'o-',label='Without mRNA', color='#b5aeb0')
                # ax[1].scatter(0,fig5_mean_errors,label=string, color='#b5aeb0')
                # ax[1].scatter(1,mRNA_mean_errors,label=string, color='#b5aeb0')
                # ax[1].set_xlim(15.5,4.5) # backwards for comparison to length
                # ax[1].set_xlabel("Sampling interval (mins)",fontsize=font_size)
                ax[1].set_xlim(0,3)
                ax[1].set_xticks([0.5,2.5])
                ax[1].set_xticklabels(['Without mRNA','With mRNA'])
                ax[1].set_ylabel("Relative mean error",fontsize=font_size)
                ax[1].tick_params(axis='x',rotation=30)
                # plt.legend()


            ax[0].plot([0.5,2.5],[fig5_mean_and_sd_covs[0],mRNA_mean_and_sd_covs[0]],color='#8d9ef1',alpha=0.5)
            ax[0].fill_between([0.5,2.5], [fig5_mean_and_sd_covs[0]-fig5_mean_and_sd_covs[1],
                                                              mRNA_mean_and_sd_covs[0]-mRNA_mean_and_sd_covs[1]],
                                                             [fig5_mean_and_sd_covs[0]+fig5_mean_and_sd_covs[1],
                                                              mRNA_mean_and_sd_covs[0]+mRNA_mean_and_sd_covs[1]],alpha=0.2,color='#8d9ef1')
            ax[1].plot([0.5,2.5],[fig5_mean_and_sd_means[0],mRNA_mean_and_sd_means[0]],color='#8d9ef1',alpha=0.5)
            ax[1].fill_between([0.5,2.5], [fig5_mean_and_sd_means[0]-fig5_mean_and_sd_means[1],
                                                              mRNA_mean_and_sd_means[0]-mRNA_mean_and_sd_means[1]],
                                                             [fig5_mean_and_sd_means[0]+fig5_mean_and_sd_means[1],
                                                              mRNA_mean_and_sd_means[0]+mRNA_mean_and_sd_means[1]],alpha=0.2,color='#8d9ef1')
            # ax[1].plot(['Without mRNA','With mRNA'],mean_and_sd_means[:,1],color='#8d9ef1',alpha=0.5)
                # ax[1].fill_between(['Without mRNA','With mRNA'], np.maximum(0,mean_and_sd_means[:,1]-mean_and_sd_means[:,2]), mean_and_sd_means[:,1]+mean_and_sd_means[:,2], alpha=0.2,color='#F18D9E')

        plt.tight_layout()
        plt.savefig(mRNA_loading_path + 'ps9_with_and_without_mRNA_error_values_frequency_prior.png')
        import pdb; pdb.set_trace()

    def fest_transcription_accuracy_with_or_without_mrna(self):
        fig5_loading_path = os.path.join(os.path.dirname(__file__),'output','figure_5/')
        mRNA_loading_path = os.path.join(os.path.dirname(__file__),'output','mRNA_output/')
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        ps6_true_parameter_values = np.load(os.path.join(saving_path,'ps6_parameter_values.npy'))
        ps9_true_parameter_values = np.load(os.path.join(saving_path,'ps9_parameter_values.npy'))


        ps6_mRNA_chain_path_strings = [i for i in os.listdir(mRNA_loading_path) if i.startswith('final_parallel_mala_output_protein_observations_with_mRNA_ps6_1_cells_15_minutes')]
        ps9_mRNA_chain_path_strings = [i for i in os.listdir(mRNA_loading_path) if i.startswith('final_parallel_mala_output_protein_observations_with_mRNA_ps9_1_cells_15_minutes')]
        ps6_fig5_chain_path_strings = [i for i in os.listdir(fig5_loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps6_1_cells_15_minutes')]
        ps9_fig5_chain_path_strings = [i for i in os.listdir(fig5_loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps9_1_cells_15_minutes')]
        ps6_mRNA_datasets = {}
        ps9_mRNA_datasets = {}
        ps6_fig5_datasets = {}
        ps9_fig5_datasets = {}

        for string in ps6_mRNA_chain_path_strings:
            ps6_mRNA_datasets[string] = np.load(mRNA_loading_path + string)
        for string in ps6_fig5_chain_path_strings:
            ps6_fig5_datasets[string] = np.load(fig5_loading_path + string)

        for string in ps9_mRNA_chain_path_strings:
            ps9_mRNA_datasets[string] = np.load(mRNA_loading_path + string)
        for string in ps9_fig5_chain_path_strings:
            ps9_fig5_datasets[string] = np.load(fig5_loading_path + string)

        plotting_strings = ['1_cells_15_minutes']
        dataset_string = ['1.npy',
                          '2.npy',
                          '3.npy',
                          '4.npy',
                          '5.npy',]

        for string in plotting_strings:
            for index, substring in enumerate(dataset_string):
                fig, ax = plt.subplots(2,1,figsize=(6.95,8.63*1.25))
                mRNA_chain = [y for x, y in ps6_mRNA_datasets.items() if substring in x][0]
                mRNA_chain = mRNA_chain.reshape(640000,5)
                fig5_chain = [y for x, y in ps6_fig5_datasets.items() if substring in x][0]
                fig5_chain = fig5_chain.reshape(640000,5)
                # xcoords = [np.int(string[string.find('lls_')+4:string.find('_min')])]*len(covs)
                heights, bins, _ = ax[0].hist(mRNA_chain[:,2],bins=30,density=True,color='#F18D9E',ec='grey',alpha=0.8)
                ax[0].vlines(np.log(ps6_true_parameter_values[4]),0,1.1*max(heights),color='k',lw=2,label='True value')
                heights, bins, _ = ax[1].hist(fig5_chain[:,2],bins=30,density=True,color='#F18D9E',ec='grey',alpha=0.8)
                ax[1].vlines(np.log(ps6_true_parameter_values[4]),0,1.1*max(heights),color='k',lw=2,label='True value')
                # ax[0].scatter(1,mRNA_cov,label='With mRNA', color='#b5aeb0')
                # ax[0].set_xlim(15.5,4.5) # backwards for comparison to length
                # ax[0].set_xlabel("Sampling interval (mins)",fontsize=font_size)
                ax[0].set_xlim(-1.5,5)
                ax[1].set_xlim(-1.5,5)
                ax[0].set_ylabel("Probability",fontsize=font_size)
                ax[1].set_ylabel("Probability",fontsize=font_size)
                ax[0].set_xlabel("$\log(\\alpha_m)$ without mRNA data",fontsize=font_size)
                ax[1].set_xlabel("$\log(\\alpha_m)$ with mRNA data",fontsize=font_size)

                plt.tight_layout()
                plt.savefig(mRNA_loading_path + 'ps6_transcription_hist_' + substring[0] + '.png')

        for string in plotting_strings:
            for index, substring in enumerate(dataset_string):
                fig, ax = plt.subplots(2,1,figsize=(6.95,8.63*1.25))
                mRNA_chain = [y for x, y in ps9_mRNA_datasets.items() if substring in x][0]
                mRNA_chain = mRNA_chain.reshape(640000,5)
                fig5_chain = [y for x, y in ps9_fig5_datasets.items() if substring in x][0]
                fig5_chain = fig5_chain.reshape(640000,5)

                heights, bins, _ = ax[0].hist(mRNA_chain[:,2],bins=30,density=True,color='#8d9ef1',ec='grey',alpha=0.8)
                ax[0].vlines(np.log(ps9_true_parameter_values[4]),0,1.1*max(heights),color='k',lw=2,label='True value')
                heights, bins, _ = ax[1].hist(fig5_chain[:,2],bins=30,density=True,color='#8d9ef1',ec='grey',alpha=0.8)
                ax[1].vlines(np.log(ps9_true_parameter_values[4]),0,1.1*max(heights),color='k',lw=2,label='True value')

                ax[0].set_xlim(-1.5,5)
                ax[1].set_xlim(-1.5,5)
                ax[0].set_ylabel("Probability",fontsize=font_size)
                ax[1].set_ylabel("Probability",fontsize=font_size)
                ax[0].set_xlabel("$\log(\\alpha_m)$ without mRNA data",fontsize=font_size)
                ax[1].set_xlabel("$\log(\\alpha_m)$ with mRNA data",fontsize=font_size)

                plt.tight_layout()
                plt.savefig(mRNA_loading_path + 'ps9_transcription_hist_' + substring[0] + '.png')
