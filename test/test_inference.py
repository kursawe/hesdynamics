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
font = {'size'   : 16}
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

class TestInference(unittest.TestCase):

    def xest_check_kalman_filter_not_broken(self):
        # load in some saved observations and correct kalman filter predictions
        saving_path                          = os.path.join(os.path.dirname(__file__), 'data','kalman_test_trace')
        fixed_langevin_trace                 = np.load(saving_path + '_true_data.npy')
        fixed_protein_observations           = np.load(saving_path + '_observations.npy')
        true_kalman_prediction_mean          = np.load(saving_path + '_prediction_mean.npy')
        true_kalman_prediction_variance      = np.load(saving_path + '_prediction_variance.npy')
        true_kalman_prediction_distributions = np.load(saving_path + '_prediction_distributions.npy')
        true_kalman_negative_log_likelihood_derivative = np.load(saving_path + '_negative_log_likelihood_derivative.npy')

        # run the current kalman filter using the same parameters and observations, then compare
        parameters = np.array([10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0])
        state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative,predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives = hes_inference.kalman_filter(fixed_protein_observations,
                                                                                                                                                                                                                                                                   parameters,
                                                                                                                                                                                                                                                                   measurement_variance=10000)
        # log_likelihood, negative_log_likelihood_derivative = hes_inference.calculate_log_likelihood_and_derivative_at_parameter_point(fixed_protein_observations,
        #                                                                                                                               parameters,
        #                                                                                                                               measurement_variance=10000)
        #
        # print(log_likelihood)
        np.testing.assert_almost_equal(state_space_mean,true_kalman_prediction_mean)
        np.testing.assert_almost_equal(state_space_variance,true_kalman_prediction_variance)
        np.testing.assert_almost_equal(predicted_observation_distributions,true_kalman_prediction_distributions)
        # np.testing.assert_almost_equal(true_kalman_negative_log_likelihood_derivative,negative_log_likelihood_derivative)
        # np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_test_trace_prediction_mean.npy'),state_space_mean)
        # np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_test_trace_prediction_variance.npy'),state_space_variance)
        # np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_test_trace_prediction_distributions.npy'),predicted_observation_distributions)
        # import pdb; pdb.set_trace()
        # If above tests fail, comment them out to look at the plot below. Could be useful for identifying problems.
        # number_of_states = state_space_mean.shape[0]
        # protein_covariance_matrix = state_space_variance[number_of_states:,number_of_states:]
        # protein_variance = np.diagonal(protein_covariance_matrix)
        # protein_error = np.sqrt(protein_variance)*2
        #
        # true_protein_covariance_matrix = true_kalman_prediction_variance[number_of_states:,number_of_states:]
        # true_protein_variance = np.diagonal(true_protein_covariance_matrix)
        # true_protein_error = np.sqrt(protein_variance)*2
        #
        # my_figure = plt.figure()
        # plt.subplot(2,1,1)
        # plt.scatter(np.arange(0,900,10),fixed_protein_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations',zorder=4)
        # plt.plot(fixed_langevin_trace[:,0],fixed_langevin_trace[:,2],label='true protein',color='#F69454',linewidth=0.89,zorder=3)
        # plt.plot(true_kalman_prediction_mean[:,0],true_kalman_prediction_mean[:,2],label='inferred protein',color='#20948B',zorder=2)
        # plt.scatter(np.arange(0,900,10),true_kalman_prediction_distributions[:,1],marker='o',s=4,c='#98DBC6',label='likelihood',zorder=2)
        # plt.errorbar(true_kalman_prediction_mean[:,0],true_kalman_prediction_mean[:,2],yerr=true_protein_error,ecolor='#98DBC6',alpha=0.1,zorder=1)
        # plt.errorbar(true_kalman_prediction_distributions[:,0],true_kalman_prediction_distributions[:,1],
        #              yerr=np.sqrt(true_kalman_prediction_distributions[:,2])*2,ecolor='#98DBC6',alpha=0.6,linestyle="None",zorder=1)
        # plt.legend(fontsize='x-small')
        # plt.title('What the Plot should look like')
        # plt.xlabel('Time')
        # plt.ylabel('Protein Copy Numbers')
        #
        # plt.subplot(2,1,2)
        # plt.scatter(np.arange(0,900,10),fixed_protein_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations',zorder=4)
        # plt.plot(fixed_langevin_trace[:,0],fixed_langevin_trace[:,2],label='true protein',color='#F69454',linewidth=0.89,zorder=3)
        # plt.plot(state_space_mean[:,0],state_space_mean[:,2],label='inferred protein',color='#20948B',zorder=2)
        # plt.scatter(np.arange(0,900,10),predicted_observation_distributions[:,1],marker='o',s=4,c='#98DBC6',label='likelihood',zorder=2)
        # plt.errorbar(state_space_mean[:,0],state_space_mean[:,2],yerr=protein_error,ecolor='#98DBC6',alpha=0.1,zorder=1)
        # plt.errorbar(predicted_observation_distributions[:,0],predicted_observation_distributions[:,1],
        #              yerr=np.sqrt(predicted_observation_distributions[:,2])*2,ecolor='#98DBC6',alpha=0.6,linestyle="None",zorder=1)
        # plt.legend(fontsize='x-small')
        # plt.title('What the current function gives')
        # plt.xlabel('Time')
        # plt.ylabel('Protein Copy Numbers')
        # plt.tight_layout()
        # my_figure.savefig(os.path.join(os.path.dirname(__file__),
        #                                'output','kalman_check.pdf'))
        #
        # likelihood = hes_inference.calculate_log_likelihood_at_parameter_point(fixed_protein_observations,parameters,measurement_variance=10000)
        # print(likelihood)
        #
        # observations = fixed_protein_observations[:,1]
        # mean = true_kalman_prediction_distributions[:,1]
        # sd = np.sqrt(true_kalman_prediction_distributions[:,2])
        #
        # from scipy.stats import norm
        # print(np.sum(norm.logpdf(observations,mean,sd)))

    def xest_likelihood_derivative_working(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','kalman_test_trace')
        fixed_protein_observations = np.load(saving_path + '_observations.npy')
        # run the current kalman filter using the same parameters and observations, then compare
        measurement_variance = 10000
        step_size = 0.000001
        print('---------------------')
        print('step size =',step_size)
        print('---------------------')
        parameter_names = np.array(['repression_threshold','hill_coefficient','mRNA_degradation_rate',
                                    'protein_degradation_rate','basal_transcription_rate','translation_rate',
                                    'transcriptional_delay'])

        for i in range(7):

            parameters = np.array([10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0])
            shifted_parameters = np.array([10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0])
            shifted_parameters[i] += step_size

            log_likelihood, log_likelihood_derivative = hes_inference.calculate_log_likelihood_and_derivative_at_parameter_point(fixed_protein_observations,
                                                                                                                                          parameters,
                                                                                                                                          measurement_variance=10000)

            shifted_log_likelihood, _ = hes_inference.calculate_log_likelihood_and_derivative_at_parameter_point(fixed_protein_observations,
                                                                                                                 shifted_parameters,
                                                                                                                 measurement_variance=10000)
            numerical_derivative = (shifted_log_likelihood-log_likelihood)/step_size
            print()
            print(parameter_names[i])
            print()
            print('log_likelihood_derivative:',log_likelihood_derivative[i])
            print('numerical derivative:',numerical_derivative)
            print('error:',np.abs(numerical_derivative-log_likelihood_derivative[i]))
            print('precentage error:',100*np.abs(numerical_derivative-log_likelihood_derivative[i])/np.abs(numerical_derivative))
            print('----------------------------')

    def xest_state_space_mean_derivative_working(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','kalman_test_trace')
        fixed_protein_observations = np.load(saving_path + '_observations.npy')
        # run the current kalman filter using the same parameters and observations, then compare
        measurement_variance = 10000
        step_sizes = [0.001,0.0001,0.00001,0.000001,0.0000001,
                      0.00000001,0.000000001,0.0000000001,0.00000000001,
                      0.000000000001,0.0000000000001,0.00000000000001,0.000000000000001]
        error = np.zeros((7,len(step_sizes)))
        percentage_error = np.zeros((7,len(step_sizes)))
        for step_size_index, step_size in enumerate(step_sizes):
            measurement_variance = 10000
            print('---------------------')
            print('step size =',step_size)
            print('---------------------')
            parameter_names = np.array(['repression_threshold','hill_coefficient','mRNA_degradation_rate',
                                        'protein_degradation_rate','basal_transcription_rate','translation_rate',
                                        'transcriptional_delay'])

            for i in range(7):

                parameters = np.array([10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0])
                shifted_parameters_up = np.array([10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0])
                shifted_parameters_up[i] += step_size/2
                shifted_parameters_down = np.array([10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0])
                shifted_parameters_down[i] -= step_size/2

                state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative, predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives = hes_inference.kalman_filter(fixed_protein_observations[:5],
                                                                                                                                                                                                                                                                            parameters,
                                                                                                                                                                                                                                                                            measurement_variance=10000)

                shifted_state_space_mean_up, shifted_state_space_variance_up, _, _, shifted_predicted_observation_distributions_up, _, _ = hes_inference.kalman_filter(fixed_protein_observations[:5],
                                                                                                                                                                       shifted_parameters_up,
                                                                                                                                                                       measurement_variance=10000)
                shifted_state_space_mean_down, shifted_state_space_variance_down, _, _, shifted_predicted_observation_distributions_down, _, _ = hes_inference.kalman_filter(fixed_protein_observations[:5],
                                                                                                                                                                             shifted_parameters_down,
                                                                                                                                                                             measurement_variance=10000)

                state_space_mean_numerical_derivative = (shifted_state_space_mean_up-shifted_state_space_mean_down)/step_size
                state_space_variance_numerical_derivative = (shifted_state_space_variance_up-state_space_variance_down)/step_size
                error[i,step_size_index] = np.abs(state_space_mean_numerical_derivative[-1,2]-state_space_mean_derivative[-1,i,1])
                percentage_error[i,step_size_index] = 100*error[i,step_size_index]/np.abs(state_space_mean_numerical_derivative[-1,2])
                # print()
                # print(parameter_names[i])
                # print()
                # print('state_space_mean_derivative:',state_space_mean_derivative[-1,i,1])
                # print('state_space_mean_numerical_derivative:',state_space_mean_numerical_derivative[-1,2])
                # print('error:',error)
                # print('precentage error:',100*error/np.abs(state_space_mean_numerical_derivative[-1,2]),'%')
                # print()
                # print('state_space_variance_derivative:',state_space_variance_derivative[i,-1,-1])
                # print('state_space_variance_numerical_derivative:',state_space_variance_numerical_derivative[-1,-1])
                # print('error:',np.abs(state_space_variance_numerical_derivative[-1,-1]-state_space_variance_derivative[i,-1,-1]))
                # print('percentage error:',100*np.abs(state_space_variance_numerical_derivative[-1,-1]-state_space_variance_derivative[i,-1,-1])/np.abs(state_space_variance_numerical_derivative[-1,-1]),'%')
                # print('----------------------------')
        fig, ax = plt.subplots(2,4,figsize=(20,10))
        # ax.set_yscale('log')

        ax[0,0].loglog(step_sizes,error[0],color='red',lw=2,label='error')
        ax[0,0].loglog(step_sizes,percentage_error[0],color='blue',lw=2,label='percentage error')
        ax[0,0].legend(loc='upper left', bbox_to_anchor=(0.0,1.0), shadow=True)
        ax[0,0].set_title('repression threshold error')
        ax[0,1].loglog(step_sizes,error[1],color='blue',lw=2)
        ax[0,1].loglog(step_sizes,percentage_error[1],color='red',lw=2)
        ax[0,1].set_title('hill coefficient error')
        ax[0,2].loglog(step_sizes,error[2],color='blue',lw=2)
        ax[0,2].loglog(step_sizes,percentage_error[2],color='red',lw=2)
        ax[0,2].set_title('mRNA degradation rate error')
        ax[0,3].loglog(step_sizes,error[3],color='blue',lw=2)
        ax[0,3].loglog(step_sizes,percentage_error[3],color='red',lw=2)
        ax[0,3].set_title('protein degradation rate error')
        ax[1,0].loglog(step_sizes,error[4],color='blue',lw=2)
        ax[1,0].loglog(step_sizes,percentage_error[4],color='red',lw=2)
        ax[1,0].set_title('transcription rate error')
        ax[1,1].loglog(step_sizes,error[5],color='blue',lw=2)
        ax[1,1].loglog(step_sizes,percentage_error[5],color='red',lw=2)
        ax[1,1].set_title('translation rate error')
        ax[1,2].loglog(step_sizes,error[6],color='blue',lw=2)
        ax[1,2].loglog(step_sizes,percentage_error[6],color='red',lw=2)
        ax[1,2].set_title('transcriptional delay error')
        plt.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','error_corrected.pdf'))

    def xest_log_likelihood_derivative_working(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output','')
        fixed_protein_observations = np.load(saving_path + 'protein_observations_90_ps3_ds1.npy')
        # run the current kalman filter using the same parameters and observations, then compare
        measurement_variance = 10000
        step_sizes = [0.01,0.001,0.0001,0.00001,0.000001,0.0000001,
                      0.00000001,0.000000001,0.0000000001,0.00000000001]
        # step_sizes = [10,1,0.1]
        error = np.zeros((7,len(step_sizes)))
        percentage_error = np.zeros((7,len(step_sizes)))
        parameter_names = np.array(['repression_threshold','hill_coefficient','mRNA_degradation_rate',
                                    'protein_degradation_rate','basal_transcription_rate','translation_rate',
                                    'transcriptional_delay'])
        for step_size_index, step_size in enumerate(step_sizes):
            print('---------------------')
            print('step size =',step_size)
            print('---------------------')
            for i in range(7):
                parameters = np.array([3000,5,np.log(2)/20, np.log(2)/70, 15, 1.5, 28.0])
                shifted_parameters_up = np.array([3000,5,np.log(2)/20, np.log(2)/70, 15, 1.5, 28.0])
                shifted_parameters_up[i] += step_size/2
                shifted_parameters_down = np.array([3000,5,np.log(2)/20, np.log(2)/70, 15, 1.5, 28.0])
                shifted_parameters_down[i] -= step_size/2

                log_likelihood, log_likelihood_derivative = hes_inference.calculate_log_likelihood_and_derivative_at_parameter_point(fixed_protein_observations,parameters,measurement_variance)
                shifted_log_likelihood_up, shifted_log_likelihood_derivative_up = hes_inference.calculate_log_likelihood_and_derivative_at_parameter_point(fixed_protein_observations,shifted_parameters_up,measurement_variance)
                shifted_log_likelihood_down, shifted_log_likelihood_derivative_down = hes_inference.calculate_log_likelihood_and_derivative_at_parameter_point(fixed_protein_observations,shifted_parameters_down,measurement_variance)

                log_likelihood_numerical_derivative = (shifted_log_likelihood_up-shifted_log_likelihood_down)/step_size
                error[i,step_size_index] = euclidean(log_likelihood_numerical_derivative, log_likelihood_derivative[i])
                percentage_error[i,step_size_index] = 100*error[i,step_size_index]/np.abs(log_likelihood_numerical_derivative)
                print()
                print(parameter_names[i])
                print()
                print('derivative: ',log_likelihood_derivative[i])
                print('numerical_derivative:',log_likelihood_numerical_derivative)
                print('error:',error[i,step_size_index])
                print('precentage error:',percentage_error[i,step_size_index],'%')
                print()
                # print('state_space_variance_derivative:',state_space_variance_derivative[i,-1,-1])
                # print('state_space_variance_numerical_derivative:',state_space_variance_numerical_derivative[-1,-1])
                # print('error:',np.abs(state_space_variance_numerical_derivative[-1,-1]-state_space_variance_derivative[i,-1,-1]))
                # print('percentage error:',100*np.abs(state_space_variance_numerical_derivative[-1,-1]-state_space_variance_derivative[i,-1,-1])/np.abs(state_space_variance_numerical_derivative[-1,-1]),'%')
                print('----------------------------')
        fig, ax = plt.subplots(2,3,figsize=(20,10))
        ax[0,0].loglog(step_sizes,percentage_error[0],color='red',lw=2)
        ax[0,0].set_title('repression threshold error')
        ax[0,0].set_xlabel('step size')
        ax[0,0].set_ylabel('percentage error')
        ax[0,1].loglog(step_sizes,percentage_error[1],color='red',lw=2)
        ax[0,1].set_title('hill coefficient error')
        ax[0,2].loglog(step_sizes,percentage_error[2],color='red',lw=2)
        ax[0,2].set_title('mRNA degradation rate error')
        ax[1,0].loglog(step_sizes,percentage_error[3],color='red',lw=2)
        ax[1,0].set_title('protein degradation rate error')
        ax[1,1].loglog(step_sizes,percentage_error[4],color='red',lw=2)
        ax[1,1].set_title('transcription rate error')
        ax[1,2].loglog(step_sizes,percentage_error[5],color='red',lw=2)
        ax[1,2].set_title('translation rate error')
        plt.tight_layout()
        fig.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','error_log_likelihood_derivative_test_corrected.pdf'))


    def xest_kalman_mala_likelihood_single_parameters(self):
        # load data and true parameter values
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'output','')

        protein_at_observations = np.array([np.load(saving_path + 'protein_observations_ps11_ds4.npy')])
        true_parameter_values = np.load(os.path.join(saving_path,'ps11_parameter_values.npy'))
        mala_output = np.load(saving_path + 'mala_output_1.npy')
        number_of_samples = 200
        measurement_variance = np.power(true_parameter_values[-1],2)

        # true parameters ps3 -- [3407.99,5.17,np.log(2)/30,np.log(2)/90,15.86,1.27,30]
        initial_position = true_parameter_values[[0,1,2,3,4,5,6]]
        initial_position[[4,5]] = np.log(initial_position[[4,5]])
        proposal_covariance = np.cov(mala_output.T)
        step_size = [55000.0,0.45,0.01,0.013,90.0]
        all_parameters = {'repression_threshold' : [0,true_parameter_values[0]],
                          'hill_coefficient' : [1,true_parameter_values[1]],
                          'mRNA_degradation_rate' : [2,np.log(2)/30],
                          'protein_degradation_rate' : [3,np.log(2)/90],
                          'basal_transcription_rate' : [4,np.log(true_parameter_values[4])],
                          'translation_rate' : [5,np.log(true_parameter_values[5])],
                          'transcription_delay' : [6,true_parameter_values[6]]}


        for index in range(5):
            if index == 0: # repression_threshold
                known_parameters = {k:all_parameters[k] for k in ('hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'translation_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_repression_ps11.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_repression_ps11.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_repression_ps11.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_repression_ps11.npy")
                x_values = np.linspace(0.5*np.mean(protein_at_observations[:,1]),1.5*np.mean(protein_at_observations[:,1]),2000)
                normal = np.trapz(np.exp(likelihood/2),x_values)
                plt.plot(x_values,20*np.exp(likelihood/2)/normal)
                import pdb; pdb.set_trace()
                _, bins, _ = plt.hist(mala,density=True,bins=30,color='#20948B',alpha=0.3,ec='black')
                plt.title("Repression Threshold Likelihood and MALA")
                plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                plt.savefig(saving_path + 'repression_likelihood_mala_ps11.png')
                plt.clf()

            elif index == 1: # hill_coefficient
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'translation_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_hill_ps11.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_hill_ps11.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_hill_ps11.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_hill_ps11.npy")
                x_values = np.linspace(2.0,6.0,2000)
                normal = np.trapz(np.exp(likelihood),x_values)
                plt.plot(x_values,np.exp(likelihood)/normal)
                _, bins, _ = plt.hist(mala,density=True,bins=30,color='#20948B',alpha=0.3,ec='black')
                plt.title("Hill Coefficient Likelihood and MALA")
                plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                plt.savefig(saving_path + 'hill_likelihood_mala_ps11.png')
                plt.clf()
            elif index == 2: # transcription_rate
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'translation_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_transcription_ps11.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_transcription_ps11.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_transcription_ps11.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_transcription_ps11.npy")
                x_values = np.linspace(0.01,60.0,2000)
                normal = np.trapz(np.exp(likelihood),x_values)
                plt.plot(x_values,np.exp(likelihood)/normal)
                _, bins, _ = plt.hist(np.exp(mala),density=True,bins=30,color='#20948B',alpha=0.3,ec='black')
                plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                plt.title("Transcription Likelihood and MALA")
                plt.savefig(saving_path + 'transcription_likelihood_mala_ps11.png')
                plt.clf()

            elif index == 3: # translation_rate
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_translation_ps11.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_translation_ps11.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_translation_ps11.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_translation_ps11.npy")
                x_values = np.linspace(np.log(0.01),np.log(40),2000)
                normal = np.trapz(np.exp(likelihood),np.exp(x_values))
                plt.plot(np.exp(x_values),np.exp(likelihood)/normal)
                _, bins, _ = plt.hist(np.exp(mala),density=True,bins=30,color='#20948B',alpha=0.3,ec='black')
                plt.title("Translation Likelihood and MALA")
                plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                plt.savefig(saving_path + 'translation_likelihood_mala_ps11.png')
                plt.clf()
            elif index == 4: # transcription_delay
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'translation_rate') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_delay_ps11.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_delay_ps11.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_delay_ps11.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_delay_ps11.npy")
                x_values = np.linspace(5.0,40.0,200)
                normal = np.trapz(np.exp(likelihood),x_values)
                plt.plot(x_values,np.exp(likelihood)/normal)
                _, bins, _ = plt.hist(mala,density=True,bins=30,color='#20948B',alpha=0.3,ec='black')
                plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                plt.title("Transcriptional Delay Likelihood and MALA")
                plt.savefig(saving_path + 'delay_likelihood_mala_ps11.png')
                plt.clf()

    def xest_kalman_mala_multiple_parameters(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')
        # specify data to use
        data_filename = 'protein_observations_360_ps3_ds4'
        protein_at_observations = np.load(saving_path + data_filename + '.npy')
        number_of_samples = 15000
        measurement_variance = 10000
        # true parameters ps3 -- [3407.99,5.17,np.log(2)/30,np.log(2)/90,15.86,1.27,30]
        initial_position = np.array([3407.99,5.17,np.log(2)/30,np.log(2)/90,np.log(15.86),np.log(1.27),30]) # ps3
        all_parameters = {'repression_threshold' : [0,3407.99],
                          'hill_coefficient' : [1,5.17],
                          'mRNA_degradation_rate' : [2,np.log(2)/30],
                          'protein_degradation_rate' : [3,np.log(2)/90],
                          'basal_transcription_rate' : [4,np.log(15.86)],
                          'translation_rate' : [5,np.log(1.27)],
                          'transcription_delay' : [6,30]}

        known_parameters = {k:all_parameters[k] for k in ('mRNA_degradation_rate',
                                                          'protein_degradation_rate') if k in all_parameters}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(initial_position)) if i not in known_parameter_indices]

        # if we already have mcmc samples, we can use them to construct a covariance matrix to make sampling better
        if os.path.exists(os.path.join(
                          os.path.dirname(__file__),
                          'output','mala_output_' + data_filename + '.npy')):

            print("Posterior samples already exist, sampling directly without warm up...")
            mala_output = np.load(saving_path + 'mala_output_' + data_filename + '.npy')
            number_of_posterior_samples = mala_output.shape[0]
            proposal_covariance = np.cov(mala_output[:int(number_of_posterior_samples/2),].T)
            step_size = 0.07
            mala = hes_inference.kalman_mala(protein_at_observations,
                                             measurement_variance,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             proposal_covariance,
                                             thinning_rate=1,
                                             known_parameter_dict=known_parameters)

            np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_' + data_filename + '_2d.npy'),
                    mala)

        else:
            print("New data set, warming up chain with " + str(number_of_samples) + " samples...")
            proposal_covariance = np.diag([5500.0,0.045,0.01,0.013,90.0])
            # proposal_covariance = np.diag([5500.0,0.045])
            step_size = 0.195
            mala = hes_inference.kalman_mala(protein_at_observations,
                                             measurement_variance,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             proposal_covariance,
                                             thinning_rate=1,
                                             known_parameter_dict=known_parameters)

            proposal_covariance = np.cov(mala[:int(number_of_samples/2),].T)
            initial_position[unknown_parameter_indices] = np.mean(mala[:int(number_of_samples/2),],axis=0)
            step_size = 0.32
            print("Warm up finished. Now sampling...")
            mala = hes_inference.kalman_mala(protein_at_observations,
                                             measurement_variance,
                                             number_of_samples,
                                             initial_position,
                                             step_size,
                                             proposal_covariance,
                                             thinning_rate=1,
                                             known_parameter_dict=known_parameters)

            np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_' + data_filename + '.npy'),
                    mala)

    def xest_plot_protein_observations(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        data = np.load(saving_path + 'true_data_ps11.npy')
        protein = np.load(saving_path + 'protein_observations_ps11_ds4.npy')

        my_figure, ax1 = plt.subplots(figsize=(7.5,4))
        ax1.scatter(data[:,0],data[:,2],marker='o',s=3,color='#20948B',alpha=0.75,label='protein')
        ax1.scatter(protein[:,0],protein[:,1],marker='o',s=10,c='#F18D9E')
        ax2 = ax1.twinx()
        ax2.scatter(data[0:-1:10,0],data[0:-1:10,1],marker='o',s=3,color='#86AC41',alpha=0.75,label='mRNA',zorder=1)
        ax1.tick_params(axis='y', labelcolor='#F18D9E')
        ax2.tick_params(axis='y', labelcolor='#86AC41')
        ax1.set_xlabel('Time (mins)')
        ax1.set_ylabel('Number of protein molecules')
        ax2.set_ylabel('Number of mRNA molecules')
        ax1.set_zorder(ax2.get_zorder()+1)
        ax1.patch.set_visible(False)
        plt.tight_layout()
        my_figure.savefig(os.path.join(saving_path,'ps11_data.pdf'))

    def xest_plot_mala_posteriors(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'data','')
        ps_string = 'ps9_ds2'
        true_parameters = np.load(loading_path + ps_string[:ps_string.find('_')] + '_parameter_values.npy')

        output = np.load(saving_path + 'parallel_mala_output_protein_observations_'+ps_string +'.npy')
        output = output.reshape(output.shape[0]*output.shape[1],output.shape[2])
        import pdb; pdb.set_trace()

        _, bins_transcription, _ = plt.hist(np.exp(output[:,2]),bins=30,density=True)
        logbins_transcription = np.geomspace(bins_transcription[0],
                                            bins_transcription[-1],
                                            len(bins_transcription))

        hist_translation, bins_translation, _ = plt.hist(np.exp(output[:,3]),bins=30,density=True)
        logbins_translation = np.geomspace(bins_translation[0],
                                          bins_translation[-1],
                                          len(bins_translation))

        mean_repression = round(np.mean(output[:,0]),-4) # round to nearest 10000

        my_figure = plt.figure(figsize=(15,4))
        # my_figure.text(.5,.005,'360 observations taken every 5 minutes',ha='center',fontsize=20)
        plt.subplot(1,5,1)
        sns.kdeplot(output[:,0])
        heights, bins, _ = plt.hist(output[:,0],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[0],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
        plt.xticks([0.5*mean_repression,1.5*mean_repression],labels=[int(0.5*mean_repression),int(1.5*mean_repression)])
        plt.title('Repression Threshold')
        # import pdb; pdb.set_trace()

        plt.subplot(1,5,2)
        sns.kdeplot(output[:,1],bw=0.4)
        heights, bins, _ = plt.hist(output[:,1],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[1],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
        plt.title('Hill Coefficient')

        plt.subplot(1,5,3)
        plt.xscale('log')
        sns.kdeplot(np.exp(output[:,2]),bw=0.1)
        heights, bins, _ = plt.hist(np.exp(output[:,2]),bins=logbins_transcription,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[4],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
        plt.xticks([.5,1,10],labels=[.5,1,10])
        plt.title('Transcription Rate')

        plt.subplot(1,5,4)
        plt.xscale('log')
        sns.kdeplot(np.exp(output[:,3]),bw=0.3)
        heights, bins, _ = plt.hist(np.exp(output[:,3]),bins=logbins_translation,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[5],0,1.1*max(heights),color='r',lw=2)
        # plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
        plt.xticks([10,50],labels=[10,50])
        plt.title('Translation Rate')

        plt.subplot(1,5,5)
        sns.kdeplot(output[:,4],bw=0.4)
        heights, bins, _ = plt.hist(output[:,4],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[6],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
        plt.title('Transcriptional Delay')

        plt.tight_layout()
        # plt.show()
        my_figure.savefig(os.path.join(saving_path,ps_string + '_posteriors_mala.pdf'))

    def xest_plot_mh_posteriors(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output','')
        loading_path             = os.path.join('/home/mbbx4jba/parallel_random_walks/')
        ps_string = 'ps2_ds2'
        mh_chains = np.array([np.load(loading_path + i) for i in os.listdir(loading_path) if ps_string in i])
        output = mh_chains[:,50000:,:].reshape(mh_chains.shape[0]*(mh_chains.shape[1]-50000),mh_chains.shape[2])
        true_parameters = np.load(loading_path + ps_string[:ps_string.find('_')] + '_parameter_values.npy')

        hist_transcription, bins_transcription, _ = plt.hist(np.exp(output[:,4]),bins=30,density=True)
        logbins_transcription = np.geomspace(bins_transcription[0],
                                             bins_transcription[-1],
                                             len(bins_transcription))
        plt.clf()

        hist_translation, bins_translation, _ = plt.hist(np.exp(output[:,5]),bins=30,density=True)
        logbins_translation = np.geomspace(bins_translation[0],
                                           bins_translation[-1],
                                           len(bins_translation))
        # import pdb; pdb.set_trace()

        plt.clf()

        my_figure = plt.figure(figsize=(15,4))
        # my_figure.text(.5,.005,'360 observations taken every 5 minutes',ha='center',fontsize=20)
        plt.subplot(1,5,1)
        sns.kdeplot(output[:,0])
        heights, bins, _ = plt.hist(output[:,0],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[0],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
        plt.title('Repression Threshold')
        # import pdb; pdb.set_trace()

        plt.subplot(1,5,2)
        sns.kdeplot(output[:,1],bw=0.4)
        heights, bins, _ = plt.hist(output[:,1],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[1],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
        plt.title('Hill Coefficient')

        plt.subplot(1,5,3)
        plt.xscale('log')
        sns.kdeplot(np.exp(output[:,4]),bw=0.1)
        heights, bins, _ = plt.hist(np.exp(output[:,4]),bins=logbins_transcription,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[2],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
        plt.xticks([1,10,50],labels=[1,10,50])
        plt.title('Transcription Rate')

        plt.subplot(1,5,4)
        plt.xscale('log')
        sns.kdeplot(np.exp(output[:,5]))
        heights, bins, _ = plt.hist(np.exp(output[:,5]),bins=logbins_translation,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[3],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
        plt.xticks([10,50],labels=[10,50])
        plt.title('Translation Rate')

        plt.subplot(1,5,5)
        sns.kdeplot(output[:,6],bw=0.4)
        heights, bins, _ = plt.hist(output[:,6],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[4],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
        plt.title('Transcriptional Delay')

        plt.tight_layout()
        # plt.show()
        my_figure.savefig(os.path.join(saving_path,ps_string + '_posteriors_mh.pdf'))

    def xest_plot_experimental_posteriors(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'output','')
        experiment_date = '280317p1'
        chains = np.array([np.load(loading_path + i) for i in os.listdir(loading_path) if 'parallel_mala_output_'+experiment_date in i
                                                                                       if 'npy' in i])
        chain_strings = np.array([i for i in os.listdir(loading_path) if 'parallel_mala_output_'+experiment_date in i
                                                                                       if 'npy' in i])
        import pdb; pdb.set_trace()
        for index, chain in enumerate(chains):
            filename = chain_strings[index][chain_strings[index].find(experiment_date):chain_strings[index].find('.npy')] + '_posterior.pdf'

            output = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])
            # import pdb; pdb.set_trace()

            hist_transcription, bins_transcription, _ = plt.hist(np.exp(output[:,2]),bins=30,density=True)
            logbins_transcription = np.geomspace(bins_transcription[0],
                                                 bins_transcription[-1],
                                                 len(bins_transcription))
            plt.clf()

            hist_translation, bins_translation, _ = plt.hist(np.exp(output[:,3]),bins=30,density=True)
            logbins_translation = np.geomspace(bins_translation[0],
                                               bins_translation[-1],
                                               len(bins_translation))
            # import pdb; pdb.set_trace()
            plt.clf()

            my_figure = plt.figure(figsize=(15,4))
            plt.subplot(1,5,3)
            # sns.kdeplot(output[:,0])
            heights, bins, _ = plt.hist(output[:,0],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
            plt.title('Repression Threshold')

            plt.subplot(1,5,5)
            # sns.kdeplot(output[:,1],bw=0.4)
            heights, bins, _ = plt.hist(output[:,1],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
            plt.title('Hill Coefficient')

            plt.subplot(1,5,1)
            plt.xscale('log')
            # sns.kdeplot(np.exp(output[:,2]))
            heights, bins, _ = plt.hist(np.exp(output[:,2]),bins=logbins_transcription,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
            plt.xticks([1,10,50],labels=[1,10,50])
            plt.title('Transcription Rate')

            plt.subplot(1,5,2)
            plt.xscale('log')
            # sns.kdeplot(np.exp(output[:,3]))
            heights, bins, _ = plt.hist(np.exp(output[:,3]),bins=logbins_translation,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
            # plt.xticks([10,50],labels=[10,50])
            plt.title('Translation Rate')

            plt.subplot(1,5,4)
            # sns.kdeplot(output[:,4],bw=0.4)
            heights, bins, _ = plt.hist(output[:,4],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
            plt.title('Transcriptional Delay')

            plt.tight_layout()
            my_figure.savefig(os.path.join(saving_path,filename))

    def xest_compare_mala_random_walk_histograms(self):
        saving_path  = os.path.join(os.path.dirname(__file__), 'output','')
        mala = np.load(saving_path + 'mala_output_hill.npy')
        random_walk = np.load(saving_path + 'random_walk_hill.npy')

        my_figure = plt.figure(figsize=(12,6))
        _,bins,_ = plt.hist(mala,density=True,bins=30,alpha=0.8,color='#20948B',label='MALA')
        sns.kdeplot(mala[:,0],bw=0.35,color='#20948B')
        plt.vlines(np.mean(mala[:,0]),0,4.1,color='#20948B')
        plt.hist(random_walk[:,1],density=True,bins=30,alpha=0.6,color='#F18D9E',label='Random Walk')
        sns.kdeplot(random_walk[:,1],bw=0.35,color='#F18D9E')
        plt.vlines(np.mean(random_walk[:,1]),0,4.1,color='#F18D9E')
        plt.vlines(5.17,0,4.1,color='k',label='True Mean')
        plt.title("Hill Coefficient (Random Walk and MALA)")
        plt.legend()
        plt.xlim(xmin=2*bins[0]-bins[2],xmax=2*bins[-1]-bins[-3])
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','algo_comparison_hill.png'))

    def xest_compare_mala_random_walk_autocorrelation(self):
        saving_path  = os.path.join(os.path.dirname(__file__), 'output','')
        mala_hill = np.load(saving_path + 'mala_output_hill.npy')
        random_walk_hill = np.load(saving_path + 'random_walk_hill.npy')
        m
        ala_repression = np.load(saving_path + 'mala_output_repression.npy')
        random_walk_repression = np.load(saving_path + 'random_walk_repression.npy')
        numlags = 20

        mala_repression_lagtime = em.autocorr.integrated_time(mala_repression[:,0])
        random_walk_repression_lagtime = em.autocorr.integrated_time(random_walk_repression[:,0])
        mala_hill_lagtime = em.autocorr.integrated_time(mala_hill[:,0])
        random_walk_hill_lagtime = em.autocorr.integrated_time(random_walk_hill[:,1])

        my_figure, ax = plt.subplots(2,2,figsize=(10,5))
        ax[0,0].acorr(mala_repression[:,0] - np.mean(mala_repression[:,0]),maxlags=numlags,color='#20948B',label='MALA',lw=2)
        # import pdb; pdb.set_trace()
        ax[0,0].set_xlim(xmin=-0.05,xmax=numlags)
        ax[0,0].set_ylabel('Repression Threshold \n autocorrelation')
        ax[0,0].set_xlabel('Lags')
        ax[0,0].set_title('MALA')
        ax[0,0].text(14.0,0.7,'$\\tau =$ ' + str(round(mala_repression_lagtime[0],2)))
        ax[0,1].acorr(random_walk_repression[:,0] - np.mean(random_walk_repression[:,0]),maxlags=numlags,color='#F18D9E',label='MH',lw=2)
        ax[0,1].set_xlim(xmin=-0.05,xmax=numlags)
        ax[0,1].set_title('MH')
        ax[0,1].text(14.0,0.7,'$\\tau =$ ' + str(round(random_walk_repression_lagtime[0],2)))
        ax[0,1].set_xlabel('Lags')
        ax[1,0].acorr(mala_hill[:,0] - np.mean(mala_hill[:,0]),maxlags=numlags,color='#20948B',lw=2)
        ax[1,0].set_xlim(xmin=-0.05,xmax=numlags)
        ax[1,0].set_xlabel('Lags')
        ax[1,0].set_ylabel('Hill Coefficient \n autocorrelation')
        ax[1,0].text(14.0,0.7,'$\\tau =$ ' + str(round(mala_hill_lagtime[0],2)))
        ax[1,1].acorr(random_walk_hill[:,1] - np.mean(random_walk_hill[:,1]),maxlags=numlags,color='#F18D9E',lw=2)
        ax[1,1].set_xlabel('Lags')
        ax[1,1].text(14.0,0.7,'$\\tau =$ ' + str(round(random_walk_hill_lagtime[0],2)))
        ax[1,1].set_xlim(xmin=-0.05,xmax=numlags)
        plt.tight_layout()

        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','autocorrelation_plot.png'))

    def xest_compare_mala_random_walk_2d_autocorrelation(self):
        saving_path  = os.path.join(os.path.dirname(__file__), 'output','')
        mala = np.load(saving_path + 'mala_output_protein_observations_360_ps3_ds4_2d.npy')
        random_walk = np.load(saving_path + 'random_walk_hill_repression.npy')
        numlags = 100

        mala_repression_lagtime = em.autocorr.integrated_time(mala[:,0])
        random_walk_repression_lagtime = em.autocorr.integrated_time(random_walk[:,0],quiet=True)
        mala_hill_lagtime = em.autocorr.integrated_time(mala[:,1])
        random_walk_hill_lagtime = em.autocorr.integrated_time(random_walk[:,1],quiet=True)

        my_figure, ax = plt.subplots(2,2,figsize=(10,5))
        ax[0,0].acorr(mala[:,0] - np.mean(mala[:,0]),maxlags=numlags,color='#20948B',label='MALA',lw=2)
        ax[0,0].set_xlim(xmin=-0.05,xmax=numlags)
        ax[0,0].set_ylabel('Repression Threshold \n autocorrelation')
        ax[0,0].set_xlabel('Lags')
        ax[0,0].set_title('MALA')
        ax[0,0].text(70.0,0.7,'$\\tau =$ ' + str(round(mala_repression_lagtime[0],2)))
        ax[0,1].acorr(random_walk[:,0] - np.mean(random_walk[:,0]),maxlags=numlags,color='#F18D9E',label='Random Walk',lw=2)
        ax[0,1].set_xlim(xmin=-0.05,xmax=numlags)
        ax[0,1].set_title('MH')
        ax[0,1].text(70.0,0.7,'$\\tau =$ ' + str(round(random_walk_repression_lagtime[0],2)))
        ax[0,1].set_xlabel('Lags')
        ax[1,0].acorr(mala[:,1] - np.mean(mala[:,1]),maxlags=numlags,color='#20948B',lw=2)
        ax[1,0].set_xlim(xmin=-0.05,xmax=numlags)
        ax[1,0].set_xlabel('Lags')
        ax[1,0].set_ylabel('Hill Coefficient \n autocorrelation')
        ax[1,0].text(70.0,0.7,'$\\tau =$ ' + str(round(mala_hill_lagtime[0],2)))
        ax[1,1].acorr(random_walk[:,1] - np.mean(random_walk[:,1]),maxlags=numlags,color='#F18D9E',lw=2)
        ax[1,1].set_xlabel('Lags')
        ax[1,1].text(70.0,0.7,'$\\tau =$ ' + str(round(random_walk_hill_lagtime[0],2)))
        ax[1,1].set_xlim(xmin=-0.05,xmax=numlags)
        plt.tight_layout()

        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','autocorrelation_plot_2d.png'))

        plt.clf()


    def xest_plot_mala(self):
        saving_path  = os.path.join(os.path.dirname(__file__), 'output','mala_output_protein_observations_360_ps3_ds4_2d')
        output = np.load(saving_path + '.npy')

        # parameters = np.array([3407.99,5.17,np.log(2)/30,np.log(2)/90,np.log(15.86),np.log(1.27),30]) # ps3

        # fig, ax = plt.subplots(output.shape[1],1,figsize=(8,12))
        # for i in range(output.shape[1]):
        #     if i in [4,5]:
        #         ax[i].semilogy(np.exp(output[:,i]))
        #         ax[i].set_xlabel('$\\theta_{}$'.format(i))
        #         ax[i].hlines(np.exp(parameters[i]),0,50000,color='r')
        #     else:
        #         ax[i].plot(output[:,i])
        #         ax[i].hlines(parameters[i],0,50000,color='r')
        #         ax[i].set_xlabel('$\\theta_{}$'.format(i))
        # plt.tight_layout()
        # fig.savefig(os.path.join(os.path.dirname(__file__),
        #                                'output','mala_traceplots.pdf'))
        g = sns.PairGrid(pd.DataFrame(output[:,[0,1]],columns=['Repression\nThreshold',
                                                      'Hill\nCoefficient']),diag_sharey=False)
        g = g.map_upper(sns.scatterplot,size=2,color='#20948B')
        g = g.map_lower(sns.kdeplot,color="#20948B",shade=True,shade_lowest=False)
        g = g.map_diag(sns.distplot,color='#20948B')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                 'output','pair_grid_mala_2d.png'))


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

    def xest_generate_multiple_protein_observations(self):
        loading_path = os.path.join(os.path.dirname(__file__), 'data','')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')
        ps_string = "ps5"
        parameters = np.load(loading_path + ps_string + "_parameter_values.npy")
        observation_duration  = 1800
        observation_frequency = 5
        no_of_observations    = np.int(observation_duration/observation_frequency)

        true_data = hes5.generate_langevin_trajectory(duration = observation_duration,
                                                      repression_threshold = parameters[0],
                                                      hill_coefficient = parameters[1],
                                                      mRNA_degradation_rate = parameters[2],
                                                      protein_degradation_rate = parameters[3],
                                                      basal_transcription_rate = parameters[4],
                                                      translation_rate = parameters[5],
                                                      transcription_delay = parameters[6],
                                                      equilibration_time = 1000)
        # np.save(loading_path + 'true_data_' + ps_string + '.npy',
        #         true_data)

        ## the F constant matrix is left out for now
        protein_at_observations = true_data[:,(0,2)]
        protein_at_observations[:,1] += np.random.randn(true_data.shape[0])*100
        protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)
        # np.save(loading_path + 'protein_observations_90_' + ps_string + '_ds1.npy',
        #             protein_at_observations[0:900:10,:])
        # np.save(loading_path + 'protein_observations_180_' + ps_string + '_ds2.npy',
        #             protein_at_observations[0:900:5,:])
        # np.save(loading_path + 'protein_observations_180_' + ps_string + '_ds3.npy',
        #             protein_at_observations[0:1800:10,:])
        # np.save(loading_path + 'protein_observations_360_' + ps_string + '_ds4.npy',
        #             protein_at_observations[0:1800:5,:])

        my_figure = plt.figure()
        plt.scatter(np.arange(0,1800),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations')
        plt.title('Protein Observations')
        plt.xlabel('Time')
        plt.ylabel('Protein Copy Numbers')
        plt.tight_layout()
        # my_figure.savefig(saving_path + 'protein_observations_' + ps_string + '.pdf')

    def xest_generate_multiple_protein_observations(self):
        loading_path = os.path.join(os.path.dirname(__file__), 'data','')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')
        ps_strings = ["ps6","ps7","ps8","ps9","ps10","ps11","ps12"]
        for ps_string in ps_strings:
            parameters = np.load(loading_path + ps_string + "_parameter_values.npy")
            observation_duration  = 736
            observation_frequency = 15
            no_of_observations    = np.int(observation_duration/observation_frequency)

            for i in range(6):
                true_data = hes5.generate_langevin_trajectory(duration = observation_duration,
                                                              repression_threshold = parameters[0],
                                                              hill_coefficient = parameters[1],
                                                              mRNA_degradation_rate = parameters[2],
                                                              protein_degradation_rate = parameters[3],
                                                              basal_transcription_rate = parameters[4],
                                                              translation_rate = parameters[5],
                                                              transcription_delay = parameters[6],
                                                              equilibration_time = 1000)

                ## the F constant matrix is left out for now
                protein_at_observations = true_data[:,(0,2)]
                protein_at_observations[:,1] += np.random.randn(true_data.shape[0])*parameters[-1]
                protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)
                np.save(loading_path + 'protein_observations_' + ps_string + '_fig5_{i}.npy'.format(i=i+1),
                protein_at_observations[0::15,:])
                # np.save(loading_path + 'protein_observations_180_' + ps_string + '_ds2.npy',
                #             protein_at_observations[0:900:5,:])
                # np.save(loading_path + 'protein_observations_180_' + ps_string + '_ds3.npy',
                #             protein_at_observations[0:1800:10,:])
                # np.save(loading_path + 'protein_observations_360_' + ps_string + '_ds4.npy',
                #             protein_at_observations[0:1800:5,:])

    def xest_kalman_filter(self):
        ## run a sample simulation to generate example protein data
        true_data = hes5.generate_langevin_trajectory(duration = 900, equilibration_time = 1000)
        np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_trace_true_data.npy'),
                    true_data)

        ## the F constant matrix is left out for now
        protein_at_observations = true_data[0:900:10,(0,2)]
        protein_at_observations[:,1] += np.random.randn(90)*100
        protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)
        np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_trace_observations.npy'),
                    protein_at_observations)

        parameters = np.array([10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0])

        ## apply kalman filter to the data
        state_space_mean, state_space_variance,_,_, predicted_observation_distributions,_,_ = hes_inference.kalman_filter(protein_at_observations,parameters,
                                                                                                                  measurement_variance=10000)
        np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_filter_mean.npy'),state_space_mean)
        np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_filter_variance.npy'),state_space_variance)
        np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_filter_distributions.npy'),predicted_observation_distributions)

        # check dimensionality of state_space_mean and the state_space_variance
        self.assertEqual(state_space_mean.shape[0],920)
        self.assertEqual(state_space_mean.shape[1],3)
        self.assertEqual(state_space_variance.shape[0],1840)
        self.assertEqual(state_space_variance.shape[1],1840)

        # variance needs to be positive definite and symmetric, maybe include quantitative check
        np.testing.assert_almost_equal(state_space_variance,state_space_variance.transpose())
        # this tests that the diagonal entries (variances) are all positive
        self.assertEqual(np.sum(np.diag(state_space_variance)>0),1840)
        ##plot data together with state-space distribution
        number_of_states = state_space_mean.shape[0]

        protein_covariance_matrix = state_space_variance[number_of_states:,number_of_states:]
        protein_variance = np.diagonal(protein_covariance_matrix)
        protein_error = np.sqrt(protein_variance)*2

        mRNA_covariance_matrix = state_space_variance[:number_of_states,:number_of_states]
        mRNA_variance = np.diagonal(mRNA_covariance_matrix)
        mRNA_error = np.sqrt(mRNA_variance)*2

        # two plots, first is only protein observations, second is true protein and predicted protein
        # with 95% confidence intervals
        my_figure = plt.figure()
        plt.subplot(2,1,1)
        plt.scatter(np.arange(0,900,10),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations')
        plt.title('Protein Observations')
        plt.xlabel('Time')
        plt.ylabel('Protein Copy Numbers')

        plt.subplot(2,1,2)
        plt.scatter(np.arange(0,900,10),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations',zorder=4)
        plt.plot(true_data[:,0],true_data[:,2],label='true protein',color='#F69454',linewidth=0.89,zorder=3)
        plt.plot(state_space_mean[:,0],state_space_mean[:,2],label='inferred protein',color='#20948B',zorder=2)
        plt.errorbar(state_space_mean[:,0],state_space_mean[:,2],yerr=protein_error,ecolor='#98DBC6',alpha=0.1,zorder=1)
        plt.legend(fontsize='x-small')
        plt.title('Predicted Protein')
        plt.xlabel('Time')
        plt.ylabel('Protein Copy Numbers')
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','kalman_test_protein.pdf'))

        # one plot with protein observations and likelihood
        my_figure = plt.figure()
        plt.scatter(np.arange(0,900,10),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations',zorder=3)
        plt.scatter(np.arange(0,900,10),predicted_observation_distributions[:,1],marker='o',s=4,c='#98DBC6',label='likelihood',zorder=2)
        plt.errorbar(predicted_observation_distributions[:,0],predicted_observation_distributions[:,1],
                     yerr=np.sqrt(predicted_observation_distributions[:,2]),ecolor='#98DBC6',alpha=0.6,linestyle="None",zorder=1)
        plt.legend(fontsize='x-small')
        plt.title('Protein likelihood')
        plt.xlabel('Time')
        plt.ylabel('Protein copy number')
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','kalman_likelihood_vs_observations.pdf'))


        # two plots, first is only protein observations, second is true mRNA and predicted mRNA
        # with 95% confidence intervals
        my_figure = plt.figure()
        plt.subplot(2,1,1)
        plt.scatter(np.arange(0,900,10),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations')
        plt.title('Protein Observations')
        plt.xlabel('Time')
        plt.ylabel('Protein Copy Numbers')

        plt.subplot(2,1,2)
        plt.plot(true_data[:,0],true_data[:,1],label='true mRNA',linewidth=0.89,zorder=3)
        plt.plot(state_space_mean[:,0],state_space_mean[:,1],label='inferred mRNA',color='#86AC41',zorder=2)
        plt.errorbar(state_space_mean[:,0],state_space_mean[:,1],yerr=mRNA_error,ecolor='#98DBC6',alpha=0.1,zorder=1)
        plt.legend(fontsize='x-small')
        plt.title('Predicted mRNA')
        plt.xlabel('Time')
        plt.ylabel('mRNA Copy Numbers')
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','kalman_test_mRNA.pdf'))

        # two plots, first is true protein and predicted protein with 95% confidence intervals,
        # second is true mRNA and predicted mRNA with 95% confidence intervals,
        my_figure = plt.figure()
        plt.subplot(2,1,1)
        plt.scatter(np.arange(0,900,10),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations',zorder=4)
        plt.plot(true_data[:,0],true_data[:,2],label='true protein',color='#F69454',linewidth=0.89,zorder=3)
        plt.plot(state_space_mean[:,0],state_space_mean[:,2],label='inferred protein',color='#20948B',zorder=2)
        plt.errorbar(state_space_mean[:,0],state_space_mean[:,2],yerr=protein_error,ecolor='#98DBC6',alpha=0.1,zorder=1)
        plt.legend(fontsize='x-small')
        plt.title('Predicted Protein')
        plt.xlabel('Time')
        plt.ylabel('Protein Copy Numbers')

        plt.subplot(2,1,2)
        plt.plot(true_data[:,0],true_data[:,1],label='true mRNA',linewidth=0.89,zorder=3)
        plt.plot(state_space_mean[:,0],state_space_mean[:,1],label='inferred mRNA',color='#86AC41',zorder=2)
        plt.errorbar(state_space_mean[:,0],state_space_mean[:,1],yerr=mRNA_error,ecolor='#98DBC6',alpha=0.1,zorder=1)
        plt.legend(fontsize='x-small')
        plt.title('Predicted mRNA')
        plt.xlabel('Time')
        plt.ylabel('mRNA Copy Numbers')
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','kalman_test_protein_vs_mRNA.pdf'))

        my_figure = plt.figure(figsize=(10,7))
        plt.subplot(2,1,1)
        plt.scatter(np.arange(0,900,10),protein_at_observations[:,1],marker='o',s=14,c='#F18D9E',label='Protein observations',zorder=4)
        # plt.plot(true_data[:,0],true_data[:,2],label='true protein',color='#F69454',linewidth=0.89,zorder=3)
        # plt.plot(state_space_mean[:,0],state_space_mean[:,2],label='inferred protein',color='#20948B',zorder=2)
        # plt.errorbar(state_space_mean[:,0],state_space_mean[:,2],yerr=protein_error,ecolor='#98DBC6',alpha=0.1,zorder=1)
        # plt.legend(fontsize='x-small')
        plt.title('Protein')
        plt.xlabel('Time')
        plt.ylabel('Protein Copy Numbers')

        true_mRNA_at_observations = true_data[0:900:10,(0,1)]
        mRNA_at_observations = np.copy(true_mRNA_at_observations)
        mRNA_at_observations[:,1] += np.random.randn(90)*np.mean(true_mRNA_at_observations[:,1],axis=0)*0.1
        mRNA_at_observations[:,1] = np.maximum(mRNA_at_observations[:,1],0)

        plt.subplot(2,1,2)
        plt.scatter(mRNA_at_observations[:,0],mRNA_at_observations[:,1],label='mRNA observations',marker='o',s=14,c='#86AC41',zorder=3)
        # plt.plot(state_space_mean[:,0],state_space_mean[:,1],label='inferred mRNA',color='#86AC41',zorder=2)
        # plt.errorbar(state_space_mean[:,0],state_space_mean[:,1],yerr=mRNA_error,ecolor='#98DBC6',alpha=0.1,zorder=1)
        # plt.legend(fontsize='x-small')
        plt.title('mRNA')
        plt.xlabel('Time')
        plt.ylabel('mRNA Copy Numbers')
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','kalman_true_protein_and_mRNA.pdf'))

        #print(predicted_observation_distributions)

    def xest_insilico_data_generation(self):
        ## run a sample simulation to generate example protein data
        in_silico_data = hes5.generate_langevin_trajectory(duration = 900, equilibration_time = 1000)

        ## the F constant matrix is left out for now
        true_protein_at_observations = in_silico_data[0:900:10,(0,2)]
        protein_at_observations = np.copy(true_protein_at_observations)
        protein_at_observations[:,1] += np.random.randn(90)*100
        protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)

        my_figure = plt.figure(figsize=(10,10),fontsize=100)
        plt.subplot(3,1,1)
        plt.plot(in_silico_data[:,0],in_silico_data[:,2],c='#F18D9E',label='Protein')
        plt.title('True Protein time course')
        plt.xlabel('Time')
        plt.ylabel('Molecule Copy Numbers')

        plt.subplot(3,1,2)
        plt.scatter(np.arange(0,900,10),true_protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='true value')
        plt.title('True Protein')
        plt.xlabel('Time')
        plt.ylabel('Molecule Copy Numbers')

        plt.subplot(3,1,3)
        plt.scatter(np.arange(0,900,10),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations',zorder=4)
        plt.title('Observed Protein')
        plt.xlabel('Time')
        plt.ylabel('Protein Copy Numbers')
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','in_silico_data.pdf'))



    def xest_get_likelihood_at_parameters(self):

        true_data = hes5.generate_langevin_trajectory(duration = 900, equilibration_time = 1000)

        protein_at_observations = true_data[0:900:10,(0,2)]
        protein_at_observations[:,1] += np.random.randn(90)*100
        protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)

        parameters = [10000,5,np.log(2)/30, np.log(2)/90, 1, 1, 29]
        #parameters2 = [10000,5,np.log(2)/30, np.log(2)/90, 2, 7, 29]

        likelihood = hes_inference.calculate_log_likelihood_at_parameter_point(protein_at_observations,parameters,measurement_variance = 10000)
        #likelihood2 = hes_inference.calculate_log_likelihood_at_parameter_point(protein_at_observations,parameters2,measurement_variance = 10000)
        print(likelihood)
        #print(likelihood2)
        #print(np.exp(likelihood2/likelihood))

    def xest_kalman_random_walk_for_profiling(self):

        true_data = hes5.generate_langevin_trajectory(duration = 900, equilibration_time = 1000)

        saving_path  = os.path.join(os.path.dirname(__file__), 'data','random_walk')
        previous_run = np.load(saving_path + '.npy')

        #true_values = [10000,5,np.log(2)/30,np.log(2)/90,1,1,29]
        protein_at_observations = true_data[0:900:10,(0,2)]
        protein_at_observations[:,1] += np.random.randn(90)*100
        protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)

        hyper_parameters = np.array([5,2000,2,2.5,5,0.1,5,0.1,3,0.333,3,0.333,3,10])
        measurement_variance = 10000
        iterations = 5
        #initial_state = np.array([np.mean(previous_run[1000:,0]),np.mean(previous_run[1000:,1]),
        #                          np.mean(previous_run[1000:,2]),np.mean(previous_run[1000:,3]),
        #                          np.mean(previous_run[1000:,4]),np.mean(previous_run[1000:,5]),
        #                          np.mean(previous_run[1000:,6])])
        #covariance = np.cov(previous_run.T)
        initial_state = np.array([8000,5,0.1,0.1,1,1,10])
        covariance = np.diag([100000000,16,0.01,0.02,2,2,50])

        time_before_call = time.time()
        random_walk, acceptance_rate = hes_inference.kalman_random_walk(iterations,protein_at_observations,hyper_parameters,measurement_variance,0.08,covariance,initial_state,adaptive='false')
        time_after_call = time.time()
        time_passed = time_after_call - time_before_call
        print('\ntime used in random walk (originally roughly 9s, now roughly 5s):')
        print(time_passed)

        print('\nrandom walk and acceptance rate')
        print(random_walk)
        print(acceptance_rate)

    def test_kalman_random_walk(self,data_filename='protein_observations_ps11_ds4.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        protein_at_observations = np.load(os.path.join(saving_path,data_filename))
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('_ds')
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        true_parameter_values = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))

        mean_protein = np.mean(protein_at_observations[:,1])

        iterations = 100000
        number_of_chains = 8
        step_size = 0.1
        measurement_variance = np.power(true_parameter_values[-1],2)
        # draw random initial states for the parallel chains
        from scipy.stats import uniform
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
        for initial_state_index, _ in enumerate(initial_states):
            initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                          np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))
        # true parameters ps3 -- [3407.99,5.17,np.log(2)/30,np.log(2)/90,15.86,1.27,30]
        hyper_parameters = np.array([0,2*mean_protein,2,4,0,1,0,1,0.01,60-0.01,0.01,40-0.01,5,35]) # uniform
        covariance = np.diag([5e+3,0.03,0.01,0.01,1.0])

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_random_walk,
                                                          args=(iterations,
                                                                protein_at_observations,
                                                                hyper_parameters,
                                                                measurement_variance,
                                                                step_size,
                                                                covariance,
                                                                initial_state))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,iterations,5))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','mh_output_' + data_filename),array_of_chains)

    def xest_compute_likelihood_in_parallel(self):

        saving_path             = os.path.join(os.path.dirname(__file__), 'data','kalman_test_trace')
        protein_at_observations = np.load(saving_path + '_observations.npy')
        likelihood_at_multiple_parameters = np.zeros((10,100,10,10,30))

        mRNA_degradation_rate    = np.log(2)/30
        protein_degradation_rate = np.log(2)/90

        pool_of_processes = mp.Pool(processes = number_of_cpus)
        process_list = []
        # hyper_parameters = np.array([100,20100,2,4,0,1,0,1,np.log10(0.1),1+np.log10(65),np.log10(0.1),1+np.log10(45),4,36])
        for repression_index, repression_threshold in enumerate(np.linspace(100,20100,10)):
            for hill_index, hill_coefficient in enumerate(np.linspace(2,6,100)):
                for basal_index, basal_transcription_rate in enumerate(np.linspace(-1,np.log10(60),10)):
                    for translation_index, translation_rate in enumerate(np.linspace(-1,np.log10(40),10)):
                        for transcription_index, transcription_delay in enumerate(np.linspace(5,40,30)):
                            process_list.append(pool_of_processes.apply_async(hes_inference.calculate_log_likelihood_at_parameter_point,
                                                                              args=(protein_at_observations,
                                                                                    np.array([repression_threshold,
                                                                                              hill_coefficient,
                                                                                              mRNA_degradation_rate,
                                                                                              protein_degradation_rate,
                                                                                              np.power(10,basal_transcription_rate),
                                                                                              np.power(10,translation_rate),
                                                                                              transcription_delay]),
                                                                                    10000)))
        # hyper_parameters = np.array([100,20100,2,4,0,1,0,1,np.log10(0.1),1+np.log10(65),np.log10(0.1),1+np.log10(45),4,36])
        process_index = 0
        for repression_index, repression_threshold in enumerate(np.linspace(100,20100,10)):
            for hill_index, hill_coefficient in enumerate(np.linspace(2,6,100)):
                for basal_index, basal_transcription_rate in enumerate(np.linspace(-1,np.log10(60),10)):
                    for translation_index, translation_rate in enumerate(np.linspace(-1,np.log10(40),10)):
                        for transcription_index, transcription_delay in enumerate(np.linspace(5,40,30)):
                            likelihood_at_multiple_parameters[repression_index,hill_index,basal_index,translation_index,transcription_index] = process_list[process_index].get()
                            process_index +=1

        np.save(os.path.join(os.path.dirname(__file__), 'output','likelihood_at_multiple_parameters_test.npy'),likelihood_at_multiple_parameters)
        pool_of_processes.close()
        pool_of_processes.join()

    def xest_compute_likelihood_at_multiple_parameters(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        protein_at_observations = np.load(saving_path + '/protein_observations_ps11_ds1.npy')
        true_parameter_values = np.load(saving_path + 'ps11_parameter_values.npy')
        number_of_evaluations = 2000
        likelihood_at_multiple_parameters = np.zeros(number_of_evaluations)
        mean_protein = np.mean(protein_at_observations[:,1])
        print(mean_protein)

        repression_threshold = true_parameter_values[0]
        hill_coefficient = true_parameter_values[1]
        mRNA_degradation_rate    = np.log(2)/30
        protein_degradation_rate = np.log(2)/90
        basal_transcription_rate = true_parameter_values[4]
        translation_rate = true_parameter_values[5]
        transcription_delay = true_parameter_values[6]
        measurement_variance = np.power(true_parameter_values[-1],2)

        for index, parameter in enumerate(np.linspace(0.5*mean_protein,1.5*mean_protein,number_of_evaluations)):
            likelihood_at_multiple_parameters[index] = hes_inference.calculate_log_likelihood_at_parameter_point(protein_at_observations,
                                                                                                                 model_parameters=np.array([parameter,
                                                                                                                                            hill_coefficient,
                                                                                                                                            mRNA_degradation_rate,
                                                                                                                                            protein_degradation_rate,
                                                                                                                                            basal_transcription_rate,
                                                                                                                                            translation_rate,
                                                                                                                                            transcription_delay]),
                                                                                                                 measurement_variance = measurement_variance)

        np.save(os.path.join(os.path.dirname(__file__), 'output','likelihood_repression_ps11.npy'),likelihood_at_multiple_parameters)

    def xest_multiple_random_walk_traces_in_parallel(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        protein_at_observations = np.load(saving_path + 'kalman_trace_observations_180_ps2_ds1.npy')
        previous_random_walk    = np.load(saving_path + 'full_random_walk_180_ps2_ds1.npy')

        hyper_parameters = np.array([100,20100,2,4,0,1,0,1,np.log10(0.1),np.log10(60)+1,np.log10(0.1),np.log10(40)+1,5,35]) # uniform
        measurement_variance = 10000.0

        covariance     = np.diag(np.array([np.var(previous_random_walk[:,0]),np.var(previous_random_walk[:,1]),
                                           0,                                0,
                                           np.var(previous_random_walk[:,2]),np.var(previous_random_walk[:,3]),
                                           np.var(previous_random_walk[:,4])]))
        # draw 8 random initial states for the parallel random walk
        from scipy.stats import uniform
        initial_states          = np.zeros((8,7))
        initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
        for initial_state_index in initial_states:
            initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([100,2,np.log10(0.1),np.log10(0.1),5]),
                        np.array([20100,4,np.log10(60)+1,np.log10(40)+1,35]))

        number_of_iterations = 25000

        pool_of_processes = mp_pool.ThreadPool(processes = 8)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_random_walk,
                                                          args=(number_of_iterations,protein_at_observations,hyper_parameters,measurement_variance,0.15,covariance,initial_state))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        list_of_random_walks = []
        list_of_acceptance_rates = []
        for process_result in process_results:
            this_random_walk, this_acceptance_rate = process_result.get()
            list_of_random_walks.append(this_random_walk)
            list_of_acceptance_rates.append(this_acceptance_rate)
        pool_of_processes.join()
        print(list_of_acceptance_rates)

        for i in range(len(initial_states)):
            np.save(os.path.join(os.path.dirname(__file__), 'output','parallel_random_walk__180_ps2_ds1_{cap}.npy').format(cap=i),list_of_random_walks[i])

        #array_of_random_walks = np.array(list_of_random_walks)
        #self.assertEqual(array_of_random_walks.shape[0], len(initial_states))
        #self.assertEqual(array_of_random_walks.shape[1], number_of_iterations)

    def xest_multiple_mala_traces_figure_5(self,data_filename = 'protein_observations_ps6_fig5_1.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        protein_at_observations = np.array([np.load(os.path.join(saving_path,data_filename))])
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('_fig')
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        true_parameter_values = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))

        mean_protein = np.mean(protein_at_observations[:,1])

        number_of_samples = 80000
        number_of_chains = 8
        measurement_variance = np.power(true_parameter_values[-1],2)
        # draw random initial states for the parallel chains
        from scipy.stats import uniform
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([true_parameter_values[2],true_parameter_values[3]])
        for initial_state_index, _ in enumerate(initial_states):
            initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                          np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))

        # define known parameters
        all_parameters = {'repression_threshold' : [0,true_parameter_values[0]],
                          'hill_coefficient' : [1,true_parameter_values[1]],
                          'mRNA_degradation_rate' : [2,true_parameter_values[2]],
                          'protein_degradation_rate' : [3,true_parameter_values[3]],
                          'basal_transcription_rate' : [4,np.log(true_parameter_values[4])],
                          'translation_rate' : [5,np.log(true_parameter_values[5])],
                          'transcription_delay' : [6,true_parameter_values[6]]}

        known_parameters = {k:all_parameters[k] for k in ('mRNA_degradation_rate',
                                                          'protein_degradation_rate') if k in all_parameters}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(initial_states[0])) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        # if we already have mcmc samples, we can use them to construct a covariance matrix to make sampling better
        if os.path.exists(os.path.join(
                          os.path.dirname(__file__),
                          'output','parallel_mala_output_' + data_filename)):
            print("Posterior samples already exist, sampling directly without warm up...")

            mala_output = np.load(saving_path + '../output/parallel_mala_output_' + data_filename)
            previous_number_of_samples = mala_output.shape[1]
            previous_number_of_chains = mala_output.shape[0]

            samples_with_burn_in = mala_output[:,int(previous_number_of_samples/2):,:].reshape(int(previous_number_of_samples/2)*previous_number_of_chains,mala_output.shape[2])
            proposal_covariance = np.cov(samples_with_burn_in.T)
            # initial_states = np.zeros((number_of_chains,7))
            # initial_states[:,(2,3)] = np.array([true_parameter_values[2],true_parameter_values[3]])
            # initial_states[:,(0,1,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

            step_size = 0.01

            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                              args=(protein_at_observations,
                                                                    measurement_variance,
                                                                    number_of_samples,
                                                                    initial_state,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    1,
                                                                    known_parameters))
                                for initial_state in initial_states ]
            ## Let the pool know that these are all so that the pool will exit afterwards
            # this is necessary to prevent memory overflows.
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            np.save(os.path.join(os.path.dirname(__file__), 'output','parallel_mala_output_' + data_filename),
            array_of_chains)

        else:
            # warm up chain
            print("New data set, warming up chain with " + str(number_of_samples) + " samples...")
            proposal_covariance = np.diag([5e+3,0.03,0.01,0.01,1.0])
            step_size = 0.01

            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                              args=(protein_at_observations,
                                                                    measurement_variance,
                                                                    number_of_samples,
                                                                    initial_state,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    1,
                                                                    known_parameters))
                                for initial_state in initial_states ]
            ## Let the pool know that these are all so that the pool will exit afterwards
            # this is necessary to prevent memory overflows.
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            # sample directly
            print("Now sampling directly...")
            # make new initial states
            initial_states = np.zeros((number_of_chains,7))
            initial_states[:,(2,3)] = np.array([true_parameter_values[2],true_parameter_values[3]])
            for initial_state_index, _ in enumerate(initial_states):
                initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                              np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))

            samples_with_burn_in = array_of_chains[:,int(number_of_samples/2):,:].reshape(int(number_of_samples/2)*number_of_chains,number_of_parameters)
            proposal_covariance = np.cov(samples_with_burn_in.T)
            step_size = 0.01

            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                              args=(protein_at_observations,
                                                                    measurement_variance,
                                                                    number_of_samples,
                                                                    initial_state,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    1,
                                                                    known_parameters))
                                for initial_state in initial_states ]
            ## Let the pool know that these are all finished so that the pool will exit afterwards
            # this is necessary to prevent memory overflows.
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            np.save(os.path.join(os.path.dirname(__file__), 'output','parallel_mala_output_' + data_filename),
            array_of_chains)

    def xest_mala_experimental_data(self,data_filename = 'protein_observations_040417_cell_1_cluster_4.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/selected_data_for_mala/')
        protein_at_observations = np.load(os.path.join(saving_path,data_filename))
        experiment_date = data_filename[data_filename.find('ns_')+3:data_filename.find('_cel')]
        mean_protein = np.mean(protein_at_observations[:,1])


        number_of_samples = 80000
        number_of_chains = 8
        measurement_variance = np.power(np.round(np.load(saving_path + experiment_date + "_measurement_variance_detrended.npy"),0),2)
        # draw random initial states for the parallel chains
        from scipy.stats import uniform
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
        for initial_state_index, _ in enumerate(initial_states):
            initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                          np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))

        # define known parameters
        all_parameters = {'repression_threshold' : [0,None],
                          'hill_coefficient' : [1,None],
                          'mRNA_degradation_rate' : [2,np.log(2)/30],
                          'protein_degradation_rate' : [3,np.log(2)/90],
                          'basal_transcription_rate' : [4,None],
                          'translation_rate' : [5,None],
                          'transcription_delay' : [6,None]}

        known_parameters = {k:all_parameters[k] for k in ('mRNA_degradation_rate',
                                                          'protein_degradation_rate') if k in all_parameters}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(initial_states[0])) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        # if we already have mcmc samples, we can use them to construct a covariance matrix to make sampling better
        if os.path.exists(os.path.join(
                          os.path.dirname(__file__),
                          'output','parallel_mala_output_' + data_filename)):
            print("Posterior samples already exist, sampling directly without warm up...")

            mala_output = np.load(os.path.join(os.path.dirname(__file__), 'output','parallel_mala_output_' + data_filename))
            previous_number_of_samples = mala_output.shape[1]
            previous_number_of_chains = mala_output.shape[0]

            samples_with_burn_in = mala_output[:,int(previous_number_of_samples/2):,:].reshape(int(previous_number_of_samples/2)*previous_number_of_chains,mala_output.shape[2])
            proposal_covariance = np.cov(samples_with_burn_in.T)
            step_size = 0.01
            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                              args=(protein_at_observations,
                                                                    measurement_variance,
                                                                    number_of_samples,
                                                                    initial_state,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    1,
                                                                    known_parameters))
                                for initial_state in initial_states ]
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            np.save(os.path.join(os.path.dirname(__file__), 'output','parallel_mala_output_' + data_filename),
            array_of_chains)

        else:
            # warm up chain
            print("New data set, warming up chain with " + str(number_of_samples) + " samples...")
            proposal_covariance = np.diag([5e+3,0.03,0.01,0.01,1.0])
            step_size = 0.001
            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                              args=(protein_at_observations,
                                                                    measurement_variance,
                                                                    number_of_samples,
                                                                    initial_state,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    1,
                                                                    known_parameters))
                                for initial_state in initial_states ]
            ## Let the pool know that these are all so that the pool will exit afterwards
            # this is necessary to prevent memory overflows.
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            # sample directly
            print("Now sampling directly...")
            # make new initial states
            initial_states = np.zeros((number_of_chains,7))
            initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
            for initial_state_index, _ in enumerate(initial_states):
                initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                              np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))

            samples_with_burn_in = array_of_chains[:,int(number_of_samples/2):,:].reshape(int(number_of_samples/2)*number_of_chains,number_of_parameters)
            proposal_covariance = np.cov(samples_with_burn_in.T)
            step_size = 0.01
            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                              args=(protein_at_observations,
                                                                    measurement_variance,
                                                                    number_of_samples,
                                                                    initial_state,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    1,
                                                                    known_parameters))
                                for initial_state in initial_states ]
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            np.save(os.path.join(os.path.dirname(__file__), 'output','parallel_mala_output_' + data_filename),
            array_of_chains)

    def xest_mala_multiple_experimental_traces(self,experiment_date='280317p1',cluster='3'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/selected_data_for_mala/')
        data_filename = experiment_date + '_cluster_' + cluster + '.npy'
        protein_at_observations = np.array([np.load(os.path.join(saving_path,i)) for i in os.listdir(saving_path) if 'detrended' in i
                                                                               if 'cluster_'+cluster in i
                                                                               if experiment_date in i])
                                                                               # mean of all protein across time

        mean_protein = np.mean([i[j,1] for i in protein_at_observations for j in range(i.shape[0])])
        number_of_samples = 50
        number_of_chains = 1
        import pdb; pdb.set_trace()
        measurement_variance = np.power(np.round(np.load(saving_path + experiment_date + "_measurement_variance_detrended.npy"),0),2)
        # draw random initial states for the parallel chains
        from scipy.stats import uniform
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
        for initial_state_index, _ in enumerate(initial_states):
            initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(1),np.log(1),10]),
                                                                          np.array([mean_protein,3,np.log(60-1),np.log(40-1),25]))

        # define known parameters
        all_parameters = {'repression_threshold' : [0,None],
                          'hill_coefficient' : [1,None],
                          'mRNA_degradation_rate' : [2,np.log(2)/30],
                          'protein_degradation_rate' : [3,np.log(2)/90],
                          'basal_transcription_rate' : [4,None],
                          'translation_rate' : [5,None],
                          'transcription_delay' : [6,None]}

        known_parameters = {k:all_parameters[k] for k in ('mRNA_degradation_rate',
                                                          'protein_degradation_rate') if k in all_parameters}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(initial_states[0])) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        # if we already have mcmc samples, we can use them to construct a covariance matrix to make sampling better
        if os.path.exists(os.path.join(
                          os.path.dirname(__file__),
                          'output','parallel_mala_output_' + data_filename)):
            print("Posterior samples already exist, sampling directly without warm up...")

            mala_output = np.load(os.path.join(os.path.dirname(__file__), 'output','parallel_mala_output_' + data_filename))
            previous_number_of_samples = mala_output.shape[1]
            previous_number_of_chains = mala_output.shape[0]

            samples_with_burn_in = mala_output[:,int(previous_number_of_samples/2):,:].reshape(int(previous_number_of_samples/2)*previous_number_of_chains,mala_output.shape[2])
            proposal_covariance = np.cov(samples_with_burn_in.T)
            step_size = 0.2
            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                              args=(protein_at_observations,
                                                                    measurement_variance,
                                                                    number_of_samples,
                                                                    initial_state,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    1,
                                                                    known_parameters))
                                for initial_state in initial_states ]
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            np.save(os.path.join(os.path.dirname(__file__), 'output','parallel_mala_output_' + data_filename),
            array_of_chains)

        else:
            # warm up chain
            print("New data set, warming up chain with " + str(number_of_samples) + " samples...")
            proposal_covariance = np.diag([500,0.05,0.01,0.01,1.0])
            step_size = 0.2
            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                              args=(protein_at_observations,
                                                                    measurement_variance,
                                                                    number_of_samples,
                                                                    initial_state,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    1,
                                                                    known_parameters))
                                for initial_state in initial_states ]
            ## Let the pool know that these are all so that the pool will exit afterwards
            # this is necessary to prevent memory overflows.
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()


            # sample directly
            print("Now sampling directly...")
            # make new initial states
            initial_states = np.zeros((number_of_chains,7))
            initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
            for initial_state_index, _ in enumerate(initial_states):
                initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                              np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))

            samples_with_burn_in = array_of_chains[:,int(number_of_samples/2):,:].reshape(int(number_of_samples/2)*number_of_chains,number_of_parameters)
            proposal_covariance = np.cov(samples_with_burn_in.T)
            step_size = 0.2
            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                              args=(protein_at_observations,
                                                                    measurement_variance,
                                                                    number_of_samples,
                                                                    initial_state,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    1,
                                                                    known_parameters))
                                for initial_state in initial_states ]
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            np.save(os.path.join(os.path.dirname(__file__), 'output','parallel_mala_output_' + data_filename),
            array_of_chains)

    def xest_mala_analysis(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('parallel_mala_output_protein_observations_ps11_ds4')]

        for chain_path_string in chain_path_strings:
            mala = np.load(loading_path + chain_path_string)
            mala = mala[[0,2,3,4,5,6,7],2000:,:]
            import pdb; pdb.set_trace()
            # mala[:,:,[2,3]] = np.exp(mala[:,:,[2,3]])
            chains = az.convert_to_dataset(mala)
            print('\n' + chain_path_string + '\n')
            print('\nrhat:\n',az.rhat(chains))
            print('\ness:\n',az.ess(chains))
            az.plot_trace(chains); plt.savefig(loading_path + 'traceplot_' + chain_path_string[:-4] + '.png'); plt.clf()
            az.plot_posterior(chains); plt.savefig(loading_path + 'posterior_' + chain_path_string[:-4] + '.png'); plt.clf()
            az.plot_pair(chains,kind='kde'); plt.savefig(loading_path + 'pairplot_' + chain_path_string[:-4] + '.png'); plt.clf()
            # import pdb; pdb.set_trace()
            # np.save(loading_path + chain_path_string,mala)

    def xest_accuracy_of_chains(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('parallel_mala_output_protein_observations_ps')
                              ]

        coherence_values_ds1 = np.zeros(len([i for i in chain_path_strings if 'ds1' in i]))
        precision_values_ds1 = np.zeros(len([i for i in chain_path_strings if 'ds1' in i]))
        coherence_values_ds2 = np.zeros(len([i for i in chain_path_strings if 'ds2' in i]))
        precision_values_ds2 = np.zeros(len([i for i in chain_path_strings if 'ds2' in i]))
        coherence_values_ds3 = np.zeros(len([i for i in chain_path_strings if 'ds3' in i]))
        precision_values_ds3 = np.zeros(len([i for i in chain_path_strings if 'ds3' in i]))
        coherence_values_ds4 = np.zeros(len([i for i in chain_path_strings if 'ds4' in i]))
        precision_values_ds4 = np.zeros(len([i for i in chain_path_strings if 'ds4' in i]))

        for chain_path_string in chain_path_strings:
            mala = np.load(loading_path + chain_path_string)[:,10000:,:]
            samples = mala.reshape(mala.shape[0]*mala.shape[1],mala.shape[2])
            # samples[:,[2,3]] = np.exp(samples[:,[2,3]])
            parameter_set_string = chain_path_string[chain_path_string.find('ps'):chain_path_string.find('_ds')]
            data_set_string = chain_path_string[chain_path_string.find('ds'):chain_path_string.find('.npy')]
            true_values = np.load(loading_path + '../data/' + parameter_set_string + '_parameter_values.npy')[[0,1,4,5,6]]
            sample_mean = np.mean(samples,axis=0)
            sample_std = np.std(samples,axis=0)
            iqr_ratio = (np.quantile(samples,.75,axis=0)-np.quantile(samples,.25,axis=0))/(np.quantile(samples,.75,axis=0)+np.quantile(samples,.25,axis=0))

            if data_set_string == 'ds1':
                coherence_values_ds1[np.where(coherence_values_ds1==0)[0][0]] = np.load(loading_path + '../data/' + parameter_set_string + '_parameter_values.npy')[-2]
                precision_values_ds1[np.where(precision_values_ds1==0)[0][0]] = np.sum((np.abs(true_values-sample_mean)+sample_std)/true_values) # product of mean difference and std
                # precision_values_ds1[np.where(precision_values_ds1==0)[0][0]] = np.sum((np.abs(true_values-sample_mean))/true_values+iqr_ratio) # product of mean difference and std
            elif data_set_string == 'ds2':
                coherence_values_ds2[np.where(coherence_values_ds2==0)[0][0]] = np.load(loading_path + '../data/' + parameter_set_string + '_parameter_values.npy')[-2]
                precision_values_ds2[np.where(precision_values_ds2==0)[0][0]] = np.sum((np.abs(true_values-sample_mean)+sample_std)/true_values) # product of mean difference and std
                # precision_values_ds2[np.where(precision_values_ds2==0)[0][0]] = np.sum((np.abs(true_values-sample_mean))/true_values+iqr_ratio) # product of mean difference and std
            elif data_set_string == 'ds3':
                coherence_values_ds3[np.where(coherence_values_ds3==0)[0][0]] = np.load(loading_path + '../data/' + parameter_set_string + '_parameter_values.npy')[-2]
                precision_values_ds3[np.where(precision_values_ds3==0)[0][0]] = np.sum((np.abs(true_values-sample_mean)+sample_std)/true_values) # product of mean difference and std
                # precision_values_ds3[np.where(precision_values_ds3==0)[0][0]] = np.sum((np.abs(true_values-sample_mean))/true_values+iqr_ratio) # product of mean difference and std
            elif data_set_string == 'ds4':
                coherence_values_ds4[np.where(coherence_values_ds4==0)[0][0]] = np.load(loading_path + '../data/' + parameter_set_string + '_parameter_values.npy')[-2]
                precision_values_ds4[np.where(precision_values_ds4==0)[0][0]] = np.sum((np.abs(true_values-sample_mean)+sample_std)/true_values) # product of mean difference and std
                # precision_values_ds4[np.where(precision_values_ds4==0)[0][0]] = np.sum((np.abs(true_values-sample_mean))/true_values+iqr_ratio) # product of mean difference and std


        plt.scatter(coherence_values_ds1,precision_values_ds1,label='DS1',s=20)
        plt.scatter(coherence_values_ds2,precision_values_ds2,label='DS2',s=20)
        plt.scatter(coherence_values_ds3,precision_values_ds3,label='DS3',s=20)
        plt.scatter(coherence_values_ds4,precision_values_ds4,label='DS4',s=20)
        plt.xlabel('Coherence')
        plt.ylabel('Accuracy ($\hat{A}$)')
        plt.tight_layout()
        plt.legend()
        plt.savefig(loading_path + 'accuracy.png')

    def xest_identify_oscillatory_parameters(self):
        loading_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(loading_path + '.npy')
        prior_samples = np.load(loading_path + '_parameters.npy')

        ps_string = "ps10"
        coherence_band = [0.27,0.3]
        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    np.logical_and(model_results[:,1]>0.05, #standard deviation
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                    np.logical_and(model_results[:,3]>coherence_band[0], #coherence
                                    np.logical_and(model_results[:,3]<coherence_band[1], #coherence
                                                   prior_samples[:,4]<3.2))))))) # hill coefficient

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        this_parameter = my_posterior_samples[0]
        this_results = my_model_results[0]

        print('this basal transcription rate')
        print(this_parameter[0])
        print('this translation rate')
        print(this_parameter[1])
        print('this repression threshold')
        print(this_parameter[2])
        print('this transcription_delay')
        print(this_parameter[3])
        print('this hill coefficient')
        print(this_parameter[4])
        print('coherence')
        print(this_results[3])

        saving_path = os.path.join(os.path.dirname(__file__), 'data','')

        measurement_variance = 1000

        parameters = np.array([this_parameter[2],
                               this_parameter[4],
                               np.log(2)/30,
                               np.log(2)/90,
                               this_parameter[0],
                               this_parameter[1],
                               this_parameter[3],
                               this_results[3],
                               measurement_variance])

        import pdb; pdb.set_trace()


        np.save(saving_path + ps_string + "_parameter_values.npy",parameters)


        # parameters = np.load(loading_path + ps_string + "_parameter_values.npy")
        observation_duration  = 1800
        observation_frequency = 5
        no_of_observations    = np.int(observation_duration/observation_frequency)

        true_data = hes5.generate_langevin_trajectory(duration = observation_duration,
                                                      repression_threshold = parameters[0],
                                                      hill_coefficient = parameters[1],
                                                      mRNA_degradation_rate = parameters[2],
                                                        protein_degradation_rate = parameters[3],
                                                      basal_transcription_rate = parameters[4],
                                                      translation_rate = parameters[5],
                                                      transcription_delay = parameters[6],
                                                      equilibration_time = 1000)
        np.save(saving_path + 'true_data_' + ps_string + '.npy',
                true_data)

        ## the F constant matrix is left out for now
        protein_at_observations = true_data[:,(0,2)]
        protein_at_observations[:,1] += np.random.randn(true_data.shape[0])*parameters[-1]
        protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)
        np.save(saving_path + 'protein_observations_' + ps_string + '_ds1.npy',
                    protein_at_observations[0:900:10,:])
        np.save(saving_path + 'protein_observations_' + ps_string + '_ds2.npy',
                    protein_at_observations[0:900:5,:])
        np.save(saving_path + 'protein_observations_' + ps_string + '_ds3.npy',
                    protein_at_observations[0:1800:10,:])
        np.save(saving_path + 'protein_observations_' + ps_string + '_ds4.npy',
                    protein_at_observations[0:1800:5,:])

        my_figure = plt.figure()
        plt.scatter(np.arange(0,1800),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations')
        plt.title('Protein Observations')
        plt.xlabel('Time')
        plt.ylabel('Protein Copy Numbers')
        plt.tight_layout()
        my_figure.savefig(saving_path + 'protein_observations_' + ps_string + '.pdf')

    def xest_infer_parameters_from_data_set(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        protein_observations    = np.load(saving_path + 'protein_observations_180_ps3_ds2.npy')
        previous_run            = np.load(saving_path + 'full_random_walk_180_ps2_ds1.npy')

        # define parameters for uniform prior distributions
        hyper_parameters = np.array([100,19900,2,4,0,1,0,1,np.log10(0.1),np.log10(60)+1,np.log10(0.1),np.log10(40)+1,5,35]) # uniform
        measurement_variance = 10000.0

        # draw 8 random initial states for the parallel random walk
        from scipy.stats import uniform
        initial_states          = np.zeros((8,7))
        initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
        for initial_state_index in range(initial_states.shape[0]):
            initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([100,2,np.log10(0.1),np.log10(0.1),5]),
                        np.array([20100,4,np.log10(60)+1,np.log10(40)+1,35]))

        # initial covariance based on prior assumptions about the data
        initial_covariance = 0.04*np.diag(np.array([np.var(previous_run[50000:,0]),np.var(previous_run[50000:,1]),
                                                    np.var(previous_run[50000:,2]),np.var(previous_run[50000:,3]),
                                                    np.var(previous_run[50000:,4])]))
        number_of_iterations = 350000

        pool_of_processes = mp_pool.ThreadPool(processes = 8)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_random_walk,
                                                          args=(number_of_iterations,protein_observations,hyper_parameters,measurement_variance,0.6,initial_covariance,initial_state),
                                                          kwds=dict(adaptive='true'))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()
        list_of_random_walks      = []
        list_of_acceptance_rates  = []
        list_of_acceptance_tuners = []
        chain_counter = 0
        for process_result in process_results:
            this_random_walk, this_acceptance_rate, this_acceptance_tuner = process_result.get()
            print('successful get ', chain_counter)
            list_of_random_walks.append(this_random_walk)
            list_of_acceptance_rates.append(this_acceptance_rate)
            list_of_acceptance_tuners.append(this_acceptance_tuner)
            chain_counter += 1
        pool_of_processes.join()
        print(list_of_acceptance_rates)
        print(list_of_acceptance_tuners)

        for i in range(len(initial_states)):
            np.save(os.path.join(os.path.dirname(__file__), 'output','parallel_random_walk_180_ps3_ds2_{cap}.npy').format(cap=i),list_of_random_walks[i])

    def xest_summary_statistics_from_inferred_parameters(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data','parallel_random_walk_')
        random_walk = np.load(saving_path + '360_ps3_ds4_4.npy')
        random_walk = random_walk[100000:,:]

        # select inferred parameter values and calculate mean mRNAs from them
        parameter_values = random_walk[:250000:100,:]
        parameter_values[:,[4,5]] = np.power(10,parameter_values[:,[4,5]])
        summary_statistics = hes_inference.calculate_langevin_summary_statistics_at_parameters(parameter_values, number_of_traces_per_sample = 100,
                                                                                               number_of_cpus = 12)
        # compute mean mRNA from true parameter values
        true_parameter_values = np.array([3407.99,5.17,np.log(2)/30,np.log(2)/90,15.86,1.27,30])
        true_summary_statistics = hes_inference.calculate_langevin_summary_statistics_at_parameter_point(true_parameter_values, number_of_traces = 100)
        inferred_mean_mRNA = summary_statistics[:,4]
        true_mean_mRNA = true_summary_statistics[4]

        # compute mean mRNA from mean inferred parameter values
        mean_inferred_parameters = np.mean(random_walk,axis=0)
        mean_summary_statistics = hes_inference.calculate_langevin_summary_statistics_at_parameter_point(mean_inferred_parameters, number_of_traces = 100)
        mean_parameters_mRNA = mean_summary_statistics[4]

        # compute mean mRNA from most likely inferred parameter values
        from scipy.stats import gaussian_kde as gkde
        most_likely_inferred_parameters = np.amax(gkde(np.transpose(random_walk)).dataset,axis=1)
        most_likely_summary_statistics = hes_inference.calculate_langevin_summary_statistics_at_parameter_point(most_likely_inferred_parameters, number_of_traces = 100)
        most_likely_parameters_mRNA = most_likely_summary_statistics[4]

        # plot results on a histogram
        my_figure = plt.figure()
        plt.hist(inferred_mean_mRNA,25)
        plt.axvline(true_mean_mRNA,label='True mean mRNA',color='#F69454')
        plt.axvline(mean_parameters_mRNA,label='Inferred mean mRNA from mean parameters',color='#20948B')
        plt.axvline(most_likely_parameters_mRNA,label='Inferred mean mRNA from most likely parameters',color='#BA1E1B')
        plt.xlabel('Number of mRNA')
        plt.ylabel('Number of occurrences')
        plt.title('mean mRNA using parameters inferred from PS3_DS4')
        plt.legend(fontsize='x-small')
        my_figure.savefig(os.path.join(os.path.dirname(__file__),'output','mRNA_histogram_ps3.pdf'))

    def xest_generate_langevin_trace(self):
        langevin_trajectory_data = hes5.generate_langevin_trajectory(duration = 720,
                                                                repression_threshold = 3581,
                                                                hill_coefficient = 4.87,
                                                                mRNA_degradation_rate = 0.03,
                                                                protein_degradation_rate = 0.08,
                                                                basal_transcription_rate = 1.5,
                                                                translation_rate = 9.7,
                                                                transcription_delay = 36,
                                                                initial_mRNA = 3,
                                                                initial_protein = 100,
                                                                equilibration_time = 0.0)

        my_dpi=96
        my_figure = plt.figure(figsize=(1680/my_dpi, 1200/my_dpi), dpi=my_dpi)
        # plt.plot(langevin_trajectory_data[:,0],langevin_trajectory_data[:,1],label='mRNA',linewidth=0.89,zorder=1)
        plt.plot(langevin_trajectory_data[:,0],0.03*langevin_trajectory_data[:,2],label='protein',linewidth=8,zorder=2,color='#2f3c7e')
        # plt.title('Protein and mRNA expression from the Langevin equation')
        plt.xlabel('Time',fontsize=80)
        plt.ylabel('Gene expression',fontsize=80)
        plt.title('Cell 1', fontsize=80)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        my_figure.patch.set_facecolor('#8aaae5')
        ax = plt.gca()
        ax.set_facecolor('#8aaae5')
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','langevin_trajectory.png'),facecolor=my_figure.get_facecolor(),dpi=my_dpi)

    def xest_make_experimental_data(self):
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','')
        # import spreadsheets as dataframes
        het_hom_df = pd.DataFrame(pd.read_excel(loading_path + "HES5Molnumber_FCS_E10_5_Het_Hom.xlsx",header=None))
        experiment_date = '280317p6'
        cell_intensity_df = pd.DataFrame(pd.read_excel(loading_path + experiment_date + "_VH5_corrected_tissmean_Zpos_correctstartt.xlsx",header=None))

        # convert to numpy arrays for plotting / fitting
        intensities = cell_intensity_df.iloc[2:,2:].astype(float).values.flatten()
        intensities = intensities[~(np.isnan(intensities))]
        het_molecule_numbers = het_hom_df.iloc[1:,0].astype(float).values.flatten()
        het_molecule_numbers = het_molecule_numbers[~(np.isnan(het_molecule_numbers))]
        hom_molecule_numbers = het_hom_df.iloc[1:,1].astype(float).values.flatten()
        hom_molecule_numbers = hom_molecule_numbers[~(np.isnan(hom_molecule_numbers))]
        het_and_hom = np.concatenate((het_molecule_numbers,hom_molecule_numbers))


        # make qqplots and calculate gradients
        # fig, ax = plt.subplots(1,3,figsize=(20,10))
        x = np.quantile(intensities,np.linspace(0.0,1.0,101))
        y = np.quantile(het_molecule_numbers,np.linspace(0.0,1.0,101))
        gradient_het = (np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[-1]))-np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[0])))/(np.unique(x[-1]-x[0]))
        # ax[0].scatter(x,y)
        # ax[0].plot(np.unique(x[5:96]), np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[5:96])),color='r')
        # ax[0].set_ylabel("Het Molecule numbers")
        # ax[0].set_xlabel("Cell intensities")
        # gradient_string = "Gradient is " + str(np.round(gradient_het[0],2))
        # ax[0].text(0.5,71000,gradient_string)
        x = np.quantile(intensities,np.linspace(0.0,1.0,101))
        y = np.quantile(hom_molecule_numbers,np.linspace(0.0,1.0,101))
        gradient_hom = (np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[-1]))-np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[0])))/(np.unique(x[-1]-x[0]))
        # ax[1].scatter(x,y)
        # ax[1].plot(np.unique(x[5:96]), np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[5:96])),color='r')
        # ax[1].set_ylabel("Hom Molecule numbers")
        # ax[1].set_xlabel("Cell intensities")
        # gradient_string = "Gradient is " + str(np.round(gradient_hom[0],2))
        # ax[1].text(0.5,120000,gradient_string)
        x = np.quantile(intensities,np.linspace(0.0,1.0,101))
        y = np.quantile(het_and_hom,np.linspace(0.0,1.0,101))
        gradient_het_and_hom = (np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[-1]))-np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[0])))/(np.unique(x[-1]-x[0]))
        # ax[2].scatter(x,y)
        # ax[2].plot(np.unique(x[5:96]), np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[5:96])),color='r')
        # ax[2].set_ylabel("Het and Hom Molecule numbers")
        # ax[2].set_xlabel("Cell intensities")
        # gradient_string = "Gradient is " + str(np.round(gradient_het_and_hom[0],2))
        # ax[2].text(0.5,120000,gradient_string)
        # plt.tight_layout()
        # plt.savefig(saving_path + 'molecule_qq_plot.png')

        np.save(loading_path + 'selected_data_for_mala/' + experiment_date + "_measurement_variance.npy",np.std((gradient_hom,gradient_het,gradient_het_and_hom)))

        # make data from each trace and save
        # for cell_index in range(1,cell_intensity_df.shape[1]):
        #     cell_intensity_values = cell_intensity_df.iloc[2:,[0,cell_index]].astype(float).values
        #     # remove NaNs
        #     cell_intensity_values = cell_intensity_values[~np.isnan(cell_intensity_values[:,1])]
        #     cell_intensity_values[:,0] *=60
        #     cell_intensity_values[:,1] *= gradient_hom
        #     cell_cluster = int(cell_intensity_df.iloc[0,cell_index])
        #     np.save(loading_path + 'protein_observations_' + experiment_date + '_cell_' + str(cell_index) + '_cluster_' + str(cell_cluster),
        #             cell_intensity_values)

    def xest_detrend_experimental_data(self):
        from scipy.signal import detrend
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/selected_data_for_mala/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','detrended_data_images/')
        # import spreadsheets as dataframes
        date = '280317p6'
        experiment_date = 'protein_observations_' + date

        experimental_data_strings = [i for i in os.listdir(loading_path) if experiment_date in i
                                     and not 'detrended' in i]
        variances = np.zeros(len(experimental_data_strings))
        detrended_variances = np.zeros(len(experimental_data_strings))

        # make data from each trace and save
        for index, cell_string in enumerate(experimental_data_strings):
            protein = np.load(loading_path + cell_string)
            detrended_protein = np.copy(protein)
            detrended_protein[:,1] = np.mean(protein[:,1]) + detrend(protein[:,1])
            variances[index] = np.var(protein[:,1])
            detrended_variances[index] = np.var(detrended_protein[:,1])
            np.save(loading_path + cell_string[:-4] + "_detrended.npy",detrended_protein)

            fig, ax = plt.subplots(1,2,figsize=(10,5))
            ax[0].plot(protein[:,0],protein[:,1])
            ax[0].set_xlabel("Time (Minutes)")
            ax[0].set_ylabel("Protein Molecule Number")
            ax[0].set_title("Observed Data")
            ax[1].plot(detrended_protein[:,0],detrended_protein[:,1])
            ax[1].set_xlabel("Time (Minutes)")
            ax[1].set_ylabel("Protein Molecule Number")
            ax[1].set_title("Detrended Data")
            plt.tight_layout()
            plt.savefig(saving_path + cell_string[:-4] + "_detrended.png")
        measurement_variance = np.sqrt(0.1*np.mean(detrended_variances))
        np.save(loading_path + date + '_measurement_variance_detrended.npy',measurement_variance)
