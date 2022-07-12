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
# import pymc3 as pm
import arviz as az
# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5
import hes_inference

number_of_cpus = mp.cpu_count()
font_size = 25
cm_to_inches = 0.3937008
class TestInference(unittest.TestCase):

    def xest_check_kalman_filter_not_broken(self):
        # load in some saved observations and correct kalman filter predictions
        saving_path                          = os.path.join(os.path.dirname(__file__), 'data','kalman_test_trace')
        fixed_langevin_trace                 = np.load(saving_path + '_true_data.npy')
        fixed_protein_observations           = np.load(saving_path + '_observations.npy')
        # true_kalman_prediction_mean          = np.load(saving_path + '_prediction_mean.npy')
        # true_kalman_prediction_variance      = np.load(saving_path + '_prediction_variance.npy')
        # true_kalman_prediction_distributions = np.load(saving_path + '_prediction_distributions.npy')
        # true_kalman_negative_log_likelihood_derivative = np.load(saving_path + '_negative_log_likelihood_derivative.npy')

        # run the current kalman filter using the same parameters and observations, then compare
        parameters = np.array([10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 5.0])
        state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative,predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives = hes_inference.kalman_filter(fixed_protein_observations,#fixed_protein_observations,
                                                                                                                                                                                                                                                                   parameters,
                                                                                                                                                                                                                                                                   measurement_variance=10000,
                                                                                                                                                                                                                                                                   derivative=True)
        # loglik, loglikd = hes_inference.calculate_log_likelihood_and_derivative_at_parameter_point(fixed_protein_observations,
                                                                                                   # parameters,
                                                                                                   # 5000,
                                                                                                   # measurement_variance=10000)
        import pdb; pdb.set_trace()
        # print(log_likelihood)
        np.testing.assert_almost_equal(state_space_mean,true_kalman_prediction_mean)
        np.testing.assert_almost_equal(state_space_variance,true_kalman_prediction_variance)
        np.testing.assert_almost_equal(predicted_observation_distributions,true_kalman_prediction_distributions)
        # np.testing.assert_almost_equal(true_kalman_negative_log_likelihood_derivative,negative_log_likelihood_derivative)
        # np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_test_trace_prediction_mean.npy'),state_space_mean)
        # np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_test_trace_prediction_variance.npy'),state_space_variance)
        # np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_test_trace_prediction_distributions.npy'),predicted_observation_distributions)
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

    def qest_runtime_mh_vs_mala(self):
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

        protein_at_observations = np.array([np.load(saving_path + 'protein_observations_90_ps3_ds1.npy')])
        mean_protein = np.mean([i[j,1] for i in protein_at_observations for j in range(i.shape[0])])
        true_parameter_values = np.load(os.path.join(saving_path,'ps3_parameter_values.npy'))
        # mala_output = np.load(saving_path + 'mala_output_1.npy')
        number_of_samples = 200
        measurement_variance = np.power(true_parameter_values[-1],2)

        # true parameters ps3 -- [3407.99,5.17,np.log(2)/30,np.log(2)/90,15.86,1.27,30]
        initial_position = true_parameter_values[[0,1,2,3,4,5,6]]
        initial_position[[4,5]] = np.log(initial_position[[4,5]])
        # proposal_covariance = np.cov(mala_output.T)
        step_size = [55000.0,0.45,0.01,0.013,90.0]
        all_parameters = {'repression_threshold' : [0,true_parameter_values[0]],
                          'hill_coefficient' : [1,true_parameter_values[1]],
                          'mRNA_degradation_rate' : [2,np.log(np.log(2)/30)],
                          'protein_degradation_rate' : [3,np.log(np.log(2)/90)],
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

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_repression.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_repression.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_repression.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_repression.npy")
                x_values = np.linspace(3300,3600,len(likelihood))
                normal = np.trapz(np.exp(likelihood),x_values)
                # import pdb; pdb.set_trace()
                # import pdb; pdb.set_trace()
                fig, ax = plt.subplots(1,1,figsize=(10.22*0.7,7.66*0.7))
                ax.plot(x_values,np.exp(likelihood)/normal,label='Likelihood')
                heights, bins, _ = ax.hist(mala,density=True,bins=30,color='#20948B',alpha=0.3,ec='black',label='MALA')
                ax.set_xlabel("$P_0$",fontsize=font_size)
                ax.set_ylabel("Probability",fontsize=font_size)
                ax.set_xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                ax.vlines(true_parameter_values[0],0,1.1*max(heights),color='k',lw=2,label='True value')
                ax.legend()
                plt.tight_layout()
                plt.savefig(loading_path + 'repression_likelihood_mala.png')
                plt.clf()

            elif index == 1: # hill_coefficient
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'translation_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_hill.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_hill.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_hill.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_hill.npy")
                x_values = np.linspace(2.0,6.0,1000)
                normal = np.trapz(np.exp(likelihood),x_values)
                fig, ax = plt.subplots(1,1,figsize=(10.22*0.7,7.66*0.7))
                heights, bins, _ = ax.hist(mala,density=True,bins=30,color='#20948B',alpha=0.3,ec='black',label='MALA')
                ax.set_xlabel("$h$",fontsize=font_size)
                ax.vlines(true_parameter_values[1],0,1.1*max(heights),color='k',lw=2,label='True value')
                ax.plot(x_values,np.exp(likelihood)/normal,label='Likelihood')
                ax.set_xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                ax.set_ylabel('Probability',fontsize=font_size)
                ax.legend()
                plt.tight_layout()
                plt.savefig(loading_path + 'hill_likelihood_mala.png')
                plt.clf()
            elif index == 2: # transcription_rate
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'translation_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_transcription.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_transcription.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_transcription.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_transcription.npy")
                x_values = np.linspace(0.01,60.0,200)
                normal = np.trapz(np.exp(likelihood),x_values)
                fig, ax = plt.subplots(1,1,figsize=(10.22*0.7,7.66*0.7))
                ax.plot(x_values,np.exp(likelihood)/normal,label='Likelihood')
                heights, bins, _ = ax.hist(np.exp(mala),density=True,bins=30,color='#20948B',alpha=0.3,ec='black',label='MALA')
                ax.set_xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                ax.vlines(true_parameter_values[4],0,1.1*max(heights),color='k',lw=2,label='True value')
                ax.set_xlabel("$\\alpha_m$ (1/min)",fontsize=font_size)
                ax.set_ylabel("Probability",fontsize=font_size)
                ax.legend()
                plt.tight_layout()
                plt.savefig(loading_path + 'transcription_likelihood_mala.png')
                plt.clf()

            elif index == 3: # translation_rate
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'transcription_delay') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_translation.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_translation.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_translation.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_translation.npy")
                x_values = np.linspace(1.1,1.55,len(likelihood))
                normal = np.trapz(np.exp(likelihood),x_values)
                fig, ax = plt.subplots(1,1,figsize=(10.22*0.7,7.66*0.7))
                ax.plot(x_values,np.exp(likelihood)/normal,label='Likelihood')
                heights, bins, _ = ax.hist(np.exp(mala),density=True,bins=30,color='#20948B',alpha=0.3,ec='black',label='MALA')
                ax.set_xlabel("$\\alpha_p$ (1/min)",fontsize=font_size)
                ax.set_ylabel("Probability",fontsize=font_size)
                ax.vlines(true_parameter_values[5],0,1.1*max(heights),color='k',lw=2,label='True value')
                ax.set_xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                ax.legend()
                plt.tight_layout()
                plt.savefig(loading_path + 'translation_likelihood_mala.png')
                plt.clf()
            elif index == 4: # transcription_delay
                known_parameters = {k:all_parameters[k] for k in ('repression_threshold',
                                                                  'hill_coefficient',
                                                                  'mRNA_degradation_rate',
                                                                  'protein_degradation_rate',
                                                                  'basal_transcription_rate',
                                                                  'translation_rate') if k in all_parameters}

                if os.path.exists(os.path.join(os.path.dirname(__file__), 'output','mala_output_delay.npy')):
                    mala = np.load(os.path.join(os.path.dirname(__file__), 'output','mala_output_delay.npy'))
                else:
                    mala = hes_inference.kalman_mala(protein_at_observations,
                                                     measurement_variance,
                                                     number_of_samples,
                                                     initial_position,
                                                     step_size[index],
                                                     thinning_rate=1,
                                                     known_parameter_dict=known_parameters)

                    np.save(os.path.join(os.path.dirname(__file__), 'output','mala_output_delay.npy'),
                            mala)

                likelihood = np.load(loading_path + "likelihood_delay.npy")
                x_values = np.linspace(5.0,40.0,36)
                unique_values, unique_indices = np.unique(likelihood,return_index=True)
                unique_indices = np.sort(unique_indices)
                normal = np.trapz(np.exp(likelihood[unique_indices]),x_values)
                fig, ax = plt.subplots(1,1,figsize=(10.22*0.7,7.66*0.7))
                ax.plot(x_values,np.exp(likelihood[unique_indices])/normal,label='Likelihood')
                heights, bins, _ = ax.hist(mala,density=True,bins=30,color='#20948B',alpha=0.3,ec='black',label='MALA')
                ax.vlines(true_parameter_values[6],0,1.1*max(heights),color='k',lw=2,label='True value')
                ax.set_xlim(xmin=2*bins[0]-bins[1],xmax=2*bins[-1]-bins[-2])
                ax.set_xlabel("$\\tau$ (mins)",fontsize=font_size)
                ax.set_ylabel("Probability",fontsize=font_size)
                ax.legend()
                plt.tight_layout()
                plt.savefig(loading_path + 'delay_likelihood_mala.png')
                plt.clf()


    def xest_plot_protein_observations(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        # data = np.load(saving_path + 'protein_observations_ps6_fig5_1_cells_5_minutes.npy')
        data = np.load(saving_path + 'true_data_ps3.npy')
        protein = np.maximum(0,data[:,2] + 100*np.random.randn(data.shape[0]))

        my_figure, ax1 = plt.subplots(figsize=(12.47*0.7,8.32*0.7))
        ax1.scatter(data[:,0],data[:,2],marker='o',s=3,color='#20948B',alpha=0.75,label='protein')
        ax1.scatter(data[0:-1:10,0],protein[0:-1:10],marker='o',s=14,color='#F18D9E')
        # ax1.scatter(protein[:,0],protein[:,1],marker='o',s=14,c='#F18D9E')
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
        my_figure.savefig(os.path.join(saving_path,'ps3_data.pdf'))

    def xest_plot_mala_and_abc_posterior(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output/single_cell_mala','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'data','')

        # first single cell
        experiment_string = '040417_cell_53_cluster_1'
        output = np.load(saving_path + 'final_parallel_mala_output_protein_observations_' + experiment_string + '_detrended.npy')
        output = output.reshape(output.shape[0]*output.shape[1],output.shape[2])
        output[:,0]/=10000

        # second single cell
        experiment_string = '280317p1_cell_43_cluster_1'
        output1 = np.load(saving_path + 'final_parallel_mala_output_protein_observations_' + experiment_string + '_detrended.npy')
        output1 = output1.reshape(output1.shape[0]*output1.shape[1],output1.shape[2])
        output1[:,0]/=10000

        # third single cell
        experiment_string = '280317p1_cell_23_cluster_2'
        output2 = np.load(saving_path + 'final_parallel_mala_output_protein_observations_' + experiment_string + '_detrended.npy')
        output2 = output2.reshape(output2.shape[0]*output2.shape[1],output2.shape[2])
        output2[:,0]/=10000

        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                    'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        accepted_indices = np.where(np.logical_and(model_results[:,0]>55000, #protein number
                                    np.logical_and(model_results[:,0]<65000, #protein_number
                                    np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                                   model_results[:,1]>0.05))))  #standard deviation

        my_posterior_samples = prior_samples[accepted_indices]
        my_posterior_samples[:,2]/=10000

        data_frame = pd.DataFrame( data = my_posterior_samples,
                                   columns= ['Transcription rate',
                                             'Translation rate',
                                             'Repression threshold/1e4',
                                             'Transcription delay',
                                             'Hill coefficient'])

        fig, ax = plt.subplots(4,5,figsize= (13*1.4,12*1.4))

        # transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
        ax[0,0].hist(np.log10(np.array(data_frame['Transcription rate'])),
                      ec = 'black',
                      density = True,
                      alpha=0.6)
        ax[0,0].set_xlim(-1,np.log10(120.0))
        ax[0,0].set_ylabel("Probability", labelpad = 20,fontsize=font_size*1.2)
        ax[0,0].set_xlabel("$\log(\\alpha_m)$ (1/min)",fontsize=font_size*1.2)
        ax[0,0].set_xticks([-1.5,0,1], [r'10$^{-1}$',r'10$^0$',r'10$^1$'])

        # translation_rate_bins = np.linspace(0,40,20)
        # import pdb; pdb.set_trace()
        ax[0,1].hist(data_frame['Translation rate'],
                     ec = 'black',
                     density = True,
                     bins = 20,
                     alpha=0.6)
        ax[0,1].set_xlim(9,40+1)
        ax[0,1].set_xticks([10,20,40])
        ax[0,1].set_xticklabels([10,20,40])
        ax[0,1].set_xlabel("$\log(\\alpha_p)$ (1/min)",fontsize=font_size*1.2)

        ax[0,2].hist(data_frame['Repression threshold/1e4'],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     alpha=0.6)
        ax[0,2].set_xlabel("$P_0$ (1e4)",fontsize=font_size*1.2)
        # ax[0,2].set_ylim(0,0.22)
        ax[0,2].set_xlim(0,12)
        ax[0,2].locator_params(axis='x', tight = True, nbins=4)
        ax[0,2].locator_params(axis='y', tight = True, nbins=2)

        time_delay_bins = np.linspace(5,40,10)
        ax[0,3].hist(data_frame['Transcription delay'],
                     ec = 'black',
                     bins = time_delay_bins,
                     density = True,
                     alpha=0.6)
        ax[0,3].set_xlim(5,40)
        # ax[0,3].set_ylim(0,0.04)
        ax[0,3].set_xlabel(" $\\tau$ (min)",fontsize=font_size*1.2)

        ax[0,4].hist(data_frame['Hill coefficient'],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     alpha=0.6)
        # ax[0,4].set_ylim(0,0.35)
        ax[0,4].set_xlim(2,6)
        ax[0,4].set_xlabel(" $h$",fontsize=font_size*1.2)

        ## MALA single cell 1
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
        ax[1,0].hist(output[:,2],
                      ec = 'black',
                      bins = transcription_rate_bins,
                      density = True,
                      color='#20948B',
                      alpha=0.6)
        ax[1,0].set_xlim(-1,np.log10(120.0))
        ax[1,0].set_ylabel("Probability", labelpad = 20,fontsize=font_size*1.2)
        ax[1,0].set_xlabel("$\log(\\alpha_m)$ (1/min)",fontsize=font_size*1.2)
        ax[1,0].set_xticks([-1.5,0,1], [r'10$^{-1}$',r'10$^0$',r'10$^1$'])

        translation_rate_bins = np.linspace(3.1,np.log(40),20)
        ax[1,1].hist(output[:,3],
                     ec = 'black',
                     density = True,
                     bins = translation_rate_bins,
                     color='#20948B',
                     alpha=0.6)
        ax[1,1].set_xlim(np.log(10),np.log(40))
        ax[1,1].set_xticks([np.log(10),np.log(20),np.log(40)])
        ax[1,1].set_xticklabels([10,20,40])
        ax[1,1].set_xlabel("$\log(\\alpha_p)$ (1/min)",fontsize=font_size*1.2)

        ax[1,2].hist(output[:,0],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        ax[1,2].set_xlabel("$P_0$ (1e4)",fontsize=font_size*1.2)
        # ax[1,2].set_ylim(0,0.22)
        ax[1,2].set_xlim(0,12)
        ax[1,2].locator_params(axis='x', tight = True, nbins=4)
        ax[1,2].locator_params(axis='y', tight = True, nbins=2)

        time_delay_bins = np.linspace(5,40,20)
        ax[1,3].hist(output[:,4],
                     ec = 'black',
                     bins = time_delay_bins,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[1,3].set_xlim(5,40)
        # ax[1,3].set_ylim(0,0.04)
        ax[1,3].set_xlabel(" $\\tau$ (mins)",fontsize=font_size*1.2)

        ax[1,4].hist(output[:,1],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[1,4].set_ylim(0,0.35)
        ax[1,4].set_xlim(2,6)
        ax[1,4].set_xlabel(" $h$",fontsize=font_size*1.2)

        ## MALA single cell 2
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
        ax[2,0].hist(output1[:,2],
                      ec = 'black',
                      bins = transcription_rate_bins,
                      density = True,
                      color='#20948B',
                      alpha=0.6)
        ax[2,0].set_xlim(-1,np.log10(120.0))
        ax[2,0].set_ylabel("Probability", labelpad = 20,fontsize=font_size*1.2)
        ax[2,0].set_xlabel("$\log(\\alpha_m)$ (1/min)",fontsize=font_size*1.2)
        ax[2,0].set_xticks([-1.5,0,1], [r'10$^{-1}$',r'10$^0$',r'10$^1$'])

        translation_rate_bins = np.linspace(3.1,np.log(40),20)
        ax[2,1].hist(output1[:,3],
                     ec = 'black',
                     density = True,
                     bins = translation_rate_bins,
                     color='#20948B',
                     alpha=0.6)
        ax[2,1].set_xlim(np.log(10),np.log(40))
        ax[2,1].set_xticks([np.log(10),np.log(20),np.log(40)])
        ax[2,1].set_xticklabels([10,20,40])
        ax[2,1].set_xlabel("$\log(\\alpha_p)$ (1/min)",fontsize=font_size*1.2)

        ax[2,2].hist(output1[:,0],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        ax[2,2].set_xlabel("$P_0$ (1e4)",fontsize=font_size*1.2)
        # ax[2,2].set_ylim(0,0.22)
        ax[2,2].set_xlim(0,12)
        ax[2,2].locator_params(axis='x', tight = True, nbins=4)
        ax[2,2].locator_params(axis='y', tight = True, nbins=2)

        time_delay_bins = np.linspace(5,40,20)
        ax[2,3].hist(output1[:,4],
                     ec = 'black',
                     bins = time_delay_bins,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[2,3].set_xlim(5,40)
        # ax[2,3].set_ylim(0,0.04)
        ax[2,3].set_xlabel(" $\\tau$ (mins)",fontsize=font_size*1.2)

        ax[2,4].hist(output1[:,1],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[2,4].set_ylim(0,0.35)
        ax[2,4].set_xlim(2,6)
        ax[2,4].set_xlabel("$h$",fontsize=font_size*1.2)

        ## MALA single cell 3
        transcription_rate_bins = np.linspace(-1,np.log10(60.0),20)
        ax[3,0].hist(output1[:,2],
                      ec = 'black',
                      bins = transcription_rate_bins,
                      density = True,
                      color='#20948B',
                      alpha=0.6)
        ax[3,0].set_xlim(-1,np.log10(120.0))
        ax[3,0].set_ylabel("Probability", labelpad = 20,fontsize=font_size*1.2)
        ax[3,0].set_xlabel("$\log(\\alpha_m)$ (1/min)",fontsize=font_size*1.2)
        ax[3,0].set_xticks([-1.5,0,1], [r'10$^{-1}$',r'10$^0$',r'10$^1$'])

        translation_rate_bins = np.linspace(3.1,np.log(40),20)
        ax[3,1].hist(output2[:,3],
                     ec = 'black',
                     density = True,
                     bins = translation_rate_bins,
                     color='#20948B',
                     alpha=0.6)
        ax[3,1].set_xlim(np.log(10),np.log(40))
        ax[3,1].set_xticks([np.log(10),np.log(20),np.log(40)])
        ax[3,1].set_xticklabels([10,20,40])
        ax[3,1].set_xlabel("$\log(\\alpha_p)$ (1/min)",fontsize=font_size*1.2)

        ax[3,2].hist(output2[:,0],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        ax[3,2].set_xlabel("$P_0$ (1e4)",fontsize=font_size*1.2)
        # ax[3,2].set_ylim(0,0.22)
        ax[3,2].set_xlim(0,12)
        ax[3,2].locator_params(axis='x', tight = True, nbins=4)
        ax[3,2].locator_params(axis='y', tight = True, nbins=2)

        time_delay_bins = np.linspace(5,40,20)
        ax[3,3].hist(output2[:,4],
                     ec = 'black',
                     bins = time_delay_bins,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[3,3].set_xlim(5,40)
        # ax[3,3].set_ylim(0,0.04)
        ax[3,3].set_xlabel(" $\\tau$ (mins)",fontsize=font_size*1.2)

        ax[3,4].hist(output2[:,1],
                     ec = 'black',
                     bins = 20,
                     density = True,
                     color='#20948B',
                     alpha=0.6)
        # ax[3,4].set_ylim(0,0.35)
        ax[3,4].set_xlim(2,6)
        ax[3,4].set_xlabel(" $h$",fontsize=font_size*1.2)

        ax[0,2].text(-4,.17,'ABC (population)',fontsize=font_size*1.2)
        ax[1,2].text(-4,.8,'Single cell (cluster 1)',fontsize=font_size*1.2)
        ax[2,2].text(-4,.6,'Single cell (cluster 1)',fontsize=font_size*1.2)
        ax[3,2].text(-4,.32,'Single cell (cluster 2)',fontsize=font_size*1.2)

        plt.tight_layout(w_pad = 0.0001)

        plt.savefig(os.path.join(os.path.dirname(__file__),
                                    'output','mala_vs_abc_test' + '.pdf'))

    def xest_visualise_cluster_posterior_variance(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output/single_cell_mala','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'data','')
        clusters = ['cluster_1',
                    'cluster_2',
                    'cluster_3',
                    'cluster_4']
        for cluster in clusters:
            strings = [i for i in os.listdir(saving_path) if '.npy' in i
                                                          if cluster in i]
            alpha = 0.35
            fig, ax = plt.subplots(1,5,figsize=(1.4*13,1.4*3.02))
            parameters = [2,3,0,4,1]
            parameter_names = np.array(["$\log(\\alpha_m)$ (1/min)",
                                        "$\log(\\alpha_p)$ (1/min)",
                                        "$P_0$",
                                        "$\\tau$ (mins)",
                                        "$h$",])
            for string in strings:
                ps_string = string[string.find('ions_')+5:string.find('_cluster')]
                output = np.load(saving_path + 'final_parallel_mala_output_protein_observations_'+ps_string +'_' + cluster + '_detrended.npy')
                output = output.reshape(output.shape[0]*output.shape[1],output.shape[2])

                for index, parameter in enumerate(parameters):
                    if parameter == 2:
                        hist, bins = np.histogram(np.exp(output[:,parameter]),density=True,bins=20)
                        logbins = np.geomspace(bins[0],bins[-1],20)
                        # import pdb; pdb.set_trace()
                        ax[index].hist(output[:,parameter],density=True,bins=logbins,alpha=alpha,color='#20948B')
                        ax[index].set_xlabel(parameter_names[index],fontsize=font_size*1.2)
                        ax[index].set_xlim(0,6)
                        ax[index].set_ylim(0,1.0)
                    if parameter == 3:
                        ax[index].hist(output[:,parameter],density=True,bins=np.geomspace(np.min(output[:,parameter]),np.max(output[:,parameter]),20),alpha=alpha,color='#20948B')
                        ax[index].set_xlabel(parameter_names[index],fontsize=font_size*1.2)
                    else:
                        ax[index].hist(output[:,parameter],density=True,bins=20,alpha=alpha,color='#20948B')
                        ax[index].set_xlabel(parameter_names[index],fontsize=font_size*1.2)
                ax[1].set_ylim(0,5)
                ax[0].set_ylabel("Probability",fontsize=font_size*1.2)


                def format_tick_labels(x, pos):
                    return '{:.0e}'.format(x)
                from matplotlib.ticker import FuncFormatter
                ax[2].yaxis.set_major_formatter(FuncFormatter(format_tick_labels))

                # ax[2].set_yticklabels()
            plt.tight_layout()
            fig.suptitle("Cluster " + cluster[-1],y=1.001,fontsize=font_size*1.2)
            plt.savefig(saving_path + cluster + "_posteriors.png")

    def xest_plot_mala_posteriors(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output/output_jan_25','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'data','')
        strings = [i for i in os.listdir(saving_path) if '.npy' in i
                                                      if 'ps11_fig5_1_cells_5_minutes' in i]
        for string in strings:
            ps_string = string[string.find('ps'):string.find('.npy')]
            true_parameters = np.load(loading_path + ps_string[:ps_string.find('_fi')] + '_parameter_values.npy')

            output = np.load(saving_path + 'final_parallel_mala_output_protein_observations_'+ps_string +'.npy')
            output = output.reshape(output.shape[0]*output.shape[1],output.shape[2])

            mean_repression = round(np.mean(output[:,0]),-4) # round to nearest 10000

            hist_transcription, bins_transcription, _ = plt.hist(np.exp(output[:,2]),bins=20,density=True)
            logbins_transcription = np.geomspace(bins_transcription[0],
                                                 bins_transcription[-1],
                                                 20)
            plt.clf()

            hist_translation, bins_translation, _ = plt.hist(np.exp(output[:,3]),bins=20,density=True)
            logbins_translation = np.geomspace(bins_translation[0],
                                               bins_translation[-1],
                                               20)
            plt.clf()

            my_figure = plt.figure(figsize=(18.87,5.66))
            # my_figure.text(.5,.005,'360 observations taken every 5 minutes',ha='center',fontsize=20)
            plt.subplot(1,5,1)
            # sns.kdeplot(output[:,0])
            heights, bins, _ = plt.hist(output[:,0]/10000,bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.vlines(true_parameters[0]/10000,0,1.1*max(heights),color='k',lw=2)
            plt.xlim(xmin=0.8,xmax=8)
            # plt.xticks([0.5*mean_repression,1.5*mean_repression],labels=[int(0.5*mean_repression),int(1.5*mean_repression)])
            plt.xlabel('$P_0$ (10e4)',fontsize=font_size)
            plt.ylabel('Probability',fontsize=font_size)

            plt.subplot(1,5,2)
            # sns.kdeplot(output[:,1],bw=0.4)
            heights, bins, _ = plt.hist(output[:,1],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.vlines(true_parameters[1],0,1.1*max(heights),color='k',lw=2)
            plt.xlim(xmin=2,xmax=6)
            plt.xlabel('$h$',fontsize=font_size)

            plt.subplot(1,5,3)
            plt.xscale('log')
            # sns.kdeplot(output[:,2])
            heights, bins, _ = plt.hist(output[:,2],bins=logbins_transcription,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.vlines(np.log(true_parameters[4]),0,1.1*max(heights),color='k',lw=2)
            plt.xlim(xmin=0.2,xmax=7)
            plt.xticks([1,6],labels=[1,6])
            plt.xlabel('log($\\alpha_m$) (1/min)',fontsize=font_size)

            plt.subplot(1,5,4)
            plt.xscale('log')
            # sns.kdeplot(output[:,3])
            heights, bins, _ = plt.hist(output[:,3],bins=np.geomspace(1.6,3.75,20),density=True,ec='black',color='#20948B',alpha=0.3)
            # import pdb; pdb.set_trace()
            plt.vlines(np.log(true_parameters[5]),0,1.1*max(heights),color='k',lw=2)
            # plt.xlim(xmin=2,xmax=4)
            plt.xlabel('log($\\alpha_p$) (1/min)',fontsize=font_size)

            plt.subplot(1,5,5)
            # sns.kdeplot(output[:,4],bw=0.4)
            heights, bins, _ = plt.hist(output[:,4],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
            plt.vlines(true_parameters[6],0,1.1*max(heights),color='k',lw=2)
            plt.xlim(xmin=5,xmax=40)
            plt.xlabel('$\\tau$ (mins)',fontsize=font_size)

            plt.tight_layout()
            # plt.show()
            # saving_path = os.path.join(os.path.dirname(__file__), 'output','')
            my_figure.savefig(os.path.join(saving_path,'final_' + ps_string + '_posteriors_mala.png'))
            plt.clf()
            parameter_names = np.array(["$P_0$",
                                        "$h$",
                                        "$\\alpha_m$",
                                        "$\\alpha_p$",
                                        "$\\tau$"])

            output[:,[2,3]] = np.exp(output[:,[2,3]])
            df = pd.DataFrame(output[::10,:],columns=parameter_names)

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

            correlation_matrix = df.corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool),k=1)
            fig, ax = plt.subplots(figsize=(7.92*0.85,5.94*0.85))
            # sns.set(font_scale=font_size*0.1)
            sns.heatmap(correlation_matrix,mask=mask,annot=True,cmap=cm,cbar_kws={'label': 'Correlation coefficient, $\\nu$'},ax=ax)
            plt.savefig(saving_path + 'correlations_' + ps_string + '.png'); plt.close()

            # Create a pair grid instance
            # import pdb; pdb.set_trace()
            grid = sns.PairGrid(data= df[parameter_names[[0,3]]])
            # Map the plots to the locations
            grid = grid.map_upper(corrfunc)
            grid = grid.map_lower(sns.scatterplot, alpha=0.002,color='#20948B')
            grid = grid.map_lower(sns.kdeplot,color='k')
            grid = grid.map_diag(sns.histplot, bins = 20,color='#20948B');
            plt.savefig(saving_path + 'low_corr_pairplot_' + ps_string + '.png'); plt.close()

    def xest_plot_experimental_mala_posteriors(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')
        loading_path = os.path.join(os.path.dirname(__file__), 'data','')
        cell_string = '280317p6_cell_11_cluster_2'
        output = np.load(saving_path + 'final_parallel_mala_output_protein_observations_' + cell_string + '_detrended.npy')
        output = output.reshape(output.shape[0]*output.shape[1],output.shape[2])

        my_figure = plt.figure(figsize=(20,6))
        plt.subplot(1,5,1)
        sns.kdeplot(output[:,0]/10000)
        heights, bins, _ = plt.hist(output[:,0]/10000,bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.xlim(xmin=1.0,xmax=7.5)
        plt.xlabel('(10e4)')
        plt.title('Repression Threshold')
        plt.ylabel('Density',fontsize=font_size)

        plt.subplot(1,5,2)
        sns.kdeplot(output[:,1])
        heights, bins, _ = plt.hist(output[:,1],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.xlim(xmin=2,xmax=6)
        plt.title('Hill Coefficient')

        plt.subplot(1,5,3)
        sns.kdeplot(output[:,2])
        heights, bins, _ = plt.hist(output[:,2],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.xlim(xmin=0,xmax=5)
        plt.xlabel('(log(1/min))',fontsize=font_size)
        plt.title('$\log($Transcription Rate)')

        plt.subplot(1,5,4)
        sns.kdeplot(output[:,3])
        heights, bins, _ = plt.hist(output[:,3],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.xlim(xmin=3.4,xmax=3.8)
        plt.xlabel('(log(1/min))',fontsize=font_size)
        plt.title('log(Translation Rate)')

        plt.subplot(1,5,5)
        sns.kdeplot(output[:,4],bw=0.4)
        heights, bins, _ = plt.hist(output[:,4],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.xlim(xmin=0,xmax=20)
        plt.xlabel('(min)',fontsize=font_size)
        plt.title('Transcriptional Delay')

        plt.tight_layout()
        # plt.show()
        my_figure.savefig(os.path.join(saving_path,'final_' + cell_string + '_posteriors_mala.pdf'))

    def xest_plot_mh_posteriors(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'data','')
        ps_string = 'ps11_ds4'
        mh_chains = np.load(saving_path + 'final_mh_output_protein_observations_' + ps_string + '.npy')
        burn_in = 0
        output = mh_chains[:,burn_in:,:].reshape(mh_chains.shape[0]*(mh_chains.shape[1]-burn_in),mh_chains.shape[2])
        true_parameters = np.load(loading_path + ps_string[:ps_string.find('_')] + '_parameter_values.npy')

        mean_repression = round(np.mean(output[:,0]),-4) # round to nearest 10000

        my_figure = plt.figure(figsize=(20,6))
        # my_figure.text(.5,.005,'360 observations taken every 5 minutes',ha='center',fontsize=20)
        plt.subplot(1,5,1)
        sns.kdeplot(output[:,0])
        heights, bins, _ = plt.hist(output[:,0],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[0],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=8000,xmax=70000)
        # plt.xticks([0.5*mean_repression,1.5*mean_repression],labels=[int(0.5*mean_repression),int(1.5*mean_repression)])
        plt.title('Repression Threshold')
        plt.ylabel('Probability',fontsize=font_size)

        plt.subplot(1,5,2)
        sns.kdeplot(output[:,1],bw=0.4)
        heights, bins, _ = plt.hist(output[:,1],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[1],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2,xmax=6)
        plt.title('Hill Coefficient')

        plt.subplot(1,5,3)
        # plt.xscale('log')
        sns.kdeplot(output[:,2])
        heights, bins, _ = plt.hist(output[:,2],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(np.log(true_parameters[4]),0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=-2,xmax=5)
        plt.xlabel('(log(1/min))',fontsize=font_size)
        plt.title('$\log($Transcription Rate)')

        plt.subplot(1,5,4)
        # plt.xscale('log')
        sns.kdeplot(output[:,3])
        heights, bins, _ = plt.hist(output[:,3],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(np.log(true_parameters[5]),0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2,xmax=4)
        plt.xlabel('(log(1/min))',fontsize=font_size)
        plt.title('log(Translation Rate)')

        plt.subplot(1,5,5)
        sns.kdeplot(output[:,4],bw=0.4)
        heights, bins, _ = plt.hist(output[:,4],bins=20,density=True,ec='black',color='#20948B',alpha=0.3)
        plt.vlines(true_parameters[6],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=5,xmax=40)
        plt.xlabel('(min)',fontsize=font_size)
        plt.title('Transcriptional Delay')

        plt.tight_layout()
        # plt.show()
        my_figure.savefig(os.path.join(saving_path,ps_string + '_posteriors_mh.pdf'))

    def xest_plot_experimental_posteriors(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'output','')
        loading_path             = os.path.join(os.path.dirname(__file__), 'output','')
        chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('final_parallel_mala_output_protein_observations_280317p6_cell_70_cluster_1_detrended.npy')]
        for index, chain_path_string in enumerate(chain_path_strings):
            chain = np.load(loading_path + chain_path_string)
            filename = chain_path_string[:chain_path_string.find('.npy')] + '_posterior.pdf'
            output = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])

            hist_transcription, bins_transcription, _ = plt.hist(np.exp(output[:,2]),bins=30,density=True)
            logbins_transcription = np.geomspace(bins_transcription[0],
                                                 bins_transcription[-1],
                                                 len(bins_transcription))
            plt.clf()

            hist_translation, bins_translation, _ = plt.hist(np.exp(output[:,3]),bins=30,density=True)
            logbins_translation = np.geomspace(bins_translation[0],
                                               bins_translation[-1],
                                               len(bins_translation))
            plt.clf()

            my_figure = plt.figure(figsize=(25,5))
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
        mala = np.load(saving_path + 'mala_output_repression.npy')
        random_walk = np.load(saving_path + 'random_walk_repression.npy')
        true_values = np.load("data/ps3_parameter_values.npy")
        # import pdb; pdb.set_trace()
        # mh_mean_error = [np.std(random_walk[:i])/3407.99 for i in range(2000)]
        # mala_mean_error = [np.std(mala[:i])/3407.99 for i in range(2000)]
        # plt.figure(figsize=(7,5))
        # plt.plot(mh_mean_error,label='MH',color='#F18D9E')
        # plt.plot(mala_mean_error,label='MALA',color='#20948B')
        # plt.xlabel('Iterations')
        # plt.ylabel('Coefficient of Variation')
        # plt.title('Repression Threshold')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(saving_path + '2d_cov_repression.png')

        height = 0.012
        bw = 0.3
        bins=50
        lw = 3

        my_figure = plt.figure(figsize=(13.35*0.7,6.68*0.7))
        _,bins,_ = plt.hist(mala,density=True,bins=bins,alpha=0.8,color='#20948B',label='MALA')
        plt.vlines(np.mean(mala),0,height,color='#20948B',label='MALA mean',lw=lw)
        plt.hist(random_walk[:,0],density=True,bins=bins,alpha=0.6,color='#F18D9E',label='MH')
        plt.vlines(np.mean(random_walk[:,0]),0,height,color='#F18D9E',linestyle='dashed',label='MH Mean',lw=lw)
        plt.vlines(true_values[0],0,height,color='k',label='True Mean',lw=lw)
        plt.xlabel("$P_0$",fontsize=font_size)
        plt.legend()
        plt.xlim(xmin=2*bins[0]-bins[2],xmax=2*bins[-1]-bins[-3])
        plt.ylabel('Probability',fontsize=font_size)
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','algo_comparison_repression.png'))

    def xest_compare_mala_random_walk_autocorrelation(self):
        import emcee as em
        saving_path  = os.path.join(os.path.dirname(__file__), 'output','')
        mala = np.load(saving_path + 'parallel_mala_output_protein_observations_ps11_ds4.npy')
        random_walk = np.load(saving_path + 'final_mh_output_protein_observations_ps11_ds4.npy')
        numlags = 20
        import pdb; pdb.set_trace()


        mala_lagtimes = np.zeros((5,8))
        random_walk_lagtimes = np.zeros((5,8))
        for i in range(5):
            for j in range(8):
                mala_lagtimes[i,j] = em.autocorr.integrated_time(mala[j,:,i],quiet=True)
                random_walk_lagtimes[i,j] = em.autocorr.integrated_time(random_walk[j,:,i],quiet=True)

        for i in range(5):
            print(i)
            print(np.mean(mala_lagtimes[i]))
            print(np.mean(random_walk_lagtimes[i]))

        parameter_names = np.array(['Repression Threshold',
                                    'Hill Coefficient',
                                    'Transcription Rate',
                                    'Translation Rate',
                                    'Transcriptional Delay'])

        myfigure = plt.figure(figsize=(10,2))
        plt.scatter(parameter_names,[np.mean(mala_lagtimes[i])/np.mean(mala_lagtimes[i]) for i in range(5)],color='#20948B',label='MALA')
        plt.scatter(parameter_names,[np.mean(random_walk_lagtimes[i])/np.mean(mala_lagtimes[i]) for i in range(5)],color='#F18D9E',label='MH')
        plt.tight_layout()
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('Relative Lag Times')
        plt.legend()
        plt.show()

        import pdb; pdb.set_trace()


        # my_figure, ax = plt.subplots(2,2,figsize=(10,5))
        # ax[0,0].acorr(mala_repression[:,0] - np.mean(mala_repression[:,0]),maxlags=numlags,color='#20948B',label='MALA',lw=2)
        # ax[0,0].set_xlim(xmin=-0.05,xmax=numlags)
        # ax[0,0].set_ylabel('Repression Threshold \n autocorrelation')
        # ax[0,0].set_xlabel('Lags')
        # ax[0,0].set_title('MALA')
        # ax[0,0].text(14.0,0.7,'$\\tau =$ ' + str(round(mala_repression_lagtime[0],2)))
        # ax[0,1].acorr(random_walk_repression[:,0] - np.mean(random_walk_repression[:,0]),maxlags=numlags,color='#F18D9E',label='MH',lw=2)
        # ax[0,1].set_xlim(xmin=-0.05,xmax=numlags)
        # ax[0,1].set_title('MH')
        # ax[0,1].text(14.0,0.7,'$\\tau =$ ' + str(round(random_walk_repression_lagtime[0],2)))
        # ax[0,1].set_xlabel('Lags')
        # ax[1,0].acorr(mala_hill[:,0] - np.mean(mala_hill[:,0]),maxlags=numlags,color='#20948B',lw=2)
        # ax[1,0].set_xlim(xmin=-0.05,xmax=numlags)
        # ax[1,0].set_xlabel('Lags')
        # ax[1,0].set_ylabel('Hill Coefficient \n autocorrelation')
        # ax[1,0].text(14.0,0.7,'$\\tau =$ ' + str(round(mala_hill_lagtime[0],2)))
        # ax[1,1].acorr(random_walk_hill[:,1] - np.mean(random_walk_hill[:,1]),maxlags=numlags,color='#F18D9E',lw=2)
        # ax[1,1].set_xlabel('Lags')
        # ax[1,1].text(14.0,0.7,'$\\tau =$ ' + str(round(random_walk_hill_lagtime[0],2)))
        # ax[1,1].set_xlim(xmin=-0.05,xmax=numlags)
        # plt.tight_layout()
        #
        # my_figure.savefig(os.path.join(os.path.dirname(__file__),
        #                                'output','autocorrelation_plot.png'))

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
        saving_path = os.path.join(os.path.dirname(__file__), 'data/figure_5','')
        ps_strings = ["ps6","ps9"]
        time = 736
        durations = [i*time for i in range(1,6)]
        frequencies = [5,8,12,15]
        batches = [1,2,3,4,5]
        for batch in batches:
            for ps_string in ps_strings:
                for obs_index, observation_duration in enumerate(durations):
                    for observation_frequency in frequencies:
                        parameters = np.load(loading_path + ps_string + "_parameter_values.npy")
                        no_of_observations = np.int(observation_duration/observation_frequency)

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
                        np.save(saving_path + 'protein_observations_' + ps_string + '_{i}_cells_{j}_minutes_{k}.npy'.format(i=obs_index+1,j=observation_frequency,k=batch),
                        protein_at_observations[0::observation_frequency,:])

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

    def xest_plot_protein_observations(self):
        loading_path = os.path.join(os.path.dirname(__file__), 'data','')
        saving_path = os.path.join(os.path.dirname(__file__), 'output','')

        data = np.load(loading_path + 'protein_observations_ps11_fig5_1_cells_5_minutes.npy')
        plt.figure(figsize=(9.13,5.71))
        plt.scatter(data[:,0],data[:,1],marker='o',s=12,c='#F18D9E')
        plt.xlabel('Time (mins)',fontsize=font_size)
        plt.ylabel('Protein molecules',fontsize=font_size)
        plt.tight_layout()
        plt.savefig(saving_path + 'best_protein.png')

    def xest_insilico_data_generation(self):
        ## run a sample simulation to generate example protein data
        in_silico_data = hes5.generate_langevin_trajectory(duration = 900,
                                                           repression_threshold = 47514,
                                                           hill_coefficient = 4.77,
                                                           mRNA_degradation_rate = np.log(2)/30,
                                                           protein_degradation_rate = np.log(2)/90,
                                                           basal_transcription_rate = 2.65,
                                                           translation_rate = 17.6,
                                                           transcription_delay = 38,
                                                           equilibration_time = 1000)

        new_in_silico_data = hes5.generate_langevin_trajectory(duration = 900,
                                                               repression_threshold = 47514,
                                                               hill_coefficient = 4.77,
                                                               mRNA_degradation_rate = np.log(2)/30,
                                                               protein_degradation_rate = np.log(2)/90,
                                                               basal_transcription_rate = 2.65,
                                                               translation_rate = 17.6,
                                                               transcription_delay = 38,
                                                               equilibration_time = 1000)

        x = np.linspace(1,900,900)
        my_figure = plt.figure(figsize=(8,6))#,fontsize=100)
        plt.subplot(2,1,1)
        plt.plot(x,1000*np.sin(x/100)+50000,c='#F18D9E',label='Protein',lw=3)
        plt.xlabel('Time')
        plt.ylabel('Molecule Number')

        plt.subplot(2,1,2)
        plt.plot(x,50000*np.ones(900)+np.random.randint(0,10,900),c='#F18D9E',label='Protein',lw=3)
        plt.xlabel('Time')
        plt.ylabel('Molecule Number')
        plt.ylim([49000,51000])


        plt.tight_layout()
        #
        # plt.subplot(3,1,2)
        # plt.scatter(np.arange(0,900,10),true_protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='true value')
        # plt.title('True Protein')
        # plt.xlabel('Time')
        # plt.ylabel('Molecule Number')
        #
        # plt.subplot(3,1,3)
        # plt.scatter(np.arange(0,900,10),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations',zorder=4)
        # plt.title('Observed Protein')
        # plt.xlabel('Time')
        # plt.ylabel('Molecule Number')
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','oscillatory_and_flat.pdf'))

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

    def jest_kalman_random_walk_for_mala_comparison(self,data_filename='protein_observations_90_ps3_ds1.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        protein_at_observations = np.array([np.load(os.path.join(saving_path,data_filename))])
        true_parameter_values = np.load(os.path.join(saving_path,'ps3_parameter_values.npy'))

        mean_protein = np.mean([i[j,1] for i in protein_at_observations for j in range(i.shape[0])])

        iterations = 2500
        number_of_chains = 1
        step_size = 1.0
        measurement_variance = np.power(100,2)
        # draw random initial states for the parallel chains
        from scipy.stats import uniform
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([true_parameter_values[2],true_parameter_values[3]])
        for initial_state_index, _ in enumerate(initial_states):
            initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                          np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))
        # true parameters ps3 -- [3407.99,5.17,np.log(2)/30,np.log(2)/90,15.86,1.27,30]
        hyper_parameters = np.array([50,2*mean_protein-50,2,6-2,0,1,0,1,0.01,120-0.01,0.01,40-0.01,1,40-1]) # uniform
        proposal_covariance = np.diag([1.0])

        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:] = np.array([true_parameter_values[0],
                                      true_parameter_values[1],
                                      true_parameter_values[2],
                                      true_parameter_values[3],
                                      np.log(true_parameter_values[4]),
                                      np.log(true_parameter_values[5]),
                                      true_parameter_values[6]])

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_random_walk,
                                                          args=(iterations,
                                                                protein_at_observations,
                                                                hyper_parameters,
                                                                measurement_variance,
                                                                step_size,
                                                                proposal_covariance,
                                                                initial_state))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all finished so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,iterations,5))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','delay_final_mh_output_' + data_filename),array_of_chains)

    def xest_kalman_random_walk(self,data_filename='protein_observations_ps9_1_cells_12_minutes_2.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        protein_at_observations = np.array([np.load(os.path.join(saving_path+'figure_5/',data_filename))])
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('_cell')-2
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        true_parameter_values = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))

        mean_protein = np.mean(protein_at_observations[:,1])

        iterations = 2000
        number_of_chains = 8
        step_size = 1.0
        measurement_variance = np.power(true_parameter_values[-1],2)
        # draw random initial states for the parallel chains
        from scipy.stats import uniform
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([true_parameter_values[2],true_parameter_values[3]])
        for initial_state_index, _ in enumerate(initial_states):
            initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                          np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))
        # true parameters ps3 -- [3407.99,5.17,np.log(2)/30,np.log(2)/90,15.86,1.27,30]
        hyper_parameters = np.array([50,2*mean_protein-50,2,6-2,0,1,0,1,0.01,120-0.01,0.01,40-0.01,1,40-1]) # uniform
        # covariance = np.diag([5e+3,0.03,0.01,0.01,1.0])

        # if we already have mcmc samples, we can use them to construct a covariance matrix to make sampling better
        if os.path.exists(os.path.join(
                          os.path.dirname(__file__),
                          'output','final_mh_output_' + data_filename)):
            print("Posterior samples already exist, sampling directly without warm up...")
            array_of_chains = np.load(saving_path + '../output/final_mh_output_' + data_filename)
            # start from mode
            initial_states = np.zeros((number_of_chains,7))
            initial_states[:] = np.array([true_parameter_values[0],
                                          true_parameter_values[1],
                                          true_parameter_values[2],
                                          true_parameter_values[3],
                                          np.log(true_parameter_values[4]),
                                          np.log(true_parameter_values[5]),
                                          true_parameter_values[6]])

            previous_number_of_samples = array_of_chains.shape[1]
            previous_number_of_chains = array_of_chains.shape[0]

            samples_with_burn_in = array_of_chains[:,int(previous_number_of_samples/2):,:].reshape(int(previous_number_of_samples/2)*previous_number_of_chains,array_of_chains.shape[-1])
            proposal_covariance = np.cov(samples_with_burn_in.T)

            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_random_walk,
                                                              args=(iterations,
                                                                    protein_at_observations,
                                                                    hyper_parameters,
                                                                    measurement_variance,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    initial_state))
                                for initial_state in initial_states ]
            ## Let the pool know that these are all finished so that the pool will exit afterwards
            # this is necessary to prevent memory overflows.
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,iterations,5))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            np.save(os.path.join(os.path.dirname(__file__), 'output','final_mh_output_' + data_filename),array_of_chains)

        else:
            print("Initial burn in of ",str(np.int(iterations*0.3)), " samples...")
            initial_burnin_number_of_samples = np.int(iterations*0.3)
            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_random_walk,
                                                              args=(initial_burnin_number_of_samples,
                                                                    protein_at_observations,
                                                                    hyper_parameters,
                                                                    measurement_variance,
                                                                    step_size,
                                                                    np.power(np.diag([2*mean_protein,4,9,8,39]),2),
                                                                    initial_state))
                                for initial_state in initial_states ]
            ## Let the pool know that these are all so that the pool will exit afterwards
            # this is necessary to prevent memory overflows.
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,initial_burnin_number_of_samples,5))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            np.save(os.path.join(os.path.dirname(__file__), 'output','first_mh_output_' + data_filename),array_of_chains)

            # second burn in
            print("Second burn in of ",str(np.int(0.7*iterations))," samples...")
            second_burnin_number_of_samples = np.int(0.7*iterations)
            # make new initial states
            initial_states = np.zeros((number_of_chains,7))
            initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
            for initial_state_index, _ in enumerate(initial_states):
                initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                              np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))

            samples_with_burn_in = array_of_chains[:,int(initial_burnin_number_of_samples/2):,:].reshape(int(initial_burnin_number_of_samples/2)*number_of_chains,array_of_chains.shape[-1])
            proposal_covariance = np.cov(samples_with_burn_in.T)

            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_random_walk,
                                                              args=(second_burnin_number_of_samples,
                                                                    protein_at_observations,
                                                                    hyper_parameters,
                                                                    measurement_variance,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    initial_state))
                                for initial_state in initial_states ]
            ## Let the pool know that these are all finished so that the pool will exit afterwards
            # this is necessary to prevent memory overflows.
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,second_burnin_number_of_samples,5))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            np.save(os.path.join(os.path.dirname(__file__), 'output','second_mh_output_' + data_filename),array_of_chains)

            # sample directly
            print("Now sampling directly...")
            # make new initial states
            initial_states = np.zeros((number_of_chains,7))
            initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
            for initial_state_index, _ in enumerate(initial_states):
                initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                              np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))

            samples_with_burn_in = array_of_chains[:,int(second_burnin_number_of_samples/2):,:].reshape(int(second_burnin_number_of_samples/2)*number_of_chains,array_of_chains.shape[-1])
            proposal_covariance = np.cov(samples_with_burn_in.T)

            pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
            process_results = [ pool_of_processes.apply_async(hes_inference.kalman_random_walk,
                                                              args=(iterations,
                                                                    protein_at_observations,
                                                                    hyper_parameters,
                                                                    measurement_variance,
                                                                    step_size,
                                                                    proposal_covariance,
                                                                    initial_state))
                                for initial_state in initial_states ]
            ## Let the pool know that these are all finished so that the pool will exit afterwards
            # this is necessary to prevent memory overflows.
            pool_of_processes.close()

            array_of_chains = np.zeros((number_of_chains,iterations,5))
            for chain_index, process_result in enumerate(process_results):
                this_chain = process_result.get()
                array_of_chains[chain_index,:,:] = this_chain
            pool_of_processes.join()

            np.save(os.path.join(os.path.dirname(__file__), 'output','final_mh_output_' + data_filename),array_of_chains)

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
        protein_at_observations = np.array([np.load(saving_path + 'protein_observations_90_ps3_ds1.npy')])
        true_parameter_values = np.load(saving_path + 'ps3_parameter_values.npy')
        number_of_evaluations = 50
        likelihood_at_multiple_parameters = np.zeros(number_of_evaluations)
        mean_protein = np.mean([i[j,1] for i in protein_at_observations for j in range(i.shape[0])])
        print(mean_protein)

        repression_threshold = true_parameter_values[0]
        hill_coefficient = true_parameter_values[1]
        mRNA_degradation_rate    = np.log(2)/30
        protein_degradation_rate = np.log(2)/90
        basal_transcription_rate = true_parameter_values[4]
        translation_rate = true_parameter_values[5]
        transcription_delay = true_parameter_values[6]
        measurement_variance = np.power(100,2)
        # import pdb; pdb.set_trace()

        for index, parameter in enumerate(np.linspace(1.1,1.55,number_of_evaluations)):
            likelihood_at_multiple_parameters[index] = -hes_inference.calculate_log_likelihood_at_parameter_point(model_parameters=np.array([repression_threshold,
                                                                                                                                            hill_coefficient,
                                                                                                                                            mRNA_degradation_rate,
                                                                                                                                            protein_degradation_rate,
                                                                                                                                            basal_transcription_rate,
                                                                                                                                            parameter,
                                                                                                                                            transcription_delay]),
                                                                                                                 protein_at_observations=protein_at_observations,
                                                                                                                 measurement_variance = measurement_variance)


        np.save(os.path.join(os.path.dirname(__file__), 'output','likelihood_translation.npy'),likelihood_at_multiple_parameters)

    def qest_make_figure_4_mh_and_mala_convergence(self,data_filename = 'protein_observations_ps11_ds4.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('ps') + 3
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        # protein_at_observations = np.array([np.load(os.path.join(saving_path,'protein_observations_'+ps_string))])
        true_parameter_values = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))
        measurement_variance = np.power(true_parameter_values[-1],2)

        true_posterior = np.load(loading_path + "final_mh_output_" + data_filename).reshape(8,200000,5)
        # import pdb; pdb.set_trace()
        mean = np.mean(true_posterior.reshape(true_posterior.shape[0]*true_posterior.shape[1],true_posterior.shape[2]),axis=0)
        std = np.std(true_posterior.reshape(true_posterior.shape[0]*true_posterior.shape[1],true_posterior.shape[2]),axis=0)

        datasets = {'MH' : np.load(loading_path + 'long_warm_up_mh_output_' + data_filename).reshape(8,50000,5),
                    'MALA' : np.load(loading_path + 'long_warm_up_parallel_mala_output_' + data_filename).reshape(8,50000,5)}
        fig, ax = plt.subplots(1,2,figsize=(14,6))
        burn_in = 0
        window = 1000
        indices = np.concatenate((np.arange(burn_in+1,burn_in+1000),np.arange(burn_in+1000,50000-10,window)))
        mean_error_dict = {}
        std_error_dict = {}
        cov_error_dict = {}

        for key in datasets.keys():
            mean_error_dict[key] = np.zeros(len(indices))
            std_error_dict[key] = np.zeros(len(indices))
            cov_error_dict[key] = np.zeros(len(indices))
            for index, chain_length in enumerate(indices):
                short_chains = datasets[key][:,burn_in:chain_length+10,:] # DIMS: 8 x chain_length x 5
                # mean
                short_chain_means = np.mean(short_chains,axis=1) # 8 x 5
                short_chain_mean_errors = np.abs(mean - short_chain_means)/mean # 8 x 5
                total_short_chain_mean_errors = np.sum(short_chain_mean_errors,axis=1) # 8
                mean_error_dict[key][index] = np.mean(total_short_chain_mean_errors)

                # std
                short_chain_stds = np.std(short_chains,axis=1) # 8 x 5
                short_chain_std_errors = np.abs(std - short_chain_stds)/std # 8 x 5
                total_short_chain_std_errors = np.sum(short_chain_std_errors,axis=1) # 8
                std_error_dict[key][index] = np.mean(total_short_chain_std_errors)

                #cov
                short_chain_cov_errors = short_chain_stds/mean # 8 x 5
                total_short_chain_cov_errors = np.sum(short_chain_cov_errors,axis=1) # 8
                cov_error_dict[key][index] = np.mean(total_short_chain_cov_errors)
                # import pdb; pdb.set_trace()

        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        for key, value in std_error_dict.items():
            ax[0].plot(indices,value)
        ax[0].set_ylabel("Relative std error")
        ax[0].set_xlabel("Iterations")

        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        for key, value in mean_error_dict.items():
            ax[1].plot(indices,value,label=key)
        ax[1].set_ylabel("Relative mean error")
        ax[1].set_xlabel("Iterations")

        plt.tight_layout()
        fig.legend(loc="center right",
                   borderaxespad=0.25)
        plt.subplots_adjust(right=.85)
        plt.savefig(loading_path + "multivariate_mh_vs_mala_convergence_speed" + "_long_warm_up_bimodal.png")
        plt.clf()

    def qest_make_figure_4_qqplots(self,data_filename = 'protein_observations_ps9_1_cells_12_minutes_2.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        # protein_at_observations = np.array([np.load(os.path.join(saving_path,data_filename))])
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('_ds')
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        # true_parameter_values = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))
        # measurement_variance = np.power(true_parameter_values[-1],2)

        parameter_labels = np.array(["$P_0$",
                                    "$h$",
                                    "$\log(\\alpha_m)$ (1/min)",
                                    "$\log(\\alpha_p)$ (1/min)",
                                    "$\\tau$ (mins)"])

        # parameter_labels = ['Repression Threshold',
        #                     'Hill Coefficient',
        #                     'log(Transcription Rate)',
        #                     'log(Translation Rate)',
        #                     'Transcriptional Delay']

        quantile_indices = np.arange(0,1,0.01)

        true_posterior = np.load(loading_path + "final_mh_output_" + data_filename).reshape(1600000,5)
        true_quantiles = np.quantile(true_posterior,quantile_indices,axis=0)

        datasets = {'MH' : np.load(loading_path + 'long_warm_up_mh_output_' + data_filename)[:,25000:,:].reshape(200000,5),
                    'MALA' : np.load(loading_path + 'long_warm_up_parallel_mala_output_' + data_filename)[:,25000:,:].reshape(200000,5)}

        fig, ax = plt.subplots(1,5,figsize=(22,5))
        quantiles_dict = {}

        for key in datasets.keys():
            quantiles_dict[key] = np.quantile(datasets[key],quantile_indices,axis=0)

        for i in range(5):
            if i == 0: # label
                ax[i].plot(true_quantiles[:,i],true_quantiles[:,i],color='grey',label='True',alpha=0.7,linewidth=2)
                ax[i].set_xlabel(parameter_labels[i])
            else:
                ax[i].plot(true_quantiles[:,i],true_quantiles[:,i],color='grey',alpha=0.7,linewidth=2)
                ax[i].set_xlabel(parameter_labels[i])
            for key, value in quantiles_dict.items():
                if i == 0: # label
                    ax[i].plot(true_quantiles[:,i],quantiles_dict[key][:,i],'--',alpha=0.5,label=key,linewidth=3)
                else:
                    ax[i].plot(true_quantiles[:,i],quantiles_dict[key][:,i],'--',alpha=0.5,linewidth=3)
        plt.tight_layout()
        fig.legend(loc="center right",
                   borderaxespad=0.25)
        plt.subplots_adjust(right=.9)
        plt.savefig(loading_path + "multivariate_mh_vs_mala_qqplots_unimodal.png")
        plt.clf()

    def xest_make_figure_4_mh_and_mala_histograms(self,data_filename = 'protein_observations_ps11_ds4.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        protein_at_observations = np.array([np.load(os.path.join(saving_path,data_filename))])
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('_ds')
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        true_parameters = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))
        measurement_variance = np.power(true_parameters[-1],2)

        true_posterior = np.load(loading_path + "final_mh_output_protein_observations_ps11_ds4.npy")
        true_posterior = true_posterior.reshape(true_posterior.shape[0]*true_posterior.shape[1],true_posterior.shape[2])

        mala_short_warm_up = np.load(loading_path + 'short_warm_up_parallel_mala_output_' + data_filename)
        mala_long_warm_up = np.load(loading_path + 'long_warm_up_parallel_mala_output_' + data_filename)
        mala_short_warm_up = mala_short_warm_up.reshape(mala_short_warm_up.shape[0]*mala_short_warm_up.shape[1],mala_short_warm_up.shape[2])
        mala_long_warm_up = mala_long_warm_up.reshape(mala_long_warm_up.shape[0]*mala_long_warm_up.shape[1],mala_long_warm_up.shape[2])
        mh_short_warm_up = np.load(loading_path + 'short_warm_up_mh_output_' + data_filename)
        mh_long_warm_up = np.load(loading_path + 'long_warm_up_mh_output_' + data_filename)
        mh_short_warm_up = mh_short_warm_up.reshape(mh_short_warm_up.shape[0]*mh_short_warm_up.shape[1],mh_short_warm_up.shape[2])
        mh_long_warm_up = mh_long_warm_up.reshape(mh_long_warm_up.shape[0]*mh_long_warm_up.shape[1],mh_long_warm_up.shape[2])

        # SET COLOURS
        mh_long_colour = '#f18d9e'
        mh_short_colour = '#8d9ef1'
        mala_long_colour = '#20948B'
        mala_short_colour = '#286dcc'

        nbins=25

        my_figure = plt.figure(figsize=(20,7))
        my_figure.suptitle('MH')
        # my_figure.text(.5,.005,'360 observations taken every 5 minutes',ha='center',fontsize=20)
        plt.subplot(1,5,1)
        sns.kdeplot(true_posterior[:,0]/10000,color='k')
        heights, bins, _ = plt.hist(mh_long_warm_up[:,0]/10000,bins=nbins,density=True,ec='grey',color=mh_long_colour,alpha=0.7,label='long warm up')
        heights, bins, _ = plt.hist(mh_short_warm_up[:,0]/10000,bins=nbins,density=True,ec='grey',color=mh_short_colour,alpha=0.5,label='short warm up')
        plt.vlines(true_parameters[0]/10000,0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=0.4,xmax=8)
        plt.legend(loc='upper left')
        # plt.xticks([0.5*mean_repression,1.5*mean_repression],labels=[int(0.5*mean_repression),int(1.5*mean_repression)])
        plt.title('Repression Threshold')
        plt.ylabel('Density',fontsize=font_size)
        plt.xlabel('(10e4)',fontsize=font_size)

        plt.subplot(1,5,2)
        sns.kdeplot(true_posterior[:,1],bw=0.4,color='k')
        heights, bins, _ = plt.hist(mh_long_warm_up[:,1],bins=nbins,density=True,ec='grey',color=mh_long_colour,alpha=0.7)
        heights, bins, _ = plt.hist(mh_short_warm_up[:,1],bins=nbins,density=True,ec='grey',color=mh_short_colour,alpha=0.5)
        plt.vlines(true_parameters[1],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=2,xmax=6)
        plt.title('Hill Coefficient')

        plt.subplot(1,5,3)
        # plt.xscale('log')
        sns.kdeplot(true_posterior[:,2],color='k')
        heights, bins, _ = plt.hist(mh_long_warm_up[:,2],bins=nbins,density=True,ec='grey',color=mh_long_colour,alpha=0.7)
        heights, bins, _ = plt.hist(mh_short_warm_up[:,2],bins=nbins,density=True,ec='grey',color=mh_short_colour,alpha=0.5)
        plt.vlines(np.log(true_parameters[4]),0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=-1,xmax=5)
        plt.xlabel('(log(1/min))',fontsize=font_size)
        plt.title('$\log($Transcription Rate)')

        plt.subplot(1,5,4)
        # plt.xscale('log')
        sns.kdeplot(true_posterior[:,3],color='k')
        heights, bins, _ = plt.hist(mh_long_warm_up[:,3],bins=nbins,density=True,ec='grey',color=mh_long_colour,alpha=0.7)
        heights, bins, _ = plt.hist(mh_short_warm_up[:,3],bins=nbins,density=True,ec='grey',color=mh_short_colour,alpha=0.5)
        plt.vlines(np.log(true_parameters[5]),0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=1.5,xmax=4)
        plt.xlabel('(log(1/min))',fontsize=font_size)
        plt.title('log(Translation Rate)')

        plt.subplot(1,5,5)
        sns.kdeplot(true_posterior[:,4],bw=0.4,color='k')
        heights, bins, _ = plt.hist(mh_long_warm_up[:,4],bins=nbins,density=True,ec='grey',color=mh_long_colour,alpha=0.7)
        heights, bins, _ = plt.hist(mh_short_warm_up[:,4],bins=nbins,density=True,ec='grey',color=mh_short_colour,alpha=0.5)
        plt.vlines(true_parameters[6],0,1.1*max(heights),color='r',lw=2)
        plt.xlim(xmin=5,xmax=40)
        plt.xlabel('(min)',fontsize=font_size)
        plt.title('Transcriptional Delay')

        my_figure.tight_layout()
        my_figure.subplots_adjust(top=0.85)
        my_figure.savefig(os.path.join(loading_path,'short_and_long_posteriors_mh.pdf'))

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

    def test_visualise_kalman_filter(self):
        saving_path = os.path.join(os.path.dirname(__file__),'test_burton_et_al_2021/data','')
        loading_path = os.path.join(os.path.dirname(__file__),'test_burton_et_al_2021/output','')

        np.random.seed(42)
        data = np.load(saving_path + 'true_data_ps3.npy')
        measurement_variance = 500
        protein = np.maximum(0,data[:,2] + measurement_variance*np.random.randn(data.shape[0]))

        my_figure, ax1 = plt.subplots(figsize=(12.47*0.7,8.32*0.7))
        ax1.scatter(data[:,0],data[:,2],marker='o',s=3,color='#20948B',alpha=0.75,label='protein')
        ax1.scatter(data[0:-1:10,0],protein[0:-1:10],marker='o',s=14,color='#F18D9E')
        # ax1.scatter(protein[:,0],protein[:,1],marker='o',s=14,c='#F18D9E')
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
        my_figure.savefig(os.path.join(saving_path,'ps3_data.pdf'))


        true_data = np.load(os.path.join(saving_path,'true_data_ps3.npy'))
        protein_observations = true_data[::10,[0,2]]
        protein_observations[:,1] = protein[0:-1:10]
        true_parameters = np.load(os.path.join(saving_path,'ps3_parameter_values.npy'))[:7]

        true_state_space_mean, true_state_space_variance, _,_,_,_,_ = hes_inference.kalman_filter(protein_observations,
                                                                                                  true_parameters*0.3,
                                                                                                  measurement_variance=np.power(measurement_variance,2)/4,
                                                                                                  derivative=False)

        number_of_states = true_state_space_mean.shape[0]

        true_protein_covariance_matrix = true_state_space_variance[number_of_states:,number_of_states:]
        true_protein_variance = np.diagonal(true_protein_covariance_matrix)
        true_protein_error = np.sqrt(true_protein_variance)*2
        # remove negatives from error bar calc
        true_protein_error = np.zeros((2,len(true_protein_variance)))
        true_protein_error[0,:] = np.minimum(true_state_space_mean[:,2],np.sqrt(true_protein_variance)*2)
        true_protein_error[1,:] = np.sqrt(true_protein_variance)*2

        true_mRNA_covariance_matrix = true_state_space_variance[:number_of_states,:number_of_states]
        true_mRNA_variance = np.diagonal(true_mRNA_covariance_matrix)
        true_mRNA_error = np.sqrt(true_mRNA_variance)*2
        # remove negatives from error bar calc
        true_mRNA_error = np.zeros((2,len(true_mRNA_variance)))
        true_mRNA_error[0,:] = np.minimum(true_state_space_mean[:,1],np.sqrt(true_mRNA_variance)*2)
        true_mRNA_error[1,:] = np.sqrt(true_mRNA_variance)*2

        fig, ax = plt.subplots(4,1,figsize=(0.76*15.38,0.76*15.38*2))
        # ground truth
        ax[0].scatter(protein_observations[:,0],protein_observations[:,1],s=18,label='observed protein (known)',color='#F18D9E',zorder=2)
        ax[0].scatter(protein_observations[:,0],true_data[::10,2],s=18,label='ground truth protein (unknown)',color='black',alpha=0.8,zorder=2)
        # state space
        ax[0].scatter(true_state_space_mean[np.int(true_parameters[-1])::10,0],true_state_space_mean[np.int(true_parameters[-1])::10,2],s=18,label='state space protein',color='#20948B',zorder=1)
        ax[0].errorbar(true_state_space_mean[30:,0],true_state_space_mean[30:,2],yerr=true_protein_error[:,30:],ecolor='#98DBC6',alpha=0.1,zorder=1)
        ax[0].set_xlabel('Time (mins)',fontsize=1.1*font_size)
        ax[0].set_ylabel('Protein Copy Numbers',fontsize=1.1*font_size)
        # ax[0].legend(fontsize=0.7*font_size)

        # ground truth
        ax[1].scatter(protein_observations[:,0],true_data[::10,1],s=14,label='ground truth mRNA (unknown)',color='#8d9ef1',zorder=2)
        # state space
        ax[1].scatter(true_state_space_mean[np.int(true_parameters[-1])::10,0],true_state_space_mean[np.int(true_parameters[-1])::10,1],s=8,label='state space mRNA',color='#F69454',zorder=1)
        ax[1].errorbar(true_state_space_mean[30:,0],true_state_space_mean[30:,1],yerr=true_mRNA_error[:,30:],ecolor='#F9be98',alpha=0.1,zorder=1)
        ax[1].set_xlabel('Time (mins)',fontsize=1.1*font_size)
        ax[1].set_ylabel('mRNA Copy Numbers',fontsize=1.1*font_size)
        # ax[1].legend(fontsize=0.7*font_size)

        ax[2].scatter(protein_observations[:,0],protein_observations[:,1],s=18,label='observed protein (known)',color='#F18D9E',zorder=2)
        ax[2].scatter(protein_observations[:,0],true_data[::10,2],s=18,label='ground truth protein (unknown)',color='black',alpha=0.8,zorder=2)
        # state space
        # ax[2].scatter(true_state_space_mean[np.int(true_parameters[-1])::10,0],true_state_space_mean[np.int(true_parameters[-1])::10,2],s=18,label='state space protein',color='#20948B',zorder=1)
        # ax[2].errorbar(true_state_space_mean[30:,0],true_state_space_mean[30:,2],yerr=true_protein_error[:,30:],ecolor='#98DBC6',alpha=0.1,zorder=1)
        ax[2].set_xlabel('Time (mins)',fontsize=1.1*font_size)
        ax[2].set_ylabel('Protein Copy Numbers',fontsize=1.1*font_size)

        ax[3].scatter(protein_observations[:,0],protein_observations[:,1],s=18,label='observed protein (known)',color='#F18D9E',zorder=2)
        # ax[3].scatter(protein_observations[:,0],true_data[::10,2],s=18,label='ground truth protein (unknown)',color='black',alpha=0.6,zorder=2)
        # state space
        # ax[2].scatter(true_state_space_mean[np.int(true_parameters[-1])::10,0],true_state_space_mean[np.int(true_parameters[-1])::10,2],s=18,label='state space protein',color='#20948B',zorder=1)
        # ax[2].errorbar(true_state_space_mean[30:,0],true_state_space_mean[30:,2],yerr=true_protein_error[:,30:],ecolor='#98DBC6',alpha=0.1,zorder=1)
        ax[3].set_xlabel('Time (mins)',fontsize=1.1*font_size)
        ax[3].set_ylabel('Protein Copy Numbers',fontsize=1.1*font_size)

        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','figure_2_kalman_visualisation_bad.pdf'))

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

    def xest_multiple_mala_traces_figure_5(self,data_filename = 'protein_observations_ps12_fig5_5.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        protein_at_observations = np.array([np.load(os.path.join(saving_path,data_filename))])
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('_fig')
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        true_parameter_values = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))
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

        run_mala_for_dataset(data_filename,
                             protein_at_observations,
                             measurement_variance,
                             number_of_parameters,
                             known_parameters,
                             step_size,
                             number_of_chains,
                             number_of_samples)

    def xest_multiple_mala_traces_figure_5_coherence(self,data_filename = 'protein_observations_coherence_0.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','figure_5_b')
        loading_path = os.path.join(os.path.dirname(__file__),'output','')
        protein_at_observations = np.array([np.load(os.path.join(saving_path,data_filename))])
        ps_string_index_start = data_filename.find('coher')
        ps_string_index_end = data_filename.find('.npy')
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        true_parameter_values = np.load(os.path.join(saving_path,"parameter_values_" + ps_string + '.npy'))
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

        run_mala_for_dataset(data_filename,
                             protein_at_observations,
                             measurement_variance,
                             number_of_parameters,
                             known_parameters,
                             step_size,
                             number_of_chains,
                             number_of_samples)

    def xest_multiple_mala_traces_figure_5b(self,data_filename = 'protein_observations_ps9_1_cells_12_minutes_2.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        protein_at_observations = np.array([np.load(os.path.join(saving_path+'/figure_5',data_filename))])
        ps_string_index_start = data_filename.find('ps')
        ps_string_index_end = data_filename.find('ps') + 3
        ps_string = data_filename[ps_string_index_start:ps_string_index_end]
        true_parameter_values = np.load(os.path.join(saving_path,ps_string + '_parameter_values.npy'))

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

        run_mala_for_dataset(data_filename,
                             protein_at_observations,
                             measurement_variance,
                             number_of_parameters,
                             known_parameters,
                             step_size,
                             number_of_chains,
                             number_of_samples)

    def xest_mala_experimental_data(self,data_filename = 'protein_observations_040417_cell_52_cluster_4_detrended.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/selected_data_for_mala/')
        protein_at_observations = np.array([np.load(os.path.join(saving_path,data_filename))])
        experiment_date = data_filename[data_filename.find('ns_')+3:data_filename.find('_cel')]
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

    def xest_mala_experimental_data_parnian(self,data_filename = 'protein_observations_cell_1_detrended.npy'):
        # load data and true parameter values
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data_parnian/detrended_data/')
        protein_at_observations = np.array([np.load(os.path.join(loading_path,data_filename))])
        measurement_variance = np.power(np.round(np.load(loading_path + "measurement_variance_detrended.npy"),4),2)

        # define known parameters
        all_parameters = {'repression_threshold' : [0,None],
                          'hill_coefficient' : [1,None],
                          'mRNA_degradation_rate' : [2,None],
                          'protein_degradation_rate' : [3,None],
                          'basal_transcription_rate' : [4,None],
                          'translation_rate' : [5,None],
                          'transcription_delay' : [6,None]}

        known_parameters = {}

        known_parameter_indices = [list(known_parameters.values())[i][0] for i in [j for j in range(len(known_parameters.values()))]]
        unknown_parameter_indices = [i for i in range(len(all_parameters)) if i not in known_parameter_indices]
        number_of_parameters = len(unknown_parameter_indices)

        number_of_samples = 80000
        number_of_chains = 8
        step_size = 0.001

        run_mala_for_dataset_parnian(data_filename,
                                     protein_at_observations,
                                     measurement_variance,
                                     number_of_parameters,
                                     known_parameters,
                                     step_size,
                                     number_of_chains,
                                     number_of_samples)

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


    def xest_mala_experimental_data_degradation(self,data_filename = 'protein_observations_040417_cell_1_cluster_4_detrended.npy'):
        # load data and true parameter values
        saving_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/selected_data_for_mala/')
        protein_at_observations = np.array([np.load(os.path.join(saving_path,data_filename))])
        experiment_date = data_filename[data_filename.find('ns_')+3:data_filename.find('_cel')]
        mean_protein = np.mean(protein_at_observations[:,1])


        number_of_samples = 80000
        number_of_chains = 8
        measurement_variance = np.power(np.round(np.load(saving_path + experiment_date + "_measurement_variance_detrended.npy"),0),2)
        # draw random initial states for the parallel chains
        from scipy.stats import uniform
        initial_states = np.zeros((number_of_chains,7))
        # initial_states[:,(2,3)] = np.array([np.log(2)/30,np.log(2)/90])
        for initial_state_index, _ in enumerate(initial_states):
            initial_states[initial_state_index,:] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(np.log(2)/200),np.log(np.log(2)/200),np.log(0.01),np.log(1),5]),
                                                                          np.array([mean_protein,3,np.log(np.log(2)/10)-np.log(np.log(2)/200),np.log(np.log(2)/10)-np.log(np.log(2)/200),np.log(60-0.01),np.log(40-1),35]))

        # define known parameters
        all_parameters = {'repression_threshold' : [0,None],
                          'hill_coefficient' : [1,None],
                          'mRNA_degradation_rate' : [2,None],
                          'protein_degradation_rate' : [3,None],
                          'basal_transcription_rate' : [4,None],
                          'translation_rate' : [5,None],
                          'transcription_delay' : [6,None]}

        known_parameters = {}#{k:all_parameters[k] for k in ('mRNA_degradation_rate',
                             #                              'protein_degradation_rate') if k in all_parameters}

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
            proposal_covariance = np.diag([5e+3,0.03,0.001,0.001,0.01,0.01,1.0])
            step_size = 0.005
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
            initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
            for initial_state_index, _ in enumerate(initial_states):
                initial_states[initial_state_index,(0,1,4,5,6)] = uniform.rvs(np.array([0.3*mean_protein,2.5,np.log(0.01),np.log(1),5]),
                                                                              np.array([mean_protein,3,np.log(60-0.01),np.log(40-1),35]))

            samples_with_burn_in = array_of_chains[:,int(number_of_samples/2):,:].reshape(int(number_of_samples/2)*number_of_chains,number_of_parameters)
            proposal_covariance = np.cov(samples_with_burn_in.T)
            step_size = 0.005
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
        loading_path = os.path.join(os.path.dirname(__file__),'output/figure_5_b','')
        chain_path_strings = [i for i in os.listdir(loading_path) if '.npy' in i]
        for chain_path_string in chain_path_strings:
            mala = np.load(loading_path + chain_path_string)
            # mala = mala[[0,1,2,4,5,6,7],:,:]
            # mala[:,:,[2,3]] = np.exp(mala[:,:,[2,3]])
            chains = az.convert_to_dataset(mala)
            # print('\n' + chain_path_string + '\n')
            # print('\nrhat:\n',az.rhat(chains))
            # print('\ness:\n',az.ess(chains))
            az.plot_trace(chains); plt.savefig(loading_path + 'traceplot_' + chain_path_string[:-4] + '.png'); plt.close()
            # az.plot_posterior(chains); plt.savefig(loading_path + 'posterior_' + chain_path_string[:-4] + '.png'); plt.close()
            # az.plot_pair(chains,kind='kde'); plt.savefig(loading_path + 'pairplot_' + chain_path_string[:-4] + '.png'); plt.close()
            # np.save(loading_path + chain_path_string,mala)

    def xest_accuracy_of_chains_by_coherence(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','output_jan_17/')
        chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps')]

        coherence_values = np.zeros(len([i for i in chain_path_strings]))
        mean_error_values = np.zeros(len([i for i in chain_path_strings]))
        mode_error_values = np.zeros(len([i for i in chain_path_strings]))
        cov_error_values = np.zeros(len([i for i in chain_path_strings]))

        for chain_path_string in chain_path_strings:
            # import pdb; pdb.set_trace()
            mean_protein = np.mean(np.load(loading_path + '../../data/figure_5_coherence/protein_observations_' +
                                           chain_path_string[chain_path_string.find('ps'):])[:,1])
            mala = np.load(loading_path + chain_path_string)
            samples = mala.reshape(mala.shape[0]*mala.shape[1],mala.shape[2])
            samples[:,[2,3]] = np.exp(samples[:,[2,3]])
            parameter_set_string = chain_path_string[chain_path_string.find('ps'):chain_path_string.find('_fig5')]
            true_values = np.load(loading_path + '../../data/' + parameter_set_string + '_parameter_values.npy')[[0,1,4,5,6]]
            # true_values[[2,3]] = np.log(true_values[[2,3]])
            sample_mean = np.mean(samples,axis=0)
            sample_std = np.std(samples,axis=0)
            prior_widths = [2*mean_protein-50,4,120,40,40]

            coherence_values[np.where(coherence_values==0)[0][0]] = np.load(loading_path + '../../data/' + parameter_set_string + '_parameter_values.npy')[-2]
            # mean_error_values[np.where(mean_error_values==0)[0][0]] = np.sum(((np.abs(true_values[[0,1,4,5,6]]-sample_mean))/true_values[[0,1,4,5,6]]))#[parameter_index]) # mean difference
            # cov_error_values[np.where(cov_error_values==0)[0][0]] = np.sum((sample_std/true_values[[0,1,4,5,6]])[2])#[parameter_index]) # std error
            cov_error_values[np.where(cov_error_values==0)[0][0]] = np.sum(sample_std/prior_widths) # std error
            # import pdb; pdb.set_trace()

        plt.figure(figsize=(8.32,5.54))
        # mean error
        mean_mean = np.zeros(len(np.unique(coherence_values)))
        mean_std= np.zeros(len(np.unique(coherence_values)))
        for index, coherence in enumerate(np.unique(coherence_values)):
            mean_error_per_coherence = cov_error_values[coherence_values==coherence]
            plt.scatter(np.array([np.unique(coherence_values)[index]]*len(mean_error_per_coherence)),
                        mean_error_per_coherence,alpha=0.6,s=50, color='#b5aeb0')
            mean_mean[index] = np.mean(cov_error_values[coherence_values==coherence])
            mean_std[index] = np.std(cov_error_values[coherence_values==coherence])

        plt.fill_between(np.unique(coherence_values),np.maximum(0,mean_mean-mean_std), mean_mean+mean_std, alpha=0.2,color='#b5aeb0',zorder=2)
        plt.plot(np.unique(coherence_values),mean_mean,c='#b5aeb0',alpha=0.5,zorder=3)
        plt.scatter(np.unique(coherence_values)[0],0*mean_mean[0]+0.75,s=150,marker="v",color='#F18D9E',zorder=4,label="Low Coherence")
        plt.scatter(np.unique(coherence_values)[3],0*mean_mean[3]+0.75,s=150,marker="v",color='#8d9ef1',zorder=5,label="High Coherence")
        # plt.errorbar(np.unique(coherence_values),variance_mean/5,variance_std/5,linestyle=None,fmt='o',label='Relative SD',alpha=0.7)
        plt.xlabel('Coherence',fontsize=font_size)
        # plt.xlim(0,0.25)
        plt.ylabel('Prior width norm',fontsize=font_size)
        # plt.ylim(0,100)
        plt.tight_layout()
        plt.legend()
        plt.savefig(loading_path + 'coherence_prior_width_error_values.png')

    def xest_accuracy_of_chains_by_coherence_per_parameter(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','output_jan_17/')
        chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps')]

        parameter_names = np.array(["$P_0$",
                                    "$h$",
                                    "$\\alpha_m$",
                                    "$\\alpha_p$",
                                    "$\\tau$"])

        for parameter_index, parameter in enumerate(parameter_names):
            coherence_values = np.zeros(len([i for i in chain_path_strings]))
            mean_error_values = np.zeros(len([i for i in chain_path_strings]))
            mode_error_values = np.zeros(len([i for i in chain_path_strings]))
            cov_error_values = np.zeros(len([i for i in chain_path_strings]))

            for chain_path_string in chain_path_strings:
                # import pdb; pdb.set_trace()
                mean_protein = np.mean(np.load(loading_path + '../../data/figure_5_coherence/protein_observations_' +
                                               chain_path_string[chain_path_string.find('ps'):])[:,1])
                mala = np.load(loading_path + chain_path_string)
                samples = mala.reshape(mala.shape[0]*mala.shape[1],mala.shape[2])
                samples[:,[2,3]] = np.exp(samples[:,[2,3]])
                parameter_set_string = chain_path_string[chain_path_string.find('ps'):chain_path_string.find('_fig5')]
                true_values = np.load(loading_path + '../../data/' + parameter_set_string + '_parameter_values.npy')[[0,1,4,5,6]]
                # true_values[[2,3]] = np.log(true_values[[2,3]])
                sample_mean = np.mean(samples,axis=0)
                sample_std = np.std(samples,axis=0)
                prior_widths = [2*mean_protein-50,4,120,40,40]

                coherence_values[np.where(coherence_values==0)[0][0]] = np.load(loading_path + '../../data/' + parameter_set_string + '_parameter_values.npy')[-2]
                # mean_error_values[np.where(mean_error_values==0)[0][0]] = np.sum(((np.abs(true_values[[0,1,4,5,6]]-sample_mean))/true_values[[0,1,4,5,6]]))#[parameter_index]) # mean difference
                # cov_error_values[np.where(cov_error_values==0)[0][0]] = np.sum((sample_std/true_values[[0,1,4,5,6]])[2])#[parameter_index]) # std error
                cov_error_values[np.where(cov_error_values==0)[0][0]] = np.sum((sample_std/prior_widths)[parameter_index]) # std error
                # import pdb; pdb.set_trace()

            plt.figure(figsize=(8.32,5.54))
            # mean error
            mean_mean = np.zeros(len(np.unique(coherence_values)))
            mean_std= np.zeros(len(np.unique(coherence_values)))
            for index, coherence in enumerate(np.unique(coherence_values)):
                mean_error_per_coherence = cov_error_values[coherence_values==coherence]
                plt.scatter(np.array([np.unique(coherence_values)[index]]*len(mean_error_per_coherence)),
                            mean_error_per_coherence,alpha=0.6,s=50, color='#b5aeb0')
                mean_mean[index] = np.mean(cov_error_values[coherence_values==coherence])
                mean_std[index] = np.std(cov_error_values[coherence_values==coherence])

            plt.fill_between(np.unique(coherence_values),np.maximum(0,mean_mean-mean_std), mean_mean+mean_std, alpha=0.2,color='#b5aeb0',zorder=2)
            plt.plot(np.unique(coherence_values),mean_mean,c='#b5aeb0',alpha=0.5,zorder=3)
            plt.scatter(np.unique(coherence_values)[0],0*mean_mean[0]+0.4,s=150,marker="v",color='#F18D9E',zorder=4,label="Low Coherence")
            plt.scatter(np.unique(coherence_values)[3],0*mean_mean[3]+0.4,s=150,marker="v",color='#8d9ef1',zorder=5,label="High Coherence")
            # plt.errorbar(np.unique(coherence_values),variance_mean/5,variance_std/5,linestyle=None,fmt='o',label='Relative SD',alpha=0.7)
            plt.xlabel('Coherence',fontsize=font_size)
            # plt.xlim(0,0.25)
            plt.ylabel('Prior width norm, ' + parameter,fontsize=font_size)
            # plt.ylim(0,100)
            plt.tight_layout()
            plt.legend()
            plt.savefig(loading_path + 'coherence_prior_width_error_values_parameter_' + str(parameter_index) + '.png')

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

    def xest_accuracy_of_chains_by_sampling_and_cells(self):
        loading_path = os.path.join(os.path.dirname(__file__),'output','figure_5/')
        saving_path = os.path.join(os.path.dirname(__file__),'data','')
        ps6_true_parameter_values = np.load(os.path.join(saving_path,'ps6_parameter_values.npy'))[[0,1,4,5,6]]
        # ps6_true_parameter_values[[2,3]] = np.exp(ps6_true_parameter_values[[2,3]])
        ps6_measurement_variance = np.power(ps6_true_parameter_values[-1],2)
        ps9_true_parameter_values = np.load(os.path.join(saving_path,'ps9_parameter_values.npy'))[[0,1,4,5,6]]
        # ps9_true_parameter_values[[2,3]] = np.exp(ps9_true_parameter_values[[2,3]])
        ps9_measurement_variance = np.power(ps9_true_parameter_values[-1],2)

        ps6_chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps6')]
        ps9_chain_path_strings = [i for i in os.listdir(loading_path) if i.startswith('final_parallel_mala_output_protein_observations_ps9')]
        ps6_datasets = {}
        ps9_datasets = {}

        for string in ps6_chain_path_strings:
            ps6_datasets[string] = np.load(loading_path + string)

        for string in ps9_chain_path_strings:
            ps9_datasets[string] = np.load(loading_path + string)
        ps6_mean_error_dict = {}
        ps6_cov_dict = {}
        ps9_mean_error_dict = {}
        ps9_cov_dict = {}
        for key in ps6_datasets.keys():
            mean_protein = np.mean(np.load(loading_path + '../../data/figure_5/protein_observations_' +
                                           key[key.find('ps'):])[:,1])
            prior_widths = [2*mean_protein-50,4,120,40,40]
            ps6_mean_error_dict[key] = 0
            ps6_cov_dict[key] = 0
            short_chains = ps6_datasets[key].reshape(ps6_datasets[key].shape[0]*ps6_datasets[key].shape[1],5)
            short_chains[:,[2,3]] = np.exp(short_chains[:,[2,3]])
            short_chains_mean = np.mean(short_chains,axis=0)
            short_chains_std = np.std(short_chains,axis=0)
            # coefficient of variation
            # short_chain_cov = np.sum(short_chains_std/ps6_true_parameter_values)
            short_chain_cov = np.sum(short_chains_std/prior_widths)
            # relative mean
            relative_mean = np.sum(np.abs(ps6_true_parameter_values - short_chains_mean)/prior_widths)
            ps6_mean_error_dict[key] = relative_mean
            ps6_cov_dict[key] = short_chain_cov

        for key in ps9_datasets.keys():
            mean_protein = np.mean(np.load(loading_path + '../../data/figure_5/protein_observations_' +
                                           key[key.find('ps'):])[:,1])
            prior_widths = [2*mean_protein-50,4,120,40,40]
            ps9_mean_error_dict[key] = 0
            ps9_cov_dict[key] = 0
            short_chains = ps9_datasets[key].reshape(ps9_datasets[key].shape[0]*ps9_datasets[key].shape[1],5)
            short_chains[:,[2,3]] = np.exp(short_chains[:,[2,3]])
            short_chains_mean = np.mean(short_chains,axis=0)
            short_chains_std = np.std(short_chains,axis=0)
            # coefficient of variation
            # short_chain_cov = np.sum(short_chains_std/ps9_true_parameter_values)
            short_chain_cov = np.sum(short_chains_std/prior_widths)
            # relative mean
            relative_mean = np.sum(np.abs(ps9_true_parameter_values - short_chains_mean)/prior_widths)
            ps9_mean_error_dict[key] = relative_mean
            ps9_cov_dict[key] = short_chain_cov

        plotting_strings = ['1_cells_5_minutes',
                            '1_cells_8_minutes',
                            '1_cells_12_minutes',
                            '1_cells_15_minutes']

        fig, ax = plt.subplots(1,2,figsize=(8.63*2,6.95))
        mean_and_sd_covs = np.zeros((4,3))
        mean_and_sd_means = np.zeros((4,3))
        for index, string in enumerate(plotting_strings):
            # 1/cov
            covs = [value for key, value in ps6_cov_dict.items() if 'ps6_' + string in key.lower()]
            xcoords = [np.int(string[string.find('lls_')+4:string.find('_min')])]*len(covs)
            ax[0].scatter(xcoords,covs,label=string, color='#b5aeb0')
            ax[0].set_xlim(15.5,4.5) # backwards for comparison to length
            ax[0].set_ylim(0.5,1.2)
            ax[0].set_xlabel("Sampling interval (mins)",fontsize=font_size)
            ax[0].set_ylabel("Uncertainty (All Parameters)",fontsize=font_size)
            # mean error
            mean_errors = [value for key, value in ps6_mean_error_dict.items() if 'ps6_' + string in key.lower()]
            ax[1].scatter(xcoords,mean_errors,label=string, color='#b5aeb0')
            ax[1].set_xlim(15.5,4.5) # backwards for comparison to length
            ax[1].set_ylim(.5,1.5)
            ax[1].set_xlabel("Sampling interval (mins)",fontsize=font_size)
            ax[1].set_ylabel("Relative mean error",fontsize=font_size)
            # plt.legend()
            mean_and_sd_covs[index,0] = xcoords[0]
            mean_and_sd_means[index,0] = xcoords[0]
            mean_and_sd_covs[index,1] = np.mean(covs)
            mean_and_sd_covs[index,2] = np.std(covs)
            mean_and_sd_means[index,1] = np.mean(mean_errors)
            mean_and_sd_means[index,2] = np.std(mean_errors)
        # print(mean_and_sd_covs[3,1])
        # print(mean_and_sd_covs[1,1])
        # print(mean_and_sd_covs[3,1]/mean_and_sd_covs[1,1])
        # print()
        # print(mean_and_sd_means[3,1])
        # print(mean_and_sd_means[1,1])
        # print(mean_and_sd_means[3,1]/mean_and_sd_means[1,1])
        # print()
        ax[0].plot(mean_and_sd_covs[:,0],mean_and_sd_covs[:,1],color='#F18D9E',alpha=0.5)
        ax[0].fill_between(mean_and_sd_covs[:,0], mean_and_sd_covs[:,1]-mean_and_sd_covs[:,2], mean_and_sd_covs[:,1]+mean_and_sd_covs[:,2], alpha=0.2,color='#F18D9E')
        ax[1].plot(mean_and_sd_means[:,0],mean_and_sd_means[:,1],color='#F18D9E',alpha=0.5)
        ax[1].fill_between(mean_and_sd_means[:,0], np.maximum(0,mean_and_sd_means[:,1]-mean_and_sd_means[:,2]), mean_and_sd_means[:,1]+mean_and_sd_means[:,2], alpha=0.2,color='#F18D9E')

        plt.tight_layout()
        plt.savefig(loading_path + 'ps6_cov_and_mean_error_values_frequency_prior.png')

        fig, ax = plt.subplots(1,2,figsize=(8.63*2,6.95))
        for index, string in enumerate(plotting_strings):
            # 1/cov
            covs = [value for key, value in ps9_cov_dict.items() if 'ps9_' + string in key.lower()]
            xcoords = [np.int(string[string.find('lls_')+4:string.find('_min')])]*len(covs)
            ax[0].scatter(xcoords,covs,label=string, color='#b5aeb0')
            ax[0].set_xlim(15.5,4.5) # backwards for comparison to length
            ax[0].set_ylim(0.5,1.2)
            ax[0].set_xlabel("Sampling interval (mins)",fontsize=font_size)
            ax[0].set_ylabel("Uncertainty (All Parameters)",fontsize=font_size)
            # mean error
            mean_errors = [value for key, value in ps9_mean_error_dict.items() if 'ps9_' + string in key.lower()]
            ax[1].scatter(xcoords,mean_errors,label=string, color='#b5aeb0')
            ax[1].set_xlim(15.5,4.5) # backwards for comparison to length
            ax[1].set_ylim(0,1)
            ax[1].set_xlabel("Sampling interval (mins)",fontsize=font_size)
            ax[1].set_ylabel("Relative mean error",fontsize=font_size)
            # plt.legend()
            mean_and_sd_covs[index,0] = xcoords[0]
            mean_and_sd_means[index,0] = xcoords[0]
            mean_and_sd_covs[index,1] = np.mean(covs)
            mean_and_sd_covs[index,2] = np.std(covs)
            mean_and_sd_means[index,1] = np.mean(mean_errors)
            mean_and_sd_means[index,2] = np.std(mean_errors)
        # print(mean_and_sd_covs[3,1])
        # print(mean_and_sd_covs[1,1])
        # print(mean_and_sd_covs[3,1]/mean_and_sd_covs[1,1])
        # print()
        # print(mean_and_sd_means[3,1])
        # print(mean_and_sd_means[1,1])
        # print(mean_and_sd_means[3,1]/mean_and_sd_means[1,1])
        # print()
        ax[0].plot(mean_and_sd_covs[:,0],mean_and_sd_covs[:,1],c='#8d9ef1',alpha=0.5)
        ax[0].fill_between(mean_and_sd_covs[:,0], mean_and_sd_covs[:,1]-mean_and_sd_covs[:,2], mean_and_sd_covs[:,1]+mean_and_sd_covs[:,2], alpha=0.2,color='#8d9ef1')
        ax[1].plot(mean_and_sd_means[:,0],mean_and_sd_means[:,1],c='#8d9ef1',alpha=0.5)
        ax[1].fill_between(mean_and_sd_means[:,0], np.maximum(0,mean_and_sd_means[:,1]-mean_and_sd_means[:,2]), mean_and_sd_means[:,1]+mean_and_sd_means[:,2], alpha=0.2,color='#8d9ef1')

        plt.tight_layout()
        plt.savefig(loading_path + 'ps9_cov_and_mean_error_values_frequency_prior.png')

        plotting_strings = ['1_cells_15_minutes',
                            '2_cells_15_minutes',
                            '3_cells_15_minutes',
                            '4_cells_15_minutes',
                            '5_cells_15_minutes']

        mean_and_sd_covs = np.zeros((5,3))
        mean_and_sd_means = np.zeros((5,3))
        fig, ax = plt.subplots(1,2,figsize=(8.63*2,6.95))
        # ax[0].set_yscale('log')
        # ax[1].set_yscale('log')
        for index, string in enumerate(plotting_strings):
            # 1/cov
            covs = [value for key, value in ps6_cov_dict.items() if 'ps6_' + string in key.lower()]
            xcoords = [np.int(string[0])*12]*len(covs)
            ax[0].scatter(xcoords,covs,label=string, color='#b5aeb0')
            ax[0].set_xticks(np.arange(6)*12)
            ax[0].set_ylim(0.5,1.2)
            ax[0].set_xlabel("Measurement duration \n(hours, interval = 15 mins)",fontsize=font_size)
            ax[0].set_ylabel("Uncertainty (All Parameters)",fontsize=font_size)
            # mean error
            mean_errors = [value for key, value in ps6_mean_error_dict.items() if 'ps6_' + string in key.lower()]
            ax[1].scatter(xcoords,mean_errors,label=string, color='#b5aeb0')
            ax[1].set_xticks(np.arange(6)*12)
            ax[1].set_ylim(.5,1.5)
            ax[1].set_xlabel("Measurement duration \n(hours, interval = 15 mins)",fontsize=font_size)
            ax[1].set_ylabel("Relative mean error",fontsize=font_size)
            # plt.legend()
            mean_and_sd_covs[index,0] = xcoords[0]
            mean_and_sd_means[index,0] = xcoords[0]
            mean_and_sd_covs[index,1] = np.mean(covs)
            mean_and_sd_covs[index,2] = np.std(covs)
            mean_and_sd_means[index,1] = np.mean(mean_errors)
            mean_and_sd_means[index,2] = np.std(mean_errors)
        # print(mean_and_sd_covs[0,1])
        # print(mean_and_sd_covs[1,1])
        # print(mean_and_sd_covs[0,1]/mean_and_sd_covs[1,1])
        # print()
        # print(mean_and_sd_means[0,1])
        # print(mean_and_sd_means[1,1])
        # print(mean_and_sd_means[0,1]/mean_and_sd_means[1,1])
        # print()
        ax[0].plot(mean_and_sd_covs[:,0],mean_and_sd_covs[:,1],color='#F18D9E',alpha=0.5)
        ax[0].fill_between(mean_and_sd_covs[:,0], mean_and_sd_covs[:,1]-mean_and_sd_covs[:,2], mean_and_sd_covs[:,1]+mean_and_sd_covs[:,2], alpha=0.2,color='#F18D9E')
        ax[1].plot(mean_and_sd_means[:,0],mean_and_sd_means[:,1],color='#F18D9E',alpha=0.5)
        ax[1].fill_between(mean_and_sd_means[:,0],np.maximum(0,mean_and_sd_means[:,1]-mean_and_sd_means[:,2]), mean_and_sd_means[:,1]+mean_and_sd_means[:,2], alpha=0.2,color='#F18D9E')

        plt.tight_layout()
        plt.savefig(loading_path + 'ps6_cov_and_mean_error_values_length_prior.png')

        fig, ax = plt.subplots(1,2,figsize=(8.63*2,6.95))
        for index, string in enumerate(plotting_strings):
            # 1/cov
            covs = [value for key, value in ps9_cov_dict.items() if 'ps9_' + string in key.lower()]
            xcoords = [np.int(string[0])*12]*len(covs)
            ax[0].scatter(xcoords,covs,label=string, color='#b5aeb0')
            ax[0].set_xticks(np.arange(6)*12)
            ax[0].set_ylim(0.5,1.2)
            # ax[0].set_xticklabels([10,20,40])
            ax[0].set_xlabel("Measurement duration \n(hours, interval = 15 mins)",fontsize=font_size)
            ax[0].set_ylabel("Uncertainty (All Parameters)",fontsize=font_size)
            # mean error
            mean_errors = [value for key, value in ps9_mean_error_dict.items() if 'ps9_' + string in key.lower()]
            ax[1].scatter(xcoords,mean_errors,label=string, color='#b5aeb0')
            ax[1].set_xticks(np.arange(6)*12)
            ax[1].set_ylim(0,1)
            ax[1].set_xlabel("Measurement duration \n(hours, interval = 15 mins)",fontsize=font_size)
            ax[1].set_ylabel("Relative mean error",fontsize=font_size)
            # plt.legend()
            mean_and_sd_covs[index,0] = xcoords[0]
            mean_and_sd_means[index,0] = xcoords[0]
            mean_and_sd_covs[index,1] = np.mean(covs)
            mean_and_sd_covs[index,2] = np.std(covs)
            mean_and_sd_means[index,1] = np.mean(mean_errors)
            mean_and_sd_means[index,2] = np.std(mean_errors)
        # print(mean_and_sd_covs[0,1])
        # print(mean_and_sd_covs[1,1])
        # print(mean_and_sd_covs[0,1]/mean_and_sd_covs[1,1])
        # print()
        # print(mean_and_sd_means[0,1])
        # print(mean_and_sd_means[1,1])
        # print(mean_and_sd_means[0,1]/mean_and_sd_means[1,1])
        # print()
        ax[0].plot(mean_and_sd_covs[:,0],mean_and_sd_covs[:,1],c='#8d9ef1',alpha=0.5)
        ax[0].fill_between(mean_and_sd_covs[:,0], mean_and_sd_covs[:,1]-mean_and_sd_covs[:,2], mean_and_sd_covs[:,1]+mean_and_sd_covs[:,2], alpha=0.2,color='#8d9ef1')
        ax[1].plot(mean_and_sd_means[:,0],mean_and_sd_means[:,1],c='#8d9ef1',alpha=0.5)
        ax[1].fill_between(mean_and_sd_means[:,0], np.maximum(0,mean_and_sd_means[:,1]-mean_and_sd_means[:,2]), mean_and_sd_means[:,1]+mean_and_sd_means[:,2], alpha=0.2,color='#8d9ef1')

        plt.tight_layout()
        plt.savefig(loading_path + 'ps9_cov_and_mean_error_values_length_prior.png')

    def xest_identify_oscillatory_parameters(self):
        loading_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(loading_path + '.npy')
        prior_samples = np.load(loading_path + '_parameters.npy')

        ps_string = "ps10"
        coherence_bands = [[0.01,0.03],[0.2,0.25]]
        observation_duration  = 600
        observation_frequency = 5
        no_of_observations    = np.int(observation_duration/observation_frequency)
        protein_at_observations = np.zeros((2,observation_duration//5,2))
        for i, coherence_band in enumerate(coherence_bands):
            accepted_indices = np.where(np.logical_and(model_results[:,0]>30000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]>0.05, #standard deviation
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                        np.logical_and(model_results[:,3]>coherence_band[0], #coherence
                                        np.logical_and(model_results[:,3]<coherence_band[1], #coherence
                                                       prior_samples[:,4]<6))))))) # hill coefficient

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



            # np.save(saving_path + ps_string + "_parameter_values.npy",parameters)


            # parameters = np.load(loading_path + ps_string + "_parameter_values.npy")

            true_data = hes5.generate_langevin_trajectory(duration = observation_duration,
                                                          repression_threshold = parameters[0],
                                                          hill_coefficient = parameters[1],
                                                          mRNA_degradation_rate = parameters[2],
                                                            protein_degradation_rate = parameters[3],
                                                          basal_transcription_rate = parameters[4],
                                                          translation_rate = parameters[5],
                                                          transcription_delay = parameters[6],
                                                          equilibration_time = 1000)
            # np.save(saving_path + 'true_data_' + ps_string + '.npy',
            #         true_data)

            ## the F constant matrix is left out for now
            protein_at_observations[i] = true_data[::5,(0,2)]
            protein_at_observations[i,:,1] += np.random.randn(true_data.shape[0]//5)*parameters[-1]
            protein_at_observations[i,:,1] = np.maximum(protein_at_observations[i,:,1],0)
            # np.save(saving_path + 'protein_observations_' + ps_string + '_ds1.npy',
            #             protein_at_observations[0:900:10,:])
            # np.save(saving_path + 'protein_observations_' + ps_string + '_ds2.npy',
            #             protein_at_observations[0:900:5,:])
            # np.save(saving_path + 'protein_observations_' + ps_string + '_ds3.npy',
            #             protein_at_observations[0:1800:10,:])
            # np.save(saving_path + 'protein_observations_' + ps_string + '_ds4.npy',
            #             protein_at_observations[0:1800:5,:])

        my_figure = plt.figure(figsize=(9.83,5.54))
        plt.scatter(np.arange(0,observation_duration,5),protein_at_observations[0,:,1],marker='o',s=12,c='#F18D9E',label='Low Coherence')
        plt.scatter(np.arange(0,observation_duration,5),protein_at_observations[1,:,1],marker='o',s=12,c='#8d9ef1',label='High Coherence')
        plt.plot(np.arange(0,observation_duration,5),protein_at_observations[0,:,1],'--',c='#F18D9E',alpha=0.5)
        plt.plot(np.arange(0,observation_duration,5),protein_at_observations[1,:,1],'--',c='#8d9ef1',alpha=0.5)
        plt.xlabel('Time (mins)',fontsize=font_size)
        plt.ylabel('Protein Molecules',fontsize=font_size)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        my_figure.savefig(saving_path + 'coherence_comparison.pdf')

    def xest_figure_5_b_make_data(self):
        loading_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        saving_path = os.path.join(os.path.dirname(__file__), 'data','figure_5_b/')
        model_results = np.load(loading_path + '.npy')
        prior_samples = np.load(loading_path + '_parameters.npy')

        bands = np.arange(0,1,.01)
        observation_duration  = 735
        observation_frequency = 15
        import random

        for index, start_point in enumerate(bands):
            coherence_band = [start_point,start_point + 0.005]

            accepted_indices = np.where(np.logical_and(model_results[:,0]>30000, #protein number
                                        np.logical_and(model_results[:,0]<65000, #protein_number
                                        np.logical_and(model_results[:,1]>0.05, #standard deviation
                                        np.logical_and(model_results[:,1]<0.15,  #standard deviation
                                        np.logical_and(model_results[:,3]>coherence_band[0], #coherence
                                        np.logical_and(model_results[:,3]<coherence_band[1], #coherence
                                                       prior_samples[:,4]<6))))))) # hill coefficient

            my_posterior_samples = prior_samples[accepted_indices]
            my_model_results = model_results[accepted_indices]

            if len(my_posterior_samples) == 0:
                continue

            this_parameter = my_posterior_samples[random.sample(range(len(my_posterior_samples)),1)][0]
            this_results = my_model_results[random.sample(range(len(my_model_results)),1)][0]

            parameters = np.array([this_parameter[2],
                                   this_parameter[4],
                                   np.log(2)/30,
                                   np.log(2)/90,
                                   this_parameter[0],
                                   this_parameter[1],
                                   this_parameter[3],
                                   this_results[3]])

            protein_at_observations = np.zeros((observation_duration//observation_frequency,2))

            true_data = hes5.generate_langevin_trajectory(duration = observation_duration,
                                                          repression_threshold = parameters[0],
                                                          hill_coefficient = parameters[1],
                                                          mRNA_degradation_rate = parameters[2],
                                                            protein_degradation_rate = parameters[3],
                                                          basal_transcription_rate = parameters[4],
                                                          translation_rate = parameters[5],
                                                          transcription_delay = parameters[6],
                                                          equilibration_time = 1000)

            measurement_variance = 1000
            parameters = np.append(parameters,measurement_variance)
            protein_at_observations = true_data[::observation_frequency,(0,2)]
            protein_at_observations[:,1] += np.random.randn(true_data.shape[0]//observation_frequency)*parameters[-1]
            protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)
            np.save(saving_path + "parameter_values_coherence_" + str(index) + ".npy",parameters)
            np.save(saving_path + 'true_data_coherence_' + str(index) + '.npy', true_data)
            np.save(saving_path + 'protein_observations_coherence_' + str(index) + '.npy', protein_at_observations)


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

        # # make data from each trace and save
        # for cell_index in range(1,cell_intensity_df.shape[1]):
        #     cell_intensity_values = cell_intensity_df.iloc[2:,[0,cell_index]].astype(float).values
        #     # remove NaNs
        #     cell_intensity_values = cell_intensity_values[~np.isnan(cell_intensity_values[:,1])]
        #     cell_intensity_values[:,0] *=60 # turn hours to minutes
        #     cell_intensity_values[:,1] *= gradient_hom # average of hom, het, het and hom?
        #     cell_cluster = int(cell_intensity_df.iloc[0,cell_index])
        #     np.save(loading_path + 'protein_observations_' + experiment_date + '_cell_' + str(cell_index) + '_cluster_' + str(cell_cluster),
        #             cell_intensity_values)

    def xest_make_experimental_data_parnian(self):
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data_parnian/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','experimental_data_parnian/detrended_data/')
        # import spreadsheets as dataframes
        FCS_df = pd.DataFrame(pd.read_excel(loading_path + "FCS_molec_different_development_stages_Hindbrain.xlsx",header=0))
        cell_intensity_df = pd.read_csv(loading_path + "190530_HV_venus.csv",header=2)

        # convert to numpy arrays for plotting / fitting
        intensities = cell_intensity_df.astype(float).values.flatten()
        intensities = intensities[~(np.isnan(intensities))]
        hpf_keys = ['19hpf','29hpf','33hpf','35hpf','48hpf']
        FCS_at_hpf = {}
        for key in hpf_keys:
            FCS_at_hpf[key] = FCS_df[key].astype(float).values.flatten()
            FCS_at_hpf[key] = FCS_at_hpf[key][~(np.isnan(FCS_at_hpf[key]))]
        FCS_at_hpf['all_hpf'] = np.hstack(FCS_at_hpf.values())

        # make qqplots and calculate gradients
        gradients = np.zeros(len(FCS_at_hpf.keys()))
        fig, ax = plt.subplots(1,len(FCS_at_hpf.keys()),figsize=(25*1.5,5*1.5))
        for index, key in enumerate(FCS_at_hpf.keys()):
            x = np.quantile(intensities,np.linspace(0.0,1.0,101))
            y = np.quantile(FCS_at_hpf[key],np.linspace(0.0,1.0,101))
            gradient = (np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[-1]))-np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[0])))/(np.unique(x[-1]-x[0]))
            gradients[index] = gradient[0]
            ax[index].scatter(x,y)
            ax[index].plot(np.unique(x[5:96]), np.poly1d(np.polyfit(x[5:96], y[5:96], 1))(np.unique(x[5:96])),color='r')
            ax[index].set_ylabel(key + 'FCS')
            ax[index].set_xlabel("Cell intensities")
            gradient_string = "Gradient is " + str(np.round(gradients[index],4))
            ax[index].text(2000,0,gradient_string)
        plt.tight_layout()
        plt.savefig(loading_path + 'molecule_qq_plot.png')

        # make data from each trace and save
        for cell_index in range(1,cell_intensity_df.shape[1]):
            cell_intensity_values = cell_intensity_df.iloc[:,[0,cell_index]].astype(float).values
            # remove NaNs
            cell_intensity_values = cell_intensity_values[~np.isnan(cell_intensity_values[:,1])]
            cell_intensity_values[:,0] *=60 # turn hours to minutes
            cell_intensity_values[:,1] *= np.mean(gradients[1:4]) # average of hpf that correspond to parnian's data
            np.save(loading_path + 'detrended_data/' + 'protein_observations_cell_' + str(cell_index),cell_intensity_values)

    def xest_detrend_trace(self,data_filename = 'protein_observations_040417_cell_2_cluster_4.npy'):
        # load data
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data_parnian/detrended_data/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','experimental_data_parnian/detrended_data_images/')
        # loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/')
        # saving_path = os.path.join(os.path.dirname(__file__),'output','detrended_data_images/')

        experimental_data_strings = [i for i in os.listdir(loading_path) if 'npy' in i
                                     and not 'detrended' in i]
        variances = np.zeros(len(experimental_data_strings))
        detrended_variances = np.zeros(len(experimental_data_strings))
        # import pdb; pdb.set_trace()

        # make data from each trace and save
        for index, cell_string in enumerate(experimental_data_strings):
            protein = np.load(loading_path + cell_string)
            protein[:,0] -= protein[0,0] # start time = 0
            mean_protein = np.mean(protein[:,1])
            protein_around_mean = protein[:,1] - mean_protein
            times = protein[:,0]
            detrended_protein, y_gpr, y_std = hes5.detrend_experimental_data(protein,length_scale=500)
            variances[index] = np.var(protein[:,1])
            detrended_variances[index] = np.var(detrended_protein[:,1])
            np.save(loading_path + cell_string[:-4] + "_detrended.npy",detrended_protein)

            # plot
            lw = 2
            fig, ax = plt.subplots(1,1,figsize=(1.4*13, 1.4*5.19))
            # plot data without trend and mean
            ax.plot(times,protein[:,1], c='k', label='data')
            X_plot = np.linspace(0, np.int(times[-1]), np.int(times[-1])+1)[:, None]
            ax.plot(times, detrended_protein[:,1], color='#20948B', lw=lw,label='detrended data')
            ax.plot(X_plot, mean_protein + y_gpr, '#F18D9E', lw=lw,label='trend')
            ax.set_xlabel('Time (mins)',fontsize=font_size*1.2)
            ax.set_ylabel('Protein',fontsize=font_size*1.2)
            ax.legend()
            # plot original data and detrended data
            plt.tight_layout()
            plt.savefig(saving_path + cell_string[:-4] + "_detrended.png")
            # import pdb; pdb.set_trace()
        measurement_variance = np.sqrt(0.1*np.mean(detrended_variances))
        np.save(loading_path + 'measurement_variance_detrended.npy',measurement_variance)

    def xest_oscillation_quality(self):
        # load data
        loading_path = os.path.join(os.path.dirname(__file__),'data','figure_5_b/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','figure_5_b/')

        experimental_data_strings = [i for i in os.listdir(loading_path) if 'npy' in i
                                                                         if 'protein_observations' in i
                                                                         if 'quality' not in i]

        # make data from each trace and save
        for index, cell_string in enumerate(experimental_data_strings):
            protein = np.load(loading_path + cell_string)
            protein[:,0] -= protein[0,0] # start time = 0
            mean_protein = np.mean(protein[:,1])
            protein_around_mean = protein[:,1] - mean_protein
            times = np.linspace(0, np.int(protein[-1,0]), np.int(protein[-1,0])+1)
            mean, var, quality = hes5.calculate_oscillation_quality(protein)
            np.save(loading_path + cell_string[:-4] + "_oscillation_quality.npy",quality)

            ## plot
            plt.figure(figsize=(12, 6))
            plt.scatter(protein[:,0], protein_around_mean, label="data")
            plt.plot(times, mean[:,0], lw=2)
            plt.fill_between(
                times,
                mean[:,0] - 1.96 * np.sqrt(var[:,0]),
                mean[:,0] + 1.96 * np.sqrt(var[:,0]),
                color="C0",
                alpha=0.2,
            )
            plt.text(50,6000,str(round(quality,2)))
            plt.savefig(saving_path + cell_string[:-4] + "_oscillation_quality.png")
            # import pdb; pdb.set_trace()

    def xest_oscillation_quality_vs_accuracy(self):
        # load data
        loading_path = os.path.join(os.path.dirname(__file__),'data','figure_5_b/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','figure_5_b/')

        experimental_data_strings = [i for i in os.listdir(saving_path) if 'npy' in i
                                                                         if 'quality' not in i
                                                                         if 'parallel_mala' in i]
        # prior width
        fig, ax = plt.subplots(figsize=(8,6))
        for data_string in experimental_data_strings:
            ps_string = data_string[data_string.find('cohe'):-4]
            quality = np.load(loading_path + 'protein_observations_' + ps_string + '_oscillation_quality.npy')
            true_values = np.load(loading_path + 'parameter_values_' + ps_string + '.npy')[[0,1,4,5,6]]
            # true_values[[2,3]] = np.log(true_values[[2,3]])
            chain = np.load(saving_path + data_string)
            chain = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])
            chain[:,[2,3]] = np.exp(chain[:,[2,3]])
            mean_protein = np.mean(np.load(loading_path + 'protein_observations_' + ps_string + '.npy')[:,1])
            prior_widths = [2*mean_protein-50,4,120,40,40]
            sample_std = np.std(chain,axis=0)
            prior_norm = np.sum(sample_std/prior_widths)
            ax.scatter(quality,prior_norm,color='#20948B')

        ax.set_xlabel("Oscillation quality")
        ax.set_ylabel("Uncertainty (All Parameters)")
        # ax.set_xscale('log')
        ax.set_xlim(0,10)
        plt.tight_layout()
        plt.savefig(saving_path + "quality_vs_uncertainty.png")

        # rel mean error
        # prior_widths = [2*50000,4,np.log(120)-np.log(0.01),np.log(40)-np.log(0.01),40]
        fig, ax = plt.subplots(figsize=(8,6))
        for data_string in experimental_data_strings:
            ps_string = data_string[data_string.find('cohe'):-4]
            quality = np.load(loading_path + 'protein_observations_' + ps_string + '_oscillation_quality.npy')
            true_values = np.load(loading_path + 'parameter_values_' + ps_string + '.npy')[[0,1,4,5,6]]
            # true_values[[2,3]] = np.log(true_values[[2,3]])
            chain = np.load(saving_path + data_string)
            chain = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])
            chain[:,[2,3]] = np.exp(chain[:,[2,3]])
            mean_protein = np.mean(np.load(loading_path + 'protein_observations_' + ps_string + '.npy')[:,1])
            prior_widths = [2*mean_protein-50,4,120,40,40]
            sample_mean = np.mean(chain,axis=0)
            prior_norm = np.sum(np.abs(sample_mean-true_values)/prior_widths)
            ax.scatter(quality,prior_norm,color='#20948B')

        ax.set_xlabel("Oscillation quality")
        ax.set_ylabel("Relative Mean Error")
        # ax.set_xscale('log')
        ax.set_xlim(0,10)
        plt.tight_layout()
        plt.savefig(saving_path + "oscillation_vs_rel_mean.png")

    def xest_oscillation_coherence_vs_accuracy(self):
        # load data
        loading_path = os.path.join(os.path.dirname(__file__),'data','figure_5_b/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','figure_5_b/')

        experimental_data_strings = [i for i in os.listdir(saving_path) if 'npy' in i
                                                                         if 'quality' not in i
                                                                         if 'parallel_mala' in i]
        coherence_values = np.zeros(len(experimental_data_strings))
        prior_norm_values = np.zeros(len(experimental_data_strings))
        # prior width
        fig, ax = plt.subplots(figsize=(8,6))
        for index, data_string in enumerate(experimental_data_strings):
            ps_string = data_string[data_string.find('cohe'):-4]
            coherence = np.load(loading_path + 'parameter_values_' + ps_string + '.npy')[-2]
            true_values = np.load(loading_path + 'parameter_values_' + ps_string + '.npy')[[0,1,4,5,6]]
            # true_values[[2,3]] = np.log(true_values[[2,3]])
            chain = np.load(saving_path + data_string)
            chain = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])
            chain[:,[2,3]] = np.exp(chain[:,[2,3]])
            mean_protein = np.mean(np.load(loading_path + 'protein_observations_' + ps_string + '.npy')[:,1])
            prior_widths = [2*mean_protein-50,4,120,40,40]
            sample_std = np.std(chain,axis=0)
            prior_norm = np.sum(sample_std/prior_widths)
            ax.scatter(coherence,prior_norm,color='#20948B')
            coherence_values[index] = coherence
            prior_norm_values[index] = prior_norm

        ax.set_xlabel("Coherence")
        ax.set_ylabel("Uncertainty (All Parameters)")
        # ax.set_xscale('log')
        # ax.set_xlim(0,10)
        plt.tight_layout()
        x = np.sort(coherence_values)[:41]
        y = [x for _,x in sorted(zip(coherence_values,prior_norm_values))][:41]
        coef = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef)
        # poly1d_fn is now a function which takes in x and returns an estimate for y
        plt.plot(x, poly1d_fn(x), '--k')
        from scipy.stats import pearsonr
        corr, _ = pearsonr(x,y)
        plt.text(0.75,1.1,"$\\nu = $" + str(round(corr,2)))
        plt.savefig(saving_path + "coherence_vs_uncertainty.png")

        # rel mean error
        coherence_values = np.zeros(len(experimental_data_strings))
        prior_norm_values = np.zeros(len(experimental_data_strings))
        fig, ax = plt.subplots(figsize=(8,6))
        for index, data_string in enumerate(experimental_data_strings):
            ps_string = data_string[data_string.find('cohe'):-4]
            coherence = np.load(loading_path + 'parameter_values_' + ps_string + '.npy')[-2]
            true_values = np.load(loading_path + 'parameter_values_' + ps_string + '.npy')[[0,1,4,5,6]]
            # true_values[[2,3]] = np.log(true_values[[2,3]])
            chain = np.load(saving_path + data_string)
            chain = chain.reshape(chain.shape[0]*chain.shape[1],chain.shape[2])
            chain[:,[2,3]] = np.exp(chain[:,[2,3]])
            mean_protein = np.mean(np.load(loading_path + 'protein_observations_' + ps_string + '.npy')[:,1])
            prior_widths = [2*mean_protein-50,4,120,40,40]
            sample_mean = np.mean(chain,axis=0)
            prior_norm = np.sum(np.abs(sample_mean-true_values)/prior_widths)
            ax.scatter(coherence,prior_norm,color='#20948B')
            coherence_values[index] = coherence
            prior_norm_values[index] = prior_norm

        ax.set_xlabel("Coherence")
        ax.set_ylabel("Relative Mean Error")
        # ax.set_xscale('log')
        # ax.set_xlim(0,10)
        plt.tight_layout()
        x = np.sort(coherence_values)[:25]
        # sort coherence values from low to high, and sort corresponding prior norm values
        y = [x for _,x in sorted(zip(coherence_values,prior_norm_values))][:25]
        coef = np.polyfit(x,y,1)
        poly1d_fn = np.poly1d(coef)
        # poly1d_fn is now a function which takes in x and returns an estimate for y
        plt.plot(x, poly1d_fn(x), '--k')
        from scipy.stats import pearsonr
        corr, _ = pearsonr(x,y)
        plt.text(0.8,1.5,"$\\nu = $" + str(round(corr,2)))
        plt.savefig(saving_path + "coherence_vs_rel_mean.png")

    def xest_multiple_traces(self):
        # load data and true parameter values
        loading_path = os.path.join(os.path.dirname(__file__),'data','experimental_data/')
        saving_path = os.path.join(os.path.dirname(__file__),'output','detrended_data_images/')

        clusters = ['cluster_1','cluster_2']
        for cluster in clusters:
            experimental_data_strings = [i for i in os.listdir(loading_path) if cluster in i
                                         if 'detrended' in i]
            means = np.zeros(len(experimental_data_strings))
            stds = np.zeros(len(experimental_data_strings))
            detrended_variances = np.zeros(len(experimental_data_strings))

            # make data from each trace and save
            fig, ax = plt.subplots(1,2,figsize=(15, 6))
            for index, cell_string in enumerate(experimental_data_strings):
                protein = np.load(loading_path + cell_string)
                protein[:,0] -= protein[0,0]
                times = protein[:,0]
                means[index] = np.mean(protein[:,1])
                stds[index] = np.std(protein[:,1])
                # plot
                ax[0].plot(times,protein[:,1],alpha=0.5)
                ax[0].set_xlabel("Time (mins)",fontsize=font_size)
                ax[0].set_ylabel("Protein",fontsize=font_size)
                ax[0].set_title("Cluster {cluster}".format(cluster = cluster[-1]),fontsize=font_size)
            ax[1].axis('off')
            ax[1].text(0.1,0.7,'$\mu \in [{i},{j}]$'.format(i=int(means.min()),j=int(means.max())),fontsize=1.5*font_size)
            ax[1].text(0.1,0.3,'$\sigma \in [{i},{j}]$'.format(i=int(stds.min()),j=int(stds.max())),fontsize=1.5*font_size)
            plt.tight_layout()
            plt.savefig(saving_path + cluster + "_test.png")
                # import pdb; pdb.set_trace()
            # measurement_variance = np.sqrt(0.1*np.mean(detrended_variances))
            # np.save(loading_path + date + '_measurement_variance_detrended.npy',measurement_variance)

def run_mala_for_dataset(data_filename,
                         protein_at_observations,
                         measurement_variance,
                         number_of_parameters,
                         known_parameters,
                         step_size = 1,
                         number_of_chains = 8,
                         number_of_samples = 80000):
    """
    A function which gives a (hopefully) decent MALA output for a given dataset with known or
    unknown parameters. If a previous output already exists, this will be used to create a
    proposal covariance matrix, otherwise one will be constructed with a two step warm-up
    process.
    """
    # make sure all data starts from time "zero"
    for i in range(protein_at_observations.shape[0]):
        protein_at_observations[i,:,0] -= protein_at_observations[i,0,0]

    mean_protein = np.mean([i[j,1] for i in protein_at_observations for j in range(i.shape[0])])

    # if we already have mcmc samples, we can use them to construct a covariance matrix to make sampling better
    if os.path.exists(os.path.join(
                      os.path.dirname(__file__),
                      'output','final_parallel_mala_output_' + data_filename)):
        print("Posterior samples already exist, sampling directly without warm up...")

        mala_output = np.load(os.path.join(os.path.dirname(__file__),
                              'output','final_parallel_mala_output_' + data_filename))
        previous_number_of_samples = mala_output.shape[1]
        previous_number_of_chains = mala_output.shape[0]

        samples_with_burn_in = mala_output[:,int(previous_number_of_samples/2):,:].reshape(int(previous_number_of_samples/2)*previous_number_of_chains,mala_output.shape[2])
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

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

        np.save(os.path.join(os.path.dirname(__file__), 'output','final_parallel_mala_output_' + data_filename),
        array_of_chains)

    else:
        # warm up chain
        print("New data set, initial warm up with " + str(np.int(number_of_samples*0.3)) + " samples...")
        # Initialise by minimising the function using a constrained optimization method witout gradients
        print("Optimizing using Powell's method for initial guess...")
        from scipy.optimize import minimize
        initial_guess = np.array([mean_protein,5.0,np.log(2)/30,np.log(2)/90,1.0,1.0,10.0])
        optimiser = minimize(hes_inference.calculate_log_likelihood_at_parameter_point,
                             initial_guess,
                             args=(protein_at_observations,measurement_variance),
                             bounds=np.array([(0.3*mean_protein,1.3*mean_protein),
                                              (2.0,5.0),
                                              (np.log(2)/30,np.log(2)/30),
                                              (np.log(2)/90,np.log(2)/90),
                                              (0.01,120.0),
                                              (0.01,40.0),
                                              (1.0,40.0)]),
                             method='Powell')

        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,6)] = optimiser.x[[0,1,6]]
        initial_states[:,(4,5)] = np.log(optimiser.x[[4,5]])

        print("Warming up...")
        initial_burnin_number_of_samples = np.int(0.3*number_of_samples)
        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                initial_burnin_number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                np.power(np.diag([2*mean_protein,4,9,8,39]),2),# initial variances are width of prior squared
                                                                1, # thinning rate
                                                                known_parameters))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,initial_burnin_number_of_samples,number_of_parameters))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','first_parallel_mala_output_' + data_filename),
        array_of_chains)

        print("Second warm up with " + str(int(number_of_samples*0.7)) + " samples...")
        second_burnin_number_of_samples = np.int(0.7*number_of_samples)

        samples_with_burn_in = array_of_chains[:,int(initial_burnin_number_of_samples/2):,:].reshape(int(initial_burnin_number_of_samples/2)*number_of_chains,number_of_parameters)
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # make new initial states
        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                second_burnin_number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                proposal_covariance,
                                                                1, # thinning rate
                                                                known_parameters))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all finished so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,second_burnin_number_of_samples,number_of_parameters))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','second_parallel_mala_output_' + data_filename),
        array_of_chains)

        # sample directly
        print("Now sampling directly...")
        samples_with_burn_in = array_of_chains[:,int(second_burnin_number_of_samples/2):,:].reshape(int(second_burnin_number_of_samples/2)*number_of_chains,number_of_parameters)
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # make new initial states
        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                proposal_covariance,
                                                                1, # thinning rate
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

        np.save(os.path.join(os.path.dirname(__file__), 'output','final_parallel_mala_output_' + data_filename),
        array_of_chains)

def run_mala_for_dataset_with_mRNA(data_filename,
                                   protein_at_observations,
                                   mRNA,
                                   measurement_variance,
                                   number_of_parameters,
                                   known_parameters,
                                   step_size = 1,
                                   number_of_chains = 8,
                                   number_of_samples = 80000):
    """
    A function which gives a (hopefully) decent MALA output for a given dataset of protein observations
    and knowledge of the mRNA distribution (mean and variance), with known or unknown parameters.
    If a previous output already exists, this will be used to create a proposal covariance matrix,
    otherwise one will be constructed with a two step warm-up process.
    """
    # make sure all data starts from time "zero"
    for protein in protein_at_observations:
        protein[:,0] -= protein[0,0]

    mean_protein = np.mean([np.mean(i[:,1]) for i in protein_at_observations])

    # if we already have mcmc samples, we can use them to construct a covariance matrix to make sampling better
    if os.path.exists(os.path.join(
                      os.path.dirname(__file__),
                      'output','final_parallel_mala_output_' + data_filename)):
        print("Posterior samples already exist, sampling directly without warm up...")

        mala_output = np.load(os.path.join(os.path.dirname(__file__),
                              'output','final_parallel_mala_output_' + data_filename))
        previous_number_of_samples = mala_output.shape[1]
        previous_number_of_chains = mala_output.shape[0]

        samples_with_burn_in = mala_output[:,int(previous_number_of_samples/2):,:].reshape(int(previous_number_of_samples/2)*previous_number_of_chains,mala_output.shape[2])
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala_with_mRNA,
                                                          args=(protein_at_observations,
                                                                mRNA,
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

        np.save(os.path.join(os.path.dirname(__file__), 'output','final_parallel_mala_output_' + data_filename),
        array_of_chains)

    else:
        # warm up chain
        print("New data set, initial warm up with " + str(np.int(number_of_samples*0.3)) + " samples...")
        # Initialise by minimising the function using a constrained optimization method witout gradients
        print("Optimizing using Powell's method for initial guess...")
        from scipy.optimize import minimize
        initial_guess = np.array([mean_protein,5.0,np.log(2)/30,np.log(2)/90,1.0,1.0,10.0])
        optimiser = minimize(hes_inference.calculate_log_likelihood_at_parameter_point,
                             initial_guess,
                             args=(protein_at_observations,measurement_variance),
                             bounds=np.array([(0.3*mean_protein,1.3*mean_protein),
                                              (2.0,5.0),
                                              (np.log(2)/30,np.log(2)/30),
                                              (np.log(2)/90,np.log(2)/90),
                                              (0.01,120.0),
                                              (0.01,40.0),
                                              (1.0,40.0)]),
                             method='Powell')

        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        # initial_states[:,(0,1,6)] = np.array([20000,4,20])
        initial_states[:,(0,1,6)] = optimiser.x[[0,1,6]]
        # initial_states[:,(4,5)] = np.array([1,1])
        initial_states[:,(4,5)] = np.log(optimiser.x[[4,5]])

        print("Warming up...")
        initial_burnin_number_of_samples = np.int(0.3*number_of_samples)
        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala_with_mRNA,
                                                          args=(protein_at_observations,
                                                                mRNA,
                                                                measurement_variance,
                                                                initial_burnin_number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                np.power(np.diag([2*mean_protein,4,9,8,39]),2),# initial variances are width of prior squared
                                                                1, # thinning rate
                                                                known_parameters))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,initial_burnin_number_of_samples,number_of_parameters))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','first_parallel_mala_output_' + data_filename),
        array_of_chains)

        print("Second warm up with " + str(int(number_of_samples*0.7)) + " samples...")
        second_burnin_number_of_samples = np.int(0.7*number_of_samples)

        samples_with_burn_in = array_of_chains[:,int(initial_burnin_number_of_samples/2):,:].reshape(int(initial_burnin_number_of_samples/2)*number_of_chains,number_of_parameters)
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # make new initial states
        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala_with_mRNA,
                                                          args=(protein_at_observations,
                                                                mRNA,
                                                                measurement_variance,
                                                                second_burnin_number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                proposal_covariance,
                                                                1, # thinning rate
                                                                known_parameters))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all finished so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,second_burnin_number_of_samples,number_of_parameters))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','second_parallel_mala_output_' + data_filename),
        array_of_chains)

        # sample directly
        print("Now sampling directly...")
        samples_with_burn_in = array_of_chains[:,int(second_burnin_number_of_samples/2):,:].reshape(int(second_burnin_number_of_samples/2)*number_of_chains,number_of_parameters)
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # make new initial states
        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(2,3)] = np.array([np.log(np.log(2)/30),np.log(np.log(2)/90)])
        initial_states[:,(0,1,4,5,6)] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala_with_mRNA,
                                                          args=(protein_at_observations,
                                                                mRNA,
                                                                measurement_variance,
                                                                number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                proposal_covariance,
                                                                1, # thinning rate
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

        np.save(os.path.join(os.path.dirname(__file__), 'output','final_parallel_mala_output_' + data_filename),
        array_of_chains)

def run_mala_for_dataset_parnian(data_filename,
                                 protein_at_observations,
                                 measurement_variance,
                                 number_of_parameters,
                                 known_parameters,
                                 step_size = 1,
                                 number_of_chains = 8,
                                 number_of_samples = 80000):
    """
    A function which gives a (hopefully) decent MALA output for a given dataset with known or
    unknown parameters. If a previous output already exists, this will be used to create a
    proposal covariance matrix, otherwise one will be constructed with a two step warm-up
    process.
    """
    # make sure all data starts from time "zero"
    for i in range(protein_at_observations.shape[0]):
        protein_at_observations[i,:,0] -= protein_at_observations[i,0,0]

    mean_protein = np.mean([i[j,1] for i in protein_at_observations for j in range(i.shape[0])])

    # if we already have mcmc samples, we can use them to construct a covariance matrix to make sampling better
    if os.path.exists(os.path.join(
                      os.path.dirname(__file__),
                      'output','final_parallel_mala_output_parnian_' + data_filename)):
        print("Posterior samples already exist, sampling directly without warm up...")

        mala_output = np.load(os.path.join(os.path.dirname(__file__),
                              'output','final_parallel_mala_output_parnian_' + data_filename))
        previous_number_of_samples = mala_output.shape[1]
        previous_number_of_chains = mala_output.shape[0]

        samples_with_burn_in = mala_output[:,int(previous_number_of_samples/2):,:].reshape(int(previous_number_of_samples/2)*previous_number_of_chains,mala_output.shape[2])
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,:] = np.mean(samples_with_burn_in,axis=0)

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

        np.save(os.path.join(os.path.dirname(__file__), 'output','final_parallel_mala_output_parnian_' + data_filename),
        array_of_chains)

    else:
        # warm up chain
        print("New data set, initial warm up with " + str(np.int(number_of_samples*0.3)) + " samples...")
        # Initialise by minimising the function using a constrained optimization method witout gradients
        print("Optimizing using Powell's method for initial guess...")
        from scipy.optimize import minimize
        initial_guess = np.array([mean_protein,5.0,np.log(2)/30,np.log(2)/90,1.0,1.0,10.0])
        optimiser = minimize(hes_inference.calculate_log_likelihood_at_parameter_point,
                             initial_guess,
                             args=(protein_at_observations,measurement_variance),
                             bounds=np.array([(0.3*mean_protein,1.3*mean_protein),
                                              (2.0,5.0),
                                              (np.log(2)/150,np.log(2)/10),
                                              (np.log(2)/150,np.log(2)/10),
                                              (0.01,120.0),
                                              (0.01,40.0),
                                              (1.0,40.0)]),
                             method='Powell')

        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,(0,1,6)] = optimiser.x[[0,1,6]]
        initial_states[:,(2,3,4,5)] = np.log(optimiser.x[[2,3,4,5]])

        print("Warming up with " + str(int(number_of_samples*0.3)) + " samples...")
        initial_burnin_number_of_samples = np.int(0.3*number_of_samples)
        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                initial_burnin_number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                np.power(np.diag([2*mean_protein,4,9,8,1.0,1.0,39]),2),# initial variances are width of prior squared
                                                                1, # thinning rate
                                                                known_parameters))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,initial_burnin_number_of_samples,number_of_parameters))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','first_parallel_mala_output_parnian_' + data_filename),
        array_of_chains)

        print("Second warm up with " + str(int(number_of_samples*0.7)) + " samples...")
        second_burnin_number_of_samples = np.int(0.7*number_of_samples)

        samples_with_burn_in = array_of_chains[:,int(initial_burnin_number_of_samples/2):,:].reshape(int(initial_burnin_number_of_samples/2)*number_of_chains,number_of_parameters)
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # make new initial states
        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,:] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                second_burnin_number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                proposal_covariance,
                                                                1, # thinning rate
                                                                known_parameters))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all finished so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()

        array_of_chains = np.zeros((number_of_chains,second_burnin_number_of_samples,number_of_parameters))
        for chain_index, process_result in enumerate(process_results):
            this_chain = process_result.get()
            array_of_chains[chain_index,:,:] = this_chain
        pool_of_processes.join()

        np.save(os.path.join(os.path.dirname(__file__), 'output','second_parallel_mala_output_parnian_' + data_filename),
        array_of_chains)

        # sample directly
        print("Now sampling directly...")
        samples_with_burn_in = array_of_chains[:,int(second_burnin_number_of_samples/2):,:].reshape(int(second_burnin_number_of_samples/2)*number_of_chains,number_of_parameters)
        proposal_covariance = np.cov(samples_with_burn_in.T)

        # make new initial states
        # start from mode
        initial_states = np.zeros((number_of_chains,7))
        initial_states[:,:] = np.mean(samples_with_burn_in,axis=0)

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
                                                          args=(protein_at_observations,
                                                                measurement_variance,
                                                                number_of_samples,
                                                                initial_state,
                                                                step_size,
                                                                proposal_covariance,
                                                                1, # thinning rate
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

        np.save(os.path.join(os.path.dirname(__file__), 'output','final_parallel_mala_output_parnian_' + data_filename),
        array_of_chains)
