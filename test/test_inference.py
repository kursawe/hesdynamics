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
import multiprocessing as mp
import multiprocessing.pool as mp_pool
from jitcdde import jitcdde,y,t
import time
from pdb_clone import pdbhandler; pdbhandler.register()

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

        # run the current kalman filter using the same parameters and observations, then compare
        parameters = [10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0]

        state_space_mean, state_space_variance, predicted_observation_distributions = hes_inference.kalman_filter(fixed_protein_observations,
                                                                                                                  parameters,measurement_variance=10000)
        np.testing.assert_almost_equal(state_space_mean,true_kalman_prediction_mean)
        np.testing.assert_almost_equal(state_space_variance,true_kalman_prediction_variance)
        np.testing.assert_almost_equal(predicted_observation_distributions,true_kalman_prediction_distributions)

        # If above tests fail, comment them out to look at the plot below. Could be useful for identifying problems.
#         my_figure = plt.figure()
#         plt.subplot(2,1,1)
#         plt.scatter(np.arange(0,900,10),fixed_protein_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations',zorder=4)
#         plt.plot(fixed_langevin_trace[:,0],fixed_langevin_trace[:,2],label='true protein',color='#F69454',linewidth=0.89,zorder=3)
#         plt.plot(true_kalman_prediction_mean[:,0],true_kalman_prediction_mean[:,2],label='inferred protein',color='#20948B',zorder=2)
#         plt.errorbar(true_kalman_prediction_mean[:,0],true_kalman_prediction_mean[:,2],yerr=protein_error,ecolor='#98DBC6',alpha=0.1,zorder=1)
#         plt.legend(fontsize='x-small')
#         plt.title('What the Plot should look like')
#         plt.xlabel('Time')
#         plt.ylabel('Protein Copy Numbers')
#
#         plt.subplot(2,1,2)
#         plt.scatter(np.arange(0,900,10),fixed_protein_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations',zorder=4)
#         plt.plot(fixed_langevin_trace[:,0],fixed_langevin_trace[:,2],label='true protein',color='#F69454',linewidth=0.89,zorder=3)
#         plt.plot(state_space_mean[:,0],state_space_mean[:,2],label='inferred protein',color='#20948B',zorder=2)
#         plt.errorbar(state_space_mean[:,0],state_space_mean[:,2],yerr=protein_error,ecolor='#98DBC6',alpha=0.1,zorder=1)
#         plt.legend(fontsize='x-small')
#         plt.title('What the current function gives')
#         plt.xlabel('Time')
#         plt.ylabel('Protein Copy Numbers')
#         plt.tight_layout()
#         my_figure.savefig(os.path.join(os.path.dirname(__file__),
#                                        'output','kalman_check.pdf'))

    def xest_generate_multiple_protein_observations(self):
        observation_duration  = 1800
        observation_frequency = 5
        no_of_observations    = np.int(observation_duration/observation_frequency)

        true_data = hes5.generate_langevin_trajectory(duration = observation_duration,
                                                      repression_threshold = 3407.99,
                                                      hill_coefficient = 5.17,
                                                      mRNA_degradation_rate = np.log(2)/30,
                                                      protein_degradation_rate = np.log(2)/90,
                                                      basal_transcription_rate = 15.86,
                                                      translation_rate = 1.27,
                                                      transcription_delay = 30,
                                                      equilibration_time = 1000)
        np.save(os.path.join(os.path.dirname(__file__), 'output','true_data_ps3.npy'),
                    true_data)

        ## the F constant matrix is left out for now
        protein_at_observations = true_data[:,(0,2)]
        protein_at_observations[:,1] += np.random.randn(true_data.shape[0])*100
        protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)
        np.save(os.path.join(os.path.dirname(__file__), 'output','protein_observations_90_ps3_ds1.npy'),
                    protein_at_observations[0:900:10,:])
        np.save(os.path.join(os.path.dirname(__file__), 'output','protein_observations_180_ps3_ds2.npy'),
                    protein_at_observations[0:900:5,:])
        np.save(os.path.join(os.path.dirname(__file__), 'output','protein_observations_180_ps3_ds3.npy'),
                    protein_at_observations[0:1800:10,:])
        np.save(os.path.join(os.path.dirname(__file__), 'output','protein_observations_360_ps3_ds4.npy'),
                    protein_at_observations[0:1800:5,:])

        my_figure = plt.figure()
        plt.scatter(np.arange(0,1800),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations')
        plt.title('Protein Observations')
        plt.xlabel('Time')
        plt.ylabel('Protein Copy Numbers')
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','protein_observations_ps3.pdf'))

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

        parameters = [10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0]

        ## apply kalman filter to the data
        state_space_mean, state_space_variance, predicted_observation_distributions = hes_inference.kalman_filter(protein_at_observations,parameters,
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

        #print(predicted_observation_distributions)


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

    def xest_kalman_random_walk(self):

        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        protein_at_observations = np.load(saving_path + 'kalman_test_trace_observations.npy')
        #previous_run            = np.load(saving_path + 'random_walk_500.npy')

        #previous_random_walk = previous_run[100000:,]

        #true_values = [10000,5,np.log(2)/30,np.log(2)/90,1,1,29]
        #hyper_parameters = np.array([20.0,500.0,4.0,1.0,5.0,0.01,5.0,0.01,3.0,0.333,3.0,0.333,5.0,4.5]) # gamma
        hyper_parameters = np.array([100,20100,2,4,0,1,0,1,np.log10(0.1),np.log10(60)+1,np.log10(0.1),np.log10(40)+1,5,35]) # uniform

        measurement_variance = 10000.0
        iterations = 500
        #initial_state = np.array([np.mean(previous_random_walk[:,0]),np.mean(previous_random_walk[:,1]),
        #                          np.mean(previous_random_walk[:,2]),np.mean(previous_random_walk[:,3]),
        #                          np.mean(previous_random_walk[:,4]),np.mean(previous_random_walk[:,5]),
        #                          np.mean(previous_random_walk[:,6])])
        #covariance    = np.cov(previous_random_walk.T)
        initial_state = np.array([500.0,3.0,np.log(2)/30,np.log(2)/90,0.5,0.5,17.0])
        covariance    = np.diag(np.array([25000000.0,0.1,0,0,0.034,0.034,4.5]))

        random_walk, acceptance_rate = hes_inference.kalman_random_walk(iterations,protein_at_observations,hyper_parameters,measurement_variance,0.3,covariance,initial_state)
        print('acceptance rate was', acceptance_rate)
        np.save(os.path.join(os.path.dirname(__file__), 'output','random_walk.npy'),random_walk)

        my_figure = plt.figure(figsize=(4,10))
        plt.subplot(7,1,1)
        plt.plot(np.arange(iterations),random_walk[:,0],color='#F69454')
        plt.title('repression_threshold')

        plt.subplot(7,1,2)
        plt.plot(np.arange(0,iterations),random_walk[:,1],color='#F69454')
        plt.title('hill_coefficient')

        plt.subplot(7,1,3)
        plt.plot(np.arange(0,iterations),random_walk[:,2],color='#F69454')
        plt.title('mRNA_degradation_rate')

        plt.subplot(7,1,4)
        plt.plot(np.arange(0,iterations),random_walk[:,3],color='#F69454')
        plt.title('protein_degradation_rate')

        plt.subplot(7,1,5)
        plt.plot(np.arange(0,iterations),np.power(10,random_walk[:,4]),color='#F69454')
        plt.title('basal_transcription_rate')

        plt.subplot(7,1,6)
        plt.plot(np.arange(0,iterations),np.power(10,random_walk[:,5]),color='#F69454')
        plt.title('translation_rate')

        plt.subplot(7,1,7)
        plt.plot(np.arange(0,iterations),random_walk[:,6],color='#F69454')
        plt.title('transcription_delay')

        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','random_walk.pdf'))

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

    def xest_compute_likelihood_at_multiple_parameters(self):

        saving_path             = os.path.join(os.path.dirname(__file__), 'data','kalman_test_trace')
        protein_at_observations = np.load(saving_path + '_observations.npy')
        likelihood_at_multiple_parameters = np.zeros((10,10,10,10,10))

        mRNA_degradation_rate    = np.log(2)/30
        protein_degradation_rate = np.log(2)/90

        # hyper_parameters = np.array([100,20100,2,4,0,1,0,1,np.log10(0.1),1+np.log10(65),np.log10(0.1),1+np.log10(45),4,36])
        for repression_index, repression_threshold in enumerate(np.linspace(100,20100,10)):
            for hill_index, hill_coefficient in enumerate(np.linspace(2,6,10)):
                for basal_index, basal_transcription_rate in enumerate(np.linspace(-1,np.log10(60),10)):
                    for translation_index, translation_rate in enumerate(np.linspace(-1,np.log10(40),10)):
                        for transcription_index, transcription_delay in enumerate(np.linspace(5,40,10)):
                            likelihood_at_multiple_parameters[repression_index,hill_index,basal_index,translation_index,transcription_index] = hes_inference.calculate_log_likelihood_at_parameter_point(protein_at_observations,
                                                                                                                                                model_parameters=np.array([repression_threshold,
                                                                                                                                                                           hill_coefficient,
                                                                                                                                                                           mRNA_degradation_rate,
                                                                                                                                                                           protein_degradation_rate,
                                                                                                                                                                           np.power(10,basal_transcription_rate),
                                                                                                                                                                           np.power(10,translation_rate),
                                                                                                                                                                           transcription_delay]),
                                                                                                                                                measurement_variance = 10000)

        np.save(os.path.join(os.path.dirname(__file__), 'output','likelihood_at_multiple_parameters_test.npy'),likelihood_at_multiple_parameters)

    def test_multiple_random_walk_traces_in_parallel(self):
        saving_path             = os.path.join(os.path.dirname(__file__), 'data','')
        protein_at_observations = np.load(saving_path + 'kalman_trace_observations_180_ps2_ds1.npy')
        previous_random_walk    = np.load(saving_path + 'full_random_walk_180_ps2_ds1.npy')

        #true_values = [10000,5,np.log(2)/30,np.log(2)/90,1,1,29]
        #hyper_parameters = np.array([20.0,500.0,4.0,1.0,5.0,0.01,5.0,0.01,3.0,0.333,3.0,0.333,5.0,4.5]) # gamma
        hyper_parameters = np.array([100,20100,2,4,0,1,0,1,np.log10(0.1),np.log10(60)+1,np.log10(0.1),np.log10(40)+1,5,35]) # uniform

        measurement_variance = 10000.0
        #initial_state = np.array([np.mean(previous_random_walk[:,0]),np.mean(previous_random_walk[:,1]),
        #                          np.mean(previous_random_walk[:,2]),np.mean(previous_random_walk[:,3]),
        #                          np.mean(previous_random_walk[:,4]),np.mean(previous_random_walk[:,5]),
        #                          np.mean(previous_random_walk[:,6])])
        #covariance    = np.cov(previous_random_walk.T)
        covariance     = np.diag(np.array([np.var(previous_random_walk[:,0]),np.var(previous_random_walk[:,1]),
                                           0,                                0,
                                           np.var(previous_random_walk[:,2]),np.var(previous_random_walk[:,3]),
                                           np.var(previous_random_walk[:,4])]))
        initial_states = np.array([[5000.0,2.0,np.log(2)/30,np.log(2)/90,0,0,4.0],       [500.0,3.0,np.log(2)/30,np.log(2)/90,0.5,0.5,17.0],
                                   [15000.0,2.0,np.log(2)/30,np.log(2)/90,-0.1,0.2,7.0], [500.0,5.0,np.log(2)/30,np.log(2)/90,0.5,0.4,25.0],
                                   [7000.0,3.5,np.log(2)/30,np.log(2)/90,0.2,-0.25,20.0],[19000.0,2.3,np.log(2)/30,np.log(2)/90,0,0,10.0],
                                   [1000.0,4.5,np.log(2)/30,np.log(2)/90,0.5,0.5,15.0],  [2000.0,2.0,np.log(2)/30,np.log(2)/90,0.2,0.1,15.0]])

        number_of_iterations = 100000

        pool_of_processes = mp_pool.ThreadPool(processes = number_of_cpus)
        process_results = [ pool_of_processes.apply_async(hes_inference.kalman_random_walk,
                                                          args=(number_of_iterations,protein_at_observations,hyper_parameters,measurement_variance,0.15,covariance,initial_state))
                            for initial_state in initial_states ]
        ## Let the pool know that these are all so that the pool will exit afterwards
        # this is necessary to prevent memory overflows.
        pool_of_processes.close()
        pool_of_processes.join()

        list_of_random_walks = []
        list_of_acceptance_rates = []
        for process_result in process_results:
            this_random_walk, this_acceptance_rate = process_result.get()
            list_of_random_walks.append(this_random_walk)
            list_of_acceptance_rates.append(this_acceptance_rate)
        print(list_of_acceptance_rates)

        for i in range(len(initial_states)):
            np.save(os.path.join(os.path.dirname(__file__), 'output','parallel_random_walk__180_ps2_ds1_{cap}.npy').format(cap=i),list_of_random_walks[i])

        #array_of_random_walks = np.array(list_of_random_walks)
        #self.assertEqual(array_of_random_walks.shape[0], len(initial_states))
        #self.assertEqual(array_of_random_walks.shape[1], number_of_iterations)

    def xest_kalman_filter_gif(self):

        # load in some saved observations and correct kalman filter predictions
        saving_path                          = os.path.join(os.path.dirname(__file__), 'data','kalman_test_trace')
        true_data                            = np.load(saving_path + '_true_data.npy')
        protein_at_observations              = np.load(saving_path + '_observations.npy')
        prediction_mean                      = np.load(saving_path + '_prediction_mean.npy')
        prediction_variance                  = np.load(saving_path + '_prediction_variance.npy')
        prediction_distributions             = np.load(saving_path + '_prediction_distributions.npy')

        # run the current kalman filter using the same parameters and observations, then compare
        # parameters = [10000.0,5.0,np.log(2)/30, np.log(2)/90, 1.0, 1.0, 29.0]
        #
        # state_space_mean, state_space_variance, predicted_observation_distributions = hes_inference.kalman_filter(fixed_protein_observations,
        #                                                                                                           parameters,measurement_variance=10000)

        my_figure = plt.figure()
        plt.scatter(np.arange(0,900,10),protein_at_observations[:,1],marker='o',s=4,c='#F18D9E',label='observations',zorder=4)
        plt.plot(true_data[:,0],true_data[:,2],label='true protein',color='#F69454',linewidth=0.89,zorder=3)
        plt.legend(fontsize='x-small')
        plt.title('Predicted Protein')
        plt.xlabel('Time')
        plt.ylabel('Protein Copy Numbers')
        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','kalman_test_gif_0.png'))

        for i in range(900):
            plt.scatter(i,prediction_mean[i+29,2])
            my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                           'output','kalman_test_gif_' + str(i) + '.png'))

    def xest_identify_oscillatory_parameters(self):
        saving_path = os.path.join(os.path.dirname(__file__), 'data',
                                   'sampling_results_extended')
        model_results = np.load(saving_path + '.npy' )
        prior_samples = np.load(saving_path + '_parameters.npy')

        coherence_band = [0.4,0.5]
        accepted_indices = np.where(np.logical_and(model_results[:,0]>5000, #protein number
                                    np.logical_and(model_results[:,0]<10000, #protein_number
                                    np.logical_and(model_results[:,1]>0.05,  #standard deviation
                                    np.logical_and(model_results[:,3]>coherence_band[0], #standard deviation
                                                   model_results[:,3]<coherence_band[1])))))#coherence

        my_posterior_samples = prior_samples[accepted_indices]
        my_model_results = model_results[accepted_indices]

        this_parameter = my_posterior_samples[3]
        this_results = my_model_results[3]

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

        this_trace = hes5.generate_langevin_trajectory(
                                                 duration = 1500,
                                                 repression_threshold = this_parameter[2],
                                                 mRNA_degradation_rate = np.log(2)/30.0,
                                                 protein_degradation_rate = np.log(2)/90,
                                                 transcription_delay = this_parameter[3],
                                                 basal_transcription_rate = this_parameter[0],
                                                 translation_rate = this_parameter[1],
                                                 initial_mRNA = 10,
                                                 hill_coefficient = this_parameter[4],
                                                 initial_protein = this_parameter[2],
                                                 equilibration_time = 1000)

        my_figure = plt.figure(figsize= (2.5,1.9))
        plt.plot(this_trace[:,0], this_trace[:,2], lw = 1)
        plt.xlabel("Time [min]")
        plt.ylabel("Hes expression")
        plt.gca().locator_params(axis='x', tight = True, nbins=5)

        plt.tight_layout()
        file_name = os.path.join(os.path.dirname(__file__), 'output',
                                   'example_oscillatory_trace_for_data')
        plt.savefig(file_name + '.pdf')
