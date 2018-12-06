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
import time

# make sure we find the right python module
sys.path.append(os.path.join(os.path.dirname(__file__),'..','src'))
import hes5
import hes_inference

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

    def test_kalman_random_walk(self):

        true_data = hes5.generate_langevin_trajectory(duration = 900, equilibration_time = 1000)

        saving_path  = os.path.join(os.path.dirname(__file__), 'data','random_walk')
        previous_run = np.load(saving_path + '.npy')

        #true_values = [10000,5,np.log(2)/30,np.log(2)/90,1,1,29]
        protein_at_observations = true_data[0:900:10,(0,2)]
        protein_at_observations[:,1] += np.random.randn(90)*100
        protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)

        hyper_parameters = np.array([5.0,2000.0,2.0,2.5,5.0,0.1,5.0,0.1,3.0,0.333,3.0,0.333,3.0,10.0]) # gamma

        measurement_variance = 10000.0
        iterations = 60000
        #initial_state = np.array([np.mean(previous_run[1000:,0]),np.mean(previous_run[1000:,1]),
        #                          np.mean(previous_run[1000:,2]),np.mean(previous_run[1000:,3]),
        #                          np.mean(previous_run[1000:,4]),np.mean(previous_run[1000:,5]),
        #                          np.mean(previous_run[1000:,6])])
        #covariance = np.cov(previous_run.T)
        initial_state = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0])
        covariance = np.diag([900000.0,2,0.06,0.06,0.1,0.1,10.0])

        random_walk, acceptance_rate = hes_inference.kalman_random_walk(iterations,protein_at_observations,hyper_parameters,measurement_variance,0.6,covariance,initial_state)
        print(acceptance_rate)
        np.save(os.path.join(os.path.dirname(__file__), 'output','random_walk.npy'),
                    random_walk)

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
        plt.plot(np.arange(0,iterations),random_walk[:,4],color='#F69454')
        plt.title('basal_transcription_rate')

        plt.subplot(7,1,6)
        plt.plot(np.arange(0,iterations),random_walk[:,5],color='#F69454')
        plt.title('translation_rate')

        plt.subplot(7,1,7)
        plt.plot(np.arange(0,iterations),random_walk[:,6],color='#F69454')
        plt.title('transcription_delay')

        plt.tight_layout()
        my_figure.savefig(os.path.join(os.path.dirname(__file__),
                                       'output','kalman_random_walk.pdf'))
