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
import hes5
import hes_inference

class TestInference(unittest.TestCase):

	def xest_inference(self):
		## run a sample simulation to generate example protein data
		true_data = hes5.generate_langevin_trajectory(duration = 900, equilibration_time = 1000)
		np.save(os.path.join(os.path.dirname(__file__), 'output','kalman_trace.npy'),
					true_data)

		## the F constant matrix is left out for now
		protein_at_observations = true_data[0:900:10,(0,2)]
		protein_at_observations[:,1] += np.random.randn(90)*100
		protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)

		parameters = [10000,5,np.log(2)/30, np.log(2)/90, 1, 1, 29]

		## apply kalman filter to the data
		state_space_mean, state_space_variance, predicted_observation_distributions = hes_inference.kalman_filter(protein_at_observations,
																												  parameters,measurement_variance=10000)

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

		# two plots, first is only protein observations, second is likelihood
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
		parameters2 = [10000,5,np.log(2)/30, np.log(2)/90, 2, 7, 29]

		likelihood = hes_inference.calculate_log_likelihood_at_parameter_point(protein_at_observations,parameters,measurement_variance = 10000)
		likelihood2 = hes_inference.calculate_log_likelihood_at_parameter_point(protein_at_observations,parameters2,measurement_variance = 10000)
		print(likelihood)
		print(likelihood2)
		#print(np.exp(likelihood2/likelihood))

	def test_kalman_random_walk(self):

		true_data = hes5.generate_langevin_trajectory(duration = 900, equilibration_time = 1000)

		#true_values = [10000,5,np.log(2)/30,np.log(2)/90,1,1,29]
		protein_at_observations = true_data[0:900:10,(0,2)]
		protein_at_observations[:,1] += np.random.randn(90)*100
		protein_at_observations[:,1] = np.maximum(protein_at_observations[:,1],0)

		hyper_parameters = [100,100,6,0.8,1,0.2,1,0.2,4,0.25,4,0.25,15,2]
		measurement_variance = 100
		iterations = 5000
		initial_state = np.array([7000,12,1,1,1,1,10])
		covariance = np.identity(7)

		random_walk, acceptance_rate = hes_inference.kalman_random_walk(iterations,protein_at_observations,hyper_parameters,measurement_variance,0.08,covariance,initial_state)
		print(random_walk)
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
